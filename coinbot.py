# -*- coding: utf-8 -*-
import ccxt
import os
import time
import json
import logging
import threading
import pandas as pd
import re
import random
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------
# [설정 & 상수]
# ---------------------------------------------------------
load_dotenv()
CONFIG_FILE = "config.json"
MODEL_NAME = "gemini-2.5-flash-lite"
LOG_FILE_NAME = "bot_master.log"

# 타임아웃 설정
AI_TIMEOUT = 30 

BASE_PROMPT = "Act as a Conservative Scalper AI for {symbol} (1m chart)."

# [전략 저장소 - V6.7 업데이트: 추격 매수 방지 (Anti-FOMO)]
PROMPTS = {
    "ultra_safe": """
        {base}
        
        **CONTEXT**: 
        - Current Leverage: {leverage}x
        - Target: {target_move}% | Stop Loss: {sl_move}%
        
        **STRATEGY**: "Anti-FOMO Sniper" (Buy the start, NOT the peak)
        
        [Strict Entry Rules]
        1. **Trend Filter (EMA50)**:
           - LONG: Price > EMA50.
           - SHORT: Price < EMA50.
        
        2. **Distance Check (CRITICAL)**:
           - **Don't Chase!** If 'Dist_from_EMA' is > 0.3%, it's too late. Output "wait".
           - We only enter when price is CLOSE to the EMA50 line (The start of the move).

        3. **Volume & Momentum**: 
           - 'Vol_Ratio' > 2.0.
           - MACD supports the direction.

        4. **RSI Safety**:
           - LONG: RSI < 70.
           - SHORT: RSI > 30.
        
        [Decision Logic]
        - **REJECT**: If Price is too far from EMA50 (Checking 'Dist_from_EMA').
        - **REJECT**: If Price vs EMA50 mismatch.
        - **REJECT**: If Volume is weak.
        
        [Final Output]
        - **LONG**: Score >= 95 + Close to EMA.
        - **SHORT**: Score >= 95 + Close to EMA.
        - **WAIT**: If price extended or signals unclear.
        
        Output JSON: {{"decision": "long/short/wait", "reason": "Mention Distance from EMA"}}
    """
}

# ---------------------------------------------------------
# [로그 설정]
# ---------------------------------------------------------
class AFCLogFilter(logging.Filter):
    def filter(self, record): return "AFC is enabled" not in record.getMessage()

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S', handlers=[logging.FileHandler(LOG_FILE_NAME, encoding='utf-8'), logging.StreamHandler()])
logging.getLogger().handlers[0].addFilter(AFCLogFilter())
logging.getLogger().handlers[1].addFilter(AFCLogFilter())
logger = logging.getLogger("V6_BOT")

for lib in ["httpx", "httpcore", "google", "urllib3"]: logging.getLogger(lib).setLevel(logging.ERROR)

# ---------------------------------------------------------
# [봇 클래스]
# ---------------------------------------------------------
class TradingBot(threading.Thread):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        try:
            self.symbol = config['symbol']
            self.leverage = config['leverage']
            self.amount = config['amount']
            self.target_roe = config['target_roe']
            self.stop_loss_roe = config['stop_loss_roe']
            self.strategy = config['strategy']
        except KeyError as e:
            logger.error(f"❌ Config Error: Missing {e}")
            raise e

        self.running = True
        self.fix_history = {} 
        
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.exchange = ccxt.binance({
            'apiKey': os.getenv("BINANCE_API_KEY"),
            'secret': os.getenv("BINANCE_SECRET_KEY"),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    def run(self):
        logger.info(f"🚀 [{self.symbol}] V6.7 Anti-FOMO Start | TP:{self.target_roe}% SL:{self.stop_loss_roe}%")
        self._init_exchange_settings()
        
        loop_count = 0
        while self.running:
            try:
                position = self._get_position()
                
                if position:
                    self._handle_active_position(position, loop_count)
                else:
                    self._scan_market_for_entry(loop_count)
                
                time.sleep(random.uniform(4, 6))
                loop_count += 1

            except Exception as e:
                self._handle_error(e)

    def _init_exchange_settings(self):
        try:
            self.exchange.load_markets()
            self.exchange.set_margin_mode('isolated', self.symbol)
            self.exchange.set_leverage(self.leverage, self.symbol)
        except: pass

    def _handle_error(self, e):
        msg = str(e)
        if "Code: -4164" in msg: logger.error(f"❌ [{self.symbol}] 잔고 부족 (Min Notional)")
        elif "503" not in msg and "429" not in msg: logger.error(f"⚠️ [{self.symbol}] Network: {e}")
        time.sleep(10)

    # --- 포지션 관리 ---
    def _handle_active_position(self, position, loop_count):
        if loop_count % 6 == 0:
            self._ensure_orders(position)
            self._log_pnl(position)

    def _log_pnl(self, position):
        try:
            pnl = float(position['unrealizedPnl'])
            roi = (pnl / float(position['initialMargin'])) * 100
            icon = "🔴" if pnl < 0 else "🟢"
            logger.info(f"{icon} [{self.symbol}] Hold | ROI: {roi:.2f}% | PnL: ${pnl:.4f}")
        except: pass

    # --- 시장 탐색 ---
    def _scan_market_for_entry(self, loop_count):
        if self.exchange.fetch_open_orders(self.symbol):
            self.exchange.cancel_all_orders(self.symbol)

        market_data = self._get_market_data()
        if not market_data: return

        decision = self._ask_llm(market_data)
        
        if decision in ['long', 'short']:
            self._execute_entry(decision)
        elif loop_count % 3 == 0:
            logger.info(f"👀 [{self.symbol}] Analyzing... (Wait)")

    # --- 데이터 수집 (이격도 계산 추가) ---
    def _get_market_data(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            close = df['c']
            
            # 지표 1: 이평선
            ema_20 = close.ewm(span=20).mean()
            ema_50 = close.ewm(span=50).mean()
            
            # [신규] 이격도 계산 (현재가와 EMA50의 거리 %)
            curr_price = close.iloc[-1]
            curr_ema50 = ema_50.iloc[-1]
            dist_ema = abs(curr_price - curr_ema50) / curr_ema50 * 100
            
            # 지표 2: 볼린저 밴드
            std = close.rolling(20).std()
            upper = ema_20 + (std * 2)
            lower = ema_20 - (std * 2)
            
            # 지표 3: RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain/loss))
            
            # 지표 4: MACD
            exp12 = close.ewm(span=12, adjust=False).mean()
            exp26 = close.ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            
            # 거래량
            vol_ma = df['v'].rolling(20).mean().iloc[-1]
            cur_vol = df['v'].iloc[-1]
            vol_ratio = cur_vol / vol_ma if vol_ma > 0 else 0
            
            curr = df.iloc[-1]
            candle_body = abs(curr['c'] - curr['o'])
            upper_wick = curr['h'] - max(curr['c'], curr['o'])
            lower_wick = min(curr['c'], curr['o']) - curr['l']
            
            return f"""
            Price: {curr['c']}
            Trends: EMA50={curr_ema50:.4f}
            Dist_from_EMA: {dist_ema:.3f}% (Must be < 0.3% to enter)
            MACD: Hist={macd_hist.iloc[-1]:.4f}
            RSI(14): {rsi.iloc[-1]:.1f}
            Volume Ratio: {vol_ratio:.2f}x
            """
        except: return None

    # --- AI 판단 ---
    def _ask_llm(self, data):
        target_price_move = self.target_roe / self.leverage
        sl_price_move = self.stop_loss_roe / self.leverage

        prompt_template = PROMPTS[self.strategy]
        
        final_prompt = prompt_template.format(
            base=BASE_PROMPT.format(symbol=self.symbol),
            leverage=self.leverage,
            target_move=f"{target_price_move:.2f}",
            sl_move=f"{sl_price_move:.2f}"
        ) + f"\nData:\n{data}"
        
        try:
            res = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=final_prompt)])],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
            )
            
            text_res = res.text.strip()
            if text_res.startswith("```json"):
                text_res = text_res[7:-3].strip()
            elif text_res.startswith("```"):
                text_res = text_res[3:-3].strip()
                
            parsed = json.loads(text_res)
            decision = parsed.get("decision", "wait").lower()
            
            if decision != "wait":
                reason = parsed.get("reason", "No reason provided")
                logger.info(f"💡 [{self.symbol}] AI Insight: {reason}")
            
            return decision
        except Exception as e:
            logger.error(f"AI Error: {e}")
            return "wait"

    # --- 진입 실행 ---
    def _execute_entry(self, side):
        try:
            self.exchange.set_margin_mode('isolated', self.symbol) 
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            raw_amount = (self.amount * self.leverage / ticker['last'])
            amount = self.exchange.amount_to_precision(self.symbol, raw_amount)
            
            order_func = self.exchange.create_market_buy_order if side == 'long' else self.exchange.create_market_sell_order
            order_func(self.symbol, amount)
            
            time.sleep(2)
            
            pos = self._get_position()
            if pos:
                self.fix_history[self.symbol] = 0
                self._ensure_orders(pos)
                logger.info(f"⚡ [{self.symbol}] Entry {side.upper()} Done")
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] Entry Fail: {e}")

    def _ensure_orders(self, position):
        if time.time() - self.fix_history.get(self.symbol, 0) < 60: return

        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            side = position['side']
            
            has_tp = any('limit' in o['type'].lower() for o in orders)
            has_sl = any('stop' in o['type'].lower() for o in orders)

            if has_tp and has_sl: return

            logger.warning(f"🔧 [{self.symbol}] Fixing Orders")
            self.fix_history[self.symbol] = time.time()
            self.exchange.cancel_all_orders(self.symbol)
            time.sleep(2)
            
            entry = float(position['entryPrice'])
            amt = float(position['contracts'])
            
            tp_rate = (self.target_roe / self.leverage) / 100
            sl_rate = (self.stop_loss_roe / self.leverage) / 100
            
            if side == 'long':
                tp_price = self.exchange.price_to_precision(self.symbol, entry * (1 + tp_rate))
                sl_price = self.exchange.price_to_precision(self.symbol, entry * (1 - sl_rate))
                
                self.exchange.create_limit_sell_order(self.symbol, amt, tp_price, {'reduceOnly': True})
                self.exchange.create_order(self.symbol, 'STOP_MARKET', 'sell', amt, None, {'stopPrice': sl_price, 'closePosition': True})
            else:
                tp_price = self.exchange.price_to_precision(self.symbol, entry * (1 - tp_rate))
                sl_price = self.exchange.price_to_precision(self.symbol, entry * (1 + sl_rate))
                
                self.exchange.create_limit_buy_order(self.symbol, amt, tp_price, {'reduceOnly': True})
                self.exchange.create_order(self.symbol, 'STOP_MARKET', 'buy', amt, None, {'stopPrice': sl_price, 'closePosition': True})
                
            logger.info(f"✅ [{self.symbol}] Orders Fixed")
        except Exception as e:
            if "-4130" not in str(e): logger.error(f"⚠️ Fix Fail: {e}")

    def _get_position(self):
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for p in positions:
                if float(p['contracts']) != 0: return p
            return None
        except: return None

if __name__ == "__main__":
    try:
        with open(CONFIG_FILE, 'r') as f: config_list = json.load(f)
        
        logger.info(f"🔥 V6.7 Anti-FOMO BOT STARTED ({len(config_list)} pairs)")
        
        threads = [TradingBot(cfg) for cfg in config_list]
        for t in threads: 
            t.start()
            time.sleep(random.uniform(0.5, 1.5)) 
            
        for t in threads: t.join()

    except KeyboardInterrupt: logger.info("👋 Exit")
    except Exception as e: logger.error(f"Main Error: {e}")