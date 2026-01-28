# -*- coding: utf-8 -*-
import ccxt
import os
import time
import json
import logging
import threading
import pandas as pd
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

BASE_PROMPT = "Act as a High-Frequency Trading AI for {symbol} (1m chart)."

# [전략 저장소]
PROMPTS = {
    "ultra_safe": """
        {base}
        **OBJECTIVE**: Secure a quick profit with MAXIMUM safety.
        **METHOD**: "Confluence Trading" (Seeking overlapping signals).
        
        [Data Analysis Task]
        Synthesize the following indicators into a single trading decision. Look for the STORY they tell together.
        
        1. **Volume Analysis**: Is smart money entering? (Look at Vol_Ratio). A reversal without volume is a trap.
        2. **Volatility Context**: Is the price Overextended relative to Bollinger Bands? (Mean Reversion logic).
        3. **Momentum Health**: Is RSI signaling a climax (Overbought/Oversold)?
        
        [Evaluation Process]
        - **Rate the Setup (0-100)**:
          * Price touching Outer Band? (+20)
          * RSI Extreme (<25 or >75)? (+30)
          * Volume Spiking (>1.5x)? (+30)
          * Counter-trend Wick detected? (+20)
        
        [Final Decision]
        - **LONG**: Only if Total Score > 90 AND Sentiment is Bullish Reversal.
        - **SHORT**: Only if Total Score > 90 AND Sentiment is Bearish Reversal.
        - **WAIT**: If Score < 90 or market is ambiguous.
        
        Output JSON: {{"decision": "long/short/wait", "reason": "Brief reason"}}
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
        
        # 설정 검증
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
        self.fix_history = {} # 주문 복구 쿨다운 관리
        
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.exchange = ccxt.binance({
            'apiKey': os.getenv("BINANCE_API_KEY"),
            'secret': os.getenv("BINANCE_SECRET_KEY"),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    def run(self):
        logger.info(f"🚀 [{self.symbol}] V6 Start | {self.strategy.upper()} | TP:{self.target_roe}% SL:{self.stop_loss_roe}%")
        self._init_exchange_settings()
        
        loop_count = 0
        while self.running:
            try:
                position = self._get_position()
                
                if position:
                    self._handle_active_position(position, loop_count)
                else:
                    self._scan_market_for_entry(loop_count)
                
                time.sleep(5)
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
        if "Code: -4164" in msg: logger.error(f"❌ [{self.symbol}] Insufficient Funds")
        elif "503" not in msg and "429" not in msg: logger.error(f"⚠️ [{self.symbol}] Error: {e}")
        time.sleep(10)

    # --- 포지션 관리 ---
    def _handle_active_position(self, position, loop_count):
        # 30초마다 주문 점검
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

    # --- 신규 진입 탐색 ---
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

    # --- 데이터 수집 & AI ---
    def _get_market_data(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            close = df['c']
            ema = close.ewm(span=20).mean()
            std = close.rolling(20).std()
            upper = ema + (std * 2)
            lower = ema - (std * 2)
            
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain/loss))
            
            vol_ma = df['v'].rolling(20).mean().iloc[-1]
            cur_vol = df['v'].iloc[-1]
            vol_ratio = cur_vol / vol_ma if vol_ma > 0 else 0
            
            curr = df.iloc[-1]
            
            candle_body = abs(curr['c'] - curr['o'])
            upper_wick = curr['h'] - max(curr['c'], curr['o'])
            lower_wick = min(curr['c'], curr['o']) - curr['l']
            
            return f"""
            Price: {curr['c']}
            RSI: {rsi.iloc[-1]:.1f}
            BB: Upper={upper.iloc[-1]:.4f}, Lower={lower.iloc[-1]:.4f}
            Volume Ratio: {vol_ratio:.2f}x (vs Avg)
            Candle Shape: Body={candle_body:.4f}, UpperWick={upper_wick:.4f}, LowerWick={lower_wick:.4f}
            """
        except: return None

    def _ask_llm(self, data):
        prompt_template = PROMPTS[self.strategy]
        final_prompt = prompt_template.format(base=BASE_PROMPT.format(symbol=self.symbol)) + f"\nData:\n{data}"
        
        try:
            res = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=final_prompt)])],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
            )
            parsed = json.loads(res.text)
            decision = parsed.get("decision", "wait").lower()
            
            if decision != "wait":
                reason = parsed.get("reason", "No reason provided")
                logger.info(f"💡 [{self.symbol}] AI Insight: {reason}")
            
            return decision
        except: return "wait"

    # --- 주문 실행 & 복구 ---
    def _execute_entry(self, side):
        try:
            self.exchange.set_margin_mode('isolated', self.symbol) 
            ticker = self.exchange.fetch_ticker(self.symbol)
            amount = self.exchange.amount_to_precision(self.symbol, (self.amount * self.leverage / ticker['last']))
            
            order_func = self.exchange.create_market_buy_order if side == 'long' else self.exchange.create_market_sell_order
            order_func(self.symbol, amount)
            
            time.sleep(2)
            pos = self._get_position()
            if pos:
                self.fix_history[self.symbol] = 0 # 쿨다운 초기화
                self._ensure_orders(pos)
                logger.info(f"⚡ [{self.symbol}] Entry {side.upper()} Done")
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] Entry Fail: {e}")

    def _ensure_orders(self, position):
        # 60초 쿨다운
        if time.time() - self.fix_history.get(self.symbol, 0) < 60: return

        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            side = position['side']
            
            # 주문 타입으로 단순 확인 (limit, stop 포함 여부)
            has_tp = any('limit' in o['type'].lower() for o in orders)
            has_sl = any('stop' in o['type'].lower() for o in orders)

            if has_tp and has_sl: return 

            logger.warning(f"🔧 [{self.symbol}] Fixing Orders (SL:{has_sl}, TP:{has_tp})")
            self.fix_history[self.symbol] = time.time()
            self.exchange.cancel_all_orders(self.symbol)
            time.sleep(2)
            
            entry = float(position['entryPrice'])
            amt = float(position['contracts'])
            
            tp_rate = (self.target_roe / self.leverage) / 100
            sl_rate = (self.stop_loss_roe / self.leverage) / 100
            
            if side == 'long':
                tp, sl = entry * (1 + tp_rate), entry * (1 - sl_rate)
                self.exchange.create_limit_sell_order(self.symbol, amt, tp, {'reduceOnly': True})
                self.exchange.create_order(self.symbol, 'STOP_MARKET', 'sell', amt, None, {'stopPrice': sl, 'closePosition': True})
            else:
                tp, sl = entry * (1 - tp_rate), entry * (1 + sl_rate)
                self.exchange.create_limit_buy_order(self.symbol, amt, tp, {'reduceOnly': True})
                self.exchange.create_order(self.symbol, 'STOP_MARKET', 'buy', amt, None, {'stopPrice': sl, 'closePosition': True})
                
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

# ---------------------------------------------------------
# [Main]
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        with open(CONFIG_FILE, 'r') as f: config_list = json.load(f)
        
        logger.info(f"🔥 V6 BOT STARTED ({len(config_list)} pairs)")
        
        threads = [TradingBot(cfg) for cfg in config_list]
        for t in threads: 
            t.start()
            time.sleep(0.5) # API 요청 분산
            
        for t in threads: t.join()

    except KeyboardInterrupt: logger.info("👋 Exit")
    except Exception as e: logger.error(f"Main Error: {e}")