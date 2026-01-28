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

BASE_PROMPT = "Act as a Conservative Scalper AI for {symbol} (1m chart)."

# [전략 저장소]
PROMPTS = {
    "ultra_safe": """
        {base}
        **MINDSET**: You are a "Cowardly Sniper". You only pull the trigger when the target is strictly locked on.
        **GOAL**: Avoid losses at all costs. Missing a trade is better than losing money.
        
        [Strict Entry Rules - NO EXCEPTIONS]
        1. **Volume is Mandatory**: 'Vol_Ratio' MUST be > 2.0. If volume is low, output "wait".
        2. **Price Extremes**: 
           - LONG: Price must be at/below Lower Bollinger Band.
           - SHORT: Price must be at/above Upper Bollinger Band.
        3. **Trend Check**: 
           - Do not catch a falling knife. Look for a 'Wick' (Rejection) candles.
        
        [Decision Logic]
        - Evaluate the setup score (0-100).
        - **Deduct points** if RSI is neutral (30-70).
        - **Deduct points** if Volume Ratio < 2.0.
        
        [Final Output]
        - **LONG**: Only if Score >= 95.
        - **SHORT**: Only if Score >= 95.
        - **WAIT**: If Score < 95. (Most of the time, you should wait).
        
        Output JSON: {{"decision": "long/short/wait", "reason": "Explain why score is high/low"}}
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
        
        # 설정 검증 및 로드
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
        
        # API 클라이언트 초기화
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.exchange = ccxt.binance({
            'apiKey': os.getenv("BINANCE_API_KEY"),
            'secret': os.getenv("BINANCE_SECRET_KEY"),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    def run(self):
        logger.info(f"🚀 [{self.symbol}] V6.1 Start | TP:{self.target_roe}% SL:{self.stop_loss_roe}%")
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
        if "Code: -4164" in msg: logger.error(f"❌ [{self.symbol}] 잔고 부족 (Min Notional)")
        elif "503" not in msg and "429" not in msg: logger.error(f"⚠️ [{self.symbol}] Error: {e}")
        time.sleep(10)

    # --- 포지션 관리 ---
    def _handle_active_position(self, position, loop_count):
        # 30초마다 주문 상태 점검 (TP/SL 누락 방지)
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
        # 이전 미체결 주문이 있다면 정리 (깨끗한 상태 유지)
        if self.exchange.fetch_open_orders(self.symbol):
            self.exchange.cancel_all_orders(self.symbol)

        market_data = self._get_market_data()
        if not market_data: return

        decision = self._ask_llm(market_data)
        
        if decision in ['long', 'short']:
            self._execute_entry(decision)
        elif loop_count % 3 == 0:
            logger.info(f"👀 [{self.symbol}] Analyzing... (Wait)")

    # --- 데이터 수집 (지표 추가됨) ---
    def _get_market_data(self):
        try:
            # 100개의 캔들 데이터 가져오기
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            close = df['c']
            
            # [지표 계산]
            ema_20 = close.ewm(span=20).mean()
            ema_50 = close.ewm(span=50).mean() # 중기 추세용 추가
            
            # 볼린저 밴드
            std = close.rolling(20).std()
            upper = ema_20 + (std * 2)
            lower = ema_20 - (std * 2)
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain/loss))
            
            # 거래량 비율
            vol_ma = df['v'].rolling(20).mean().iloc[-1]
            cur_vol = df['v'].iloc[-1]
            vol_ratio = cur_vol / vol_ma if vol_ma > 0 else 0
            
            curr = df.iloc[-1]
            
            # 캔들 모양 수치화
            candle_body = abs(curr['c'] - curr['o'])
            upper_wick = curr['h'] - max(curr['c'], curr['o'])
            lower_wick = min(curr['c'], curr['o']) - curr['l']
            
            # AI에게 제공할 데이터 포맷
            return f"""
            Price: {curr['c']}
            Trends: EMA20={ema_20.iloc[-1]:.4f}, EMA50={ema_50.iloc[-1]:.4f}
            RSI(14): {rsi.iloc[-1]:.1f}
            Bollinger: Upper={upper.iloc[-1]:.4f}, Lower={lower.iloc[-1]:.4f}
            Volume Ratio: {vol_ratio:.2f}x (Critical check: MUST be > 2.0)
            Candle Shape: Body={candle_body:.4f}, UpperWick={upper_wick:.4f}, LowerWick={lower_wick:.4f}
            """
        except: return None

    # --- AI 판단 ---
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

    # --- 진입 및 주문 복구 ---
    def _execute_entry(self, side):
        try:
            self.exchange.set_margin_mode('isolated', self.symbol) 
            ticker = self.exchange.fetch_ticker(self.symbol)
            # 금액 계산 (USDT -> Coin 수량)
            amount = self.exchange.amount_to_precision(self.symbol, (self.amount * self.leverage / ticker['last']))
            
            # 시장가 진입
            order_func = self.exchange.create_market_buy_order if side == 'long' else self.exchange.create_market_sell_order
            order_func(self.symbol, amount)
            
            time.sleep(2)
            
            # 진입 후 즉시 TP/SL 설정
            pos = self._get_position()
            if pos:
                self.fix_history[self.symbol] = 0 # 쿨다운 리셋
                self._ensure_orders(pos)
                logger.info(f"⚡ [{self.symbol}] Entry {side.upper()} Done")
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] Entry Fail: {e}")

    def _ensure_orders(self, position):
        # 60초 쿨다운 (API 과부하 방지)
        if time.time() - self.fix_history.get(self.symbol, 0) < 60: return

        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            side = position['side']
            
            # 기존 주문 검증
            has_tp = any('limit' in o['type'].lower() for o in orders)
            has_sl = any('stop' in o['type'].lower() for o in orders)

            if has_tp and has_sl: return # 정상

            # 주문 재설정
            logger.warning(f"🔧 [{self.symbol}] Fixing Orders (SL:{has_sl}, TP:{has_tp})")
            self.fix_history[self.symbol] = time.time()
            self.exchange.cancel_all_orders(self.symbol)
            time.sleep(2)
            
            entry = float(position['entryPrice'])
            amt = float(position['contracts'])
            
            # Config 값 적용
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
            # -4130 오류(주문 금액 너무 작음 등)는 로그만 찍고 넘어감
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
        
        logger.info(f"🔥 V6.1 BOT STARTED ({len(config_list)} pairs)")
        
        threads = [TradingBot(cfg) for cfg in config_list]
        for t in threads: 
            t.start()
            time.sleep(0.5) # API 요청 분산
            
        for t in threads: t.join()

    except KeyboardInterrupt: logger.info("👋 Exit")
    except Exception as e: logger.error(f"Main Error: {e}")