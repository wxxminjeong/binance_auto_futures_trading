# -*- coding: utf-8 -*-
import ccxt
import os
import time
import json
import logging
import threading
import pandas as pd
import random
import re
from types import SimpleNamespace
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Configuration & Constants ---
load_dotenv()
CONFIG_FILE = "config.json"
MODEL_NAME = "gemini-2.5-flash-lite"
LOG_FILE_NAME = "bot_master.log"
AI_TIMEOUT = 30

# JSON Extraction Pattern
JSON_PATTERN = re.compile(r"```json\s*(.*?)\s*```|```\s*(.*?)\s*```", re.DOTALL)

BASE_PROMPT = "Act as a Conservative Scalper AI for {symbol} (1m chart)."

# Strategy Prompts
PROMPTS = {
    "ultra_safe": """
        {base}
        
        **CONTEXT**: 
        - Current Leverage: {leverage}x
        - Target Real Move: {target_move}% | Stop Loss Risk: {sl_move}%
        
        **STRATEGY**: "Anti-FOMO Sniper" (Buy the start, NOT the peak)
        
        [Strict Entry Rules]
        1. **Trend Filter (EMA50)**:
           - LONG: Price > EMA50.
           - SHORT: Price < EMA50.
        
        2. **Distance Check (CRITICAL)**:
           - **Don't Chase!** If 'Dist_from_EMA' is > 0.3%, it's too late. Output "wait".
           - We only enter when price is CLOSE to the EMA50 line.

        3. **Volume & Momentum**: 
           - 'Vol_Ratio' MUST be > 2.0.
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
        
        Output JSON format: 
        {{
            "decision": "long/short/wait", 
            "reason": "Mention Distance from EMA",
            "checked_vol": "value",
            "checked_dist": "value"
        }}
    """
}

# --- Logging Setup ---
class AFCLogFilter(logging.Filter):
    def filter(self, record): return "AFC is enabled" not in record.getMessage()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(message)s', 
    datefmt='%H:%M:%S', 
    handlers=[
        logging.FileHandler(LOG_FILE_NAME, encoding='utf-8'), 
        logging.StreamHandler()
    ]
)
logging.getLogger().handlers[0].addFilter(AFCLogFilter())
logging.getLogger().handlers[1].addFilter(AFCLogFilter())
logger = logging.getLogger("BOT")

for lib in ["httpx", "httpcore", "google", "urllib3"]: 
    logging.getLogger(lib).setLevel(logging.ERROR)


class TradingBot(threading.Thread):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.cfg = SimpleNamespace(**config)
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
        """Main loop: initializes settings and alternates between position management and market scanning."""
        logger.info(f"🚀 [{self.cfg.symbol}] Bot Started | TP:{self.cfg.target_roe}% SL:{self.cfg.stop_loss_roe}%")
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
            self.exchange.set_margin_mode('isolated', self.cfg.symbol)
            self.exchange.set_leverage(self.cfg.leverage, self.cfg.symbol)
        except Exception: 
            pass

    def _handle_error(self, e: Exception):
        msg = str(e)
        if "Code: -4164" in msg: 
            logger.error(f"❌ [{self.cfg.symbol}] Insufficient Balance (Min Notional)")
        elif "503" not in msg and "429" not in msg: 
            logger.error(f"⚠️ [{self.cfg.symbol}] Error: {e}")
        time.sleep(10)

    def _handle_active_position(self, position: Dict, loop_count: int):
        """Monitors active positions and ensures TP/SL orders exist."""
        if loop_count % 6 == 0:  # Check every ~30s
            self._ensure_orders(position)
            self._log_pnl(position)

    def _log_pnl(self, position: Dict):
        try:
            pnl = float(position['unrealizedPnl'])
            roi = (pnl / float(position['initialMargin'])) * 100
            icon = "🔴" if pnl < 0 else "🟢"
            logger.info(f"{icon} [{self.cfg.symbol}] Hold | ROI: {roi:.2f}% | PnL: ${pnl:.4f}")
        except Exception: 
            pass

    def _scan_market_for_entry(self, loop_count: int):
        """Analyzes market data via LLM and executes entry if signals match."""
        market_data = self._get_market_data()
        if not market_data: return

        decision = self._ask_llm(market_data)
        
        if decision in ['long', 'short']:
            # Cancel stale orders before entry
            if self.exchange.fetch_open_orders(self.cfg.symbol):
                self.exchange.cancel_all_orders(self.cfg.symbol)
            self._execute_entry(decision)
            
        elif loop_count % 3 == 0:
            logger.info(f"👀 [{self.cfg.symbol}] Analyzing... (Wait)")

    def _get_market_data(self) -> Optional[str]:
        """Fetches OHLCV and calculates indicators (EMA, RSI, Bollinger, MACD)."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.cfg.symbol, '1m', limit=300)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            close = df['c']
            
            # Indicators
            ema_50 = close.ewm(span=50).mean()
            ema_200 = close.ewm(span=200).mean()
            
            curr_price = close.iloc[-1]
            curr_ema50 = ema_50.iloc[-1]
            
            # Anti-FOMO: Check distance from EMA50
            dist_ema50 = abs(curr_price - curr_ema50) / curr_ema50 * 100
            
            # Bollinger Bands
            std = close.rolling(20).std()
            upper = close.rolling(20).mean() + (std * 2)
            lower = close.rolling(20).mean() - (std * 2)
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain/loss))
            
            # MACD
            exp12 = close.ewm(span=12, adjust=False).mean()
            exp26 = close.ewm(span=26, adjust=False).mean()
            macd_hist = (exp12 - exp26) - (exp12 - exp26).ewm(span=9, adjust=False).mean()
            
            # Volume Ratio
            vol_ma = df['v'].rolling(20).mean().iloc[-1]
            cur_vol = df['v'].iloc[-1]
            vol_ratio = cur_vol / vol_ma if vol_ma > 0 else 0
            
            curr = df.iloc[-1]
            candle_info = f"Body={abs(curr['c']-curr['o']):.4f}"

            return f"""
            Symbol: {self.cfg.symbol}
            Price: {curr_price}
            Trends: EMA50={curr_ema50:.4f}, EMA200={ema_200.iloc[-1]:.4f}
            Dist_from_EMA50: {dist_ema50:.3f}% (Limit: < 0.3%)
            Volume Ratio: {vol_ratio:.2f}x (Limit: > 2.0x)
            RSI(14): {rsi.iloc[-1]:.1f}
            MACD Hist: {macd_hist.iloc[-1]:.4f}
            Bollinger: Up={upper.iloc[-1]:.4f}, Low={lower.iloc[-1]:.4f}
            Candle: {candle_info}
            """
        except Exception: 
            return None

    def _ask_llm(self, data: str) -> str:
        """Sends market data to Gemini and parses the JSON decision."""
        target_price_move = self.cfg.target_roe / self.cfg.leverage
        sl_price_move = self.cfg.stop_loss_roe / self.cfg.leverage

        try:
            final_prompt = PROMPTS[self.cfg.strategy].format(
                base=BASE_PROMPT.format(symbol=self.cfg.symbol),
                leverage=self.cfg.leverage,
                target_move=f"{target_price_move:.2f}",
                sl_move=f"{sl_price_move:.2f}"
            ) + f"\nData:\n{data}"
        except KeyError as e:
            logger.error(f"❌ Prompt Format Error: {e}")
            return "wait"
        
        try:
            res = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=final_prompt)])],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
            )
            
            text_res = res.text.strip()
            match = JSON_PATTERN.search(text_res)
            if match:
                text_res = match.group(1) or match.group(2)
                
            parsed = json.loads(text_res)
            decision = parsed.get("decision", "wait").lower()
            
            if decision != "wait":
                check_log = f"Vol:{parsed.get('checked_vol', 'N/A')} Dist:{parsed.get('checked_dist', 'N/A')}"
                logger.info(f"💡 [{self.cfg.symbol}] AI Entry: {decision.upper()} | {parsed.get('reason')} | {check_log}")
            
            return decision
        except Exception:
            return "wait"

    def _execute_entry(self, side: str):
        """Executes a market order for entry."""
        try:
            self.exchange.set_margin_mode('isolated', self.cfg.symbol) 
            ticker = self.exchange.fetch_ticker(self.cfg.symbol)
            
            raw_amount = (self.cfg.amount * self.cfg.leverage / ticker['last'])
            amount = self.exchange.amount_to_precision(self.cfg.symbol, raw_amount)
            
            order_func = self.exchange.create_market_buy_order if side == 'long' else self.exchange.create_market_sell_order
            order_func(self.cfg.symbol, amount)
            
            time.sleep(2)
            
            pos = self._get_position()
            if pos:
                self.fix_history[self.cfg.symbol] = 0
                self._ensure_orders(pos)
                logger.info(f"⚡ [{self.cfg.symbol}] Entry {side.upper()} Done")
        except Exception as e:
            logger.error(f"❌ [{self.cfg.symbol}] Entry Fail: {e}")

    def _ensure_orders(self, position: Dict):
        """Places TP/SL orders if they are missing."""
        if time.time() - self.fix_history.get(self.cfg.symbol, 0) < 60: return

        try:
            orders = self.exchange.fetch_open_orders(self.cfg.symbol)
            side = position['side']
            
            has_tp = any('limit' in o['type'].lower() for o in orders)
            has_sl = any('stop' in o['type'].lower() for o in orders)

            if has_tp and has_sl: return

            logger.warning(f"🔧 [{self.cfg.symbol}] Fixing Orders")
            self.fix_history[self.cfg.symbol] = time.time()
            self.exchange.cancel_all_orders(self.cfg.symbol)
            time.sleep(2)
            
            entry = float(position['entryPrice'])
            amt = float(position['contracts'])
            
            tp_rate = (self.cfg.target_roe / self.cfg.leverage) / 100
            sl_rate = (self.cfg.stop_loss_roe / self.cfg.leverage) / 100
            
            if side == 'long':
                tp_price = self.exchange.price_to_precision(self.cfg.symbol, entry * (1 + tp_rate))
                sl_price = self.exchange.price_to_precision(self.cfg.symbol, entry * (1 - sl_rate))
                self.exchange.create_limit_sell_order(self.cfg.symbol, amt, tp_price, {'reduceOnly': True})
                self.exchange.create_order(self.cfg.symbol, 'STOP_MARKET', 'sell', amt, None, {'stopPrice': sl_price, 'closePosition': True})
            else:
                tp_price = self.exchange.price_to_precision(self.cfg.symbol, entry * (1 - tp_rate))
                sl_price = self.exchange.price_to_precision(self.cfg.symbol, entry * (1 + sl_rate))
                self.exchange.create_limit_buy_order(self.cfg.symbol, amt, tp_price, {'reduceOnly': True})
                self.exchange.create_order(self.cfg.symbol, 'STOP_MARKET', 'buy', amt, None, {'stopPrice': sl_price, 'closePosition': True})
                
            logger.info(f"✅ [{self.cfg.symbol}] Orders Fixed")
        except Exception as e:
            if "-4130" not in str(e): logger.error(f"⚠️ Fix Fail: {e}")

    def _get_position(self) -> Optional[Dict]:
        try:
            positions = self.exchange.fetch_positions([self.cfg.symbol])
            for p in positions:
                if float(p['contracts']) != 0: return p
            return None
        except Exception: 
            return None

if __name__ == "__main__":
    try:
        with open(CONFIG_FILE, 'r') as f: config_list = json.load(f)
        logger.info(f"🔥 Binance AI Bot Started ({len(config_list)} pairs)")
        threads = [TradingBot(config) for config in config_list]
        for t in threads: 
            t.start()
            time.sleep(random.uniform(0.5, 1.5)) 
        for t in threads: t.join()
    except KeyboardInterrupt: logger.info("👋 Exit")
    except Exception as e: logger.error(f"Main Error: {e}")
