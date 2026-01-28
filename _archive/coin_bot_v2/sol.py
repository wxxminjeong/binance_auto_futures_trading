# -*- coding: utf-8 -*-
import ccxt
import os
import time
import json
import logging
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------
# [SETTINGS] TRX Configuration
# ---------------------------------------------------------
SYMBOL = "SOL/USDT"
LEVERAGE = 40

INVEST_AMOUNT_USDT = 0.5  

TIMEFRAME = '1m'
TARGET_ROE = 13.0  
MODEL_NAME = "gemini-2.5-flash-lite"
LOG_FILE = "bot_SOL.log"

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
logger = logging.getLogger("SOL_BOT")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

if logger.hasHandlers(): logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
load_dotenv()
try:
    exchange = ccxt.binance({
        'apiKey': os.getenv("BINANCE_API_KEY"),
        'secret': os.getenv("BINANCE_SECRET_KEY"),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    logger.info(f"💎 [TRX] Bot Started! (Target ROE {TARGET_ROE}%)")
except Exception as e:
    logger.error(f"❌ Init Failed: {e}")
    exit()

# ---------------------------------------------------------
# Functions
# ---------------------------------------------------------
def set_leverage():
    try:
        exchange.load_markets()
        exchange.set_leverage(LEVERAGE, SYMBOL)
    except: pass

def get_margin_balance():
    try: return float(exchange.fetch_balance()['total'].get('USDT', 0))
    except: return 0.0

def calculate_indicators(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['ema'] = df['close'].ewm(span=20, adjust=False).mean()
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    return df

def get_market_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = calculate_indicators(df)
        recent_df = df.tail(10)
        latest = df.iloc[-1]
        
        data_str = "Index | Time | Open | High | Low | Close | Vol | EMA | RSI\n"
        data_str += "-" * 60 + "\n"
        for index, row in recent_df.iterrows():
            ts = time.strftime('%H:%M', time.localtime(row['timestamp']/1000))
            vol_signal = "*" if row['volume'] > row['vol_ma'] else "" 
            data_str += f"{index} | {ts} | {row['open']:.4f} | {row['high']:.4f} | {row['low']:.4f} | {row['close']:.4f} | {row['volume']:.0f}{vol_signal} | {row['ema']:.4f} | {row['rsi']:.1f}\n"
            
        summary = f"""
        Current Price: {latest['close']:.4f}
        RSI(14): {latest['rsi']:.2f}
        EMA(20): {latest['ema']:.4f}
        Price vs EMA: {"ABOVE" if latest['close'] > latest['ema'] else "BELOW"}
        Volume Status: {"HIGH" if latest['volume'] > latest['vol_ma'] else "Normal"}
        """
        return summary + "\nRecent Candles (Last 10):\n" + data_str
    except Exception as e:
        logger.error(f"Data Error: {e}")
        return None

def get_open_position():
    try:
        positions = exchange.fetch_positions([SYMBOL]) 
        for p in positions:
            if float(p['contracts']) != 0: return p
        return None
    except: return None

def check_and_cancel_orders():
    try:
        if exchange.fetch_open_orders(SYMBOL):
            exchange.cancel_all_orders(SYMBOL)
            logger.info("🧹 Cancelled open orders")
    except: pass

# ---------------------------------------------------------
# AI Logic (With Retry)
# ---------------------------------------------------------
def ask_llm_decision():
    data = get_market_data()
    if not data: return "wait"
    
    prompt = f"""
    Act as an Expert Crypto Price Action Trader specializing in Scalping (1m chart) for Tron ({SYMBOL}).
    
    Analyze the provided market data (OHLCV candles & Indicators).
    
    Analysis Logic:
    1. **Candle Patterns**: Look for Reversal patterns (Hammer, Engulfing) or Continuation patterns.
    2. **Volume Analysis**: 
       - Price UP + Volume UP -> Strong Bullish (GOOD for LONG)
       - Price UP + Volume DOWN -> Weak Bullish (Warning)
       - Price DOWN + Volume UP -> Strong Bearish (GOOD for SHORT)
    3. **Trend**: Use EMA(20) as the baseline.
    4. **RSI**: Use it only for momentum context.

    Decision Rules:
    - **LONG**: Strong uptrend, Bullish pattern at EMA support, or Breakout with volume.
    - **SHORT**: Strong downtrend, Bearish pattern at EMA resistance, or Breakdown with volume.
    - **WAIT**: Choppy market, Doji candles, or conflicting signals.

    Output strict JSON: {{"decision": "long"}} or {{"decision": "short"}} or {{"decision": "wait"}}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model=MODEL_NAME,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt), types.Part.from_text(text=data)])],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
            )
            decision = json.loads(res.text).get("decision", "wait").lower()
            logger.info(f"🤖 AI Decision: {decision.upper()}")
            return decision

        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "overloaded" in error_msg:
                logger.warning(f"⚠️ Server Overloaded (503). Retrying... ({attempt+1}/{max_retries})")
                time.sleep(2)
                continue
            else:
                logger.error(f"❌ Gemini Error: {e}")
                return "wait"
    
    return "wait"

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
def enter_position_with_safe_tp(side):
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        amount = exchange.amount_to_precision(SYMBOL, (INVEST_AMOUNT_USDT * LEVERAGE / current_price))
        
        if side == 'long': exchange.create_market_buy_order(SYMBOL, amount)
        else: exchange.create_market_sell_order(SYMBOL, amount)
        logger.info(f"🚀 {side.upper()} Entry sent...")
        
        time.sleep(3)
        position = get_open_position()
        if not position: return

        entry = float(position['entryPrice'])
        amt = float(position['contracts'])
        logger.info(f"✅ Filled! Entry: {entry:.4f}")

        move = (TARGET_ROE / LEVERAGE) / 100
        tp = entry * (1 + move) if side == 'long' else entry * (1 - move)
        tp = float(exchange.price_to_precision(SYMBOL, tp))
        
        if side == 'long': exchange.create_limit_sell_order(SYMBOL, amt, tp, {'reduceOnly': True})
        else: exchange.create_limit_buy_order(SYMBOL, amt, tp, {'reduceOnly': True})
        logger.info(f"🎯 TP Set: {tp:.4f}")
    except Exception as e:
        logger.error(f"❌ Order Failed: {e}")

def main():
    set_leverage()
    try: exchange.cancel_all_orders(SYMBOL)
    except: pass
    
    while True:
        try:
            position = get_open_position()
            bal = get_margin_balance()
            
            if position:
                side = position['side'].upper()
                pnl = float(position['unrealizedPnl'])
                roe = (pnl / INVEST_AMOUNT_USDT) * 100
                entry = float(position['entryPrice'])
                logger.info(f"👀 [{side}] Entry:{entry:.4f} | ROE:{roe:.2f}% (Waiting for TP)")
            else:
                check_and_cancel_orders()
                decision = ask_llm_decision()
                
                if decision in ['long', 'short']:
                    enter_position_with_safe_tp(decision)
                    time.sleep(10)
                else:
                    logger.info(f"🧘 Waiting... (💰 {bal:.2f} USDT)")
            
            time.sleep(5) 
            
        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()