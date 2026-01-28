import ccxt
import os
import time
import json
import logging
import argparse
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. 설정 및 파라미터
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--symbol', type=str, required=True, help='Trading Symbol (e.g., BTC/USDT)')
args = parser.parse_args()

SYMBOL = args.symbol
LEVERAGE = 40
INVEST_AMOUNT_USDT = 0.2

TIMEFRAME = '1m'
# 목표 수익률 15% (수수료 떼고 확실히 먹기 위함)
TARGET_ROE = 15.0  

MODEL_NAME = "gemini-2.5-flash-lite"
safe_symbol = SYMBOL.replace('/', '')
LOG_FILE = f"bot_{safe_symbol}.log"

# ---------------------------------------------------------
# 2. 로깅 설정
# ---------------------------------------------------------
logger = logging.getLogger(safe_symbol)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ---------------------------------------------------------
# 3. 초기화
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
    logger.info(f"🔥 [진짜 노빠꾸] {SYMBOL} 시작! (목표 ROE {TARGET_ROE}% / 손절 절대 없음)")
except Exception as e:
    logger.error(f"❌ 초기화 실패: {e}")
    exit()

# ---------------------------------------------------------
# 4. 함수들
# ---------------------------------------------------------
def set_leverage():
    try:
        exchange.load_markets()
        exchange.set_leverage(LEVERAGE, SYMBOL)
    except Exception as e:
        logger.warning(f"⚠️ 레버리지 설정 실패: {e}")

def get_margin_balance():
    try:
        return float(exchange.fetch_balance()['total'].get('USDT', 0))
    except:
        return 0.0

def calculate_indicators(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['ema'] = df['close'].ewm(span=20, adjust=False).mean()
    return df

def get_market_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        
        if pd.isna(latest['rsi']) or pd.isna(latest['ema']): return None
        
        trend = "UP" if latest['close'] > latest['ema'] else "DOWN"
        candles = ""
        for i in range(5):
            r = df.iloc[-(5-i)]
            ts = time.strftime('%H:%M', time.localtime(r['timestamp']/1000))
            # [수정] 가격은 4자리, 거래량은 1자리로 깔끔하게
            candles += f"[{ts}] {r['close']:.4f}\n"
            
        # [수정] RSI, EMA 소수점 2자리로 제한
        return f"Price: {latest['close']:.4f}, RSI: {latest['rsi']:.2f}, EMA: {latest['ema']:.2f}, Trend: {trend}\nCandles:\n{candles}"
    except:
        return None

def get_open_position():
    """현재 포지션 정보를 정확하게 가져옴"""
    try:
        positions = exchange.fetch_positions([SYMBOL]) 
        for p in positions:
            if float(p['contracts']) != 0:
                return p
        return None
    except Exception as e:
        logger.error(f"포지션 조회 실패: {e}")
        return None

def check_and_cancel_orders():
    """포지션이 없는데 남아있는 좀비 주문 정리"""
    try:
        open_orders = exchange.fetch_open_orders(SYMBOL)
        if open_orders:
            exchange.cancel_all_orders(SYMBOL)
            logger.info(f"🧹 잔여 주문 {len(open_orders)}개 취소 (포지션 없음)")
    except:
        pass

def ask_llm_decision():
    data = get_market_data()
    if not data: return "wait"
    
    prompt = f"""
    Act as a SCALPER trading {SYMBOL} (1m chart).
    Strategy: Trend Following (EMA20) + RSI.
    - Price > EMA & RSI < 70 -> LONG
    - Price < EMA & RSI > 30 -> SHORT
    Output JSON: {{"decision": "long"|"short"|"wait"}}
    """
    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt), types.Part.from_text(text=data)])],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return json.loads(res.text).get("decision", "wait").lower()
    except:
        return "wait"

def enter_position_with_safe_tp(side):
    """
    1. 시장가 진입
    2. 실제 평단가 확인
    3. TP 설정 (로그 깔끔하게 수정됨)
    """
    try:
        # 1. 진입 (Market Order)
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        amount = exchange.amount_to_precision(SYMBOL, (INVEST_AMOUNT_USDT * LEVERAGE / current_price))
        
        if side == 'long': 
            exchange.create_market_buy_order(SYMBOL, amount)
        else: 
            exchange.create_market_sell_order(SYMBOL, amount)
            
        logger.info(f"🚀 {side.upper()} 주문 전송 완료! 체결 대기중...")
        
        time.sleep(3) 
        
        # 3. 실제 내 포지션 정보 확인
        position = get_open_position()
        if not position:
            logger.error("❌ 진입 실패? 포지션이 안 보입니다.")
            return

        real_entry_price = float(position['entryPrice'])
        position_amt = float(position['contracts']) 
        
        # [수정] 평단가는 4자리, 수량은 2자리로 표기
        logger.info(f"✅ 체결 확인! 평단가: {real_entry_price:.4f} (수량: {position_amt:.2f})")

        # 4. TP 계산
        required_price_move = (TARGET_ROE / LEVERAGE) / 100
        
        if side == 'long':
            tp_price = real_entry_price * (1 + required_price_move)
            tp_side = 'sell'
        else:
            tp_price = real_entry_price * (1 - required_price_move)
            tp_side = 'buy'

        tp_price = float(exchange.price_to_precision(SYMBOL, tp_price))
        
        # 5. TP 주문 걸기
        if tp_side == 'sell':
            exchange.create_limit_sell_order(SYMBOL, position_amt, tp_price, {'reduceOnly': True})
        else:
            exchange.create_limit_buy_order(SYMBOL, position_amt, tp_price, {'reduceOnly': True})

        # [수정] 익절가는 4자리로 표기
        logger.info(f"🎯 안전 익절 설정 완료: {tp_price:.4f} (목표 ROE {TARGET_ROE:.2f}%)")

    except Exception as e:
        logger.error(f"❌ 주문 프로세스 실패: {e}")

# ---------------------------------------------------------
# 5. 메인 루프
# ---------------------------------------------------------
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
                
                # [수정] 평단가는 4자리, ROE는 2자리로 깔끔하게
                logger.info(f"👀 [{side}] 평단:{entry:.4f} | ROE:{roe:.2f}% (익절 대기중... 버티기)")
                
            else:
                check_and_cancel_orders()
                
                decision = ask_llm_decision()
                if decision in ['long', 'short']:
                    enter_position_with_safe_tp(decision)
                    time.sleep(10)
                else:
                    # [수정] 잔고 2자리로 표기
                    logger.info(f"🧘 관망 중... (💰 {bal:.2f} USDT)")
            
            time.sleep(5) 

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"에러: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()