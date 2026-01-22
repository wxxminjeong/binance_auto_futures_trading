import ccxt
import os
import time
import json
import logging
import pandas as pd      # 데이터 분석용
import pandas_ta as ta   # 보조지표 계산용
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. 로깅 설정
# ---------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('trading_bot.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# ---------------------------------------------------------
# 2. 환경 설정
# ---------------------------------------------------------
load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYMBOL = "DOGE/USDT"
LEVERAGE = 40
INVEST_AMOUNT_USDT = 0.2
TARGET_ROE = 5.0
TIMEFRAME = '15m'
MODEL_NAME = "gemini-2.5-flash-lite"

# ---------------------------------------------------------
# 3. 초기화
# ---------------------------------------------------------
try:
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET_KEY,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info(f"✅ 초기화 성공: {MODEL_NAME} (RSI+EMA 지표 적용)")

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
        logger.info(f"⚙️ 레버리지 {LEVERAGE}배 설정 완료")
    except Exception as e:
        logger.warning(f"⚠️ 레버리지 설정 실패: {e}")

def get_market_data():
    """RSI와 EMA 등 보조지표를 계산해서 텍스트로 반환"""
    try:
        # [변경] 지표 계산을 위해 캔들을 50개 가져옴 (최소 20개 이상 필요)
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=50)
        
        # 1. 데이터프레임(표)으로 변환
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 2. 보조지표 계산 (pandas_ta)
        # RSI (기간 14)
        df['rsi'] = df.ta.rsi(length=14)
        # EMA (기간 20)
        df['ema'] = df.ta.ema(length=20)
        
        # 3. 최신 값 추출 (맨 마지막 줄)
        latest = df.iloc[-1]
        rsi_val = latest['rsi']
        ema_val = latest['ema']
        curr_price = latest['close']
        
        # 추세 판단 (가격이 이평선 위면 상승세)
        trend = "UP (Bullish)" if curr_price > ema_val else "DOWN (Bearish)"
        
        # 4. 최근 5개 캔들만 텍스트로 정리 (LLM에게는 요약본 전달)
        candles_str = ""
        for i in range(5):
            row = df.iloc[-(5-i)] # 뒤에서 5번째부터 순서대로
            ts = time.strftime('%Y-%m-%d %H:%M', time.localtime(row['timestamp']/1000))
            candles_str += f"[{ts}] Close: {row['close']}, Vol: {row['volume']}\n"

        # 5. LLM에게 줄 최종 리포트 작성
        report = f"""
        Symbol: {SYMBOL}
        Current Price: {curr_price}
        
        *** Technical Indicators ***
        - RSI(14): {rsi_val:.2f} (Over 70=Overbought, Under 30=Oversold)
        - EMA(20): {ema_val:.5f}
        - Current Trend: {trend}
        
        *** Recent 5 Candles ***
        {candles_str}
        """
        return report

    except Exception as e:
        logger.error(f"❌ 데이터 분석 실패: {e}")
        return None

def get_open_position():
    """현재 포지션 조회"""
    try:
        positions = exchange.fetch_positions()
        for p in positions:
            current_symbol = p['symbol'].split(':')[0]
            if current_symbol == SYMBOL and float(p['contracts']) != 0:
                return p 
        return None
    except Exception as e:
        logger.error(f"❌ 포지션 조회 에러: {e}")
        return None

def ask_llm_decision():
    market_data = get_market_data()
    if not market_data: return "wait"

    # [변경] 지표를 활용하라는 구체적인 지시 추가
    system_prompt = f"""
    You are a professional crypto scalper bot.
    Analyze the provided technical indicators (RSI, EMA) and price data.
    
    Goal: Quick 5% ROE (1.25% price move).
    
    Strategy:
    1. Trend Following: If Trend is UP, prefer LONG. If DOWN, prefer SHORT.
    2. RSI Check: 
       - If RSI > 70, be careful of LONG (Overbought). Consider SHORT if trend shows weakness.
       - If RSI < 30, be careful of SHORT (Oversold). Consider LONG if trend shows strength.
    3. If signals are mixed or weak, chose 'wait'.
    
    Output JSON strictly: {{"decision": "long"}} or {{"decision": "short"}} or {{"decision": "wait"}}
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=system_prompt),
                        types.Part.from_text(text=f"Market Report:\n{market_data}")
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        result = json.loads(response.text)
        decision = result.get("decision", "wait").lower()
        logger.info(f"🧠 Gemini 분석: {decision.upper()}")
        return decision

    except Exception as e:
        logger.error(f"❌ Gemini 에러: {e}")
        return "wait"

def enter_position_and_set_tp(side):
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        
        notional_value = INVEST_AMOUNT_USDT * LEVERAGE
        amount_coin = notional_value / current_price
        amount_coin = exchange.amount_to_precision(SYMBOL, amount_coin)
        
        logger.info(f"🚀 {side.upper()} 진입 시도: {amount_coin}개")

        order = None
        if side == 'long':
            order = exchange.create_market_buy_order(SYMBOL, amount_coin)
        elif side == 'short':
            order = exchange.create_market_sell_order(SYMBOL, amount_coin)
            
        entry_price = float(order['average'] if order.get('average') else current_price)
        logger.info(f"✅ 진입 완료! 평단가: {entry_price}")

        # TP 계산 (ROE 5%)
        required_move = (TARGET_ROE / LEVERAGE) / 100
        
        if side == 'long':
            tp_price = entry_price * (1 + required_move)
            tp_side = 'sell'
        else:
            tp_price = entry_price * (1 - required_move)
            tp_side = 'buy'

        tp_price = float(exchange.price_to_precision(SYMBOL, tp_price))
        
        # 익절 주문
        params = {'reduceOnly': True}
        if tp_side == 'sell':
            exchange.create_limit_sell_order(SYMBOL, amount_coin, tp_price, params)
        else:
            exchange.create_limit_buy_order(SYMBOL, amount_coin, tp_price, params)
            
        logger.info(f"🎯 5% 익절(TP) 설정 완료: {tp_price}")
        
    except Exception as e:
        logger.error(f"❌ 주문 실패: {e}")

# ---------------------------------------------------------
# 5. 메인 루프
# ---------------------------------------------------------
def main():
    logger.info(f"🤖 봇 시작: {SYMBOL}, ${INVEST_AMOUNT_USDT}, x{LEVERAGE}, 목표수익 {TARGET_ROE}%")
    set_leverage()
    
    while True:
        try:
            position = get_open_position()
            
            if position:
                side = position['side'].upper()
                pnl = position['unrealizedPnl']
                roe = (pnl / INVEST_AMOUNT_USDT) * 100
                logger.info(f"👀 [{side}] 추적 중... ROE: {roe:.2f}% (목표: {TARGET_ROE}%)")
                
            else:
                logger.info("🔍 포지션 없음. 시장 분석 중...")
                decision = ask_llm_decision()
                
                if decision in ['long', 'short']:
                    enter_position_and_set_tp(decision)
                    time.sleep(5)
                else:
                    logger.info("🧘 관망")
            
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("🛑 종료")
            break
        except Exception as e:
            logger.error(f"에러: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
