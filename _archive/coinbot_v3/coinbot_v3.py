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
# [V3] 설정 및 로깅
# ---------------------------------------------------------
load_dotenv()
CONFIG_FILE = "config.json"
MODEL_NAME = "gemini-2.5-flash-lite"

# 1. 로거 설정 (시간 | 메시지 형태로 깔끔하게)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("bot_v3_master.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MASTER_BOT")

# 2. [핵심] 시끄러운 외부 라이브러리 조용히 시키기 (WARNING 이상만 출력)
# 이 부분이 없으면 HTTP Request 로그가 계속 뜹니다.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Gemini 클라이언트
try:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    logger.error(f"❌ Gemini Key Error: {e}")
    exit()

# ---------------------------------------------------------
# [AI 전략 창고]
# ---------------------------------------------------------
def get_strategy_prompt(symbol, strategy_type, data):
    base_prompt = f"Act as a Crypto Scalper for {symbol} (1m chart)."
    
    if strategy_type == "hybrid":
        return f"""
        {base_prompt}
        Your Goal: Consistent profit. Minimize losses.
        
        Step 1: IDENTIFY MARKET REGIME
        - **TRENDING**: Price moving away from EMA with volume.
        - **RANGING**: Price chopping around EMA, RSI oscillating.

        Step 2: DECIDE
        [IF TRENDING] Follow the trend (Buy dips in Uptrend).
        [IF RANGING] Mean Reversion (Buy Low, Sell High).
        
        Output strict JSON: {{"decision": "long"}} or {{"decision": "short"}} or {{"decision": "wait"}}
        """ + f"\nData:\n{data}"

    elif strategy_type == "aggressive":
        return f"""
        {base_prompt}
        You are trading a high-volatility Meme Coin.
        
        Strategy: "Momentum & Volatility"
        1. **Ignore Safety**: Do not wait for perfect confirmation.
        2. **Momentum**: Price Pump + Volume Spike = **LONG**.
        3. **Panic**: Price Dump + Volume Spike = **SHORT**.
        
        Output strict JSON: {{"decision": "long"}} or {{"decision": "short"}} or {{"decision": "wait"}}
        """ + f"\nData:\n{data}"
    
    else:
        return f"{base_prompt} Analyze data and decide. Output JSON." + f"\nData:\n{data}"

# ---------------------------------------------------------
# [봇 클래스] 독립 실행 유닛
# ---------------------------------------------------------
class TradingBot(threading.Thread):
    def __init__(self, config):
        threading.Thread.__init__(self)
        self.symbol = config['symbol']
        self.leverage = config['leverage']
        self.invest_amount = config['amount']
        self.target_roe = config.get('target_roe', 10.0)
        self.strategy = config.get('strategy', 'hybrid')
        self.running = True
        
        self.exchange = ccxt.binance({
            'apiKey': os.getenv("BINANCE_API_KEY"),
            'secret': os.getenv("BINANCE_SECRET_KEY"),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    def run(self):
        # 시작 로그
        logger.info(f"🚀 [{self.symbol}] 가동 | {self.leverage}x | ${self.invest_amount}")
        self.set_leverage()
        
        loop_count = 0
        
        while self.running:
            try:
                # 1. 포지션 확인
                position = self.get_open_position()
                
                if position:
                    # 포지션 잡고 있을 때는 6번 루프(약 1분)마다 한 번씩만 상태 로그 출력
                    if loop_count % 6 == 0:
                        pnl = float(position['unrealizedPnl'])
                        try:
                            # ROI 계산
                            entry_price = float(position['entryPrice'])
                            current_price = float(position['markPrice'])
                            if position['side'] == 'long':
                                roi = ((current_price - entry_price) / entry_price) * self.leverage * 100
                            else:
                                roi = ((entry_price - current_price) / entry_price) * self.leverage * 100
                            
                            icon = "🔴" if pnl < 0 else "🟢"
                            logger.info(f"{icon} [{self.symbol}] 보유중 | ROI: {roi:.2f}% | PnL: ${pnl:.4f}")
                        except:
                            logger.info(f"✊ [{self.symbol}] 보유중... (익절 대기)")

                    time.sleep(10)
                    loop_count += 1
                    continue

                # 2. 미체결 주문 정리
                self.check_and_cancel_orders()

                # 3. 데이터 수집
                market_data = self.get_market_data()
                if not market_data:
                    time.sleep(5)
                    continue

                # 4. AI 판단
                decision = self.ask_llm(market_data)

                # 5. 주문 실행
                if decision in ['long', 'short']:
                    self.enter_position(decision)
                    loop_count = 0 
                    time.sleep(10) 
                else:
                    time.sleep(5) 

            except Exception as e:
                # 에러 메시지 간소화
                err_msg = str(e)
                if "Code: -4164" in err_msg:
                    logger.error(f"❌ [{self.symbol}] 주문 거절: 최소 주문 금액 부족 ($0.6 이상으로 올리세요)")
                elif "503" in err_msg or "429" in err_msg:
                    pass 
                else:
                    logger.error(f"⚠️ [{self.symbol}] 오류: {e}")
                time.sleep(10)

    # --- 유틸리티 함수 ---
    def set_leverage(self):
        try:
            self.exchange.load_markets()
            self.exchange.set_leverage(self.leverage, self.symbol)
        except: pass

    def get_open_position(self):
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for p in positions:
                if float(p['contracts']) != 0: return p
            return None
        except: return None

    def check_and_cancel_orders(self):
        try:
            if self.exchange.fetch_open_orders(self.symbol):
                self.exchange.cancel_all_orders(self.symbol)
        except: pass

    def get_market_data(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df['ema'] = df['close'].ewm(span=20, adjust=False).mean()
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            latest = df.iloc[-1]
            return f"Price: {latest['close']}, EMA: {latest['ema']:.4f}, RSI: {latest['rsi']:.1f}, Vol: {latest['volume']}"
        except: return None

    def ask_llm(self, data):
        prompt = get_strategy_prompt(self.symbol, self.strategy, data)
        for _ in range(3):
            try:
                res = gemini_client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
                )
                decision = json.loads(res.text).get("decision", "wait").lower()
                
                if decision != "wait":
                    logger.info(f"🤖 [{self.symbol}] AI 판단: {decision.upper()} 진입 시도")
                return decision
                
            except Exception as e:
                if "503" in str(e): time.sleep(2); continue
                return "wait"
        return "wait"

    # --- [핵심] 주문 함수 ---
    def enter_position(self, side):
        try:
            # 1. 시장가 진입
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            amount = self.exchange.amount_to_precision(self.symbol, (self.invest_amount * self.leverage / price))
            
            if side == 'long': self.exchange.create_market_buy_order(self.symbol, amount)
            else: self.exchange.create_market_sell_order(self.symbol, amount)
            
            time.sleep(3) 
            
            # 2. 평단가 확인 및 TP 계산
            pos = self.get_open_position()
            if pos:
                entry = float(pos['entryPrice'])
                amt = float(pos['contracts'])
                move = (self.target_roe / self.leverage) / 100 
                
                if side == 'long':
                    tp_price = entry * (1 + move)
                    self.exchange.create_limit_sell_order(self.symbol, amt, tp_price, {'reduceOnly': True})
                else:
                    tp_price = entry * (1 - move)
                    self.exchange.create_limit_buy_order(self.symbol, amt, tp_price, {'reduceOnly': True})
                
                logger.info(f"⚡ [{self.symbol}] {side.upper()} 체결 완료 | 평단: {entry} | 목표가: {tp_price:.4f}")
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] 주문 실패: {e}")

# ---------------------------------------------------------
# [메인 실행]
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        with open(CONFIG_FILE, 'r') as f:
            bot_configs = json.load(f)
        
        threads = []
        logger.info(f"=================================================")
        logger.info(f"🔥 V3 마스터 봇 가동 시작 (총 {len(bot_configs)}개 코인)")
        logger.info(f"=================================================")

        for config in bot_configs:
            bot = TradingBot(config)
            bot.start()
            threads.append(bot)
            time.sleep(1)

        for t in threads:
            t.join()

    except KeyboardInterrupt:
        logger.info("👋 봇을 종료합니다.")
    except Exception as e:
        logger.error(f"실행 오류: {e}")
