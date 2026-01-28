# Binance Futures AI Trading Bot

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![Binance](https://img.shields.io/badge/Binance_Futures-FCD535?style=for-the-badge&logo=binance&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu_Server-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)

**Intel N100 (Ubuntu Server)** 환경에서 24시간 운용되도록 설계된 바이낸스 선물 자동매매 시스템입니다.
Google Gemini LLM을 활용하여 정형화된 지표 분석의 한계를 보완하고, 시장 상황(거래량, 변동성, 추세)을 종합적으로 해석하여 매매를 수행합니다.

## Project Overview

* **Objective:** 저전력 개인 서버 환경에서의 안정적인 24/7 알고리즘 트레이딩 시스템 구축
* **Environment:** Intel N100 Mini PC, Ubuntu Linux (Headless)
* **Core Logic:** 1분봉 데이터 기반의 기술적 지표(RSI, Bollinger Bands, MACD)와 LLM의 추론 능력을 결합한 하이브리드 의사결정

## Key Features

### 1. LLM 기반 시장 분석 (AI Analysis)
* 단순한 골든크로스/데드크로스 로직 대신, 수치화된 차트 데이터(OHLCV, 보조지표, 거래량 비율)를 프롬프트로 구성하여 AI에게 전달합니다.
* AI는 현재 시장이 추세장인지 횡보장인지, 혹은 거래량이 동반된 유의미한 변동인지 판단하여 `Long`, `Short`, `Wait` 결정을 내립니다.

### 2. 리스크 관리 (Risk Management)
* **격리 모드(Isolated Margin):** 모든 포지션은 격리 모드로 진입하여 개별 포지션의 리스크가 전체 자산으로 전이되는 것을 방지합니다.
* **진입 필터링:** 'Ultra Safe' 전략을 적용하여 단순 과매도/과매수가 아닌, 평소 대비 1.5배 이상의 거래량(Volume Spike)이 동반된 반전 신호에서만 진입합니다.

### 3. 주문 무결성 보장 (Order Integrity)
* 네트워크 불안정이나 서버 재부팅 직후에도 시스템이 현재 보유 중인 포지션과 미체결 주문 상태를 대조합니다.
* 익절(TP) 또는 손절(SL) 주문이 누락된 경우, 계산된 비율에 맞춰 즉시 주문을 재생성하여 포지션 관리를 자동화했습니다.

### 4. 모니터링 시스템 (Monitoring)
* **Flask Web Dashboard:** 외부에서 접근 가능한 웹 대시보드를 통해 실시간 로그, 수익률(ROI), 포지션 상태를 시각화했습니다.
* **Ngrok Tunneling:** 별도의 포트포워딩 없이 외부 네트워크에서 안전하게 모니터링 페이지에 접속할 수 있도록 구성했습니다.

## Tech Stack & Environment

### Core & AI
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini%202.5-8E75B2?style=for-the-badge&logo=google&logoColor=white)

### Server & Infra
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![PM2](https://img.shields.io/badge/PM2-2B037B?style=for-the-badge&logo=pm2&logoColor=white)
![Ngrok](https://img.shields.io/badge/Ngrok-1F1E37?style=for-the-badge&logo=ngrok&logoColor=white)

### Web & API
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![CCXT](https://img.shields.io/badge/CCXT-121212?style=for-the-badge&logo=github&logoColor=white)

## Disclaimer
본 프로젝트는 개인 연구 및 포트폴리오 목적으로 개발되었습니다. 실제 트레이딩에 사용할 경우 발생할 수 있는 금전적 손실에 대한 책임은 사용자 본인에게 있습니다.
