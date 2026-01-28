# -*- coding: utf-8 -*-
from flask import Flask, render_template_string, jsonify
from pyngrok import ngrok
from dotenv import load_dotenv
import os
import re

# ---------------------------------------------------------
# [설정] .env 로드
# ---------------------------------------------------------
load_dotenv()
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
LOG_FILE = "bot_master.log"
PORT = 5000
MAX_LINES = 300

app = Flask(__name__)

# ---------------------------------------------------------
# HTML 템플릿 (V6 Cyberpunk Design)
# ---------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>V6 Command Center</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --panel-color: #161b22;
            --text-color: #c9d1d9;
            --accent-color: #58a6ff;
            --success-color: #2ea043;
            --error-color: #ff7b72;
            --warn-color: #d29922;
            --border-color: #30363d;
        }
        body { 
            background-color: var(--bg-color); 
            color: var(--text-color); 
            font-family: 'Consolas', 'Monaco', monospace; 
            margin: 0; padding: 0; 
            height: 100vh; display: flex; flex-direction: column;
        }
        header {
            background-color: var(--panel-color);
            padding: 15px;
            border-bottom: 1px solid var(--border-color);
            display: flex; justify-content: space-between; align-items: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        h1 { margin: 0; font-size: 1.2rem; color: var(--accent-color); letter-spacing: 1px; }
        .status-dot { height: 10px; width: 10px; background-color: var(--success-color); border-radius: 50%; display: inline-block; margin-right: 8px; box-shadow: 0 0 8px var(--success-color); }
        
        .controls { padding: 10px 15px; background-color: var(--bg-color); }
        input { 
            width: 100%; box-sizing: border-box;
            padding: 12px; font-size: 16px; 
            background-color: var(--panel-color); 
            border: 1px solid var(--border-color); 
            color: white; border-radius: 6px; outline: none;
            transition: border-color 0.3s;
        }
        input:focus { border-color: var(--accent-color); }

        #log-container { 
            flex: 1; 
            overflow-y: auto; 
            padding: 10px 15px; 
            font-size: 13px; line-height: 1.5;
            scroll-behavior: smooth;
        }
        .log-line { 
            padding: 4px 8px; 
            border-bottom: 1px solid #21262d; 
            animation: fadeIn 0.3s ease;
        }
        .log-line:hover { background-color: rgba(88, 166, 255, 0.1); }
        
        /* Syntax Highlighting */
        .ts { color: #8b949e; margin-right: 10px; font-size: 0.85em; }
        .tag { font-weight: bold; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; margin-right: 8px; }
        
        .info { border-left: 3px solid var(--accent-color); }
        .success { border-left: 3px solid var(--success-color); color: #7ee787; }
        .error { border-left: 3px solid var(--error-color); background-color: rgba(255, 123, 114, 0.1); color: #ff7b72; }
        .warn { border-left: 3px solid var(--warn-color); color: #e3b341; }

        .coin { color: #d2a8ff; font-weight: bold; }
        .profit { color: #7ee787; }
        .loss { color: #ff7b72; }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }
        
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-color); }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
    </style>
</head>
<body>
    <header>
        <div><span class="status-dot"></span>V6 COMMAND CENTER</div>
        <div style="font-size: 0.8rem; color: #8b949e;">LIVE</div>
    </header>
    
    <div class="controls">
        <input type="text" id="filterInput" placeholder="🔍 Search Logs (e.g. BTC, Error, Profit...)" onkeyup="filterLogs()">
    </div>

    <div id="log-container">Initializing...</div>

    <script>
        let allLogs = [];
        const container = document.getElementById('log-container');

        // 로그 파싱 및 스타일링
        function formatLine(line) {
            let cssClass = 'info';
            if (line.includes('ERROR') || line.includes('❌')) cssClass = 'error';
            else if (line.includes('WARNING') || line.includes('⚠️') || line.includes('🔧')) cssClass = 'warn';
            else if (line.includes('✅') || line.includes('🚀') || line.includes('⚡')) cssClass = 'success';

            // 시간 파싱 (HH:MM:SS)
            const timeMatch = line.match(/^(\d{2}:\d{2}:\d{2})/);
            const timestamp = timeMatch ? `<span class="ts">${timeMatch[1]}</span>` : '';
            let content = line.replace(/^(\d{2}:\d{2}:\d{2})\s?\|\s?/, '');

            // 키워드 하이라이팅
            content = content.replace(/\[([A-Z0-9\/]+)\]/g, '<span class="coin">[$1]</span>');
            content = content.replace(/(ROI: \+?[\d\.]+%\s?\|\s?PnL: \$[\d\.]+)/g, '<span class="profit">$1</span>'); // 수익
            content = content.replace(/(ROI: -[\d\.]+%\s?\|\s?PnL: \$-[\d\.]+)/g, '<span class="loss">$1</span>'); // 손실

            return `<div class="log-line ${cssClass}">${timestamp}${content}</div>`;
        }

        async function fetchLogs() {
            try {
                const response = await fetch('/data');
                const data = await response.json();
                
                // 새로운 로그만 렌더링하지 않고 전체를 다시 그리는 방식 (필터링 유지 위해)
                // 실제 프로덕션에서는 diff만 그리는 게 좋지만 여기선 간단히 구현
                if (JSON.stringify(data.logs) !== JSON.stringify(allLogs)) {
                    allLogs = data.logs;
                    renderLogs();
                }
            } catch (error) {
                console.error('Fetch error:', error);
            }
        }

        function renderLogs() {
            const filterText = document.getElementById('filterInput').value.toUpperCase();
            const filteredHTML = allLogs
                .filter(line => line.toUpperCase().includes(filterText))
                .map(line => formatLine(line))
                .join('');
            
            container.innerHTML = filteredHTML;
        }

        function filterLogs() {
            renderLogs();
        }

        setInterval(fetchLogs, 2000);
        fetchLogs();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/data')
def get_data():
    if not os.path.exists(LOG_FILE):
        return jsonify({"logs": ["waiting for bot..."]})
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 최신 로그가 위로 오도록 역순 정렬
            last_lines = lines[-MAX_LINES:]
            last_lines.reverse() 
            return jsonify({"logs": [line.strip() for line in last_lines]})
    except Exception as e:
        return jsonify({"logs": [f"Error: {str(e)}"]})

if __name__ == '__main__':
    try:
        ngrok.kill()
        if NGROK_AUTH_TOKEN:
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
            print(f"🔑 Ngrok Authenticated")
        
        url = ngrok.connect(PORT).public_url
        print(f"\n=======================================================")
        print(f"📲 DASHBOARD URL: {url}")
        print(f"=======================================================\n")
        
    except Exception as e:
        print(f"❌ Ngrok Error: {e}")

    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)