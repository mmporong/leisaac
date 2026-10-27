"""Local page that follows a read_so101_joints.py --watch-seconds session: live wrist view, progress, result.

Serves http://127.0.0.1:<port>/ from the session status file. Reads only; it never opens the serial port.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import urllib.request

PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>자세 포착</title>
<style>
:root{--bg:#fff;--ink:#1d2433;--muted:#667085;--line:#e4e7ec;--accent:#2e90fa;--ok:#12b76a;--warn:#f79009}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Noto Sans KR",sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px} .muted{color:var(--muted)}
.banner{border-radius:12px;padding:16px 18px;margin:12px 0;font-size:22px;font-weight:700;background:#eff8ff;color:#175cd3}
.banner.done{background:#ecfdf3;color:#027a48} .banner.warn{background:#fffaeb;color:#b54708}
.grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:16px}
@media (max-width:720px){.grid{grid-template-columns:1fr}}
.card{border:1px solid var(--line);border-radius:12px;padding:12px}
img.live{width:100%;border-radius:8px;background:#f2f4f7;aspect-ratio:352/288;object-fit:cover}
.steps{display:flex;gap:8px;margin:8px 0} .step{flex:1;text-align:center;padding:8px 0;border-radius:8px;background:#f2f4f7;font-weight:600}
.step.on{background:var(--accent);color:#fff} .step.ok{background:var(--ok);color:#fff}
.bar{height:12px;background:#f2f4f7;border-radius:6px;overflow:hidden} .bar>div{height:100%;background:var(--warn);width:0}
.thumbs{display:grid;grid-template-columns:repeat(4,1fr);gap:6px} .thumbs img{width:100%;border-radius:6px}
table{width:100%;border-collapse:collapse;font-size:14px} td,th{border-bottom:1px solid var(--line);padding:6px;text-align:left}
tr.best td{background:#ecfdf3;font-weight:700}
</style></head><body><main>
<h1>SO101 관절 대응 측정 — 자세 포착</h1>
<div class="muted">읽기 전용(모터 쓰기 없음). 큐브는 그대로 두고, 팔을 손으로 옮겨 큐브를 위에서 내려다본 채 2초 멈추세요.</div>
<div id="banner" class="banner">연결 중…</div>
<div class="grid">
 <div class="card"><img class="live" id="live" alt="손목 카메라"><div class="muted" id="clock"></div></div>
 <div class="card">
  <div class="steps" id="steps"></div>
  <div class="muted">멈춤 진행</div><div class="bar"><div id="hold"></div></div>
  <p class="muted" id="hint"></p>
  <div class="thumbs" id="thumbs"></div>
 </div>
</div>
<div class="card" id="resultCard" style="display:none;margin-top:16px">
 <h2 style="font-size:17px;margin:0 0 8px">결과: 자세마다 계산한 큐브 위치가 얼마나 모이는가</h2>
 <table><thead><tr><th>관절 대응 후보</th><th>자세 수</th><th>평균 위치 (m)</th><th>평균에서 최대 벗어남</th></tr></thead><tbody id="rows"></tbody></table>
 <p class="muted">가장 작게 모이는 후보가 실물과 시뮬의 관절 대응으로 맞는 쪽입니다.</p>
</div>
</main><script>
const names={leisaac_linear_rad:"LeIsaac 선형 대응",ticks_from_2047_rad:"눈금 2047 기준",ticks_from_range_mid_rad:"범위 중점 기준"};
async function tick(){
 try{
  const s=await (await fetch('/status',{cache:'no-store'})).json();
  const b=document.getElementById('banner');
  b.className='banner'+(s.phase==='done'?' done':(s.phase==='no_cube'||s.phase==='timeout')?' warn':'');
  b.textContent=s.message||s.phase;
  const n=s.captures?s.captures.length:0, t=s.target||4;
  const names={hover:'큐브 위',grasp:'잡기',box:'상자 위',rest:'휴식'};
  document.getElementById('steps').innerHTML=Array.from({length:t},(_,i)=>`<div class="step ${i<n?'ok':(i===n&&s.phase!=='done'?'on':'')}">${s.labels?(names[s.labels[i]]||s.labels[i]):'자세 '+(i+1)}</div>`).join('');
  document.getElementById('hold').style.width=(s.phase==='done'?0:Math.round(100*(s.hold_fraction||0)))+'%';
  const left=s.deadline&&s.now?Math.max(0,Math.round(s.deadline-s.now)):null;
  document.getElementById('hint').textContent=left!==null&&s.phase!=='done'?`남은 시간 ${Math.floor(left/60)}분 ${left%60}초`:'';
  document.getElementById('thumbs').innerHTML=(s.captures||[]).map(c=>`<img src="/capture/${c.jpg}" alt="${c.label}">`).join('');
  if(s.phase==='done'&&s.result){
   document.getElementById('resultCard').style.display='block';
   const m=s.result.mappings.filter(x=>x.max_deviation_from_mean_m!==null);
   const best=m.reduce((a,x)=>a===null||x.max_deviation_from_mean_m<a.max_deviation_from_mean_m?x:a,null);
   document.getElementById('rows').innerHTML=s.result.mappings.map(x=>`<tr class="${best&&x.mapping===best.mapping?'best':''}"><td>${names[x.mapping]||x.mapping}</td><td>${x.poses}</td><td>${x.mean_xy_m?x.mean_xy_m.map(v=>v.toFixed(3)).join(', '):'-'}</td><td>${x.max_deviation_from_mean_m===null?'-':(x.max_deviation_from_mean_m*100).toFixed(1)+' cm'}</td></tr>`).join('');
  }
 }catch(e){document.getElementById('banner').textContent='세션 상태를 기다리는 중…';}
 document.getElementById('live').src='/frame.jpg?'+Date.now();
 document.getElementById('clock').textContent=new Date().toLocaleTimeString('ko-KR');
}
setInterval(tick,500); tick();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--frame-url", default="http://127.0.0.1:8765/frame.jpg")
    args = parser.parse_args()
    session = args.session_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/status":
                status = session / "status.json"
                waiting = {"phase": "waiting", "message": "세션 시작을 기다리는 중"}
                body = status.read_bytes() if status.exists() else json.dumps(waiting, ensure_ascii=False).encode()
                self._send(200, body, "application/json")
            elif path == "/frame.jpg":
                try:
                    with urllib.request.urlopen(args.frame_url, timeout=2) as response:
                        self._send(200, response.read(), "image/jpeg")
                except OSError:
                    self._send(503, b"", "text/plain")
            elif path.startswith("/capture/"):
                target = (session / path.removeprefix("/capture/")).resolve()
                if target.parent == session and target.suffix == ".jpg" and target.exists():
                    self._send(200, target.read_bytes(), "image/jpeg")
                else:
                    self._send(404, b"", "text/plain")
            else:
                self._send(404, b"", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"POSE_CAPTURE_DASHBOARD http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
