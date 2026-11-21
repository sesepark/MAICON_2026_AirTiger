#!/usr/bin/env python3
"""임시 시험용: 웹에서 정해진 거리·각도(또는 짧은 이동) 명령으로 /cmd_vel 을 내보내고, /odom 이 얼마나 변했는지와 줄자로 잰 실측 값을 비교해 로그에 남긴다.

- 로봇의 ROS 컨테이너(`sudo ugv-shell`) 안에서 실행한다. PC 브라우저로 http://<로봇IP>:8790 접속.
- 새 노드·토픽·서비스를 만들지 않는다 (노드 `tmp_teleop` 은 시험하는 동안만 존재). /cmd_vel 발행, /odom 구독만 한다.
- 이동은 한 번에 한 개만, 거리·각도·속도 상한 있음. STOP 은 항상 즉시. 이 파일은 `navigation/시험기록/` 의 시험 도구이며 정식 navigation 노드가 아니다.
- 모든 요청·명령·odom 샘플·결과·실측은 jsonl 로그로 남는다 (README.md 참고).

명령 방식
  time : 속도 × 시간(= 목표 / 속도) 만큼 일정 속도로 보낸다 (개방 루프). 명령 대비 실제 오차를 잴 때.
  odom : /odom 이 목표에 닿을 때까지 보내고 멈춘다. 정지 후 미끄러짐(오버슈트)을 잴 때. 못 닿으면 시간 상한에서 멈춘다.
"""
import json
import math
import os
import signal
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

PORT = 8790         # 8765 는 ~/team 의 UGV_Sensor_Server.py 가 쓰고 있다
MAX_LIN = 0.20      # m/s  (로봇 브리지 상한은 0.6)
MAX_ANG = 0.80      # rad/s (브리지 상한은 3.0)
MAX_DIST = 1.5      # m   (한 번에 가는 거리 상한)
MAX_TURN = 360.0    # deg (한 번에 도는 각도 상한)
MAX_DUR = 20.0      # s
RATE_HZ = 20.0      # 브리지 cmd_vel_timeout 이 0.5 s 이므로 그보다 훨씬 자주 보낸다
SETTLE_S = 0.8      # 정지 후 미끄러짐까지 기다린 뒤 odom 을 읽는다
LOG_DIR = os.environ.get('TELEOP_LOG_DIR', '/home/ws/team/tmp_teleop_logs')   # 컨테이너 경로. 호스트에서는 /home/jetson/team/tmp_teleop_logs


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Log:
    """한 줄에 JSON 하나(jsonl). 서버가 켜질 때마다 새 파일. 모든 줄을 쓰자마자 flush 한다."""

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path = os.path.join(LOG_DIR, 'teleop_%s.jsonl' % datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.f = open(self.path, 'a', encoding='utf-8')
        self.lock = threading.Lock()

    def w(self, ev, **kw):
        rec = {'t': time.time(), 'iso': datetime.now().isoformat(timespec='milliseconds'), 'ev': ev}
        rec.update(kw)
        with self.lock:
            self.f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            self.f.flush()


class Teleop(Node):
    def __init__(self, log):
        super().__init__('tmp_teleop')
        self.log = log
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom = None
        self.match_state = None
        self.move_id = 0
        self.rec = False                        # True 인 동안 odom 을 한 샘플도 빠짐없이 기록
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(String, '/mil/match_state', self._on_match, 10)
        self.busy = threading.Lock()
        self.abort = threading.Event()
        self.results = {}                       # move id -> 결과 (실측 입력 때 오차 계산용)

    def _on_match(self, m):
        if m.data != self.match_state:
            self.log.w('match_state', old=self.match_state, new=m.data)
        self.match_state = m.data

    def _on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom = (m.pose.pose.position.x, m.pose.pose.position.y, yaw, time.time())
        if self.rec:
            self.log.w('odom', move=self.move_id, x=m.pose.pose.position.x, y=m.pose.pose.position.y, yaw=yaw,
                       vx=m.twist.twist.linear.x, wz=m.twist.twist.angular.z,
                       stamp=m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)

    def zero(self, n=3, why=''):
        for _ in range(n):
            self.pub.publish(Twist())
            time.sleep(0.02)
        self.log.w('cmd_zero', move=self.move_id, why=why)

    def snapshot(self):
        o = self.odom
        return None if o is None else {'x': o[0], 'y': o[1], 'yaw': o[2], 'age': round(time.time() - o[3], 2)}

    # ---- 명령 해석 -------------------------------------------------------------
    def plan(self, p):
        """요청 -> (lin, ang, 시간상한, 목표종류, 목표값, 모드). 목표값은 dist=m(부호 있음), turn=deg(부호 있음), raw=이상적 이동량."""
        kind = p.get('kind', 'raw')
        mode = p.get('mode', 'time')
        if mode not in ('time', 'odom'):
            raise ValueError('mode')
        if kind == 'dist':
            v = max(-MAX_DIST, min(MAX_DIST, float(p['value'])))
            sp = max(0.05, min(MAX_LIN, abs(float(p['speed']))))
            dur = abs(v) / sp
            lin, ang = math.copysign(sp, v), 0.0
            tgt = v
        elif kind == 'turn':
            v = max(-MAX_TURN, min(MAX_TURN, float(p['value'])))
            sp = max(0.1, min(MAX_ANG, abs(float(p['speed']))))
            dur = math.radians(abs(v)) / sp
            lin, ang = 0.0, math.copysign(sp, v)
            tgt = v
        else:   # raw: 화살표 버튼
            lin = max(-MAX_LIN, min(MAX_LIN, float(p['lin'])))
            ang = max(-MAX_ANG, min(MAX_ANG, float(p['ang'])))
            dur = max(0.1, float(p['dur']))
            tgt = lin * dur if lin else math.degrees(ang * dur)
            mode = 'time'
            kind = 'dist' if lin else 'turn'
        if dur <= 0:
            raise ValueError('zero')
        if mode == 'odom':
            dur = dur * 2 + 2.0                 # 못 닿을 때를 위한 시간 상한
        return lin, ang, min(dur, MAX_DUR), kind, tgt, mode

    def move(self, p):
        lin, ang, dur, kind, tgt, mode = self.plan(p)
        if not self.busy.acquire(blocking=False):
            self.log.w('move_rejected', reason='busy', req=p)
            return 409, {'error': '이미 이동 중'}
        try:
            self.abort.clear()
            a = self.odom
            if a is None or time.time() - a[3] > 1.0:
                self.log.w('move_rejected', reason='no_odom', req=p)
                return 503, {'error': '/odom 이 들어오지 않음'}
            self.move_id += 1
            mid = self.move_id
            self.log.w('move_start', move=mid, req=p, lin=lin, ang=ang, time_limit=round(dur, 3), kind=kind, target=tgt, mode=mode,
                       match_state=self.match_state, odom_start={'x': a[0], 'y': a[1], 'yaw': a[2], 'age': round(time.time() - a[3], 3)})
            state = {'last_yaw': a[2], 'acc': 0.0}

            def progress():
                o = self.odom
                if kind == 'dist':
                    return (o[0] - a[0]) * math.cos(a[2]) + (o[1] - a[1]) * math.sin(a[2])      # 출발 방향 기준 앞으로 간 거리
                state['acc'] += wrap(o[2] - state['last_yaw'])
                state['last_yaw'] = o[2]
                return math.degrees(state['acc'])

            self.rec = True
            t0 = time.time()
            msg = Twist()
            msg.linear.x, msg.angular.z = lin, ang
            n_pub, reached = 0, False
            while time.time() - t0 < dur and not self.abort.is_set():
                if mode == 'odom' and math.copysign(1, tgt) * progress() >= abs(tgt):
                    reached = True
                    break
                self.pub.publish(msg)
                n_pub += 1
                self.log.w('cmd', move=mid, lin=lin, ang=ang, since_start=round(time.time() - t0, 3))
                time.sleep(1.0 / RATE_HZ)
            aborted = self.abort.is_set()
            self.zero(why='abort' if aborted else ('odom_target_reached' if reached else 'end_of_move'))
            t_stop = time.time()
            time.sleep(SETTLE_S)
            self.rec = False
            b = self.odom
            dx, dy = b[0] - a[0], b[1] - a[1]
            fwd = dx * math.cos(a[2]) + dy * math.sin(a[2])
            dyaw_deg = math.degrees(state['acc'] + wrap(b[2] - state['last_yaw']))    # 누적(여러 바퀴 가능) 회전각
            res = {
                'move': mid, 'kind': kind, 'mode': mode, 'target': round(tgt, 3), 'aborted': aborted, 'reached': reached,
                'cmd': {'lin': lin, 'ang': ang, 'drive_s': round(t_stop - t0, 3)},
                'odom_forward_m': round(fwd, 3), 'odom_dist_m': round(math.hypot(dx, dy), 3), 'odom_dyaw_deg': round(dyaw_deg, 1),
            }
            self.results[mid] = res
            self.log.w('move_end', move=mid, published=n_pub, result=res, odom_end={'x': b[0], 'y': b[1], 'yaw': b[2]})
            return 200, res
        except Exception:
            self.rec = False
            self.zero(why='exception')
            self.log.w('move_error', move=self.move_id, trace=traceback.format_exc())
            return 500, {'error': '서버 오류 (로그 참고)'}
        finally:
            self.rec = False
            self.busy.release()

    def measured(self, p):
        mid = int(p['move'])
        r = self.results.get(mid)
        val = float(p['value'])                  # 실측: dist 는 m, turn 은 deg (크기)
        out = {'move': mid, 'measured': val, 'note': str(p.get('note', ''))}
        if r:
            odom_val = abs(r['odom_forward_m']) if r['kind'] == 'dist' else abs(r['odom_dyaw_deg'])
            out.update(kind=r['kind'], mode=r['mode'], target=abs(r['target']), odom=odom_val,
                       err_vs_target=round(val - abs(r['target']), 3), err_odom_vs_measured=round(odom_val - val, 3))
        self.log.w('measured', **out)
        return 200, out


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>UGV-08 임시 조종</title>
<style>
body{font:15px system-ui,'Malgun Gothic',sans-serif;margin:0;background:#111;color:#ddd;padding:12px;max-width:760px}
.pad{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0}
button{font-size:17px;padding:14px 0;border:0;border-radius:10px;background:#2b6cb0;color:#fff;cursor:pointer}
button:active{background:#1e4e82}.stop{background:#c53030;grid-column:1/4;font-size:24px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}.row button{flex:1;min-width:92px;padding:12px 6px}
input[type=number],select{background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:6px;width:84px}
label{display:block;margin:8px 0 2px}input[type=range]{width:100%}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #333;padding:3px 5px;text-align:right}
#st{color:#9c9}.warn{color:#f6ad55}h4{margin:14px 0 4px}.fwd{background:#2f855a}.rot{background:#6b46c1}
</style>
<h3>UGV-08 임시 조종 · 오차 측정 (시험용)</h3>
<div class=warn>정해진 만큼 움직이고 멈춥니다. 한 번에 하나만. STOP/스페이스바는 즉시 정지. 모든 기록은 로봇의 로그 파일에 남습니다.</div>
<div id=st>연결 중…</div>
<div class=pad><button class=stop onclick="stop()">STOP (스페이스)</button></div>

<h4>정해진 거리·각도 명령</h4>
<div class=row>방식 <select id=mode><option value=time>시간 기준 (개방 루프)</option><option value=odom>odom 기준</option></select>
 직진 속도 <input id=sl type=number value=0.1 step=0.05 min=0.05 max=0.2> m/s
 회전 속도 <input id=sa type=number value=0.3 step=0.1 min=0.1 max=0.8> rad/s</div>
<div class=row>
 <button class=fwd onclick="go('dist',0.5)">전진 0.5m</button><button class=fwd onclick="go('dist',1)">전진 1m</button>
 <button class=fwd onclick="go('dist',-0.5)">후진 0.5m</button><button class=fwd onclick="go('dist',-1)">후진 1m</button></div>
<div class=row>
 <button class=rot onclick="go('turn',90)">좌 90°</button><button class=rot onclick="go('turn',-90)">우 90°</button>
 <button class=rot onclick="go('turn',180)">좌 180°</button><button class=rot onclick="go('turn',360)">좌 360°</button></div>
<div class=row>직접 입력 <input id=cv type=number value=0.3 step=0.1>
 <button class=fwd onclick="go('dist',+$('cv').value)">전진(m)</button><button class=fwd onclick="go('dist',-$('cv').value)">후진(m)</button>
 <button class=rot onclick="go('turn',+$('cv').value)">좌(°)</button><button class=rot onclick="go('turn',-$('cv').value)">우(°)</button></div>

<h4>짧은 이동 (화살표)</h4>
<div class=pad>
 <span></span><button onclick="mv(1,0)">▲</button><span></span>
 <button onclick="mv(0,1)">◀</button><button onclick="mv(-1,0)">▼</button><button onclick="mv(0,-1)">▶</button></div>
<label>직진 <b id=lv></b> m/s · 회전 <b id=av></b> rad/s · 시간 <b id=dv></b> s</label>
<input id=l type=range min=0.05 max=0.2 step=0.05 value=0.1 oninput=up()>
<input id=a type=range min=0.1 max=0.8 step=0.1 value=0.3 oninput=up()>
<input id=d type=range min=0.2 max=3 step=0.1 value=1 oninput=up()>

<h4>결과 · 실측 입력 (줄자 m / 각도 °, 크기만, 입력 후 Enter)</h4>
<table><thead><tr><th>#</th><th>명령</th><th>odom</th><th>실측</th><th>실측−목표</th><th>odom−실측</th></tr></thead><tbody id=log></tbody></table>
<script>
const $=id=>document.getElementById(id);const R={};
function up(){$('lv').textContent=$('l').value;$('av').textContent=$('a').value;$('dv').textContent=$('d').value;}up();
function ph(s){try{const o=JSON.parse(s);return o.mode+'/'+o.phase;}catch(e){return s||'?';}}
function add(j,label){R[j.move]=j;const tr=document.createElement('tr');tr.id='r'+j.move;
 const u=j.kind=='dist'?'m':'°',o=j.kind=='dist'?Math.abs(j.odom_forward_m):Math.abs(j.odom_dyaw_deg);
 tr.innerHTML='<td>'+j.move+'</td><td>'+label+(j.mode=='odom'?' (odom)':'')+(j.aborted?' ⛔중단':'')+'</td><td>'+o+u+'</td>'+
  '<td><input type=number step=any style="width:70px" onkeydown="if(event.key==\\'Enter\\')meas('+j.move+',this.value)"></td><td id=e1_'+j.move+'></td><td id=e2_'+j.move+'></td>';
 $('log').prepend(tr);}
async function call(path,body){const r=await fetch(path,{method:'POST',body:JSON.stringify(body)});const j=await r.json();
 if(!r.ok){$('st').textContent='오류: '+j.error;return null;}return j;}
async function go(kind,v){const j=await call('/move',{kind,value:v,speed:kind=='dist'?$('sl').value:$('sa').value,mode:$('mode').value});
 if(j)add(j,(kind=='dist'?(v>0?'전진 ':'후진 ')+Math.abs(v)+'m':(v>0?'좌 ':'우 ')+Math.abs(v)+'°'));}
async function mv(s,t){const j=await call('/move',{kind:'raw',lin:s*$('l').value,ang:t*$('a').value,dur:$('d').value});
 if(j)add(j,s?(s>0?'▲ ':'▼ ')+Math.abs(j.target).toFixed(2)+'m':(t>0?'◀ ':'▶ ')+Math.abs(j.target).toFixed(0)+'°');}
async function meas(id,v){if(v==='')return;const j=await call('/measured',{move:id,value:v});if(!j)return;
 $('e1_'+id).textContent=j.err_vs_target;$('e2_'+id).textContent=j.err_odom_vs_measured;}
async function stop(){await fetch('/stop',{method:'POST'});}
addEventListener('keydown',e=>{if(e.code=='Space'&&e.target.tagName!='INPUT'){e.preventDefault();stop();}});
setInterval(async()=>{try{const j=await (await fetch('/state')).json();
 $('st').textContent='odom '+(j.odom?('x '+j.odom.x.toFixed(2)+' y '+j.odom.y.toFixed(2)+' yaw '+(j.odom.yaw*57.3).toFixed(0)+'°'):'없음')+' · 매치 '+ph(j.match_state)+(j.busy?' · 이동 중':'');}
 catch(e){$('st').textContent='로봇과 연결 끊김';}},700);
</script>"""


def make_handler(node):
    log = node.log

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj=None, body=None, ctype='application/json'):
            data = body if body is not None else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype + '; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/state':
                self._send(200, {'odom': node.snapshot(), 'match_state': node.match_state, 'busy': node.busy.locked()})
            else:
                self._send(200, body=PAGE.encode(), ctype='text/html')

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get('Content-Length', 0)) or 0)
            log.w('http', method='POST', path=self.path, client=self.client_address[0], body=raw.decode('utf-8', 'replace'))
            if self.path == '/stop':
                node.abort.set()
                node.zero(why='stop_button')
                self._send(200, {'ok': True})
            elif self.path in ('/move', '/measured'):
                try:
                    p = json.loads(raw or b'{}')
                    code, res = node.move(p) if self.path == '/move' else node.measured(p)
                except (KeyError, ValueError, TypeError):
                    code, res = 400, {'error': '잘못된 요청'}
                    log.w('bad_request', path=self.path, body=raw.decode('utf-8', 'replace'), trace=traceback.format_exc())
                self._send(code, res)
            else:
                self._send(404, {'error': 'not found'})

        def log_message(self, *a):
            pass
    return H


def main():
    log = Log()
    log.w('server_start', port=PORT, max_lin=MAX_LIN, max_ang=MAX_ANG, max_dist=MAX_DIST, max_turn=MAX_TURN, rate_hz=RATE_HZ, pid=os.getpid())
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)     # rclpy 가 SIGINT/SIGTERM 을 가로채면 웹 서버만 남는다
    signal.signal(signal.SIGTERM, signal.default_int_handler)      # kill(TERM)도 Ctrl+C 처럼 -> finally 에서 0 속도 + 로그 마감
    node = Teleop(log)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    srv = ThreadingHTTPServer(('0.0.0.0', PORT), make_handler(node))
    print('teleop 서버 시작: http://0.0.0.0:%d  (Ctrl+C 로 종료)  로그: %s' % (PORT, log.path), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception:
        log.w('server_error', trace=traceback.format_exc())
        raise
    finally:
        node.zero(why='server_exit')
        log.w('server_stop')
        srv.server_close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
