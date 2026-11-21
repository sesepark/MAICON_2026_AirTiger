"""경기 중 내부 상태를 이번 실행의 기록 폴더에 남긴다 (2026-11-19). ROS 와 무관.

경기 · 포털 팀 테스트 중에는 로봇이 팀 폴더의 다른 프로그램을 끄므로 로봇 안에서 ros2 로 들여다볼 수 없다
(로그_파일/1118_팀테스트_경기장1_R2/상황.md 5절 — 거점 진입이 왜 막혔는지 디버그 값을 못 읽었다).
그래서 navigation 두 노드는 /team/navigation/debug 에 내던 것과 같은 내부 상태를 파일에도 쓴다.

- 어디: team_node.py 가 정해 준 TEAM_LOG_DIR 아래 <이름>/debug.jsonl (예: log/20261119_101500/navigation/debug.jsonl)
- 언제 안 쓰나: TEAM_LOG_DIR 이 없을 때(시험 · 웹 시험기 · 손으로 켰을 때), TEAM_LOG_OFF=1 일 때(디스크가 거의 참)
- 한 줄 = JSON 하나. 쓰는 일은 따로 도는 스레드가 한다 — 제어 주기는 줄을 큐에 넣기만 하고 디스크를 기다리지 않는다.
  줄마다 flush 한다 (경기가 끝나면 로봇이 프로그램을 끄므로)
- 한 파일 max_bytes 까지만 쓰고, 넘으면 {"log": "truncated"} 한 줄을 남기고 그만 쓴다. 큐가 차면(디스크가 멈춤) 줄을 버리고 센다
- 쓰다가 오류(디스크 참 등)가 나면 그 뒤로는 조용히 쓰지 않는다 — 주행 · 위치 계산을 막지 않는다
"""
import json
import os
import queue
import threading

SESSION_ENV = 'TEAM_LOG_DIR'
OFF_ENV = 'TEAM_LOG_OFF'
LIGHT_ENV = 'TEAM_LOG_LIGHT'
FILE_NAME = 'debug.jsonl'
MAX_BYTES = 50_000_000        # 한 파일 상한 (2 Hz 로 경기 한 판 수 MB [추정])
LIGHT_MAX_BYTES = 10_000_000  # TEAM_LOG_LIGHT=1 (디스크가 1.5 GB 아래) 일 때
QUEUE_LINES = 2000

_STOP = object()


class RunLog:
    """name: 기록 폴더 아래 하위 폴더 이름 (프로그램 이름과 같게 — navigation, navigation_slam)."""

    def __init__(self, name, environ=None, max_bytes=None):
        env = os.environ if environ is None else environ
        self.light = env.get(LIGHT_ENV) == '1'
        self.max_bytes = int(max_bytes if max_bytes is not None else (LIGHT_MAX_BYTES if self.light else MAX_BYTES))
        self.path = None
        self.written = 0      # 쓴 바이트
        self.lines = 0        # 쓴 줄
        self.dropped = 0      # 큐가 차서 버린 줄
        self.truncated = False
        self._file = None
        self._queue = None
        self._thread = None
        session = env.get(SESSION_ENV)
        if not session or env.get(OFF_ENV) == '1':
            return
        try:
            folder = os.path.join(session, name)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, FILE_NAME)
            self._file = open(path, 'a', encoding='utf-8')
        except OSError:
            self._file = None
            return
        self.path = path
        self._queue = queue.Queue(maxsize=QUEUE_LINES)
        self._thread = threading.Thread(target=self._run, name='run_log:%s' % name, daemon=True)
        self._thread.start()

    @property
    def enabled(self):
        return self._file is not None

    def write(self, record):
        """record(dict) 를 한 줄로 쓰도록 맡긴다. 바로 돌아온다. 맡기지 못했으면(꺼짐 · 상한 · 큐 참) False."""
        if self._file is None:
            return False
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            return False
        return True

    def flush(self, timeout=2.0):
        """맡긴 줄을 다 쓸 때까지 기다린다 (시험 · 끝낼 때). 다 썼으면 True."""
        if self._queue is None:
            return True
        done = threading.Event()
        try:
            self._queue.put(done, timeout=timeout)
        except queue.Full:
            return False
        return done.wait(timeout)

    def close(self, timeout=2.0):
        """남은 줄을 쓰고 파일을 닫는다."""
        if self._thread is None:
            return
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)

    # ---- 쓰는 스레드
    def _run(self):
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            if isinstance(item, threading.Event):
                item.set()
                continue
            if self._file is not None:
                self._write_line(item)
        self._close()

    def _write_line(self, record):
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(',', ':'), default=str) + '\n'
        except (TypeError, ValueError) as error:
            line = json.dumps({'log': 'unserializable', 'error': repr(error)}) + '\n'
        size = len(line.encode('utf-8'))
        try:
            if self.written + size > self.max_bytes:
                self._file.write(json.dumps({'log': 'truncated', 'bytes': self.written, 'lines': self.lines, 'dropped': self.dropped}) + '\n')
                self._file.flush()
                self.truncated = True
                self._close()
                return
            self._file.write(line)
            self._file.flush()
        except (OSError, ValueError):
            self._close()
            return
        self.written += size
        self.lines += 1

    def _close(self):
        f, self._file = self._file, None
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
