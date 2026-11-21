#!/usr/bin/env python3
"""team_node.py — 로봇이 경기 '시작 대기' 때 `python3 ~/team/team_node.py` 로 실행하는 시작 파일 (이름 · 위치 고정).

하는 일
  1. ~/team/ws/install 에 빌드해 둔 팀 작업공간의 환경으로 자기 자신을 다시 시작한다 (프로세스 번호는 그대로)
  2. programs.json 에 적힌 나머지 프로그램(navigation · perception · server_bridge · llm_worker …)을 자식으로 켠다
  3. 이 프로세스에서 supervisor 를 돌린다 (경기 신호 stop 을 받으면 돌아온다)
  4. 자식들을 끄고 종료 코드 0 으로 끝난다

공식 뼈대의 규칙: 다른 노드는 이 파일이 띄우고, 종료 신호를 받으면 함께 끝낸다. stop 뒤 1.5초 안에 끝나지 않으면 종료 신호(SIGINT),
그 뒤 3초면 강제 종료. 종료 코드가 0 이 아니면 로봇이 다시 실행한다(경기당 3번). 정상 종료는 0, supervisor 가 뜻밖의 예외로 죽었을 때만 1 로 끝낸다.

로봇의 폴더 (이 파일이 있는 곳 = ~/team)
  team_node.py      이 파일
  programs.json     켤 프로그램 목록
  supervisor.json   supervisor 설정 (없어도 됨)
  secrets.json      서버 주소 · 임무코드 · LLM 키 (저장소에 올리지 않는다 — secrets.example.json 참고)
  ws/               팀 작업공간 (src 를 복사해 로봇 안에서 colcon build → ws/install)
  log/<날짜_시각>/   이번 실행의 기록 폴더 (실행할 때마다 하나). 프로그램들이 환경변수 TEAM_LOG_DIR 로 받아 그 아래에 쓴다
      team_node.log · <프로그램>.log   이 파일과 자식 프로그램의 출력
      version.json                    그때 돌린 코드의 지문 (어느 커밋의 코드였는지 나중에 맞춰 볼 수 있게)
      supervisor/ · server_bridge/ · llm_worker/ · ros/   프로그램별 기록

  python3 team_node.py --fingerprint [폴더]   그 폴더(기본 ws/src) 아래 패키지들의 코드 지문만 찍고 끝낸다
  python3 team_node.py --pack-logs [--all] [--no-photos]
        log/ 아래의 기록을 team_logs_<날짜_시각>.tgz 하나로 묶는다 (아직 묶지 않은 실행만. --all 은 전부, --no-photos 는 사진 · 드론 화면 제외).
        이 파일을 내려받아 저장소의 로그_파일/ 에 넣는다. 비밀 파일(secrets.json)은 들어가지 않는다
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INSTALL = os.path.join(HERE, "ws", "install")
LOG_DIR = os.path.join(HERE, "log")
STOP_WAIT_S = 0.8  # 자식에게 종료 신호를 보낸 뒤 기다리는 시간 (전체가 1.5초 안에 끝나야 한다)
MAX_RESTARTS = 3
LIGHT_BELOW = 1.5e9  # 남은 디스크가 이보다 적으면 사진 · 드론 화면은 남기지 않는다 (글 기록만)
OFF_BELOW = 2.0e8  # 이보다 적으면 기록을 아예 남기지 않는다
SOURCE_EXT = (".py", ".msg", ".srv", ".action", ".xml", ".cfg", ".yaml", ".yml", ".json", ".txt")
SKIP_DIRS = ("__pycache__", "build", "install", "log", ".git", ".pytest_cache", ".ruff_cache")
SESSION = None  # 이번 실행의 기록 폴더 (없으면 None)
# programs.json 을 읽지 못했을 때 켜는 목록 (파일이 깨져도 경기는 치르게). 평소에는 programs.json 이 이것을 대신한다
DEFAULT_PROGRAMS = [
    {"name": "lidar", "cmd": ["bash", "lidar.sh"]},  # /scan 이 없을 때만 라이다를 켠다 (lidar.sh 머리말)
    {"name": "navigation_slam", "package": "navigation", "executable": "navigation_slam"},
    {"name": "navigation", "package": "navigation", "executable": "navigation_drive"},
    {"name": "perception", "package": "perception", "executable": "perception_node"},
    {"name": "server_bridge", "package": "server_bridge", "executable": "server_bridge_node"},
    {"name": "llm_worker", "package": "llm_worker", "executable": "llm_worker_node"},
]


def say(text):
    print("[team_node] " + text, flush=True)
    if SESSION:
        try:
            with open(os.path.join(SESSION, "team_node.log"), "a", encoding="utf-8") as fh:
                fh.write("%s %s\n" % (time.strftime("%H:%M:%S"), text))
        except OSError:
            pass


def fingerprint(root):
    """폴더 아래 소스 파일들의 이름과 내용으로 만든 지문 (sha256 앞 12자) 과 파일 수. 줄바꿈(CRLF · LF) 차이는 무시한다."""
    digest, count = hashlib.sha256(), 0
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for name in sorted(files):
            if not name.endswith(SOURCE_EXT) or name == "secrets.json":
                continue
            path = os.path.join(folder, name)
            try:
                with open(path, "rb") as fh:
                    data = fh.read().replace(b"\r\n", b"\n")
            except OSError:
                continue
            digest.update(os.path.relpath(path, root).replace(os.sep, "/").encode("utf-8") + b"\0" + data + b"\0")
            count += 1
    return digest.hexdigest()[:12], count


def package_fingerprints(src):
    out = {}
    if os.path.isdir(src):
        for name in sorted(os.listdir(src)):
            if os.path.isdir(os.path.join(src, name)) and name not in SKIP_DIRS:
                out[name] = fingerprint(os.path.join(src, name))[0]
    return out


def write_version(programs):
    """이번 실행에 쓴 코드가 무엇이었는지 적어 둔다 (기록을 나중에 볼 때 어느 커밋이었는지 맞춰 보려고). 배경 스레드에서 돈다."""
    if not SESSION:
        return
    info = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "team_node": fingerprint_file(os.path.abspath(__file__)),
            "packages": package_fingerprints(os.path.join(HERE, "ws", "src")), "programs": [p.get("name") for p in programs], "version_file": None, "git": None}
    try:
        with open(os.path.join(HERE, "VERSION"), encoding="utf-8") as fh:
            info["version_file"] = fh.read().strip()[:400]
    except OSError:
        pass
    try:  # 저장소를 통째로 내려받아 둔 경우
        out = subprocess.run(["git", "-C", os.path.realpath(os.path.join(HERE, "ws", "src")), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=3)
        info["git"] = out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(os.path.join(SESSION, "version.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=1)
    except OSError:
        pass


def fingerprint_file(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()[:12]
    except OSError:
        return None


def open_session():
    """이번 실행의 기록 폴더를 만들고 환경변수로 알린다. 다시 시작(enter_workspace)한 뒤에는 앞에서 만든 것을 그대로 쓴다."""
    global SESSION
    if os.environ.get("TEAM_NODE_WS") == INSTALL and (os.environ.get("TEAM_LOG_DIR") or os.environ.get("TEAM_LOG_OFF") == "1"):
        SESSION = os.environ.get("TEAM_LOG_DIR") or None
        return
    for name in ("TEAM_LOG_DIR", "TEAM_LOG_OFF", "TEAM_LOG_LIGHT"):
        os.environ.pop(name, None)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        free = shutil.disk_usage(LOG_DIR).free
    except OSError:
        free = 0
    if free < OFF_BELOW:
        os.environ["TEAM_LOG_OFF"] = "1"
        say("디스크가 %.0f MB 밖에 남지 않아 기록을 남기지 않습니다" % (free / 1e6))
        return
    folder = os.path.join(LOG_DIR, time.strftime("%Y%m%d_%H%M%S"))
    try:
        os.makedirs(os.path.join(folder, "ros"), exist_ok=True)
    except OSError:
        os.environ["TEAM_LOG_OFF"] = "1"
        return
    SESSION = folder
    os.environ["TEAM_LOG_DIR"] = folder
    os.environ["ROS_LOG_DIR"] = os.path.join(folder, "ros")  # 각 프로그램의 ROS 기록 파일도 이 폴더로
    if free < LIGHT_BELOW:
        os.environ["TEAM_LOG_LIGHT"] = "1"
        say("디스크가 %.1f GB 남아 사진 · 드론 화면은 남기지 않습니다 (글 기록만)" % (free / 1e9))


def workspace_env():
    """ws/install/setup.bash 를 읽은 뒤의 환경 변수들. 못 읽으면 None."""
    setup = os.path.join(INSTALL, "setup.bash")
    if not os.path.exists(setup):
        say("팀 작업공간이 없습니다: %s (로봇 안에서 빌드해야 합니다 — README 참고)" % setup)
        return None
    try:
        out = subprocess.run(["bash", "-c", 'source "%s" >/dev/null 2>&1 && python3 -c "import os, json; print(json.dumps(dict(os.environ)))"' % setup],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        return json.loads(out.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        say("작업공간 환경을 읽지 못했습니다: %r" % (exc,))
        return None


def enter_workspace():
    """팀 작업공간의 환경으로 이 프로그램을 다시 시작한다 (같은 프로세스 번호 그대로).

    메시지 형식의 라이브러리는 프로그램이 시작될 때의 환경(LD_LIBRARY_PATH)으로만 찾으므로, 켜진 뒤에 환경만 바꿔서는 못 읽는다.
    이미 그 환경이면 True, 작업공간이 없으면 False 를 돌려준다. 다시 시작하면 돌아오지 않는다.
    """
    if os.environ.get("TEAM_NODE_WS") == INSTALL:
        return True
    env = workspace_env()
    if env is None:
        return False
    env["TEAM_NODE_WS"] = INSTALL
    sys.stdout.flush()
    os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env)
    return False  # 여기는 오지 않는다


def command_of(prog):
    """programs.json 의 한 줄 → 실행 명령. package/executable 이면 설치된 실행 파일을 바로 부른다 (ros2 run 을 거치지 않는다)."""
    if "cmd" in prog:
        return [str(c) for c in prog["cmd"]]
    exe = os.path.join(INSTALL, prog["package"], "lib", prog["package"], prog["executable"])
    if not os.path.exists(exe):
        return ["ros2", "run", prog["package"], prog["executable"]] + [str(a) for a in prog.get("args", [])]
    return [exe] + [str(a) for a in prog.get("args", [])]


class Children:
    def __init__(self, programs):
        self.programs = programs
        self.procs = {}  # 이름 → Popen
        self.restarts = {}
        self.closing = False

    def start(self, prog):
        name = prog["name"]
        try:
            log = open(os.path.join(SESSION, name + ".log") if SESSION else os.devnull, "ab")
            env = dict(os.environ, **{k: os.path.expandvars(str(v)) for k, v in prog.get("env", {}).items()})  # 값에 ${PYTHONPATH} 처럼 쓸 수 있다
            self.procs[name] = subprocess.Popen(command_of(prog), cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
            say("켬: %s (pid %d)" % (name, self.procs[name].pid))
        except Exception as exc:  # noqa: BLE001 — 프로그램 하나가 안 켜져도 나머지는 돈다
            say("못 켬: %s — %r" % (name, exc))

    def start_all(self):
        for prog in self.programs:
            self.start(prog)

    def watch(self):
        """죽은 자식을 다시 켠다 (프로그램마다 MAX_RESTARTS 번까지)."""
        while not self.closing:
            time.sleep(1.0)
            for prog in self.programs:
                name = prog["name"]
                proc = self.procs.get(name)
                if self.closing or proc is None or proc.poll() is None or not prog.get("restart", True):
                    continue
                count = self.restarts.get(name, 0)
                if count >= MAX_RESTARTS:
                    continue
                self.restarts[name] = count + 1
                say("%s 가 끝났습니다 (코드 %s) → 다시 켭니다 (%d/%d)" % (name, proc.returncode, count + 1, MAX_RESTARTS))
                self.start(prog)

    def stop_all(self):
        self.closing = True
        alive = [p for p in self.procs.values() if p.poll() is None]
        for p in alive:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:  # noqa: BLE001
                pass
        end = time.monotonic() + STOP_WAIT_S
        while time.monotonic() < end and any(p.poll() is None for p in alive):
            time.sleep(0.05)
        for p in alive:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass


def pack_logs(args):
    """기록 폴더들을 압축 파일 하나로 묶는다. 돌려주는 값: 만든 파일 경로 (묶을 것이 없으면 None)."""
    import io
    import tarfile

    everything, photos = "--all" in args, "--no-photos" not in args
    sessions = sorted(d for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))) if os.path.isdir(LOG_DIR) else []
    chosen = [d for d in sessions if everything or not os.path.exists(os.path.join(LOG_DIR, d, ".packed"))]
    if not chosen:
        print("묶을 새 기록이 없습니다 (%s 아래 %d개 실행은 이미 묶었습니다. 다시 묶으려면 --all)" % (LOG_DIR, len(sessions)))
        return None
    out = os.path.join(HERE, "team_logs_%s.tgz" % time.strftime("%Y%m%d_%H%M%S"))

    def keep(info):
        name = os.path.basename(info.name)
        if name in ("secrets.json", ".packed"):
            return None
        if not photos and info.isfile() and ("/photos/" in info.name or "/drone/" in info.name):
            return None
        return info

    with tarfile.open(out, "w:gz") as tar:
        for d in chosen:
            tar.add(os.path.join(LOG_DIR, d), arcname="log/" + d, filter=keep)
        for path in ("~/mil/log/team_node.log", "/home/jetson/mil/log/team_node.log", "/home/ws/mil/log/team_node.log"):  # 로봇이 team_node.py 의 출력을 받아 적는 파일
            path = os.path.expanduser(path)
            if os.path.isfile(path):
                with open(path, "rb") as fh:
                    fh.seek(max(0, os.path.getsize(path) - 2000000))  # 끝의 2 MB 만
                    data = fh.read()
                info = tarfile.TarInfo("mil_log/team_node.log")
                info.size, info.mtime = len(data), os.path.getmtime(path)
                tar.addfile(info, io.BytesIO(data))
                break
    now = time.time()
    for d in chosen:
        folder = os.path.join(LOG_DIR, d)
        newest = max([os.path.getmtime(os.path.join(root, f)) for root, _dirs, files in os.walk(folder) for f in files] or [0.0])
        if now - newest > 30.0:  # 아직 돌고 있는 실행은 다음에 다시 묶는다
            try:
                with open(os.path.join(folder, ".packed"), "w", encoding="utf-8") as fh:
                    fh.write(os.path.basename(out))
            except OSError:
                pass
    print("묶음: %s (%d개 실행, %.1f MB%s)" % (out, len(chosen), os.path.getsize(out) / 1e6, "" if photos else ", 사진 제외"))
    return out


def load_programs():
    path = os.path.join(HERE, "programs.json")
    try:
        with open(path, encoding="utf-8") as fh:
            listed = json.load(fh)["programs"]
        return [p for p in listed if isinstance(p, dict) and p.get("name") and p.get("enabled", True)]
    except Exception as exc:  # noqa: BLE001 — 파일이 없거나 깨졌다
        say("programs.json 을 읽지 못해 기본 목록으로 켭니다: %r" % (exc,))
        return [dict(p) for p in DEFAULT_PROGRAMS]


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--fingerprint":
        src = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(HERE, "ws", "src")
        print(json.dumps({"team_node": fingerprint_file(os.path.abspath(__file__)), "packages": package_fingerprints(src)}, ensure_ascii=False, indent=1))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "--pack-logs":
        pack_logs(sys.argv[2:])
        return
    children = None
    code = 0  # 정상 종료 · 종료 신호 · 작업공간 없음은 0. 뜻밖의 예외만 1 (로봇이 다시 실행 — supervisor 는 저장해 둔 상태를 읽어 이어 간다)
    try:
        os.chdir(HERE)  # 설정 · 비밀 · 상태 파일을 모두 이 폴더 기준으로 읽고 쓴다 (로봇이 어느 폴더에서 실행하든)
        open_session()
        programs = load_programs()
        ready = enter_workspace()
        children = Children(programs)
        if ready:
            say("기록 폴더: %s" % (SESSION or "없음"))
            threading.Thread(target=write_version, args=(programs,), daemon=True).start()
            children.start_all()
            threading.Thread(target=children.watch, daemon=True).start()
            os.environ.setdefault("SUPERVISOR_CONFIG", os.path.join(HERE, "supervisor.json"))
            from supervisor.node import main as supervisor_main

            supervisor_main()  # stop 을 받으면 돌아온다
        else:
            say("supervisor 를 돌릴 수 없어 그대로 끝냅니다")
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001 — supervisor 가 예외로 죽었다 → 로봇이 다시 실행하도록 0 이 아닌 코드로 끝낸다
        import traceback

        say("예외로 끝남: %r\n%s" % (exc, traceback.format_exc()))
        code = 1
    finally:
        try:
            if children is not None:
                children.stop_all()
        except KeyboardInterrupt:
            pass
        except Exception:  # noqa: BLE001
            pass
    sys.stdout.flush()
    os._exit(code)  # 남은 스레드를 기다리지 않고 끝낸다


if __name__ == "__main__":
    main()
    sys.exit(0)
