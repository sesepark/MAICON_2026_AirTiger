"""llm_worker 의 ROS 연결: /team/llm/analyze 액션 (M1 전술명령 해석).

  PARSE_ORDER   명령문 → target · search_sectors · base.
                1. 낱말 규칙으로 바로 푼다 (LLM 없이도 답이 나온다)
                2. 대회 제공 LLM 에 같은 명령문을 묻고, 답이 wait_s 안에 오면 항목별로 덮어써서 돌려준다
                3. 그 안에 안 오면 낱말 해석을 돌려주되 error_code 를 LLM_PENDING 으로 둔다. LLM 요청은 뒤에서 계속 돌고,
                   같은 명령문으로 다시 물으면 그 답을 (이미 왔으면 바로) 돌려준다. LLM 이 실패했으면 LLM_FAILED — 다시 물으면 새로 요청한다
                어느 경우에도 success=true 이고 command 가 채워져 있다 (낱말 해석도 못 했을 때만 false)
  ASSESS_THREAT 하지 않는다 (supervisor 가 목숨 · 탄으로 직접 순위를 낸다) → success=false, error_code UNSUPPORTED

우리 진영은 /team/mission/state 의 camp 로 안다. LLM 키는 비밀 파일(secrets.json 의 llm_api_key)이나 환경변수 LLM_API_KEY 에서,
모델 · 주소는 설정 파일(llm_worker.json)에서 읽는다. 키는 기록 어디에도 찍지 않는다. 키가 없으면 낱말 해석만 한다.
"""
from __future__ import annotations

import json
import os
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from team_interfaces.action import Analyze
from team_interfaces.msg import MissionState, NodeStatus, ParsedCommand

from . import constants as C
from . import interpret

DEFAULTS = {
    "provider": "anthropic",  # anthropic · openai
    "model": "claude-opus-5-5",
    "extra": {"output_config": {"effort": "low"}},  # 요청 본문에 더 넣을 것 (모델에 맞춰)
    "max_tokens": 2000,
    "timeout_s": 10.0,  # LLM 요청 하나의 시한
    "retries": 0,
    "retry_gaps_s": [],
    "wait_s": 4.0,  # 액션 한 번이 LLM 답을 기다리는 시간. 넘으면 낱말 해석을 돌려준다 (LLM_PENDING)
    "retry_after_s": 3.0,  # LLM 이 실패한 뒤 다시 물어 오면, 실패하고 이만큼 지났을 때 새로 요청한다
    "url": "",  # 비우면 공급자의 기본 주소
}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_llm_config(folder):
    """설정(llm_worker.json) + 비밀(secrets.json 의 llm_api_key 또는 환경변수 LLM_API_KEY). 키가 없으면 LLM 을 쓰지 않는다."""
    cfg = dict(DEFAULTS)
    cfg.update(read_json(os.environ.get("LLM_WORKER_CONFIG", os.path.join(folder, "llm_worker.json"))))
    cfg.pop("api_key", None)  # 키는 설정 파일에 두지 않는다 (저장소에 올라갈 수 있다)
    key = os.environ.get("LLM_API_KEY") or read_json(os.environ.get("TEAM_SECRETS", os.path.join(folder, "secrets.json"))).get("llm_api_key")
    if key:
        cfg["api_key"] = str(key).strip()
    return cfg


class CallLog:
    """LLM 에 물은 것과 받은 것을 한 줄씩 남긴다 (나중에 분석용). 키는 여기로 오지 않는다."""

    def __init__(self, folder):
        self.path = None
        self.lock = threading.Lock()
        if folder:
            try:
                os.makedirs(folder, exist_ok=True)
                self.path = os.path.join(folder, "llm.jsonl")
            except OSError:
                pass

    def write(self, **record):
        if self.path is None:
            return
        try:
            with self.lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(record, wall=round(time.time(), 3)), ensure_ascii=False, default=str) + "\n")
        except (OSError, TypeError, ValueError):
            pass


class LlmWorker(Node):
    def __init__(self):
        super().__init__("llm_worker")
        self.cfg = load_llm_config(os.getcwd())
        self.interpreter = interpret.Interpreter(self.cfg, say=lambda text: self.get_logger().info(text))
        shared = os.environ.get("TEAM_LOG_DIR")  # team_node.py 가 정해 준 이번 실행의 기록 폴더. 따로 켰을 때는 log/llm_worker/
        self.calls = CallLog(None if os.environ.get("TEAM_LOG_OFF") == "1" else os.path.join(shared, "llm_worker") if shared else os.path.join("log", "llm_worker"))
        self.camp = ""
        self.run_id = ""
        self.lock = threading.Lock()
        self.jobs = {}  # (진영, 명령문) → {"done": Event, "out": (해석, 한 줄 기록), "ok": LLM 답을 반영했는가, "ended": 끝난 시각}
        group = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(MissionState, "/team/mission/state", self.on_mission, latched, callback_group=group)
        self.pub_status = self.create_publisher(NodeStatus, "/team/node_status", QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.executing = 0  # 지금 처리 중인 요청 수 (한 번에 하나만 받는다)
        self.create_timer(1.0, self.publish_status, callback_group=group)
        self.server = ActionServer(self, Analyze, "/team/llm/analyze", self.execute, goal_callback=self.on_goal,
                                   cancel_callback=lambda handle: CancelResponse.ACCEPT, callback_group=group)
        self.get_logger().info("llm_worker 켜짐 (LLM %s)" % ("%s · %s" % (self.cfg.get("provider"), self.cfg.get("model")) if self.interpreter.uses_llm else "없음 — 낱말 해석만"))

    def on_mission(self, m):
        self.run_id = m.run_id
        if m.camp in C.CAMPS:
            self.camp = m.camp

    def publish_status(self):
        s = NodeStatus()
        s.header.stamp = self.get_clock().now().to_msg()
        s.run_id, s.node_name, s.state = self.run_id, self.get_fully_qualified_name(), NodeStatus.READY
        s.detail = "LLM 사용" if self.interpreter.uses_llm else "LLM 없음 (낱말 해석만)"
        self.pub_status.publish(s)

    def on_goal(self, request):
        with self.lock:
            if self.executing:  # 한 번에 하나만. 줄 세우지 않는다 (부르는 쪽이 조금 뒤에 다시 묻는다)
                return GoalResponse.REJECT
            if request.kind not in (Analyze.Goal.PARSE_ORDER, Analyze.Goal.ASSESS_THREAT):
                return GoalResponse.REJECT
            self.executing += 1
            return GoalResponse.ACCEPT

    # ------------------------------------------------------------------ LLM 에 묻기 (뒤에서 돈다)
    def job_for(self, camp, text, quick):
        """이 명령문을 LLM 에 묻는 일. 이미 묻는 중이거나 답을 받았으면 그것을, 실패한 지 retry_after_s 가 지났으면 새로."""
        key = (camp, text)
        with self.lock:
            job = self.jobs.get(key)
            failed = job is not None and job["done"].is_set() and not job["ok"]
            if job is None or (failed and time.monotonic() - job["ended"] >= float(self.cfg["retry_after_s"])):
                job = self.jobs[key] = {"done": threading.Event(), "out": None, "ok": False, "ended": 0.0}
                threading.Thread(target=self.ask, args=(camp, text, quick, job), name="llm-ask", daemon=True).start()
            return job

    def ask(self, camp, text, quick, job):
        began = time.monotonic()
        try:
            final, note = self.interpreter.refine(text, camp, quick)  # 어떤 경우에도 예외를 내지 않는다
        except Exception as error:  # noqa: BLE001 — 그래도 막는다
            final, note = dict(quick), "LLM 실패 (예외 %s) → 낱말 해석 그대로" % type(error).__name__
        took = time.monotonic() - began
        job["out"], job["ok"], job["ended"] = (final, "%s · LLM %.1f초" % (note, took)), note.startswith(interpret.OK_NOTE), time.monotonic()
        job["done"].set()
        self.calls.write(e="llm", camp=camp, order=text, model=self.cfg.get("model"), provider=self.cfg.get("provider"), took_s=round(took, 2), ok=job["ok"], note=note,
                         quick=interpret.payload(quick), final=interpret.payload(final))
        self.get_logger().info("LLM 요청 끝 (%.1f초): %s" % (took, note))

    # ------------------------------------------------------------------ 액션
    def execute(self, handle):
        try:
            return self.answer(handle)
        finally:
            with self.lock:
                self.executing = max(0, self.executing - 1)

    def feedback(self, handle, stage, began):
        try:
            handle.publish_feedback(Analyze.Feedback(stage=stage, elapsed_ms=int((time.monotonic() - began) * 1000)))
        except Exception:  # noqa: BLE001 — 알림을 못 보내도 답은 낸다
            pass

    def answer(self, handle):
        goal = handle.request
        result = Analyze.Result()
        if goal.kind != Analyze.Goal.PARSE_ORDER:
            result.success, result.error_code, result.detail = False, "UNSUPPORTED", "M6 순위는 supervisor 가 직접 낸다"
            handle.abort()
            return result
        camp = self.camp
        text = goal.tactical_command
        if camp not in C.CAMPS or not text.strip():
            result.success, result.error_code = False, "NO_CAMP" if camp not in C.CAMPS else "EMPTY_ORDER"
            result.detail = "우리 진영을 아직 모름 (/team/mission/state 의 camp)" if camp not in C.CAMPS else "명령문이 비어 있음"
            handle.abort()
            return result
        began = time.monotonic()
        quick = self.interpreter.quick(text, camp)
        final, note, code = quick, "낱말 해석 (LLM 설정 없음)", ""
        if self.interpreter.uses_llm:
            job = self.job_for(camp, text, quick)
            limit = float(self.cfg["wait_s"])
            if goal.execution_timeout_ms:
                limit = min(limit, max(0.5, goal.execution_timeout_ms / 1000.0 - 1.0))  # 요청이 준 시한보다 1초 먼저 답한다
            while not job["done"].is_set():
                left = limit - (time.monotonic() - began)
                if left <= 0 or job["done"].wait(min(0.5, left)):
                    break
                if handle.is_cancel_requested:  # 기다리기를 그만둔다 (LLM 요청은 뒤에서 끝까지 돌고, 다음에 물으면 그 답을 준다)
                    result.success, result.error_code, result.detail = False, "CANCELED", "취소됨"
                    handle.canceled()
                    return result
                self.feedback(handle, "requesting", began)
            self.feedback(handle, "validating", began)
            if job["done"].is_set():
                final, note = job["out"]
                code = "" if job["ok"] else "LLM_FAILED"
            else:
                note, code = "LLM 답이 아직 없음 (%.0f초 기다림) → 낱말 해석" % limit, "LLM_PENDING"
        body = interpret.payload(final)
        result.success = bool(body)
        result.command = ParsedCommand(target=body.get(C.M1_TARGET, ""), search_sectors=list(body.get(C.M1_SECTORS, [])), base=body.get(C.M1_BASE, ""))
        result.error_code = code if body else "NOTHING_PARSED"
        result.detail = "%s (답하는 데 %.1f초)" % (note, time.monotonic() - began)
        self.calls.write(e="answer", camp=camp, order=text, body=body, code=result.error_code, detail=result.detail)
        self.get_logger().info("M1 해석: %s — %s%s" % (json.dumps(body, ensure_ascii=False), result.detail, " [%s]" % code if code else ""))
        handle.succeed()
        return result


def main():
    rclpy.init()
    node = LlmWorker()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
