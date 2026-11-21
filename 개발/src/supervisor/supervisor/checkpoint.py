"""경기 중 상태를 파일에 적어 두었다가, 프로그램이 죽고 다시 켜지면 읽어 온다.

왜: M1 · M3 는 다시 보내면 통째로 바뀐다. 다시 켜진 프로그램이 빈 면 기록으로 M3 를 보내면 앞에서 얻은 점수를 덮어쓴다.
파일은 새 경기(시작 대기)가 오면 지운다. 너무 오래된 파일(다른 경기의 것)은 읽지 않는다.
"""
from __future__ import annotations

import json
import os
import time


def save(path, data):
    """통째로 바꿔 쓴다 (쓰다가 죽어도 반쯤 쓴 파일이 남지 않게 임시 파일에 쓰고 이름을 바꾼다)."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dict(data, saved_at=time.time()), fh, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def load(path, max_age_s=330.0):
    """돌려주는 값: 저장해 둔 것(dict) 또는 None (없음 · 못 읽음 · 너무 오래됨)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or time.time() - float(data.get("saved_at", 0.0)) > max_age_s:
        return None
    return data


def clear(path):
    for name in (path, path + ".tmp"):
        try:
            os.remove(name)
        except OSError:
            pass
