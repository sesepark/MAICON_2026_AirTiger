import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from supervisor.recorder import OFF_ENV, SESSION_ENV, Recorder, session_folder


class RecorderTests(unittest.TestCase):
    def test_events_and_photos_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(os.path.join(tmp, "run"), {"scout_mode": "face"})
            rec.clock = lambda: 12.345
            rec.event("go", x=1.0, y=2.0)
            rec.event("report", kind=3, name="x")  # 칸 이름이 kind · name 이어도 된다
            rec.photo_record({"tag": ["face", 3], "pairs": None})
            img = SimpleNamespace(width=4, height=2, encoding="rgb8", data=bytes(range(24)))
            name = rec.save_photo(("face", 3, "N"), img)
            rec.close()
            self.assertTrue(name and os.path.getsize(os.path.join(tmp, "run", "photos", name)) > 0)
            line = json.loads(Path(tmp, "run", "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual((line["e"], line["x"], line["t"]), ("go", 1.0, 12.35))
            second = json.loads(Path(tmp, "run", "events.jsonl").read_text(encoding="utf-8").splitlines()[1])
            self.assertEqual((second["e"], second["kind"], second["name"]), ("report", 3, "x"))
            self.assertEqual(json.loads(Path(tmp, "run", "meta.json").read_text(encoding="utf-8"))["scout_mode"], "face")
            self.assertEqual(len(Path(tmp, "run", "photos.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_lines_from_two_threads_stay_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(os.path.join(tmp, "run"))
            workers = [threading.Thread(target=lambda k=k: [rec.event("tick", k=k, i=i, pad="가" * 200) for i in range(300)]) for k in range(4)]
            for w in workers:
                w.start()
            for w in workers:
                w.join()
            rec.close()
            lines = Path(tmp, "run", "events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1200)
            for line in lines:
                json.loads(line)

    def test_no_folder_means_no_files_and_no_errors(self):
        rec = Recorder(None)
        rec.event("go", x=1)
        self.assertIsNone(rec.save_photo(("a",), SimpleNamespace(width=1, height=1, encoding="rgb8", data=b"abc")))
        bad = Recorder("/proc/없는/곳")  # 만들 수 없는 폴더 → 조용히 끈다
        bad.event("go", x=1)
        self.assertIsNone(bad.dir)

    def test_session_folder_follows_the_start_file(self):
        saved = {name: os.environ.pop(name, None) for name in (SESSION_ENV, OFF_ENV)}
        try:
            self.assertIsNone(session_folder("", "abc"))
            self.assertTrue(session_folder("log/supervisor", "abc").endswith("_abc"))
            os.environ[SESSION_ENV] = "/x/log/20261118_190000"
            self.assertEqual(session_folder("log/supervisor", "abc"), "/x/log/20261118_190000/supervisor")
            os.environ[OFF_ENV] = "1"  # 디스크가 거의 찼다 → 남기지 않는다
            self.assertIsNone(session_folder("log/supervisor", "abc"))
        finally:
            for name, value in saved.items():
                os.environ.pop(name, None)
                if value is not None:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
