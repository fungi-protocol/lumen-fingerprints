import json, os, sys, tempfile, unittest
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dss", os.path.join(os.path.dirname(__file__), "dashboard-server.py"))
dss = importlib.util.module_from_spec(spec); spec.loader.exec_module(dss)

def _epochs(*rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close(); return f.name

class Helpers(unittest.TestCase):
    def test_resume_height_reads_last_end_height(self):
        p = _epochs({"start_height": 939969, "end_height": 940112},
                    {"start_height": 940113, "end_height": 940256})
        self.assertEqual(dss.resume_height(p), 940256)
        self.assertIsNone(dss.resume_height("/nonexistent.jsonl"))

    def test_progress_from_epochs(self):
        p = _epochs({"start_height": 939969, "end_height": 940112},
                    {"start_height": 940113, "end_height": 940256})
        pr = dss.progress(p, 939969, 939969 + 2000)
        self.assertEqual(pr["height"], 940256)
        self.assertEqual(pr["epochs"], 2)
        self.assertEqual(pr["blocks_done"], 940256 - 939969)
        self.assertEqual(pr["blocks_target"], 2000)
        self.assertTrue(0 < pr["pct"] < 100)

    def test_throttle(self):
        self.assertTrue(dss.should_reaggregate(100.0, 80.0, 0))   # 20s >= T
        self.assertTrue(dss.should_reaggregate(100.0, 99.0, 5))   # K epochs
        self.assertFalse(dss.should_reaggregate(100.0, 99.0, 1))  # neither

class ServerStatus(unittest.TestCase):
    def test_status_and_static(self):
        import threading, urllib.request
        book = tempfile.mkdtemp()
        open(os.path.join(book, "index.html"), "w").write("<h1>hi</h1>")
        epochs = _epochs({"start_height": 939969, "end_height": 940112})
        srv = dss.make_server(book, "/some/datadir", epochs)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        base = f"http://127.0.0.1:{port}"
        st = json.load(urllib.request.urlopen(base + "/api/status"))
        self.assertTrue(st["backend"]); self.assertTrue(st["datadir_set"])
        self.assertEqual(st["resume_height"], 940112); self.assertFalse(st["running"])
        body = urllib.request.urlopen(base + "/index.html").read().decode()
        self.assertIn("hi", body)
        srv.shutdown()

class ScanSSE(unittest.TestCase):
    def test_scan_streams_progress_and_updated_and_done(self):
        import threading, urllib.request, subprocess
        book = tempfile.mkdtemp(); work = tempfile.mkdtemp()
        epochs = os.path.join(work, "epochs.jsonl")
        env = dict(os.environ,
                   LUMEN_BIN=f"{sys.executable} {os.path.dirname(__file__)}/fake_survey.py",
                   DASHBOARD_EMIT=f"{os.path.dirname(__file__)}/dashboard-data.py")
        # The server runs in-process (via a thread below), not as a subprocess, so it
        # reads os.environ directly -- update the live environment, not just a local dict.
        os.environ.update(env)
        srv = dss.make_server(book, work, epochs, 939969, 940257)  # datadir=work for the fake
        srv.emit_target = os.path.join(work, "explorer-data.json")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/scan?mode=fresh")  # GET: the dashboard's EventSource client is GET-only
        kinds = []
        with urllib.request.urlopen(req) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    kinds.append(line.split(":", 1)[1].strip())
                if "done" in kinds:
                    break
        self.assertIn("phase", kinds); self.assertIn("progress", kinds)
        self.assertIn("updated", kinds); self.assertIn("done", kinds)
        srv.shutdown()

    def test_final_reaggregate_copies_to_emit_upstream(self):
        import threading, urllib.request
        book = tempfile.mkdtemp(); work = tempfile.mkdtemp()
        epochs = os.path.join(work, "epochs.jsonl")
        env = dict(os.environ,
                   LUMEN_BIN=f"{sys.executable} {os.path.dirname(__file__)}/fake_survey.py",
                   DASHBOARD_EMIT=f"{os.path.dirname(__file__)}/dashboard-data.py")
        os.environ.update(env)
        srv = dss.make_server(book, work, epochs, 939969, 940257)
        emit_target = os.path.join(work, "explorer-data.json")
        upstream = os.path.join(work, "upstream-explorer-data.json")
        srv.emit_target = emit_target
        srv.emit_upstream = upstream
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/scan?mode=fresh")
        kinds = []
        with urllib.request.urlopen(req) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    kinds.append(line.split(":", 1)[1].strip())
                if "done" in kinds:
                    break
        srv.shutdown()
        # The final reaggregate must have mirrored the live emit into the upstream
        # (committed docs/src) path, byte-for-byte.
        self.assertTrue(os.path.exists(upstream),
                        "emit_upstream file must exist after a completed scan")
        with open(emit_target) as a, open(upstream) as b:
            self.assertEqual(a.read(), b.read())

    def test_final_reaggregate_copies_cross_csv_upstream(self):
        import threading, urllib.request
        book = tempfile.mkdtemp(); work = tempfile.mkdtemp()
        epochs = os.path.join(work, "epochs.jsonl")
        env = dict(os.environ,
                   LUMEN_BIN=f"{sys.executable} {os.path.dirname(__file__)}/fake_survey.py",
                   DASHBOARD_EMIT=f"{os.path.dirname(__file__)}/dashboard-data.py")
        os.environ.update(env)
        srv = dss.make_server(book, work, epochs, 939969, 940257)
        srv.emit_target = os.path.join(work, "explorer-data.json")
        cross_upstream = os.path.join(work, "upstream-wallet-axis-cross.csv")
        srv.cross_upstream = cross_upstream
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/scan?mode=fresh")
        kinds = []
        with urllib.request.urlopen(req) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    kinds.append(line.split(":", 1)[1].strip())
                if "done" in kinds:
                    break
        srv.shutdown()
        self.assertTrue(os.path.exists(cross_upstream),
                        "wallet-axis-cross.csv must be copied to cross_upstream after a scan")
        with open(cross_upstream) as f:
            self.assertTrue(f.readline().startswith("start_height,end_height,wallet_era,axis,value"))

    def test_no_emit_upstream_leaves_no_extra_file(self):
        # Without emit_upstream set (the default), a scan writes only emit_target.
        import threading, urllib.request
        book = tempfile.mkdtemp(); work = tempfile.mkdtemp()
        epochs = os.path.join(work, "epochs.jsonl")
        env = dict(os.environ,
                   LUMEN_BIN=f"{sys.executable} {os.path.dirname(__file__)}/fake_survey.py",
                   DASHBOARD_EMIT=f"{os.path.dirname(__file__)}/dashboard-data.py")
        os.environ.update(env)
        srv = dss.make_server(book, work, epochs, 939969, 940257)
        srv.emit_target = os.path.join(work, "explorer-data.json")
        self.assertIsNone(srv.emit_upstream)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/scan?mode=fresh")
        kinds = []
        with urllib.request.urlopen(req) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    kinds.append(line.split(":", 1)[1].strip())
                if "done" in kinds:
                    break
        srv.shutdown()
        self.assertIn("done", kinds)

    def test_bad_mode_does_not_crash(self):
        import threading, urllib.request
        book = tempfile.mkdtemp(); work = tempfile.mkdtemp()
        epochs = os.path.join(work, "epochs.jsonl")
        env = dict(os.environ,
                   LUMEN_BIN=f"{sys.executable} {os.path.dirname(__file__)}/fake_survey.py",
                   DASHBOARD_EMIT=f"{os.path.dirname(__file__)}/dashboard-data.py")
        os.environ.update(env)
        srv = dss.make_server(book, work, epochs, 939969, 940257)
        srv.emit_target = os.path.join(work, "explorer-data.json")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        # mode=bogus is not "fresh" or "resume" and must never crash the handler
        # thread or abort the connection before SSE headers -- it should degrade
        # to the "resume" default and stream a normal SSE session.
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/scan?mode=bogus")  # GET: the dashboard's EventSource client is GET-only
        kinds = []
        with urllib.request.urlopen(req) as r:
            self.assertEqual(r.status, 200)
            self.assertIn("text/event-stream", r.headers.get("Content-Type", ""))
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    kinds.append(line.split(":", 1)[1].strip())
                if "done" in kinds or "error" in kinds:
                    break
        # a clean SSE stream: either it started a normal scan (phase -> ... -> done)
        # or it rejected cleanly with a single "error" event -- never a dropped
        # connection / raised exception with no frames at all.
        self.assertTrue(kinds)
        self.assertIn(kinds[0], ("phase", "error"))
        srv.shutdown()

class ConcurrencyAndStop(unittest.TestCase):
    def test_second_concurrent_scan_attaches(self):
        # A second /api/scan while one is already running must ATTACH -- it
        # replays the buffered event log and rides it through to the same
        # terminal event -- not get rejected with an "error". The scan itself
        # is a background thread now; both requests are just followers of it.
        import threading, urllib.request, time
        book = tempfile.mkdtemp(); work = tempfile.mkdtemp()
        epochs = os.path.join(work, "epochs.jsonl")
        env = dict(os.environ,
                   LUMEN_BIN=f"{sys.executable} {os.path.dirname(__file__)}/fake_survey.py",
                   DASHBOARD_EMIT=f"{os.path.dirname(__file__)}/dashboard-data.py")
        os.environ.update(env)
        srv = dss.make_server(book, work, epochs, 939969, 940257)
        srv.emit_target = os.path.join(work, "explorer-data.json")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        base = f"http://127.0.0.1:{port}"

        first_kinds = []
        def run_first():
            req = urllib.request.Request(base + "/api/scan?mode=fresh")
            with urllib.request.urlopen(req) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if line.startswith("event:"):
                        first_kinds.append(line.split(":", 1)[1].strip())
                    if "done" in first_kinds:
                        break
        t1 = threading.Thread(target=run_first)
        t1.start()

        # Wait for the background worker to actually be running before the
        # second request attaches -- that's the scenario under test.
        for _ in range(100):
            if srv.state["running"]:
                break
            time.sleep(0.05)
        self.assertTrue(srv.state["running"])

        second_kinds = []
        req2 = urllib.request.Request(
            base + "/api/scan?mode=fresh")  # mode is ignored: a scan is already running
        with urllib.request.urlopen(req2) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    second_kinds.append(line.split(":", 1)[1].strip())
                if "done" in second_kinds:
                    break

        t1.join(timeout=10)
        self.assertIn("phase", second_kinds)
        self.assertIn("done", second_kinds)
        self.assertIn("done", first_kinds)
        srv.shutdown()

    def test_reattach_after_disconnect_replays_buffered_events(self):
        # A viewer that disconnects mid-scan (e.g. navigates away) and a
        # second/returning viewer that opens /api/scan again must see the full
        # replay from the start of the event log, then ride it to `done` --
        # the scan itself never stopped running in the background.
        import threading, urllib.request, time
        book = tempfile.mkdtemp(); work = tempfile.mkdtemp()
        epochs = os.path.join(work, "epochs.jsonl")
        env = dict(os.environ,
                   LUMEN_BIN=f"{sys.executable} {os.path.dirname(__file__)}/fake_survey.py",
                   DASHBOARD_EMIT=f"{os.path.dirname(__file__)}/dashboard-data.py")
        os.environ.update(env)
        srv = dss.make_server(book, work, epochs, 939969, 940257)
        srv.emit_target = os.path.join(work, "explorer-data.json")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        base = f"http://127.0.0.1:{port}"

        req1 = urllib.request.Request(base + "/api/scan?mode=fresh")
        r1 = urllib.request.urlopen(req1)
        got = []
        for raw in r1:
            line = raw.decode().strip()
            if line.startswith("event:"):
                got.append(line.split(":", 1)[1].strip())
            if len(got) >= 2:
                break
        r1.close()  # simulate the viewer navigating away mid-scan

        self.assertTrue(srv.state["running"])
        self.assertGreaterEqual(len(srv.state["events"]), 2)

        second_kinds = []
        req2 = urllib.request.Request(base + "/api/scan?mode=fresh")
        with urllib.request.urlopen(req2) as r2:
            for raw in r2:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    second_kinds.append(line.split(":", 1)[1].strip())
                if "done" in second_kinds:
                    break

        self.assertEqual(second_kinds[0], "phase")  # replay starts from index 0
        self.assertIn("done", second_kinds)
        srv.shutdown()

    def test_stop_with_no_scan_is_safe(self):
        import threading, urllib.request
        book = tempfile.mkdtemp(); work = tempfile.mkdtemp()
        epochs = os.path.join(work, "epochs.jsonl")
        srv = dss.make_server(book, work, epochs)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/stop", method="POST", data=b"")
        with urllib.request.urlopen(req) as r:
            self.assertEqual(r.status, 200)
            body = json.loads(r.read())
        self.assertIsInstance(body, dict)
        srv.shutdown()

class MainCLI(unittest.TestCase):
    def test_main_queries_tip(self):
        from unittest import mock
        book = tempfile.mkdtemp(); work = tempfile.mkdtemp()
        epochs = os.path.join(work, "epochs.jsonl")
        emit = os.path.join(work, "explorer-data.json")
        argv = ["dashboard-server.py", "--book", book, "--datadir", work,
                "--epochs", epochs, "--emit", emit, "--port", "0"]
        os.environ["LUMEN_BIN"] = \
            f"{sys.executable} {os.path.dirname(__file__)}/fake_survey.py"

        import time
        captured = {}
        orig_make_server = dss.make_server
        def fake_make_server(*args, **kwargs):
            srv = orig_make_server(*args, **kwargs)
            captured["srv"] = srv
            srv.serve_forever = lambda: None  # never actually block the test
            return srv

        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(dss, "make_server", fake_make_server):
            dss.main()

        # main() serves immediately and queries the tip on a background thread; wait for it
        # to land, then check floor/tip came from fake_survey.py's `tip` subcommand output.
        srv = captured["srv"]
        for _ in range(60):
            if srv.floor is not None:
                break
            time.sleep(0.05)
        self.assertEqual(srv.floor, 939969)
        self.assertEqual(srv.tip, 940257)
        srv.server_close()

if __name__ == "__main__":
    unittest.main()
