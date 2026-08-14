#!/usr/bin/env python3
"""Local control server for the Fingerprint dashboard: serves the built book and
exposes /api/status, /api/scan (SSE), /api/stop to run a Floresta-backed scan and
stream progress. Localhost only. Python stdlib only."""
import argparse, json, os, queue, shutil, subprocess, sys, threading, time
import http.server
import urllib.parse

def _last_epoch(epochs_path):
    try:
        last = None
        with open(epochs_path) as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last) if last else None
    except (OSError, ValueError):
        return None

def resume_height(epochs_path):
    row = _last_epoch(epochs_path)
    return row["end_height"] if row else None

def _count_epochs(epochs_path):
    try:
        with open(epochs_path) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0

def progress(epochs_path, start_height, target_height):
    row = _last_epoch(epochs_path)
    height = row["end_height"] if row else (start_height or 0)
    # Before the live tip is known, report height/epoch only (no % / target).
    if start_height is None or target_height is None:
        return {"height": height, "epochs": _count_epochs(epochs_path),
                "blocks_done": None, "blocks_target": None, "pct": None}
    target_span = max(1, target_height - start_height)
    done = max(0, height - start_height)
    return {
        "height": height,
        "epochs": _count_epochs(epochs_path),
        "blocks_done": done,
        "blocks_target": target_height - start_height,
        "pct": min(100.0, round(done / target_span * 100, 1)),
    }

def should_reaggregate(now, last_ts, epochs_since, *, t=15.0, k=5):
    return (now - last_ts) >= t or epochs_since >= k

def _pump_stdout(proc, line_queue):
    """Runs on a dedicated daemon thread: reads the scanner's stdout line-by-line
    and pushes each line onto line_queue. A blocking readline() here can never stall
    the progress/throttle loop, which lives on the main thread and only ever does
    non-blocking queue reads."""
    try:
        for line in proc.stdout:
            line_queue.put(line)
    except (ValueError, OSError):
        pass  # stdout closed out from under us during teardown; nothing left to read
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass

def run_scan(server, mode):
    """Runs an entire scan to completion on a background thread, independent of
    any HTTP request/response: spawns `lumen scan`, drives the progress/
    reaggregate loop, and appends every event to server.state["events"] instead
    of writing SSE directly. /api/scan requests are just followers that tail
    this log (see DashboardHandler._handle_scan) -- so the scan survives page
    navigation and any number of viewers can watch the same run."""
    state = server.state

    def send(event, data):
        with server.events_lock:
            state["events"].append((event, data))

    try:
        epochs_path = server.epochs_path
        datadir = server.datadir
        # floor/tip are read LIVE from the server (below) each time progress is
        # computed, so a tip that lands mid-scan (async query) starts driving the %.

        if mode == "fresh":
            try:
                os.remove(epochs_path)
            except OSError:
                pass

        survey_bin = os.environ.get("LUMEN_BIN", "./target/release/lumen")
        dashboard_emit = os.environ.get("DASHBOARD_EMIT", "contrib/dashboard-data.py")
        emit_target = getattr(server, "emit_target", None) or \
            os.path.join(datadir, "explorer-data.json")

        send("phase", {"phase": "connecting"})

        cmd = survey_bin.split() + [
            "scan", "--datadir", datadir, "--out", epochs_path,
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        with server.events_lock:
            state["proc"] = proc

        emit_upstream = getattr(server, "emit_upstream", None)
        cross_upstream = getattr(server, "cross_upstream", None)

        def reaggregate(final=False):
            report_dir = os.path.join(datadir, "report")
            subprocess.run(
                survey_bin.split() + [
                    "report", "--epochs", epochs_path, "--out-dir", report_dir,
                ], check=True)
            report_json = os.path.join(report_dir, "report.json")
            subprocess.run(
                [sys.executable, dashboard_emit, report_json, emit_target],
                check=True)
            # The live emit (`emit_target`) drives the served book and is rewritten every
            # reaggregate. The upstream copies are the committed `docs/src` artifacts (the
            # Explorer's JSON and the ML-facing wallet×axis cross CSV `lumen report` writes
            # next to report.json); they are written ONLY on the final reaggregate so a
            # scan leaves the source tree with exactly the artifacts to commit, not files
            # that churned on every throttle tick.
            if final and emit_upstream:
                shutil.copyfile(emit_target, emit_upstream)
                send("log", {"line": f"upstream explorer-data.json updated: {emit_upstream}"})
            if final and cross_upstream:
                shutil.copyfile(os.path.join(report_dir, "wallet-axis-cross.csv"), cross_upstream)
                send("log", {"line": f"upstream wallet-axis-cross.csv updated: {cross_upstream}"})
            send("updated", {})

        scanning = False
        last_ts = time.time()
        last_epoch_count = progress(epochs_path, server.floor, server.tip)["epochs"]
        scan_started = time.time()
        stall_warned = False
        STALL_WARN_SECS = 120  # first epoch (144 blocks) should arrive within ~1-2 min

        # Stdout draining is decoupled onto its own daemon thread so a
        # partial/unterminated line sitting in the pipe can never block this
        # loop's progress/throttle work -- the loop below only ever does
        # non-blocking queue reads plus a short sleep.
        log_queue = queue.Queue()
        stdout_thread = threading.Thread(
            target=_pump_stdout, args=(proc, log_queue), daemon=True)
        stdout_thread.start()

        def drain_log_queue():
            nonlocal scanning
            while True:
                try:
                    line = log_queue.get_nowait()
                except queue.Empty:
                    break
                line = line.rstrip("\n")
                if line:
                    send("log", {"line": line})
                    if not scanning and "height" in line:
                        scanning = True
                        send("phase", {"phase": "scanning"})

        while True:
            drain_log_queue()

            now = time.time()
            pr = progress(epochs_path, server.floor, server.tip)
            epochs_since = pr["epochs"] - last_epoch_count
            if should_reaggregate(now, last_ts, epochs_since):
                reaggregate()
                last_ts = now
                last_epoch_count = pr["epochs"]
            send("progress", pr)

            # No epoch after a couple of minutes almost always means the node can't
            # reach a utreexo-serving peer (block download needs one; header sync does
            # not). Say so once instead of leaving a silent 0%.
            if not stall_warned and pr["epochs"] == 0 and (now - scan_started) > STALL_WARN_SECS:
                stall_warned = True
                send("phase", {"phase": "stalled"})
                send("log", {"line": "No blocks yet after 2 min — the node is still "
                             "looking for a utreexo peer (block download needs one; the "
                             "utreexo network can be sparse or slow). It keeps trying; "
                             "Stop and retry, or run a local utreexod bridge for reliability."})

            # Exit once the scanner has exited *and* the queue is fully
            # drained (the stdout thread having died is what guarantees no
            # further lines can appear after this check).
            exited = proc.poll() is not None
            drained = log_queue.empty() and not stdout_thread.is_alive()
            if exited and drained:
                break
            time.sleep(0.5)

        # stdout thread has (or is about to have) hit EOF; join it before
        # touching the process/pipe further so teardown never races the reader.
        stdout_thread.join(timeout=5.0)
        drain_log_queue()  # pick up anything queued in the join window
        proc.wait()

        if proc.returncode != 0:
            send("log", {"line": f"scan exited with status {proc.returncode}"})

        # Final re-aggregation is best-effort: a failure here (e.g. the
        # very last `report`/emit invocation) must not turn a successful
        # scan into an `error` from the client's perspective -- it still
        # gets `phase:done` + `done`, just possibly with stale aggregated
        # output. Log the failure to stderr for visibility.
        try:
            reaggregate(final=True)
        except Exception as e:
            print(f"final reaggregate failed: {e}", file=sys.stderr)
        final = progress(epochs_path, server.floor, server.tip)
        send("phase", {"phase": "done"})
        send("done", final)
    except Exception as e:
        send("error", {"msg": str(e)})
    finally:
        with server.events_lock:
            state["running"] = False
            state["proc"] = None

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the built book as static files and answers /api/status.
    Additional /api/* endpoints (scan, stop) are added in later tasks."""

    def end_headers(self):
        # No browser caching: this is a local dev tool whose pages and data change on
        # every scan/edit, and a stale cached page is exactly the confusion to avoid.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/api/status":
            self._send_status()
        elif parsed.path == "/api/scan":
            # EventSource (the dashboard's SSE client) is GET-only, so /api/scan is
            # served on GET as well as POST.
            self._handle_scan(parsed)
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/api/scan":
            self._handle_scan(parsed)
        elif parsed.path == "/api/stop":
            self._handle_stop()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_stop(self):
        state = self.server.state
        proc = state.get("proc")
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
        with self.server.events_lock:
            state["stopped"] = True
        body = json.dumps({"stopped": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_scan(self, parsed):
        server = self.server
        state = server.state

        # Parsing happens up front so a bad query string can never abort the
        # connection without an SSE `error` frame -- it degrades to safe
        # defaults instead. mode is only consulted when THIS request is the
        # one that starts a new scan; an attach to an already-running scan
        # ignores it entirely.
        qs = urllib.parse.parse_qs(parsed.query)
        mode = qs.get("mode", ["resume"])[0]
        if mode not in ("fresh", "resume"):
            mode = "resume"

        # Atomic check-and-set: two concurrent /api/scan requests on the
        # ThreadingHTTPServer must not both observe running==False before either
        # sets it. Hold server.scan_lock for the whole check-and-set. Whichever
        # request wins starts the background worker and resets the event log;
        # every other request (this one included, if it loses) just attaches
        # below as a follower of whatever is currently running.
        with server.scan_lock:
            if not state["running"]:
                with server.events_lock:
                    state["running"] = True
                    state["proc"] = None
                    state["stopped"] = False
                    state["events"] = []
                threading.Thread(
                    target=run_scan, args=(server, mode), daemon=True).start()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        # Follow: replay everything buffered so far from the start, then keep
        # polling for new events, writing each as SSE to THIS response. This
        # runs identically whether this request just started the scan above or
        # is attaching to one already in flight -- every viewer gets the full
        # replay + live tail. Returns on a terminal event or a dead connection;
        # the scan thread is entirely independent and keeps running either way.
        idx = 0
        while True:
            with server.events_lock:
                pending = state["events"][idx:]
            for event, data in pending:
                idx += 1
                try:
                    self.wfile.write(
                        f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return  # client disconnected; the scan thread keeps running
                if event in ("done", "error"):
                    return
            time.sleep(0.2)

    def _send_status(self):
        server = self.server
        body = json.dumps({
            "backend": True,
            "datadir_set": bool(server.datadir),
            "resume_height": resume_height(server.epochs_path),
            "running": server.state["running"],
            "floor": server.floor,
            "tip": server.tip,
            "repo_name": server.repo_name,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # keep test/CLI output quiet; stdlib default logs to stderr per request

def make_server(book_dir, datadir, epochs_path, floor=0, tip=0, port=0):
    def handler_factory(*args, **kwargs):
        return DashboardHandler(*args, directory=book_dir, **kwargs)
    # ThreadingHTTPServer: /api/scan streams for the duration of a scan, so /api/stop
    # (and /api/status) must be servable concurrently on another thread.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler_factory)
    server.daemon_threads = True
    server.datadir = datadir
    server.epochs_path = epochs_path
    server.floor = floor
    server.tip = tip
    server.state = {"running": False, "proc": None, "stopped": False, "events": []}
    # scan_lock guards the atomic start check-and-set; events_lock guards
    # server.state's mutable fields (the append-only "events" log plus
    # running/proc/stopped) against the worker thread and any number of
    # concurrent follower requests.
    server.scan_lock = threading.Lock()
    server.events_lock = threading.Lock()
    server.emit_target = os.path.join(book_dir, "explorer-data.json")
    server.emit_upstream = None
    server.cross_upstream = None
    server.repo_name = os.path.basename(os.path.abspath("."))
    return server

def main():
    parser = argparse.ArgumentParser(
        description="Local control server for the Fingerprint dashboard.")
    parser.add_argument("--book", required=True, help="built book directory to serve")
    parser.add_argument("--datadir", required=True, help="Floresta/survey datadir")
    parser.add_argument("--epochs", required=True, help="epochs .jsonl path")
    parser.add_argument("--emit", required=True, help="explorer-data.json emit target")
    parser.add_argument("--emit-upstream", default=None,
                        help="also copy the final explorer-data.json here "
                             "(e.g. the committed docs/src file the public site serves)")
    parser.add_argument("--cross-upstream", default=None,
                        help="also copy the final wallet-axis-cross.csv here "
                             "(the committed ML-facing wallet×axis cross)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true",
                        help="open the dashboard in a browser once it is serving")
    args = parser.parse_args()

    survey_bin = os.environ.get("LUMEN_BIN", "./target/release/lumen")

    # Serve immediately with an unknown tip; query the live tip in the background so the
    # page and the Run button appear instantly instead of blocking on a header sync
    # (which can take minutes on a fresh datadir). floor/tip fill in when the query
    # returns; progress shows height/epoch meanwhile and gains a % once the tip is known.
    server = make_server(args.book, args.datadir, args.epochs, None, None, args.port)
    server.emit_target = args.emit
    server.emit_upstream = args.emit_upstream
    server.cross_upstream = args.cross_upstream

    def query_tip():
        cmd = survey_bin.split() + ["tip", "--datadir", args.datadir]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            info = json.loads([l for l in out.splitlines() if l.strip()][-1])
            server.floor, server.tip = info["floor"], info["tip"]
            sys.stderr.write(f"range: floor {server.floor} .. tip {server.tip}\n")
            sys.stderr.flush()
        except Exception as e:  # noqa: BLE001 - background best-effort
            sys.stderr.write(f"warning: `lumen tip` failed ({e}); "
                             "progress %% disabled until a tip is known\n")
            sys.stderr.flush()

    threading.Thread(target=query_tip, daemon=True).start()
    sys.stderr.write(f"Dashboard on http://127.0.0.1:{args.port} "
                     "(serving now; querying the live tip in the background)\n")
    sys.stderr.flush()
    if args.open:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    server.serve_forever()

if __name__ == "__main__":
    main()
