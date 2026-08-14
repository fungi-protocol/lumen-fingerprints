# contrib/fake_survey.py — stands in for the real `survey` binary in tests.
# `scan`: append 2 epoch rows to --out, print a log line each, then exit.
#   Always scans a small fixed number of epochs (2 epochs = 288 blocks) so
#   tests stay fast.
# `report`: write a minimal report.json with the keys the emitter reads, plus a
#   wallet-axis-cross.csv (header only) mirroring what the real `lumen report` emits.
# `tip`: print one JSON line with floor/tip/in_ibd, like `lumen tip`.
import json, sys, time

def arg(name):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else None

if sys.argv[1] == "scan":
    out = arg("--out")
    start = 939969; h = start
    stop = start + 2 * 144
    print("connecting to peers...", flush=True)
    with open(out, "a") as f:
        while h < stop:
            end = min(h + 143, stop)
            f.write(json.dumps({"start_height": h, "end_height": end,
                                "txs": 100, "defects": 0, "axis_counts": {},
                                "vectors": {}, "vectors_extended": {},
                                "aux_counts": {}, "template_matches": {}}) + "\n")
            f.flush()
            print(f"scanned to height {end}", flush=True)
            h = end + 1
            time.sleep(0.01)
    print("done", flush=True)
elif sys.argv[1] == "tip":
    print(json.dumps({"tip": 940257, "floor": 939969, "in_ibd": False}), flush=True)
elif sys.argv[1] == "report":
    out_dir = arg("--out-dir"); import os; os.makedirs(out_dir, exist_ok=True)
    json.dump({"window": {"start_height": 939969, "end_height": 940112, "epochs": 1},
               "totals": {"txs": 100, "defects": 0},
               "axis_summaries": {}, "encoding_families": {},
               "conditional_anonymity": {}},
              open(os.path.join(out_dir, "report.json"), "w"))
    with open(os.path.join(out_dir, "wallet-axis-cross.csv"), "w") as f:
        f.write("start_height,end_height,wallet_era,axis,value,share,in_set_lt10,in_set_lt100\n")
