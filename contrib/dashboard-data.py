#!/usr/bin/env python3
"""report.json -> explorer-data.json : the compact JSON the Fingerprint Explorer reads.
Usage: dashboard-data.py <report.json> <explorer-data.json>"""
import json, sys
d = json.load(open(sys.argv[1]))
out = {
    "window": d["window"],
    "totals": d["totals"],
    "axis_summaries": d["axis_summaries"],
    "encoding_families": d["encoding_families"],
    "cond": d["conditional_anonymity"],
    "template_series": d.get("template_series", {}),
    "axis_value_series": d.get("axis_value_series", {}),
    "template_axes": d.get("template_axes", {}),
    "fields": { f["name"]: f for f in d.get("fields", []) },
}
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
print(f"wrote {sys.argv[2]} ({len(json.dumps(out))} bytes, {len(out['axis_summaries'])} axes, cond {len(out['cond'])} axes, {len(out['template_series'])} template series)")
