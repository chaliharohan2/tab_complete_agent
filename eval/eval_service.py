"""
Run the test sets against the DEPLOYED router service over HTTP -- i.e. the
exact path the app uses. Runs locally (no GPU needed).

    python eval/eval_service.py [http://host:8800/classify]
"""
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(__file__))
import test_cases  # noqa: E402
import test_cases_holdout  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CLASSIFIER_URL", "http://spark-19de.local:8800/classify")


def fmt(s):
    return "{" + ",".join(sorted(s)) + "}" if s else "{}"


def run(name, cases):
    ok, lat = 0, []
    fails = []
    for text, req, opt in cases:
        t0 = time.time()
        r = httpx.post(URL, json={"text": text}, timeout=10).json()
        lat.append((time.time() - t0) * 1000)
        pred = set(r["entity_types"])
        if req <= pred <= (req | opt):
            ok += 1
        else:
            fails.append((text, pred, req, opt, r["source"]))
    lat.sort()
    print(f"### {name}: {ok}/{len(cases)} = {ok / len(cases):.1%}   "
          f"(round-trip latency median {lat[len(lat) // 2]:.0f}ms, p90 {lat[int(len(lat) * 0.9)]:.0f}ms)")
    for text, pred, req, opt, src in fails:
        exp = fmt(req) + (f" (+opt {fmt(opt)})" if opt else "")
        print(f"  FAIL [{src}] {text!r:55s} got {fmt(pred):35s} exp {exp}")
    print()


if __name__ == "__main__":
    run("main", test_cases.CASES)
    run("holdout", test_cases_holdout.CASES)
