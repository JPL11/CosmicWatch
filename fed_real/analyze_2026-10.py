"""Summaries for the October 2026 real-WAN campaigns (campaigns_2026-10.sh).

  python3 analyze_2026-10.py            # prints A, B, BBR tables + data checks
  python3 analyze_2026-10.py --json out.json

Tail = mean global test weighted MSE over the last quarter of the run's wall
clock (the paper's metric). Paired differences use the same seed in both arms.
Standard library only.
"""
import argparse
import glob
import gzip
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = [1, 2, 3, 4, 5]
CUBIC_ONLY = {"jetson", "nano"}          # kernels without BBR


def load(tag):
    """results/<tag>.json, or the committed results/<tag>.json.gz."""
    p = os.path.join(RES, tag + ".json")
    if os.path.exists(p):
        return json.load(open(p))
    if os.path.exists(p + ".gz"):
        return json.load(gzip.open(p + ".gz", "rt"))
    return None


def tail_async(r):
    u = r["updates"]
    t_end = u[-1]["t_s"]
    vals = [x["global_test_weighted_mse"] for x in u if x["t_s"] >= 0.75 * t_end]
    return st.mean(vals)


def tail_sync(r):
    rs = r["rounds"]
    t_end = rs[-1]["wall_s"]
    vals = [x["global_test_weighted_mse"] for x in rs if x["wall_s"] >= 0.75 * t_end]
    return st.mean(vals)


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return "%.4f (n=1)" % xs[0]
    return "%.4f ± %.4f (n=%d)" % (st.mean(xs), st.stdev(xs), len(xs))


def paired(a, b):
    """mean paired difference b-a and a t-based 95% CI (n small, so wide)."""
    d = [y - x for x, y in zip(a, b) if x is not None and y is not None]
    if len(d) < 2:
        return "n/a"
    m, s = st.mean(d), st.stdev(d)
    tcrit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}.get(len(d) - 1, 2.0)
    h = tcrit * s / math.sqrt(len(d))
    return "%+.4f  95%% CI [%+.4f, %+.4f]  (n=%d)" % (m, m - h, m + h, len(d))


def checks():
    print("== data checks")
    bad = 0
    tags = sorted({os.path.basename(p).split(".json")[0] for p in
                   glob.glob(os.path.join(RES, "[AB]*_s[0-9].json*"))})
    for tag in tags:
        r = load(tag)
        meta = r.get("run_meta", {})
        issues = []
        if r.get("dropped"):
            issues.append("dropped=" + ",".join(d["host"] for d in r["dropped"]))
        if r.get("aborted"):
            issues.append("aborted")
        for k in ("credits_before", "credits_after"):
            try:
                if float(meta.get(k, "nan")) <= 1:
                    issues.append(k + "<=1")
            except ValueError:
                issues.append(k + " missing")
        if issues:
            bad += 1
            print("  %-22s %s" % (tag, "; ".join(issues)))
    print("  runs with issues: %d" % bad)


def campaign_a(out):
    print("\n== A: sync vs async, clean-power hosts {desktop,pi5,jetson,nano,pi02w}, 120 s")
    ta, ts, idle, rt = [], [], [], []
    for s in SEEDS:
        a, y = load("A_async_s%d" % s), load("A_sync_s%d" % s)
        ta.append(tail_async(a) if a else None)
        ts.append(tail_sync(y) if y else None)
        if y:
            rs = y["rounds"][1:]                 # drop the torch warm-up round
            walls = [r_["wall_s"] for r_ in rs]  # server logs per-round duration
            rt.extend(walls)
            for r_, w in zip(rs, walls):
                d = r_["client_timings"].get("desktop", {}).get("train_s")
                if d is not None and w > 0:
                    idle.append(1 - d / w)
    print("  tail loss async:", ms(ta))
    print("  tail loss sync :", ms(ts))
    print("  sync - async   :", paired(ta, ts))
    if rt:
        print("  sync round time: median %.3f s; desktop idle per round: median %.0f%%"
              % (st.median(rt), 100 * st.median(idle)))
    out["A"] = {"tail_async": ta, "tail_sync": ts}


def campaign_b(out):
    print("\n== B: wire ladder, all 10 hosts, 120 s")
    base = [tail_async(r) if r else None for r in (load("B_fp32_s%d" % s) for s in SEEDS)]
    print("  %-16s %-26s %-38s %8s %7s" % ("wire", "tail loss", "vs fp32 (paired)", "MB", "updates"))
    out["B"] = {}
    for w in ("fp32", "int8", "int8_delta2_ef", "topk_ef"):
        runs = [load("B_%s_s%d" % (w, s)) for s in SEEDS]
        tails = [tail_async(r) if r else None for r in runs]
        mb = [r["bytes_total"] / 1e6 for r in runs if r]
        upd = [len(r["updates"]) for r in runs if r]
        print("  %-16s %-26s %-38s %8.0f %7.0f" % (
            w, ms(tails), "-" if w == "fp32" else paired(base, tails),
            st.mean(mb) if mb else float("nan"), st.mean(upd) if upd else float("nan")))
        out["B"][w] = {"tail": tails, "mb": mb, "updates": upd}


def campaign_bbr(out):
    print("\n== BBR: per-socket cubic vs bbr, fp32, all 10 hosts, adjacent pairs")
    rows = {"cubic": [], "bbr": []}
    for s in SEEDS:
        for cc in ("cubic", "bbr"):
            r = load("BBR_%s_s%d" % (cc, s))
            if not r:
                rows[cc].append(None)
                continue
            per = r["updates_per_host"]
            comm = [u["comm_s"] for u in r["updates"]]
            rows[cc].append({
                "tail": tail_async(r), "updates": len(r["updates"]),
                "upd_bbr_capable": sum(v for h, v in per.items() if h not in CUBIC_ONLY),
                "upd_cubic_only": sum(v for h, v in per.items() if h in CUBIC_ONLY),
                "comm_med": st.median(comm), "mb": r["bytes_total"] / 1e6,
                "cc": r.get("cc", {})})

    def col(cc, k):
        return [x[k] if x else None for x in rows[cc]]
    for k, label in (("updates", "total updates"),
                     ("upd_bbr_capable", "updates, 8 BBR-capable hosts"),
                     ("upd_cubic_only", "updates, jetson+nano (cubic only)"),
                     ("comm_med", "median comm s/exchange"),
                     ("mb", "MB moved"), ("tail", "tail loss")):
        c, b = col("cubic", k), col("bbr", k)
        print("  %-34s cubic %s | bbr %s | bbr-cubic %s" % (label, ms(c), ms(b), paired(c, b)))
    fallbacks = sorted({h for x in rows["bbr"] if x for h, v in x["cc"].items()
                        if v.get("client") != "bbr" and h not in CUBIC_ONLY})
    print("  BBR requested but not in force on:", fallbacks or "none")
    out["BBR"] = rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    out = {}
    checks()
    campaign_a(out)
    campaign_b(out)
    campaign_bbr(out)
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1, default=str)
