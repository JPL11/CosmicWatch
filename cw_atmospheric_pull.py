import json, time, urllib3; urllib3.disable_warnings()
import requests, datetime
env = dict(l.strip().split('=',1) for l in open('.env') if '=' in l and not l.startswith('#'))
url = env['CREDO_ES_URL'].rstrip('/'); idx = env['CREDO_INDEX']
auth = (env['CREDO_USER'], env['CREDO_PASS'])

def post(path, body, tries=4, to=120):
    for _ in range(tries):
        try:
            r = requests.post(f"{url}/{idx}/{path}", json=body, auth=auth,
                              verify=False, timeout=to)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(5)
    return None

lo, hi, step = 1.755e9, 1.792e9, 5e5
active = []
t = lo
while t < hi:
    j = post("_count", {"query": {"range": {"wall_time": {"gte": t, "lt": t + step}}}})
    n = j["count"] if j else -1
    if n > 0:
        active.append((t, n))
    print(f"slice {t:.0f}: {n}", flush=True)
    t += step
print("non-empty slices:", len(active), flush=True)

rows = []
for t, n in active:
    q = {"size": 0, "query": {"range": {"wall_time": {"gte": t, "lt": t + step}}},
         "aggs": {"rate": {"histogram": {"field": "wall_time", "interval": 3600,
                                         "min_doc_count": 1},
                  "aggs": {"p": {"avg": {"field": "pressure_pa"}},
                           "t": {"avg": {"field": "temperature_c"}}}}}}
    j = post("_search", q, to=240)
    if j:
        bs = j["aggregations"]["rate"]["buckets"]
        rows += [(b["key"], b["doc_count"], b["p"]["value"], b["t"]["value"])
                 for b in bs]
        print(f"agg slice {t:.0f}: +{len(bs)} buckets", flush=True)
    else:
        print(f"AGG FAILED {t:.0f}", flush=True)
json.dump(rows, open("/tmp/cw_hourly_wall.json", "w"))
full = [r_ for r_ in rows if r_[1] > 4000]
days = sorted(set(datetime.datetime.utcfromtimestamp(k).date()
                  for k, _, _, _ in full))
print("DONE buckets:", len(rows), "near-full hours:", len(full),
      "active days:", len(days), flush=True)
print("first/last:", days[0] if days else None, days[-1] if days else None, flush=True)
