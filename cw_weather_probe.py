"""Weather probe on the frozen archive's hourly series.

The CosmicWatch-schema docs carry an onboard barometer/thermometer but
no geo field, so step 1 locates the detector empirically: correlate the
device's hourly pressure series against Open-Meteo archive weather at
candidate sites and keep the site that explains it. Step 2 then uses
that site's EXTERNAL weather (pressure, temperature, humidity,
precipitation) as regressors for the muon rate on the stable post-
change segment, complementing the device-sensor-only analysis in
PHYSICS_PROBE_NOTES.md.

Input: the hourly (wall_time, count, device_pressure_pa, device_temp_c)
buckets cached by cw_atmospheric_pull.py. Output: cw_weather_probe.json.
"""
import datetime
import json
import math
import urllib.request

HOURLY = "/tmp/cw_hourly_wall.json"
CHANGE_POINT = datetime.datetime(2026, 6, 11)  # known instrument step

CANDIDATES = {
    "MIT/Boston": (42.36, -71.09),
    "Krakow": (50.06, 19.94),
    "Warsaw": (52.23, 21.01),
    "Pomona CA": (34.06, -117.82),
    "Long Beach CA": (33.77, -118.19),
}
VARS = ("surface_pressure,pressure_msl,temperature_2m,"
        "relative_humidity_2m,precipitation")


def fetch(lat, lon):
    u = ("https://archive-api.open-meteo.com/v1/archive"
         f"?latitude={lat}&longitude={lon}"
         "&start_date=2026-05-16&end_date=2026-06-20"
         f"&hourly={VARS}&timezone=UTC")
    with urllib.request.urlopen(u, timeout=60) as r:
        return json.load(r)["hourly"]


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((a - my) ** 2 for a in y))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def ols(y, cols):
    """Tiny OLS with intercept; returns coefs and their std errors."""
    import numpy as np
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, res, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
    dof = len(y) - X.shape[1]
    s2 = float(res[0]) / dof if len(res) else float(
        ((np.array(y) - X @ beta) ** 2).sum()) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


rows = [r for r in json.load(open(HOURLY)) if r[1] > 4000 and r[2] and r[3]]
by_hour = {int(k // 3600 * 3600): (c, p / 100.0, t) for k, c, p, t in rows}

out = {"n_hours": len(by_hour), "site_match": {}, "regressions": {}}

# ---- step 1: which site's weather does the device barometer track? ----
best, weather = None, None
for site, (lat, lon) in CANDIDATES.items():
    h = fetch(lat, lon)
    ext = {}
    for i, ts in enumerate(h["time"]):
        t = int(datetime.datetime.fromisoformat(ts)
                .replace(tzinfo=datetime.timezone.utc).timestamp())
        ext[t] = {k: h[k][i] for k in h if k != "time"}
    common = sorted(set(by_hour) & set(ext))
    dev_p = [by_hour[t][1] for t in common]
    r_sfc = pearson(dev_p, [ext[t]["surface_pressure"] for t in common])
    r_msl = pearson(dev_p, [ext[t]["pressure_msl"] for t in common])
    off = (sum(ext[t]["surface_pressure"] for t in common) / len(common)
           - sum(dev_p) / len(common))
    out["site_match"][site] = {"r_surface": round(r_sfc, 3),
                               "r_msl": round(r_msl, 3),
                               "surface_offset_hPa": round(off, 1),
                               "n": len(common)}
    print(site, out["site_match"][site], flush=True)
    score = max(r_sfc, r_msl)
    if best is None or score > best[1]:
        best, weather = (site, score), ext

out["best_site"] = {"site": best[0], "r": round(best[1], 3)}
print("BEST:", out["best_site"], flush=True)

# ---- step 2: rate vs external weather on the stable segment ----
cp = CHANGE_POINT.replace(tzinfo=datetime.timezone.utc).timestamp()
seg = [t for t in sorted(set(by_hour) & set(weather)) if t >= cp]
y = [100.0 * math.log(by_hour[t][0]) for t in seg]      # % (log-rate)
mean_rate = sum(by_hour[t][0] for t in seg) / len(seg)
poisson_pct = 100.0 / math.sqrt(mean_rate)

names = ["pressure_msl", "temperature_2m", "relative_humidity_2m"]
cols = [[weather[t][n] for t in seg] for n in names]
cols.append([by_hour[t][2] for t in seg])               # device temp covariate
beta, se = ols(y, cols)
out["regressions"]["stable_segment"] = {
    "n_hours": len(seg),
    "mean_counts_per_hour": round(mean_rate),
    "hourly_poisson_noise_pct": round(poisson_pct, 2),
    "coef_pct_per_unit": {n: [round(float(b), 3), round(float(s), 3)]
                          for n, b, s in zip(names + ["device_temp_c"],
                                             beta[1:], se[1:])},
}

# univariate external-pressure slope (the classic barometric fit)
bp, sp = ols(y, [[weather[t]["pressure_msl"] for t in seg]])
out["regressions"]["barometric_external_only"] = {
    "beta_pct_per_hPa": round(float(bp[1]), 3),
    "se": round(float(sp[1]), 3)}

# precipitation: rainy vs dry hours
wet = [math.log(by_hour[t][0]) for t in seg
       if weather[t]["precipitation"] and weather[t]["precipitation"] > 0]
dry = [math.log(by_hour[t][0]) for t in seg
       if not weather[t]["precipitation"]]
if wet:
    d = 100.0 * (sum(wet) / len(wet) - sum(dry) / len(dry))
    out["regressions"]["precip_wet_minus_dry_pct"] = {
        "delta_pct": round(d, 2), "n_wet": len(wet), "n_dry": len(dry)}
else:
    out["regressions"]["precip_wet_minus_dry_pct"] = {
        "delta_pct": None, "n_wet": 0, "n_dry": len(dry)}

json.dump(out, open("cw_weather_probe.json", "w"), indent=1)
print(json.dumps(out["regressions"], indent=1), flush=True)
print("WROTE cw_weather_probe.json", flush=True)
