#!/usr/bin/env bash
# One federation run on the Azure server (fedsrv) with every edge host over
# WireGuard, one real detector per host. Run from the Mac in fed_real/.
#
#   ./run_campaign.sh --tag wan_fp32_s1 --mode async --wire fp32 --seed 1 --seconds 120
#   ./run_campaign.sh --tag wan_sync_s1 --mode sync --rounds 20 --seed 1
#   ./run_campaign.sh --tag wan_adapt_s1 --mode async --adapt --seed 1
#
# Writes results/<tag>.json (server output + run metadata: per-host ping RTT,
# VM CPU credits before/after, detector assignment) and results/<tag>/ logs.
# B-series VM: a run whose credits hit 0 is flagged and must not be reported.
set -uo pipefail
cd "$(dirname "$0")"

TAG=""; MODE=async; WIRE=fp32; ADAPT=""; SEED=1; SECONDS_RUN=120; ROUNDS=20
DATA=legacy_fl_hardware_10.npz
while [[ $# -gt 0 ]]; do
  case $1 in
    --tag) TAG=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --wire) WIRE=$2; shift 2 ;;
    --adapt) ADAPT=--adapt; shift ;;
    --seed) SEED=$2; shift 2 ;;
    --seconds) SECONDS_RUN=$2; shift 2 ;;
    --rounds) ROUNDS=$2; shift 2 ;;
    --data) DATA=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n $TAG ]] || { echo "--tag required" >&2; exit 2; }
[[ $MODE == async || $MODE == sync ]] || { echo "--mode async|sync" >&2; exit 2; }

SERVER=fedsrv            # ssh alias of the Azure VM
SERVER_VPN=10.8.0.1
SERVER_PY='~/fl/bin/python'
# Detector -> host assignment: seeded permutation (numpy default_rng(2026))
# so partition size is not tied to host speed. Fixed for every campaign.
# ssh-alias   code dir                 python                         --host  detector
HOSTS=(
  "desktop     ~/CosmicWatch/fed_real   ~/CosmicWatch/.venv/bin/python desktop 1"
  "pi5         ~/fed                    python3                        pi5     9"
  "orin        ~/fed                    python3                        jetson  0"
  "pi5borrowed ~/fed                    ~/fl/bin/python                pi5b    6"
  "pi4         ~/fed                    python3                        pi4     8"
  "pi4b        ~/fed                    ~/fl/bin/python                pi4b    3"
  "pi400       ~/fed                    ~/fl/bin/python                pi400   5"
  "pi3         ~/fed                    ~/fl/bin/python                pi3     2"
  "pi02w       ~/fed                    ~/fl/bin/python                pi02w   7"
  "eclipse     ~/fed                    python3                        nano    4"
)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
OUT=results/$TAG; mkdir -p "$OUT"
META=$OUT/meta.txt; : >"$META"

credits() { ./azure_setup.sh status 2>/dev/null | awk '/cpu credits/{print $NF}'; }
# Pi power health: sticky since-boot flags; bit 16 = under-voltage occurred
throttle_flags() {
  for h in "${HOSTS[@]}"; do
    read -r alias _ _ host _ <<<"$h"
    t=$("${SSH[@]}" -n "${alias}" 'vcgencmd get_throttled 2>/dev/null | cut -d= -f2' 2>/dev/null)
    [[ -n $t ]] && echo "throttled_$1[$host]=$t" | tee -a "$META"
  done
}
echo "tag=$TAG mode=$MODE wire=$WIRE adapt=${ADAPT:-no} seed=$SEED data=$DATA" | tee -a "$META"
echo "credits_before=$(credits)" | tee -a "$META"
throttle_flags before
for h in "${HOSTS[@]}"; do
  read -r alias _ _ host _ <<<"$h"
  rtt=$("${SSH[@]}" -n "${alias}" "ping -c 5 -q $SERVER_VPN | tail -1 | cut -d/ -f5" 2>/dev/null)
  echo "ping_ms[$host]=${rtt:-unreachable}" | tee -a "$META"
done

if [[ $MODE == async ]]; then
  SRV_CMD="fed_async_server.py --hosts ${#HOSTS[@]} --seconds $SECONDS_RUN --seed $SEED --data $DATA --out run.json"
  [[ -n $ADAPT ]] && SRV_CMD+=" --adapt" || SRV_CMD+=" --wire $WIRE"
  CLIENT=fed_async_client.py
else
  SRV_CMD="fed_server.py --hosts ${#HOSTS[@]} --rounds $ROUNDS --seed $SEED --data $DATA --out run.json"
  CLIENT=fed_client.py
fi
# separate pkill call with a [f] pattern: it must not match its own shell
"${SSH[@]}" -n "$SERVER" 'pkill -f "[f]ed_async_server|[f]ed_server.py"; true'
"${SSH[@]}" -n "$SERVER" "cd fed && rm -f run.json server.log &&
  setsid -f $SERVER_PY $SRV_CMD </dev/null >server.log 2>&1"
for _ in $(seq 30); do
  "${SSH[@]}" -n "$SERVER" 'grep -q listening fed/server.log' && break; sleep 1
done

for h in "${HOSTS[@]}"; do
  read -r alias dir py host det <<<"$h"
  "${SSH[@]}" -n "${alias}" "cd $dir && timeout $((SECONDS_RUN + 600)) $py $CLIENT \
    --server $SERVER_VPN --host $host --clients $det --data $DATA" >"$OUT/$host.log" 2>&1 &
done
wait

scp -q "$SERVER:fed/run.json" "$OUT/run.json" && scp -q "$SERVER:fed/server.log" "$OUT/server.log"
echo "credits_after=$(credits)" | tee -a "$META"
throttle_flags after
python3 - "$OUT" "results/$TAG.json" <<'EOF'
import json, re, sys, collections
out, dst = sys.argv[1], sys.argv[2]
r = json.load(open(f"{out}/run.json"))
meta = dict(l.strip().split("=", 1) for l in open(f"{out}/meta.txt") if "=" in l)
r["run_meta"] = meta
json.dump(r, open(dst, "w"), indent=1)
if "updates" in r:
    ups = r["updates"]
    print("updates per host:", dict(collections.Counter(u["host"] for u in ups)))
    print(f"test MSE {ups[0]['global_test_weighted_mse']:.4f} -> {ups[-1]['global_test_weighted_mse']:.4f}"
          f" over {len(ups)} mixes, {r['bytes_total']/1e6:.1f} MB")
else:
    rs = r["rounds"]
    print(f"{len(rs)} sync rounds, test MSE {rs[0]['global_test_weighted_mse']:.4f} -> "
          f"{rs[-1]['global_test_weighted_mse']:.4f}, wall {r['total_wall_s']} s")
for d in r.get("dropped", []):
    print(f"WARNING: host {d['host']} dropped at {d['t_s']} s ({d['error']})")
if "aborted" in r:
    print(f"WARNING: sync run aborted: {r['aborted']}")
uv = [k for k, v in meta.items() if k.startswith("throttled_after")
      and v.startswith("0x") and int(v, 16) & 0x10000]
if uv:
    print("WARNING: under-voltage flagged (since boot) on:", ", ".join(uv))
cb, ca = meta.get("credits_before"), meta.get("credits_after")
try:
    if min(float(cb), float(ca)) <= 1.0:
        print("WARNING: VM CPU credits near zero: run may be throttled, do not report")
except (TypeError, ValueError):
    print("WARNING: could not read VM CPU credits")
print("wrote", dst)
EOF
for f in "$OUT"/*.log; do
  [[ $f == */server.log ]] && continue
  grep -qiE "traceback|error" "$f" && { echo "== $(basename "$f") errors:"; tail -4 "$f"; }
done
