#!/usr/bin/env bash
# October 2026 real-WAN campaigns (Azure server, WireGuard). Resumable: a tag
# whose results/<tag>.json exists is skipped. Seeds are interleaved across
# arms so slow drift (network, time of day) spreads evenly over the arms.
#
#   A   sync vs async, clean-power hosts only (timing claim), 120 s, 5 seeds
#   B   wire ladder, all 10 hosts (participation/quality claim), 120 s, 5 seeds
#   BBR cubic vs bbr per-socket congestion control, fp32, all 10, adjacent pairs
#   L   long runs: fp32 / int8 / int8_delta2_ef, 600 s, all 10, 5 seeds (does
#       the paper's LAN int8 late-run instability reproduce over the WAN?)
#
# Usage: ./campaigns_2026-10.sh [A] [B] [BBR] [L] (default: A B BBR, in order)
set -uo pipefail
cd "$(dirname "$0")"
CLEAN=desktop,pi5,jetson,nano,pi02w      # hosts with clean power/thermal flags
SEEDS="1 2 3 4 5"
LOG=results/campaign_log.txt
mkdir -p results

credits() { ./azure_setup.sh status 2>/dev/null | awk '/cpu credits/{print int($NF)}'; }

run() {                                   # run <tag> <runner args...>
  local tag=$1; shift
  if [[ -f results/$tag.json ]]; then echo "skip $tag (done)" | tee -a "$LOG"; return; fi
  local c; c=$(credits)
  while [[ -n $c && $c -lt 30 ]]; do      # B-series: let credits recharge
    echo "$(date '+%F %T') credits $c < 30, waiting 10 min" | tee -a "$LOG"
    sleep 600; c=$(credits)
  done
  echo "$(date '+%F %T') start $tag ($*)" | tee -a "$LOG"
  ./run_campaign.sh --tag "$tag" "$@" >"results/$tag.out" 2>&1
  grep -E "^(updates|test MSE|[0-9]+ sync|cc per|WARNING)" "results/$tag.out" | sed "s/^/    /" | tee -a "$LOG"
  [[ -f results/$tag.json ]] || echo "    FAILED: no results/$tag.json (see results/$tag.out)" | tee -a "$LOG"
}

ARMS=" ${*:-A B BBR} "
has() { [[ $ARMS == *" $1 "* ]]; }      # whole-word: "BBR" must not match "B"
for s in $SEEDS; do
  if has A; then
    run "A_async_s$s" --mode async --wire fp32 --seed "$s" --seconds 120 --hosts "$CLEAN"
    run "A_sync_s$s"  --mode sync --seed "$s" --seconds 120 --hosts "$CLEAN"
  fi
done
for s in $SEEDS; do
  if has B; then
    for w in fp32 int8 int8_delta2_ef topk_ef; do
      run "B_${w}_s$s" --mode async --wire "$w" --seed "$s" --seconds 120
    done
  fi
done
for s in $SEEDS; do
  if has BBR; then
    run "BBR_cubic_s$s" --mode async --wire fp32 --seed "$s" --seconds 120
    run "BBR_bbr_s$s"   --mode async --wire fp32 --seed "$s" --seconds 120 --cc bbr
  fi
done
for s in $SEEDS; do
  if has L; then
    for w in fp32 int8 int8_delta2_ef; do
      run "L_${w}_s$s" --mode async --wire "$w" --seed "$s" --seconds 600
    done
  fi
done
echo "$(date '+%F %T') campaigns done: $ARMS" | tee -a "$LOG"
