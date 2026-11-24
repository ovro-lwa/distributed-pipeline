#!/bin/bash
# One night on a gsi deploy (production layout), ONLY if that date's calibration is good:
# the main (LST 22h) table, else the backup (21h), must be successful, have 15 bands in frequency order with < 50%
# flagged, pass run_nightly_lst.check_table against REF and the within-band shape check (bp_ramp.py).
# Otherwise nothing runs.  usage: night_guarded.sh <YYYY-MM-DD>
#
# gsi-v4's /opt/devel/pipeline/gsi-v4/night_guarded_v4.sh with the deploy as a variable:
#   G         deploy dir (default /opt/devel/pipeline/gsi-v5); its gsi_env.sh is sourced, the logs, run labels and
#             the generated config are written there, the template is read from there
#   TEMPLATE  config template in $G (default config-lst-<v5>-template, <v5> from $G's name gsi-v5)
#   REF_TABLE reference bandpass table (daily_guarded.sh sets it from $G/last_good_bp.txt)
# bp_ramp.py and etcd_ok.sh are the ones next to this script.
G=${G:-/opt/devel/pipeline/gsi-v5}
TAG=$(basename $G)                  # gsi-v5: the log line daily_guarded.sh greps says "on gsi-v5"
V=${TAG#gsi-}                       # v5
TEMPLATE=${TEMPLATE:-config-lst-$V-template}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=$1
REF=${REF_TABLE:-/lustre/pipeline/calibration/results/2026-11-05/22h/successful/20261105_063757/tables/calibration_2026-11-05_22h.B.flagged}   # post-change reference (bandpass shape changed 2026-11-04 ~03 UTC)
W=/fast/pipeline/calibration/watch_state_lst
LOG=$G/night_guard_${D}.log
say() { echo "[$(date -u '+%F %T')] $*" | tee -a $LOG; }
[ -d $G/$TEMPLATE ] || { say "no template $G/$TEMPLATE"; exit 1; }
source $G/gsi_env.sh
check() {   # <state json> -> "OK <table> ..." or "REJECT ..."
python - "$1" "$REF" "$D" "$HERE/bp_ramp.py" <<'PY'
import json, os, subprocess, sys, numpy as np
sys.path.insert(0, os.environ["GSI_HOME"] + "/scripts")
st = json.load(open(sys.argv[1]))
if st.get("status") != "successful" or not st.get("bandpass"):
    print("REJECT %s status=%s" % (st.get("role"), st.get("status"))); sys.exit()
bp = st["bandpass"]
from casacore.tables import table
t = table(bp, ack=False); f = t.getcol("FLAG"); s = t.getcol("SPECTRAL_WINDOW_ID")
fr = table(t.getkeyword("SPECTRAL_WINDOW"), ack=False).getcol("REF_FREQUENCY") / 1e6
flags = [float(f[s == k].mean()) for k in range(len(fr))]
if len(fr) != 15 or not np.all(np.diff(fr) > 0) or max(flags) >= 0.5:
    print("REJECT nspw=%d ordered=%s maxflag=%.2f" % (len(fr), bool(np.all(np.diff(fr) > 0)), max(flags))); sys.exit()
from pathlib import Path
import run_nightly_lst as rn
ok, why = rn.check_table(bp, sys.argv[2], Path("/lustre/pipeline/snapshots_gpu_pipeline/_state/bpcheck_%s_%s" % (sys.argv[3], st.get("role"))))
if not ok:
    print("REJECT " + bp + " | " + why.replace("\n", " ")[:400]); sys.exit()
r = subprocess.run([sys.executable, "-W", "ignore", sys.argv[4], bp, sys.argv[2]],
                   capture_output=True, text=True).stdout.strip()
print(("OK " + bp + " | " + r) if r.startswith("OK") else ("REJECT " + bp + " | " + r))
PY
}
BPT=
for role in main backup; do
    until [ -f $W/${D}_$role.json ] && grep -q '"status"' $W/${D}_$role.json; do
        [ $(date -u +%s) -gt $(date -u -d "$D 16:00" +%s) ] && break
        sleep 120
    done
    [ -f $W/${D}_$role.json ] || { say "$role: no calibration state by 16 UTC"; continue; }
    R=$(check $W/${D}_$role.json)
    say "$role calibration: $R"
    case "$R" in OK*) BPT=$(echo "$R" | awk '{print $2}'); break;; esac
done
[ -n "$BPT" ] || { say "no good calibration for $D: not starting the night"; exit 0; }
[ -n "$CHECK_ONLY" ] && { say "CHECK_ONLY: would start with $BPT"; exit 0; }
# the MNC etcd anthealth service must answer (the night's pinned bad-antenna list): wait for it, until 23:00 UTC of
# the date; then the night is not started (never run without the bad antennas; the next date goes on)
until bash $HERE/etcd_ok.sh; do
    if [ $(date -u +%s) -gt $(date -u -d "$D 23:00" +%s) ]; then
        say "anthealth / etcd not answering by 23:00 UTC: not starting the night"; exit 0
    fi
    say "anthealth / etcd not answering: waiting"; sleep 600
done
cd $G
RL=$(cat $G/run_label_${D}.txt 2>/dev/null || echo Run_$(date -u +%Y%m%d_%H%M%S))
echo $RL > $G/run_label_${D}.txt
say "starting night $D on $TAG, run label $RL, bandpass $BPT"
python $GSI_HOME/scripts/run_nightly_lst.py --date $D --run-label $RL --bp $BPT \
    --template $TEMPLATE --output-config config-lst-$V-auto 2>&1 | tee -a nightly_${D}.log
say "night $D finished rc=${PIPESTATUS[0]}"
