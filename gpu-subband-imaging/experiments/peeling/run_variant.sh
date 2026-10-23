#!/usr/bin/env bash
# Peel + image one variant over a set of prepped snapshots.
#   GPU=0 run_variant.sh VARIANT SNAP [SNAP ...]      (SNAP = e.g. 23MHz_0800)
# Output: $PX/runs/study/<SNAP>/<VARIANT>.ms and img/<VARIANT>-{I,V}-image.fits
#
# Variants
#   prep            no peeling (reference)
#   cpu_sky         main pipeline sky zest (julia060 TTCal)
#   cpu_sky_rfi     main pipeline sky zest + RFI zest (ttcal_dev)
#   gpu_prod        deployed TTCalX + production daemon, production settings
#   gpu_exp         experimental TTCalX/daemon, production settings (as_is UVW)
#   gpu_geom        + geometric (precessed) UVW
#   gpu_cpumatch    geom + CPU TTCal settings (rk4, maxiter 5, tol 1e-4, minuvw 5,
#                   all sources above horizon, no convergence gate)
#   gpu_geom_rfi    gpu_geom + RFI pass (correct emitter position)
#   gpu_geom_rficpu gpu_geom + RFI pass at ttcal_dev's (misread) emitter position
#   gpu_rfi_oldbr   gpu_exp + RFI pass with the old branch's geometry (no ŝ₀ term)
#   gpu_reg<D>      gpu_geom_rfi with the Tukey phase prior, max offset D deg
#                   (ZEST_REG_TAPER_DEG / ZEST_REG_LAMBDA / ZEST_REG_TARGETS from env)
set -eo pipefail
VARIANT=$1; shift
source "$(dirname "$0")/env.sh"
DAEMON_EXP=$PX_REPO/workers/zest_daemon.jl
DAEMON_PROD=$PX/prod_daemon/zest_daemon.jl
CPU_RFI_ITRF="6.368747415744499e6,-364625.3282460041,114589.56805339623"
STUDY=$PX/runs/study

MSES=()
for snap in "$@"; do
    prep=( "$PX"/data/$snap/*_prep.ms )
    out=$STUDY/$snap/$VARIANT.ms
    mkdir -p "$STUDY/$snap/img"
    rm -rf "$out"; cp -r "${prep[0]}" "$out"
    MSES+=("$out")
done

gpu_run() {  # TTCALX DAEMON SOURCES
    "$PX_SCRIPTS/gpu_zest.sh" "$1" "$2" "$3" "${MSES[@]}"
}
geom() { export ZEST_UVW_CONVENTION=geom; }
rfi() { export ZEST_RFI_SOURCES=$RFI_MODEL; }

t0=$(date +%s)
case $VARIANT in
    prep) ;;
    cpu_sky) for ms in "${MSES[@]}"; do "$PX_SCRIPTS/cpu_zest.sh" "$ms" sky; done ;;
    cpu_sky_rfi)
        for ms in "${MSES[@]}"; do
            "$PX_SCRIPTS/cpu_zest.sh" "$ms" sky; "$PX_SCRIPTS/cpu_zest.sh" "$ms" rfi
        done ;;
    gpu_prod) gpu_run "$TTCALX_BASE" "$DAEMON_PROD" "$SKY_MODEL" ;;
    gpu_exp) gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_geom) geom; gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_cpumatch)
        geom
        export ZEST_SOLVER=rk4 ZEST_MAXITER=5 ZEST_TOLERANCE=1e-4 ZEST_MINUVW=5
        export ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_REQUIRE_CONVERGENCE=false
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_geom_rfi) geom; rfi; gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_geom_rficpu)
        geom; rfi; export ZEST_RFI_POSITION_ITRF=$CPU_RFI_ITRF
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_rfi_oldbr)
        rfi; export ZEST_RFI_GEOMETRY=none
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_reg*)
        geom; rfi
        export ZEST_REG_MAX_OFFSET_DEG=${VARIANT#gpu_reg}
        export ZEST_REG_MAX_OFFSET_DEG=${ZEST_REG_MAX_OFFSET_DEG%%_*}
        export ZEST_SAVE_GAINS_DIR=$STUDY/gains; mkdir -p "$ZEST_SAVE_GAINS_DIR"
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
esac
echo "VARIANT_PEELED $VARIANT n=${#MSES[@]} elapsed=$(( $(date +%s) - t0 ))s"

SNAPS=("$@")
for i in "${!MSES[@]}"; do
    snap=${SNAPS[$i]}
    band=${snap%%MHz*}
    "$PX_SCRIPTS/image.sh" "${MSES[$i]}" "$STUDY/$snap/img/$VARIANT" "$band" > /dev/null
done
echo "VARIANT_DONE $VARIANT"
