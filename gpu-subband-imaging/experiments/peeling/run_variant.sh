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
#   gpu_joint       gpu_geom with the RFI emitter zested jointly with the sky
#   gpu_jreg<D>     gpu_joint with the Tukey phase prior, max offset D deg
#   gpu_prodfix     production code path and settings + the three minimal fixes:
#                   no convergence gate, elevation cut 0, model flags (ZEST_FLAG_MODEL)
#   gpu_prodfix_rfi gpu_prodfix + joint RFI source (damped solver)
#   gpu_cm_exact    gpu_cpumatch with the CPU-exact host model (ZEST_GENVIS=exact)
#   gpu_cm_exact_joint  gpu_cm_exact + joint RFI
#   gpu_cm_after    gpu_cpumatch + RFI pass after sky (the main pipeline's scheme)
#   gpu_cm_aftercpu gpu_cm_after at ttcal_dev's (misread) emitter position
#   gpu_cm_joint    gpu_cpumatch with the RFI zested jointly with the sky
#   gpu_cm_jreg<D>  gpu_cm_joint + Tukey phase prior, max offset D deg
#   gpu_cm_areg<D>  gpu_cm_after with the prior on the sky pass only (stops the
#                   sky gains absorbing the RFI before the RFI pass)
#   abl_*           one-at-a-time changes to production settings (exp code):
#                   abl_tol (damped, tol 1e-4, maxiter 100), abl_el0 (min elevation 0),
#                   abl_nogate (no convergence gate), abl_rk4 (rk4, tol 1e-4, maxiter 30,
#                   no gate), abl_geom_rk4 (abl_rk4 + geometric UVW)
#                   abl_nogate_{uvw5,it5}: abl_nogate + minuvw 5 / maxiter 5;
#                   abl_cm_{uvw10,it30}: gpu_cpumatch with production minuvw / maxiter
#   gpu_reg<D>      gpu_geom_rfi with the Tukey phase prior, max offset D deg
#                   (ZEST_REG_TAPER_DEG / ZEST_REG_LAMBDA / ZEST_REG_TARGETS from env)
set -eo pipefail
VARIANT=$1; shift
source "$(dirname "$0")/env.sh"
DAEMON_EXP=$PX_REPO/workers/zest_daemon.jl
DAEMON_PROD=$PX/prod_daemon/zest_daemon.jl
CPU_RFI_ITRF="6.368747415744499e6,-364625.3282460041,114589.56805339623"
STUDY=${STUDY:-$PX/runs/study}
DATA=${DATA:-$PX/data}

MSES=()
for snap in "$@"; do
    prep=( "$DATA"/$snap/*_prep.ms )
    out=$STUDY/$snap/$VARIANT.ms
    mkdir -p "$STUDY/$snap/img"
    rm -rf "$out"; cp -r "${prep[0]}" "$out"
    MSES+=("$out")
done

gpu_run() {  # TTCALX DAEMON SOURCES
    "$PX_SCRIPTS/gpu_zest.sh" "$1" "$2" "$3" "${MSES[@]}"
}
geom() { export ZEST_UVW_CONVENTION=geom; }
cpum() {
    geom
    export ZEST_SOLVER=rk4 ZEST_MAXITER=5 ZEST_TOLERANCE=1e-4 ZEST_MINUVW=5
    export ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_REQUIRE_CONVERGENCE=false
    export ZEST_FLAG_MODEL=${ZEST_FLAG_MODEL:-1} ZEST_RESUBTRACT=1
}
reg() {  # max offset from the variant name suffix
    export ZEST_REG_MAX_OFFSET_DEG=$1
    export ZEST_SAVE_GAINS_DIR=$STUDY/gains; mkdir -p "$ZEST_SAVE_GAINS_DIR"
}
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
    gpu_prodfix) export ZEST_REQUIRE_CONVERGENCE=false ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_FLAG_MODEL=1 ZEST_RESUBTRACT=1
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_prodfix_rfi) export ZEST_REQUIRE_CONVERGENCE=false ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_FLAG_MODEL=1 ZEST_RESUBTRACT=1
        rfi; export ZEST_RFI_MODE=joint
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_cm_exact) cpum; export ZEST_GENVIS=exact
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_cm_exact_joint) cpum; rfi; export ZEST_GENVIS=exact ZEST_RFI_MODE=joint
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_cm_after) cpum; rfi; gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_cm_aftercpu) cpum; rfi; export ZEST_RFI_POSITION_ITRF=$CPU_RFI_ITRF
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_cm_joint) cpum; rfi; export ZEST_RFI_MODE=joint; reg 0
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_cm_jreg*) cpum; rfi; export ZEST_RFI_MODE=joint; reg "${VARIANT#gpu_cm_jreg}"
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_cm_areg*) cpum; rfi; export ZEST_REG_TARGETS=sky; reg "${VARIANT#gpu_cm_areg}"
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_nogate_uvw5) export ZEST_REQUIRE_CONVERGENCE=false ZEST_MINUVW=5
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_nogate_it5) export ZEST_REQUIRE_CONVERGENCE=false ZEST_MAXITER=5
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_cm_uvw10) cpum; export ZEST_MINUVW=10
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_cm_it30) cpum; export ZEST_MAXITER=30
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_nogate_el0) export ZEST_REQUIRE_CONVERGENCE=false ZEST_MIN_SOURCE_ELEVATION_DEG=0
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_nogate_el5) export ZEST_REQUIRE_CONVERGENCE=false ZEST_MIN_SOURCE_ELEVATION_DEG=5
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_cm_skipshort) cpum; export ZEST_SKIP_SHORT=1
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_cm_after_skipshort) cpum; rfi; export ZEST_SKIP_SHORT=1
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_cm_flagmodel) cpum; export ZEST_FLAG_MODEL=1
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_nogate_el0_flagmodel) export ZEST_REQUIRE_CONVERGENCE=false ZEST_MIN_SOURCE_ELEVATION_DEG=0 ZEST_FLAG_MODEL=1
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_tol) export ZEST_TOLERANCE=1e-4 ZEST_MAXITER=100
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_el0) export ZEST_MIN_SOURCE_ELEVATION_DEG=0
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_nogate) export ZEST_REQUIRE_CONVERGENCE=false
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_rk4) export ZEST_SOLVER=rk4 ZEST_TOLERANCE=1e-4 ZEST_REQUIRE_CONVERGENCE=false
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    abl_geom_rk4) geom; export ZEST_SOLVER=rk4 ZEST_TOLERANCE=1e-4 ZEST_REQUIRE_CONVERGENCE=false
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_joint) geom; rfi; export ZEST_RFI_MODE=joint
        gpu_run "$TTCALX_EXP" "$DAEMON_EXP" "$SKY_MODEL" ;;
    gpu_jreg*)
        geom; rfi; export ZEST_RFI_MODE=joint
        export ZEST_REG_MAX_OFFSET_DEG=${VARIANT#gpu_jreg}
        export ZEST_REG_MAX_OFFSET_DEG=${ZEST_REG_MAX_OFFSET_DEG%%_*}
        export ZEST_SAVE_GAINS_DIR=$STUDY/gains; mkdir -p "$ZEST_SAVE_GAINS_DIR"
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
    band=$(basename "$snap"); band=${band%%MHz*}
    "$PX_SCRIPTS/image.sh" "${MSES[$i]}" "$STUDY/$snap/img/$VARIANT" "$band" > /dev/null
done
echo "VARIANT_DONE $VARIANT"
