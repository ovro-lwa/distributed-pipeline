# Environment for running the gsi-v5 GPU pipeline from calim02 (source before gsi / run_nightly_lst.py)
# = /opt/devel/pipeline/gsi-v5/gsi_env.sh (gsi-v4's with gsi-v4 -> gsi-v5); night_guarded.sh sources $G/gsi_env.sh
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor
export GSI_HOME=/opt/devel/pipeline/gsi-v5/distributed-pipeline/gpu-subband-imaging
export PYTHONPATH=$GSI_HOME/src
export GSI_WORKER_SH=$GSI_HOME/workers/run_worker.sh
export GSI_TASKSET_CPUS=0-23            # data recorders use cores 24-47
export GSI_NICE=10
export GSI_IONICE="-c2 -n7"
export NUMPY_MADVISE_HUGEPAGE=0
export CASARCFILES=/opt/devel/pipeline/gsi-v2/casarc   # casacore measures (fix_uvw)
export GSI_ALLOWED_LUSTRE_WRITE_ROOT=/lustre/pipeline/snapshots_gpu_pipeline
