# Sync code to astm and notebooks from astm.
REMOTE_DIR=/opt/astro/devel/yuping/orca/
LOCAL_DIR=/home/yuping/radio_projects/lwa/distributed-pipeline/
REMOTE_HOST=astmhead
REMOTE_USER=yuping

rsync -aP --delete --exclude 'notebooks' --exclude '.pytest_cache' --exclude '.venv' \
    --exclude '*__pycache__*' --exclude 'ovro_lwa_orca.egg-info' \
    --exclude '.*cache' \
    --exclude '.idea' $LOCAL_DIR ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} 
