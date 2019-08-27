# Sync code to astm and notebooks from astm.
REMOTE_DIR=/home/yuping/transient/
LOCAL_DIR=/home/yuping/radio_projects/lwa/astm-transient-stuff/
REMOTE_HOST=astmhead
REMOTE_USER=yuping

rsync -aP --delete --exclude 'notebooks' --exclude '.pytest_cache' --exclude '.venv' \
    --exclude '*__pycache__*' \
    --exclude 'Pipfile.lock' --exclude '.idea' $LOCAL_DIR ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} 
rsync -aP ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/notebooks $LOCAL_DIR/
