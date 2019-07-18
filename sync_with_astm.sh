# Sync code to astm and notebooks from astm.
REMOTE_DIR=/home/yuping/transient/
LOCAL_DIR=/home/dynamic/astm-workspace/astm-transient-stuff/
REMOTE_HOST=astmhead
REMOTE_USER=yuping

rsync -aP --delete --exclude 'notebooks' --exclude 'Pipfile.lock' $LOCAL_DIR ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} 
rsync -aP ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/notebooks $LOCAL_DIR/
