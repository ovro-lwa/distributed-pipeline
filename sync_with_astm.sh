# Sync code to astm and notebooks from astm.
set -e

if [ -z "$ASTM_USER_NAME" ]; then
  echo 'Environmental variable ASTM_USER_NAME is not set.'
  exit 1
fi

REMOTE_USER=$ASTM_USER_NAME
REMOTE_DIR=/opt/astro/devel/${REMOTE_USER}/
LOCAL_DIR=`pwd`
REMOTE_HOST=astm13.lwa.ovro.caltech.edu

rsync -aP --delete --exclude 'notebooks' --exclude '.pytest_cache' --exclude '.venv' \
    --exclude '*__pycache__*' --exclude 'ovro_lwa_orca.egg-info' \
    --exclude '.*cache' \
    --exclude '.idea' $LOCAL_DIR ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} 
