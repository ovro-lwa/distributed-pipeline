#!/usr/bin/env bash
# Deploy the committed subproject and record its parent-repository revision.
set -euo pipefail

HOST=${1:-calim0}
DEST=${2:-/opt/devel/pipeline/gpu-subband-imaging}
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$PROJECT" rev-parse --show-toplevel)
REL=${PROJECT#"$REPO"/}

if [[ -n $(git -C "$REPO" status --porcelain -- "$REL") ]]; then
    echo "Commit $REL changes before deployment." >&2
    exit 1
fi

REVISION=$(git -C "$REPO" rev-parse HEAD)
rsync -a \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude 'config/' \
    --exclude 'config-*/' \
    --exclude 'old/' \
    "$PROJECT/" "$HOST:$DEST/"
ssh "$HOST" "printf '%s\\n' '$REVISION' > '$DEST/.gsi-version'"
echo "Deployed $REVISION to $HOST:$DEST"
