#!/bin/bash
set -euo pipefail

export ROCR_VISIBLE_DEVICES="$SLURM_LOCALID"
exec "$@"