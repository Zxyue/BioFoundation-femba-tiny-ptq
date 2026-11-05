#!/usr/bin/env bash
#SBATCH --job-name=tusl-qap-six
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=tusl-qap-six-%j.log
# Set account/partition/time/memory according to your cluster before submission.
set -euo pipefail
: "${TOOLS:?Export TOOLS as the extracted complete tool directory}"
: "${REPO:?Export REPO as xiaochen/femba-qap}"
: "${DATA:?Export DATA as tusl-events-20261026-v2}"
: "${OUT:?Export OUT as the new combined output directory}"
PYTHON_BIN=${PYTHON_BIN:-python}
exec "$PYTHON_BIN" "$TOOLS/run_all_tusl_qap.py" run \
  --repo "$REPO" --data-root "$DATA" --out "$OUT" "$@"
