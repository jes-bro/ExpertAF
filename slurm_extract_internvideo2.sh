#!/bin/bash
#SBATCH --account=simurgh
#SBATCH --partition=simurgh --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --job-name="iv2-feats"
#SBATCH --output=/simurgh2/projects/ret-hoi/InterMimic/baselines_data/iv2-feats-%A_%a.out
#SBATCH --mail-user=jesb@stanford.edu
#SBATCH --mail-type=END,FAIL
# InternVideo2 ego/exo features for every ExpertAF window (extract_internvideo2_feats.py), one array task
# per shard of takes. Resumable: rerun the same array and finished windows are skipped.
# Submit from the ExpertAF clone:  sbatch --exclude=simurgh6,simurgh2 slurm_extract_internvideo2.sh
# Smoke test first (one take, no array):
#   SMOKE_TAKE=sfu_basketball013_11 sbatch --exclude=simurgh6,simurgh2 --array=0 slurm_extract_internvideo2.sh
# Override any path per submission, e.g. IV2_ENV=<conda env name>.
set -u
source ~/.bashrc
conda deactivate 2>/dev/null
conda activate "${IV2_ENV:-internvideo2}"

ROOT=/simurgh2/projects/ret-hoi
JSON_DIR="${JSON_DIR:-$ROOT/InterMimic/baselines_data/expertaf_json}"
TAKES_JSON="${TAKES_JSON:-/vision/group/egoexo4d/takes.json}"
EGOEXO_ROOT="${EGOEXO_ROOT:-/vision/group/egoexo4d}"
IV2_ROOT="${IV2_ROOT:-$ROOT/InternVideo/InternVideo2/multi_modality}"
WEIGHTS="${WEIGHTS:-$ROOT/baselines_data/internvideo2/InternVideo2-stage2_1b-224p-f4.pt}"
OUT="${OUT:-$ROOT/InterMimic/baselines_data/egoexo_feats}"
N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"

for p in "$JSON_DIR/train.json" "$JSON_DIR/test.json" "$TAKES_JSON" "$IV2_ROOT/demo/utils.py" "$WEIGHTS"; do
    [ -e "$p" ] || { echo "[iv2] ERROR: missing $p" >&2; exit 1; }
done
python -c "import flash_attn, timm, einops, cv2, torch; print('[iv2] torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

cd "$(dirname "$0")"
echo "[iv2] shard $I of $N -> $OUT (job ${SLURM_JOB_ID:-none}, host $(hostname))"
python -u extract_internvideo2_feats.py --expertaf-json-dir "$JSON_DIR" --takes-json "$TAKES_JSON" \
    --egoexo-root "$EGOEXO_ROOT" --internvideo-root "$IV2_ROOT" --weights "$WEIGHTS" --out-dir "$OUT" \
    --shard "$I" --num-shards "$N" ${SMOKE_TAKE:+--takes "$SMOKE_TAKE"}
