#!/usr/bin/env bash
# run_all_experiments.sh
# Submit all EWL sweep jobs to SLURM, or run them locally on a multi-GPU machine.
#
# Usage:
#   ./scripts/run_all_experiments.sh                           # submit all sweeps to SLURM
#   ./scripts/run_all_experiments.sh --local                   # run locally (multi-GPU machine)
#   ./scripts/run_all_experiments.sh --sweep ablations noise_imbalance  # subset of sweeps
#   ./scripts/run_all_experiments.sh --no_wandb                # disable wandb for all runs
#   ./scripts/run_all_experiments.sh --gpus 0 1                # override GPU list (local mode)
#   ./scripts/run_all_experiments.sh --entity my-team          # override wandb entity
#   ./scripts/run_all_experiments.sh --project my-project      # override wandb project
#   ./scripts/run_all_experiments.sh --epochs 2 --no_wandb     # smoke test (2 epochs)
#
# Wandb credentials:
#   Store your API key in ~/.wandb_secrets:
#     echo 'export WANDB_API_KEY="your_key_here"' > ~/.wandb_secrets && chmod 600 ~/.wandb_secrets
#   Entity defaults to null (your personal account).  Pass --entity to log to a team.
#
# Available sweep names:
#   all_datasets          — SFT / EWL / EWL-no-proxy × 3 datasets × 3 seeds  (27 runs)
#   noise_all_datasets    — Noise ablation × 3 datasets × 5 levels × 3 seeds  (90 runs)
#   aircraft_ablations    — Rank / alpha / temperature sweep on Aircraft        (~66 runs)
#   noise_imbalance       — Noise + imbalance ablation on Aircraft              (20 runs)

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLURM_DIR="$PROJECT_ROOT/scripts/slurm"
LOCAL=false
NO_WANDB=false
GPUS="0"
EPOCHS=""
WANDB_ENTITY_OVERRIDE=""
WANDB_PROJECT_OVERRIDE=""
SELECTED_SWEEPS=()

ALL_SWEEPS=(all_datasets noise_all_datasets aircraft_ablations noise_imbalance)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)        LOCAL=true;                           shift ;;
        --no_wandb)     NO_WANDB=true;                        shift ;;
        --gpus)         shift; GPUS="$1";                     shift ;;
        --epochs)       shift; EPOCHS="$1";                   shift ;;
        --entity)       shift; WANDB_ENTITY_OVERRIDE="$1";    shift ;;
        --project)      shift; WANDB_PROJECT_OVERRIDE="$1";   shift ;;
        --sweep)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SELECTED_SWEEPS+=("$1"); shift
            done
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ${#SELECTED_SWEEPS[@]} -eq 0 ]]; then
    SELECTED_SWEEPS=("${ALL_SWEEPS[@]}")
fi

# Build extra args for local mode
EXTRA_ARGS=""
[[ "$NO_WANDB" == true ]]  && EXTRA_ARGS="$EXTRA_ARGS --no_wandb"
[[ -n "$EPOCHS" ]]         && EXTRA_ARGS="$EXTRA_ARGS --epochs $EPOCHS"

GPU_LIST=$(echo "$GPUS" | tr ' ' '\n' | head -1 | xargs)   # used for display
GPU_ARGS="--gpus $GPUS"

cd "$PROJECT_ROOT"
mkdir -p slurm/logs

# ── Wandb setup (local mode) ──────────────────────────────────────────────────
if [[ $NO_WANDB == false ]]; then
    # Source credentials from ~/.wandb_secrets if present
    [[ -f ~/.wandb_secrets ]] && source ~/.wandb_secrets

    # Apply overrides from CLI flags
    [[ -n "$WANDB_PROJECT_OVERRIDE" ]] && export WANDB_PROJECT="$WANDB_PROJECT_OVERRIDE"
    [[ -n "$WANDB_ENTITY_OVERRIDE"  ]] && export WANDB_ENTITY="$WANDB_ENTITY_OVERRIDE"

    # Defaults — override with --project / --entity flags
    export WANDB_PROJECT="${WANDB_PROJECT_OVERRIDE:-${WANDB_PROJECT:-ewl-vision-ensf619}}"
    export WANDB_ENTITY="${WANDB_ENTITY_OVERRIDE:-${WANDB_ENTITY:-sayannath235}}"

    if [[ $LOCAL == true ]]; then
        python -c "import wandb; wandb.login()" \
            || { echo "[ERROR] wandb login failed — set WANDB_API_KEY or use --no_wandb"; exit 1; }
        echo "[wandb] Authenticated  project=${WANDB_PROJECT}${WANDB_ENTITY:+  entity=${WANDB_ENTITY}}"
    fi
fi

echo "============================================================"
echo "EWL — launching experiment sweeps"
echo "Mode:    $( [[ $LOCAL == true ]] && echo 'local' || echo 'SLURM' )"
echo "Sweeps:  ${SELECTED_SWEEPS[*]}"
[[ $LOCAL == true ]] && echo "GPUs:    $GPUS"
[[ $NO_WANDB == true ]]  && echo "Wandb:   DISABLED (--no_wandb)" \
                         || echo "Wandb:   project=${WANDB_PROJECT}${WANDB_ENTITY:+  entity=${WANDB_ENTITY}}"
[[ -n "$EPOCHS" ]]       && echo "Epochs:  $EPOCHS  (smoke-test override)"
echo "============================================================"
echo ""

# ── Helper: run one sweep locally ────────────────────────────────────────────
run_local() {
    local name="$1"
    local script="$2"
    shift 2
    echo "[LOCAL] Starting: $name"
    # shellcheck disable=SC2086
    python "$script" $GPU_ARGS $EXTRA_ARGS "$@"
    echo "[LOCAL] Finished: $name"
    echo ""
}

# ── Helper: submit one SLURM job ─────────────────────────────────────────────
submit_slurm() {
    local name="$1"
    local slurm_file="$2"
    shift 2

    if [[ ! -f "$slurm_file" ]]; then
        echo "[WARN] SLURM script not found: $slurm_file — skipping $name"
        return
    fi

    # Patch --no_wandb / --epochs into the SLURM script via sbatch --export
    local extra_env=""
    [[ "$NO_WANDB" == true ]] && extra_env="${extra_env}EWL_NO_WANDB=1,"
    [[ -n "$EPOCHS" ]]        && extra_env="${extra_env}EWL_EPOCHS=$EPOCHS,"

    local job_id
    if [[ -n "$extra_env" ]]; then
        job_id=$(sbatch --export="ALL,${extra_env%,}" "$slurm_file" | awk '{print $NF}')
    else
        job_id=$(sbatch "$slurm_file" | awk '{print $NF}')
    fi
    echo "[SLURM] Submitted $name → job $job_id"
    echo "        Log: logs/slurm/${name}_%j.out (replace %j with $job_id)"
}

# ── Dispatch each requested sweep ─────────────────────────────────────────────
declare -A SUBMITTED_IDS

for sweep in "${SELECTED_SWEEPS[@]}"; do
    case "$sweep" in

        all_datasets)
            if [[ $LOCAL == true ]]; then
                run_local "sweep_all_datasets" \
                    scripts/sweep_all_datasets.py
            else
                submit_slurm "sweep_all_datasets" \
                    "$SLURM_DIR/sweep_all_datasets.slurm"
            fi
            ;;

        noise_all_datasets)
            if [[ $LOCAL == true ]]; then
                run_local "sweep_noise_all_datasets" \
                    scripts/sweep_noise_all_datasets.py
            else
                submit_slurm "sweep_noise_all_datasets" \
                    "$SLURM_DIR/sweep_noise_all_datasets.slurm"
            fi
            ;;

        aircraft_ablations)
            if [[ $LOCAL == true ]]; then
                run_local "sweep_aircraft_ablations" \
                    scripts/sweep_aircraft_ablations.py
            else
                submit_slurm "sweep_aircraft_ablations" \
                    "$SLURM_DIR/sweep_aircraft_ablations.slurm"
            fi
            ;;

        noise_imbalance)
            if [[ $LOCAL == true ]]; then
                run_local "sweep_noise_imbalance" \
                    scripts/sweep_noise_imbalance.py
            else
                submit_slurm "sweep_noise_imbalance" \
                    "$SLURM_DIR/sweep_noise_imbalance.slurm"
            fi
            ;;

        *)
            echo "[WARN] Unknown sweep name: '$sweep' — skipping."
            echo "       Valid names: ${ALL_SWEEPS[*]}"
            ;;
    esac
done

echo ""
echo "============================================================"
echo "All requested sweeps launched."
if [[ $LOCAL == false ]]; then
    echo "Monitor with:  squeue -u \$USER"
    echo "Live log:      tail -f logs/slurm/<sweep>_<jobid>.out"
fi
echo "Results under: outputs/vision/"
echo "============================================================"
