#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_NAME="${0##*/}"
readonly IDLE_MEMORY_LIMIT_MIB=512
readonly IDLE_UTILIZATION_LIMIT_PERCENT=5
readonly ALLOWED_GPU_IDS=(0 2)

usage() {
    cat <<EOF
Usage:
  ${SCRIPT_NAME} --list
  ${SCRIPT_NAME} <sglang|sdc> <auto|0|2> -- <command> [args...]

Examples:
  ${SCRIPT_NAME} --list
  ${SCRIPT_NAME} sglang 0 -- python -m sglang.launch_server --help
  ${SCRIPT_NAME} sdc auto -- python speculative/speculative_gsm8k_FI.py --help

The selected physical A100 is exposed to the command as cuda:0.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_nvidia_smi() {
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available."
}

gpu_snapshot() {
    local gpu_id="$1"
    nvidia-smi -i "$gpu_id" \
        --query-gpu=index,name,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
}

gpu_has_compute_process() {
    local gpu_id="$1"
    nvidia-smi -i "$gpu_id" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ { found=1 } END { exit !found }'
}

gpu_is_idle() {
    local gpu_id="$1"
    local snapshot index name memory_used utilization

    snapshot="$(gpu_snapshot "$gpu_id")" || return 1
    IFS=',' read -r index name memory_used utilization <<<"$snapshot"
    name="${name# }"
    memory_used="${memory_used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"

    [[ "$name" == *A100* ]] || return 1
    gpu_has_compute_process "$gpu_id" && return 1
    (( memory_used <= IDLE_MEMORY_LIMIT_MIB )) || return 1
    (( utilization <= IDLE_UTILIZATION_LIMIT_PERCENT )) || return 1
}

list_gpus() {
    local gpu_id snapshot state

    printf '%-8s %-28s %-14s %-12s %s\n' \
        "GPU" "NAME" "MEMORY(MiB)" "UTIL(%)" "STATE"
    for gpu_id in "${ALLOWED_GPU_IDS[@]}"; do
        snapshot="$(gpu_snapshot "$gpu_id")" || {
            printf '%-8s %s\n' "$gpu_id" "unavailable"
            continue
        }
        state="busy"
        gpu_is_idle "$gpu_id" && state="idle"
        IFS=',' read -r index name memory_used utilization <<<"$snapshot"
        printf '%-8s %-28s %-14s %-12s %s\n' \
            "${index//[[:space:]]/}" "${name# }" \
            "${memory_used//[[:space:]]/}" "${utilization//[[:space:]]/}" "$state"
    done
}

is_allowed_gpu() {
    local requested="$1" gpu_id
    for gpu_id in "${ALLOWED_GPU_IDS[@]}"; do
        [[ "$requested" == "$gpu_id" ]] && return 0
    done
    return 1
}

select_gpu() {
    local requested="$1" gpu_id

    if [[ "$requested" != "auto" ]]; then
        is_allowed_gpu "$requested" || die "GPU must be auto, 0, or 2."
        gpu_is_idle "$requested" || die "GPU $requested is not idle. Run ${SCRIPT_NAME} --list."
        printf '%s\n' "$requested"
        return
    fi

    for gpu_id in "${ALLOWED_GPU_IDS[@]}"; do
        if gpu_is_idle "$gpu_id"; then
            printf '%s\n' "$gpu_id"
            return
        fi
    done
    die "No idle A100 is available."
}

resolve_environment() {
    case "$1" in
        sglang) printf '%s\n' "/data6t/hpk/sglang_env" ;;
        sdc) printf '%s\n' "/data6t/hpk/sdc_env" ;;
        *) die "Environment must be sglang or sdc." ;;
    esac
}

main() {
    require_nvidia_smi

    if [[ "${1:-}" == "--list" ]]; then
        [[ "$#" -eq 1 ]] || die "--list does not accept additional arguments."
        list_gpus
        return
    fi

    [[ "$#" -ge 4 ]] || { usage >&2; exit 2; }

    local environment_name="$1"
    local requested_gpu="$2"
    shift 2
    [[ "$1" == "--" ]] || die "Expected -- before the experiment command."
    shift
    [[ "$#" -gt 0 ]] || die "Missing experiment command."

    local environment_path selected_gpu lock_path
    environment_path="$(resolve_environment "$environment_name")"
    [[ -x "$environment_path/bin/python" ]] || die "Invalid environment: $environment_path"

    selected_gpu="$(select_gpu "$requested_gpu")"
    lock_path="/tmp/sdc_experiment_gpu_${selected_gpu}.lock"
    exec 9>"$lock_path"
    flock -n 9 || die "GPU $selected_gpu is reserved by another harness process."

    # Recheck after taking the harness lock to narrow the check-to-launch race.
    gpu_is_idle "$selected_gpu" || die "GPU $selected_gpu became busy before launch."

    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$selected_gpu"
    export VIRTUAL_ENV="$environment_path"
    export PATH="$environment_path/bin:$PATH"
    unset PYTHONHOME

    echo "Environment: $environment_name ($environment_path)"
    echo "Physical GPU: $selected_gpu; experiment-visible device: cuda:0"
    echo "Command: $*"

    exec "$@"
}

main "$@"
