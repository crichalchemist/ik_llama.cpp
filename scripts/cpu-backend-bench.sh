#!/usr/bin/env bash
set -euo pipefail

this=$(realpath "$0"); readonly this
repo_root=$(dirname "$(dirname "$this")"); readonly repo_root

usage() {
    cat <<EOF
Usage:
  $(basename "$0") --model /absolute/path/to/model.gguf [--build-dir /abs/build] [--threads N] [--prompt-tokens N] [--gen-tokens N] [--perf] [--callgrind]

Examples:
  $(basename "$0") --model /models/Qwen3-8B-Q4_K_M.gguf
  $(basename "$0") --model /models/Qwen3-8B-Q4_K_M.gguf --threads 16 --perf
  $(basename "$0") --model /models/Qwen3-8B-Q4_K_M.gguf --callgrind
EOF
}

model=""
build_dir="$repo_root/build"
threads=""
prompt_tokens=512
gen_tokens=128
run_perf=0
run_callgrind=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model="${2:-}"
            shift 2
            ;;
        --build-dir)
            build_dir="${2:-}"
            shift 2
            ;;
        --threads)
            threads="${2:-}"
            shift 2
            ;;
        --prompt-tokens)
            prompt_tokens="${2:-}"
            shift 2
            ;;
        --gen-tokens)
            gen_tokens="${2:-}"
            shift 2
            ;;
        --perf)
            run_perf=1
            shift
            ;;
        --callgrind)
            run_callgrind=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$model" ]]; then
    echo "--model is required." >&2
    usage >&2
    exit 1
fi

if [[ "$model" != /* ]]; then
    echo "--model must be an absolute path (got: $model)." >&2
    exit 1
fi

if [[ ! -f "$model" ]]; then
    echo "Model file not found: $model" >&2
    exit 1
fi

bench_bin="$build_dir/bin/llama-bench"
if [[ ! -x "$bench_bin" ]]; then
    echo "llama-bench not found at: $bench_bin" >&2
    echo "Build first: cmake -B $build_dir -DGGML_NATIVE=ON && cmake --build $build_dir --config Release -j$(nproc)" >&2
    exit 1
fi

out_dir="$repo_root/tmp/cpu-bench/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out_dir"

cpu_cores="$(nproc)"
if [[ -z "$threads" ]]; then
    threads="$cpu_cores"
fi

bench_args=(
    -m "$model"
    -ngl 0
    -t "$threads"
    -p "$prompt_tokens"
    -n "$gen_tokens"
)

echo "Repository: $repo_root"
echo "Output dir: $out_dir"
echo "CPU cores: $cpu_cores"
echo "Threads: $threads"
echo "Model: $model"

"$bench_bin" "${bench_args[@]}" | tee "$out_dir/llama-bench.log"

if [[ "$run_perf" -eq 1 ]]; then
    if ! command -v perf >/dev/null 2>&1; then
        echo "perf requested but not installed." >&2
        exit 1
    fi
    perf record -g -- "$bench_bin" "${bench_args[@]}" >/dev/null 2>&1
    perf report --stdio > "$out_dir/perf-report.txt"
    mv perf.data "$out_dir/perf.data"
fi

if [[ "$run_callgrind" -eq 1 ]]; then
    if ! command -v valgrind >/dev/null 2>&1; then
        echo "callgrind requested but valgrind is not installed." >&2
        exit 1
    fi
    valgrind --tool=callgrind --callgrind-out-file="$out_dir/callgrind.out" "$bench_bin" "${bench_args[@]}" >/dev/null 2>&1
fi

echo "Done."
