#!/usr/bin/env bash
set -euo pipefail

script_path="$0"
if [[ "$script_path" != /* ]]; then
    script_path="$PWD/$script_path"
fi
this="$(cd "$(dirname "$script_path")" && pwd -P)/$(basename "$script_path")"; readonly this
repo_root="$(cd "$(dirname "$(dirname "$this")")" && pwd -P)"; readonly repo_root

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

get_cpu_cores() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
        return
    fi
    if command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
        return
    fi
    echo 1
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
    echo "Build first: cmake -B $build_dir -DGGML_NATIVE=ON && cmake --build $build_dir --config Release -j" >&2
    exit 1
fi

out_parent="$repo_root/tmp/cpu-bench"
mkdir -p "$out_parent"
out_dir="$(mktemp -d "$out_parent/run-XXXXXX")"
if [[ -z "$out_dir" || ! -d "$out_dir" ]]; then
    echo "Failed to create output directory in $out_parent" >&2
    exit 1
fi

cpu_cores="$(get_cpu_cores)"
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
    perf record -o "$out_dir/perf.data" -g -- "$bench_bin" "${bench_args[@]}" 2>&1 | tee "$out_dir/perf-record.log"
    if [[ ! -f "$out_dir/perf.data" ]]; then
        echo "perf did not produce $out_dir/perf.data" >&2
        exit 1
    fi
    perf report -i "$out_dir/perf.data" --stdio > "$out_dir/perf-report.txt"
fi

if [[ "$run_callgrind" -eq 1 ]]; then
    if ! command -v valgrind >/dev/null 2>&1; then
        echo "callgrind requested but valgrind is not installed." >&2
        exit 1
    fi
    valgrind --tool=callgrind --callgrind-out-file="$out_dir/callgrind.out" "$bench_bin" "${bench_args[@]}" 2>&1 | tee "$out_dir/callgrind.log"
    if [[ ! -f "$out_dir/callgrind.out" ]]; then
        echo "callgrind did not produce $out_dir/callgrind.out" >&2
        exit 1
    fi
fi

echo "Done. Results: $out_dir"
