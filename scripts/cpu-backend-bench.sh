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
  $(basename "$0") --model /absolute/path/to/model.gguf [--build-dir /abs/build] [--threads N] [--prompt-tokens N] [--gen-tokens N] [--matrix P:G[,P:G...]] [--op-profile] [--cpu-util] [--perf] [--callgrind]

Examples:
  $(basename "$0") --model /models/Qwen3-8B-Q4_K_M.gguf
  $(basename "$0") --model /models/Qwen3-8B-Q4_K_M.gguf --threads 16 --perf
  $(basename "$0") --model /models/Qwen3-8B-Q4_K_M.gguf --matrix 512:0,0:128
  $(basename "$0") --model /models/Qwen3-8B-Q4_K_M.gguf --op-profile --cpu-util
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
matrix="512:0,0:128"
prompt_tokens_set=0
gen_tokens_set=0
matrix_set=0
run_op_profile=0
run_cpu_util=0
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
            prompt_tokens_set=1
            shift 2
            ;;
        --gen-tokens)
            gen_tokens="${2:-}"
            gen_tokens_set=1
            shift 2
            ;;
        --matrix)
            matrix="${2:-}"
            matrix_set=1
            shift 2
            ;;
        --op-profile)
            run_op_profile=1
            shift
            ;;
        --cpu-util)
            run_cpu_util=1
            shift
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

if [[ "$matrix_set" -eq 0 && ( "$prompt_tokens_set" -eq 1 || "$gen_tokens_set" -eq 1 ) ]]; then
    matrix="${prompt_tokens}:${gen_tokens}"
fi

echo "Repository: $repo_root"
echo "Output dir: $out_dir"
echo "CPU cores: $cpu_cores"
echo "Threads: $threads"
echo "Model: $model"
echo "Matrix: $matrix"

extract_tps() {
    local log_file="$1"
    python3 - "$log_file" <<'PY'
import json,sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    data = json.loads(p.read_text())
except Exception:
    print("NA")
    raise SystemExit(0)
if isinstance(data, dict):
    rows = data.get("results") or data.get("benchmarks") or data.get("data") or []
else:
    rows = data
if not isinstance(rows, list):
    print("NA")
    raise SystemExit(0)
vals = []
for row in rows:
    if not isinstance(row, dict):
        continue
    for k in ("avg_ts","tps","tokens_per_second","tok_per_s","tok/s"):
        v = row.get(k)
        if isinstance(v, (int,float)):
            vals.append(float(v))
            break
if not vals:
    print("NA")
else:
    print(f"{max(vals):.3f}")
PY
}

summary_tsv="$out_dir/summary.tsv"
{
    echo -e "scenario\tprompt_tokens\tgen_tokens\ttok_per_s\tcpu_percent"
} > "$summary_tsv"

IFS=',' read -r -a matrix_entries <<< "$matrix"
if [[ "${#matrix_entries[@]}" -eq 0 ]]; then
    echo "Invalid --matrix value: $matrix" >&2
    exit 1
fi

first_entry="${matrix_entries[0]}"
first_pp="${first_entry%%:*}"
first_tg="${first_entry##*:}"
if [[ -z "$first_pp" || -z "$first_tg" || "$first_pp" == "$first_entry" || ! "$first_pp" =~ ^[0-9]+$ || ! "$first_tg" =~ ^[0-9]+$ ]]; then
    echo "Invalid first matrix entry '$first_entry' (expected P:G)." >&2
    exit 1
fi

primary_run_args=(
    -m "$model"
    -ngl 0
    -t "$threads"
    -p "$first_pp"
    -n "$first_tg"
)

profiled=0

for entry in "${matrix_entries[@]}"; do
    pp="${entry%%:*}"
    tg="${entry##*:}"
    if [[ -z "$pp" || -z "$tg" || "$pp" == "$entry" ]]; then
        echo "Invalid matrix entry '$entry' (expected P:G)." >&2
        exit 1
    fi
    if [[ ! "$pp" =~ ^[0-9]+$ || ! "$tg" =~ ^[0-9]+$ ]]; then
        echo "Invalid matrix entry '$entry' (P and G must be integers)." >&2
        exit 1
    fi

    scenario="pp${pp}_tg${tg}"
    scenario_dir="$out_dir/$scenario"
    mkdir -p "$scenario_dir"

    run_args=(
        -m "$model"
        -ngl 0
        -t "$threads"
        -p "$pp"
        -n "$tg"
        -o json
    )

    cpu_percent="NA"
    if [[ "$run_cpu_util" -eq 1 && -x /usr/bin/time ]]; then
        /usr/bin/time -v -o "$scenario_dir/time.log" "$bench_bin" "${run_args[@]}" > "$scenario_dir/llama-bench.json" 2> "$scenario_dir/llama-bench.stderr.log"
        cpu_percent="$(awk -F': *' '/Percent of CPU this job got/ {gsub(/%/,"",$2); print $2}' "$scenario_dir/time.log" | tail -n 1)"
        if [[ -z "$cpu_percent" ]]; then
            cpu_percent="NA"
        fi
    else
        "$bench_bin" "${run_args[@]}" > "$scenario_dir/llama-bench.json" 2> "$scenario_dir/llama-bench.stderr.log"
    fi

    tok_per_s="$(extract_tps "$scenario_dir/llama-bench.json")"
    printf "%s\t%s\t%s\t%s\t%s\n" "$scenario" "$pp" "$tg" "$tok_per_s" "$cpu_percent" >> "$summary_tsv"

    if [[ "$run_op_profile" -eq 1 && "$profiled" -eq 0 && "$tg" -gt 0 ]]; then
        if command -v perf >/dev/null 2>&1; then
            perf record -o "$scenario_dir/perf.data" -g -- "$bench_bin" "${run_args[@]}" > "$scenario_dir/perf.stdout.log" 2> "$scenario_dir/perf.stderr.log"
            perf report -i "$scenario_dir/perf.data" --stdio > "$scenario_dir/perf-report.txt"
            {
                echo -e "symbol\tshare_percent"
                awk '
                    /ggml_compute_forward_mul_mat/ { print "mul_mat\t" $1; next }
                    /ggml_compute_forward_flash_attn_ext_f16/ { print "flash_attn\t" $1; next }
                    /ggml_compute_forward_rope_f32|ggml_compute_forward_rope_f16/ { print "rope\t" $1; next }
                    /ggml_compute_forward_norm_f32|ggml_compute_forward_fused_norm_f32/ { print "norm\t" $1; next }
                    /ggml_compute_forward_rms_norm_f32/ { print "rms_norm\t" $1; next }
                    /ggml_backend_sched_split_graph/ { print "scheduler_split\t" $1; next }
                ' "$scenario_dir/perf-report.txt" | sed 's/%//g' | awk -F'\t' '!seen[$1]++'
            } > "$scenario_dir/op-hotspots.tsv"
        else
            echo "perf is not installed; skipping --op-profile output." >&2
        fi
        profiled=1
    fi
done

if [[ "$run_perf" -eq 1 ]]; then
    if ! command -v perf >/dev/null 2>&1; then
        echo "perf requested but not installed." >&2
        exit 1
    fi
    perf record -o "$out_dir/perf.data" -g -- "$bench_bin" "${primary_run_args[@]}" 2>&1 | tee "$out_dir/perf-record.log"
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
    valgrind --tool=callgrind --callgrind-out-file="$out_dir/callgrind.out" "$bench_bin" "${primary_run_args[@]}" 2>&1 | tee "$out_dir/callgrind.log"
    if [[ ! -f "$out_dir/callgrind.out" ]]; then
        echo "callgrind did not produce $out_dir/callgrind.out" >&2
        exit 1
    fi
fi

echo "Done. Results: $out_dir"
echo "Baseline summary: $summary_tsv"
