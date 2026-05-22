# CPU backend contribution workflow

This workflow turns the CPU-backend guidance in the main README into a repeatable process for contributors who want to improve AVX2/AVX-512 and ARM_NEON/SVE performance and feature coverage.

## 1) Baseline the current CPU backend state

Use these files as the primary map:

- Core CPU execution path:
  - `ggml/src/ggml.c`
- ARM-specific kernels and helpers:
  - `ggml/src/ggml-aarch64.c`
- Quantized CPU matmul and CPU ops (major hot path):
  - `ggml/src/iqk/iqk_mul_mat.cpp`
  - `ggml/src/iqk/iqk_gemm_*.cpp`
  - `ggml/src/iqk/iqk_cpu_ops.cpp`
  - `ggml/src/iqk/iqk_config.h`

Support policy anchor:

- `README.md` (backend support policy note near the top of the file)

## 2) Define explicit optimization targets

Pick concrete targets before changing code:

- **Performance targets**: measurable prompt/gen throughput gains for specific model + quant + context settings.
- **Feature targets**: missing or partial SIMD paths and missing kernel variants.
- **Compatibility targets**: better use of AVX-512/VNNI/BF16 on x86_64 or DOTPROD/SVE on ARM.

Split every target by architecture:

- **x86_64**: AVX2 baseline, AVX-512/VNNI optional fast paths.
- **ARM64**: NEON baseline, SVE (if available) where it materially helps.

## 3) Measure before changing code

Use reproducible CPU-only runs:

```bash
cd <repo-root>
./scripts/cpu-backend-bench.sh \
  --model /absolute/path/to/model.gguf \
  --matrix 512:0,0:128 \
  --cpu-util \
  --op-profile
```

The helper script can:

- Run `llama-bench` in CPU-only mode over a fixed prefill/decode matrix.
- Write a reproducible `summary.tsv` with prompt/decode scenario, tok/s, and CPU utilization.
- Optionally collect `perf` samples and extract hotspot shares for:
  - `ggml_compute_forward_mul_mat*`
  - `ggml_compute_forward_flash_attn_ext_f16`
  - `ggml_compute_forward_rope_f32/f16`
  - `ggml_compute_forward_norm_f32` / `ggml_compute_forward_fused_norm_f32`
  - `ggml_compute_forward_rms_norm_f32`
  - `ggml_backend_sched_split_graph`
- Optionally collect valgrind callgrind output.

## 4) Implement CPU kernel improvements

Focus on hot kernels first (typically IQK GEMM and related quantized paths):

- Improve SIMD lane utilization.
- Improve cache locality and memory access patterns.
- Avoid regressions in existing AVX2/NEON kernels.
- Add architecture-guarded code paths only when a measurable gain exists.

Keep each PR narrow (one kernel family or one optimization class per PR).

## 5) Validate correctness and regressions

For each change:

1. Re-run your benchmark baseline and report deltas.
2. Rebuild and run project tests.
3. Verify CPU output correctness against known-good runs (including CUDA when relevant).

Suggested commands:

```bash
cd <repo-root>
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j
cd build && ctest --output-on-failure
```

## 6) Submit maintainer-friendly contributions

In each PR:

- State the exact bottleneck and why this change was chosen.
- Include benchmark setup and before/after numbers.
- Include correctness checks and test commands.
- Keep scope small and reviewable.

## 7) Engage early for larger design changes

Before major refactors or new backend abstractions:

- Open an issue/discussion with design goals and trade-offs.
- Confirm direction with maintainers before deeper implementation.

This reduces review overhead and avoids rework.
