# VABench T2AV (GPU, eval-only)

Use an external pinned VABench checkout and one Conda Python 3.11 scoring
environment for all 15 dimensions. Keep the AISBench main environment separate.

From the AISBench source root, using the scoring Conda's interpreter:

```bash
python3 -m ais_bench.benchmark.vabench.environment install \
  --source-root /workspace/code/VABench-main \
  --cache-dir /workspace/vabench-cache \
  --work-dir /workspace/vabench-env-audit
```

This uses `requirements/datasets/vabench*.txt`, pinned Git revisions, and a
metadata-only NISQA wheel patch. Download/provide model weights separately.
Use `plan` instead of `install` for a read-only command preview.
On Ubuntu/Debian containers, root may add `--install-system-deps` to prepare
FFmpeg, git, libsndfile and the C/C++ build tools through apt before pip runs.
Without that explicit option, the tool does not invoke the system package manager.

Then edit the paths and `BACKEND_PYTHON` in `eval_vabench.py` (Mini10/778) or
`eval_vabench_smoke.py` (one CLAP sample). Follow
`docs/source_zh_cn/get_started/vabench_gpu_environment.md` for installation and
`docs/source_zh_cn/get_started/vabench_t2av.md` for media preparation/scoring.

Environment checks and mock-scoring tests do not establish real GPU score
alignment. The evaluator exports all 15 native means and fixed 0–100 display scores,
coverage, and a complete/preview/unavailable equal-weight total for the selected
samples. Scene-level error attribution and checkpoint resume remain separate work.
