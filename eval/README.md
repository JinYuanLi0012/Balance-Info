## Evaluation

### Entry points
- Step-level PRM evaluation (VisualProcessBench-style): `eval/prm/evaluate_visualprocessbench_prm.py`
- Best-of-N PRM scoring on rollout annotations:
  - `eval/prm/evaluate_k12_prm.py`
  - `eval/prm/evaluate_mathvista_prm.py`
  - `eval/prm/evaluate_mathverse_prm.py`
  - `eval/prm/evaluate_mathvision_prm.py`
  - `eval/prm/evaluate_olympiadbench_prm.py`

### Inputs (high-level)
- VisualProcessBench-style: JSONL annotations + images
- Best-of-N: rollout annotation JSON (containing question, images, and `solutions_splits`)

### Main parameters

Common:
- `--checkpoint`: model checkpoint path
- `--out-dir`: output directory for results
- `--num-workers`: dataloader workers
- `--seed`: random seed
- `--load-in-8bit`, `--load-in-4bit`: optional quantized loading
- `--auto`: optional device mapping behavior (see model loader)

VisualProcessBench step-level evaluator:
- `--annotation`: JSONL annotation path
- `--image-root`: image root directory
- `--threshold`: score threshold for step classification
- `--dynamic`, `--max-num`, `--grid-max-cols`: image tiling / dynamic patch settings
- `--mini-batch-size`: mini-batch size for PRM scoring

Best-of-N evaluators:
- `--datasets`: dataset key(s) (comma-separated)
- `--root`, `--annotation`: optional overrides for image root and annotation JSON path (available in some scripts)

### Outputs
- Timestamped JSON results under `--out-dir`
- Optional post-processing utilities: `eval/prm/extract_calculate.py`

