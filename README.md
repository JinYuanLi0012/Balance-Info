## PRM Training (VisualPRM400K-style)

### Contents
- `src/internvl/`: training code
- `configs/zero_stage3_config.json`: DeepSpeed configuration
- `data/meta_visualprm400k_soft.example.json`: dataset meta template
- `requirements.txt`: dependency list
- `LICENSE.txt`: license

### Entry point
- `src/internvl/train/internvl_chat_finetune.py`

### Configuration (high-level)
The entry point supports arguments such as:
- `--model_name_or_path`: pretrained model path
- `--meta_path`: dataset meta JSON path
- `--output_dir`: output directory
- `--deepspeed`: DeepSpeed config JSON path
- `--conv_style`: template style (e.g., `internvl2_5`)
- image / sequence settings: `--force_image_size`, `--max_seq_length`, `--dynamic_image_size`, `--use_thumbnail`, etc.

### Dataset meta schema
The training code reads a **meta JSON** mapping dataset names to:
- `root`: image root directory
- `annotation`: a JSONL file path
- `data_augment`: boolean
- `repeat_time`: integer
- `length`: sample count (informational; some codepaths may not require it)

See: `data/meta_visualprm400k_soft.example.json`.

### Annotation JSONL schema (expected fields)

The dataset loader expects each JSONL line to contain fields like:
- `conversations`: conversation list (with `from`/`value`)
- `image`: image path (relative to `root`) or list of images
- (optional) `length`: token length hint for bucketing

Exact preprocessing depends on `--conv_style internvl2_5` and the dataset's annotation format.

