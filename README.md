### What is in this folder
- `src/internvl/`: training code used in our setup
- `configs/zero_stage3_config.json`: DeepSpeed config used as a reference
- `data/meta_visualprm400k.json`: a *template* for describing datasets
- `requirements.txt`: dependencies
- `LICENSE.txt`: license

### Main entry
- `src/internvl/train/internvl_chat_finetune.py`

### Expected inputs (minimal)
The entry script reads:
- **a pretrained checkpoint path** (`--model_name_or_path`)
- **a meta JSON** that points to your local data (`--meta_path`)
- **an output directory** (`--output_dir`)
- optionally a DeepSpeed JSON (`--deepspeed`)

We use `--conv_style internvl2_5`. If you change the conversation template, you will need to keep the annotation format consistent.

### Dataset meta JSON
The meta file maps dataset names to a small config. See `data/meta_visualprm400k.json`.

Example (illustrative):
```json
{
  "my_dataset": {
    "root": "/path/to/images",
    "annotation": "/path/to/ann.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 12345
  }
}
```

### Annotation JSONL
Each line is a JSON object. The loader expects at least:
- `conversations`: a list of turns with `from` / `value`
- `image`: a relative path (w.r.t. `root`) or a list of image paths
- optional `length`: a token-length hint used for bucketing in some setups

Example (single line):
```json
{"image":"subdir/000001.png","conversations":[{"from":"human","value":"..."} ,{"from":"gpt","value":"..."}]}
```

### Notes / assumptions
- This is a research artifact and assumes you already have model weights and data prepared locally.
- If your JSONL fields differ (naming, nesting, or multi-image structure), adjust the dataset loader accordingly.

