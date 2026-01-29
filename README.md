### Files
- `src/internvl/`: training code used in our experiments
- `configs/zero_stage3_config.json`: DeepSpeed config used as a reference
- `data/meta_visualprm400k.json`: meta file describing datasets
- `requirements.txt`: dependencies

### Main entry
- `src/internvl/train/internvl_chat_finetune.py`

### What the script needs
The training script expects:
- a pretrained checkpoint path (`--model_name_or_path`)
- a meta JSON that points to your local data (`--meta_path`)
- an output directory (`--output_dir`)
- optionally a DeepSpeed JSON (`--deepspeed`)


### Example training command
One way to call the script is:

```bash
python -m torch.distributed.run \
  --nproc_per_node=4 src/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path /path/to/checkpoint \
  --meta_path data/meta_visualprm400k.json \
  --output_dir /path/to/output_dir \
  --deepspeed configs/zero_stage3_config.json \
  --conv_style internvl2_5
```

### Dataset meta JSON
The meta file maps dataset names to a small config. See `data/meta_visualprm400k.json`.

A small example:
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

### Evaluation
Evaluation scripts are under `eval/`. See `eval/README.md` for more details.

One example call:

```bash
python eval/prm/evaluate_visualprocessbench_prm.py \
  --checkpoint /path/to/checkpoint \
  --annotation /path/to/annotations.jsonl \
  --image-root /path/to/images \
  --out-dir /path/to/output_dir
```

