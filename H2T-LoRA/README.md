# H2T-LoRA Training and Inference
## Prompt Template ##
Please see the template in `templates/H2T-LoRA.json`

You can modify it using your own task description.

## Path ##
`--data_path`: Your data path includes Hypo-Trans pairs.

`--output_dir`: The path to save LoRA weight

## Other Hyperparameters ##

`--base_model`: The foundation model for H2T learning

`--lora_r`: LoRA rank

`--lora_target_modules`: The modules to apply the LoRA update matrices, e.g., ["q_proj", "k_proj", "v_proj", "o_proj"], 

`--val_set_size`: The validation set size from training set.

## Usage ##

Finetune LLaMA-7b with LoRA tuning:
```bash
python finetune.py \
    --base_model 'yahma/llama-7b-hf' \
    --data_path './data/train_wsj.json' \
    --output_dir './wsj' \
    --lora_target_modules='[q_proj,k_proj, v_proj, o_proj]' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --learning_rate 2e-4 \
    --micro_batch_size=64 \
    --batch_size=256 \
    --lora_r=16
```

Then, inference with LLaMA-7b + well-trained LoRA:
```bash
python inference.py \
    --ckpt_path './wsj'
    --test_data_path './data/test_wsj.json'
```

