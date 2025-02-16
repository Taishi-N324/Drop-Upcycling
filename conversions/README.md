# Conversions

## From scratch

```bash
python from_scratch.py \
    --config_path /path/to/target_model_config.json \
    --save_path /path/to/output_model_directory \
    --seed 1234
```

## Branch-Train-Mix

```bash
python branch_train_mix.py \
    --source_model_paths /path/to/source/model /path/to/expert_model_1 /path/to/expert_model_2 /path/to/expert_model_3 \
    --target_config_path /path/to/target_model_config.json \
    --output_path /path/to/output_model_directory \
    --num_experts 8 \
    --num_layers 12 \
    --seed 1234 \
    --init_method torch_rand_002
```

## Random Noise Upcycling

```bash
python noise_upcycling.py \
    --ffn_init_ratio 0.5 \
    --source_model_path /path/to/source/model \
    --target_config_path /path/to/target_model_config.json \
    --output_path /path/to/output_model_directory \
    --num_experts 8 \
    --num_layers 12 \
    --seed 1234 \
    --init_method torch_rand_002
```

## Na¨ıve Upcycling

```bash
python naive_upcycling.py \
    --source_model_path /path/to/source/model \
    --target_config_path /path/to/target_model_config.json \
    --output_path /path/to/output_model_directory \
    --num_experts 8 \
    --num_layers 12 \
    --seed 1234 \
    --init_method torch_rand_002
```

## Drop-Upcycling

```bash
python drop_upcycling.py \
    --ffn_init_ratio 0.5 \
    --source_model_path /path/to/source/model \
    --target_config_path /path/to/target_model_config.json \
    --output_path /path/to/output_model_directory \
    --num_experts 8 \
    --num_layers 12 \
    --seed 1234 \
    --init_method torch_rand_002
```
