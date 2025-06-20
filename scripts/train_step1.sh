#!/bin/bash
MODEL_QWEN25_PATH=
model_base="qwen2.5_7b"
template=qwen

lr=2e-5
num_train_epochs=2

declare -A dataset_dict
dataset_dict=(
    ["data_9task"]=""
)

model_path_sft=

export NCCL_DEBUG=WARN

declare -A model_dict
for data_name in "${!dataset_dict[@]}"; do
    model_dict[$data_name]="${model_path_sft}/${data_name}_${model_base}_tem${template}_${lr}_ep${num_train_epochs}"
    python add_datasets.py ${data_name} ${dataset_dict[$data_name]}
done

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 29700
  "

cd LLaMA-Factory

for data_name in "${!dataset_dict[@]}"; do
    data_path=${data_name}
    model_path=${model_dict[$data_name]}
    echo "$data_path::$model_path"

    torchrun $DISTRIBUTED_ARGS src/train.py \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --stage sft \
        --do_train \
        --use_fast_tokenizer \
        --flash_attn auto \
        --model_name_or_path $MODEL_QWEN25_PATH \
        --dataset $data_path \
        --template $template \
        --finetuning_type full \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target q_proj,v_proj \
        --output_dir $model_path \
        --overwrite_cache \
        --overwrite_output_dir \
        --warmup_ratio 0 \
        --weight_decay 0. \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --ddp_timeout 9000 \
        --learning_rate $lr \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --cutoff_len 15000 \
        --save_strategy epoch \
        --save_steps 1 \
        --save_total_limit 5 \
        --save_only_model True \
        --plot_loss \
        --num_train_epochs $num_train_epochs \
        --bf16 True 
done
