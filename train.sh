data="nq320k"
docid="bert_10_100"

mkdir -p data/$data/run
mkdir -p data/$data/run/model

python -m torch.distributed.launch --nproc_per_node 2 --master_port=12345 src/run.py \
    --task train \
    --kmeans_path data/$data/kmeans/$docid.tsv \
    --train_path data/$data/train_data/train.json \
    --valid_path data/$data/train_data/valid.json \
    --output_dir data/$data/run/model \
    --overwrite_output_dir True \
    --logging_dir data/$data/run/log \
    --model_name_or_path t5-base \
    --max_steps  120000 \
    --max_length 128 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 8 \
    --valid_callback True \
    --num_beams 10 \
    --learning_rate 1e-3 \
    --dataloader_drop_last False \
    --warmup_ratio 0 \
    --dataloader_num_workers 8 \
    --max_grad_norm 5.0 \
    --save_strategy steps \
    --save_steps  20000 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy no \
    --report_to wandb
