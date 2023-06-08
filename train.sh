data="nq320k"
docid="bert_10_100"
docid_strategy="expand_vocab" # expand_vocab / tokenize / token

mkdir -p data/$data/run

python -m torch.distributed.launch --nproc_per_node 2 --master_port=12345 src/run.py \
    --task train \
    --docid_path data/$data/kmeans/$docid.tsv \
    --train_path data/$data/train_data/${docid}_train.json \
    --valid_path data/$data/train_data/${docid}_valid.json \
    --output_dir data/$data/run/${docid}_${docid_strategy} \
    --overwrite_output_dir True \
    --logging_dir data/$data/run/${docid}_${docid_strategy}/log \
    --model_name_or_path t5-base \
    --docid_strategy $docid_strategy \
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
