data="nq320k"
docid="bert_10_100"
model_path="data/nq320k/run/model"

mkdir -p data/$data/run

CUDA_VISIBLE_DEVICES=0 python src/run.py \
    --task inference \
    --output_dir data/$data/run/model \
    --logging_dir data/$data/run/log \
    --kmeans_path data/$data/kmeans/$docid.tsv \
    --test_path data/$data/train_data/dev.json \
    --output_run_path data/$data/run/$docid.json \
    --model_name_or_path $model_path \
    --max_length 128 \
    --dataloader_num_workers 8 \
    --per_device_eval_batch_size 4 \
    --num_beams 100