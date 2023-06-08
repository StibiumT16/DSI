data="nq320k"
docid="bert_10_100"
docid_strategy="expand_vocab"

model_path="data/nq320k/run/${docid}_${docid_strategy}"

CUDA_VISIBLE_DEVICES=0 python src/run.py \
    --task inference \
    --output_dir data/$data/run/model \
    --logging_dir data/$data/run/log \
    --docid_path data/$data/kmeans/$docid.tsv \
    --test_path data/$data/train_data/${docid}_dev.json \
    --output_run_path data/$data/run/$docid.$docid_strategy.json \
    --docid_strategy $docid_strategy \
    --model_name_or_path $model_path \
    --backbone_model t5-base \
    --max_length 128 \
    --dataloader_num_workers 8 \
    --per_device_eval_batch_size 4 \
    --num_beams 100