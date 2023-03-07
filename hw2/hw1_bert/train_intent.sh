python3 train_intent.py \
    --train_file data/intent/train_organized.json \
    --valid_file data/intent/eval_organized.json \
    --model_name bert-base-uncased \
    --train_batch_size 32 \
    --lr 3e-5 \
    --weight_decay 1e-2;