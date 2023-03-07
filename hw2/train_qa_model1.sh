python run_qa_no_trainer.py \
  --model_name_or_path=IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese \
  --cat-n=2 --cat-op=sum \
  --train_file=work/train_squad.json \
  --validation_file=work/valid_squad.json \
  --max_seq_length=512 \
  --per_device_train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=3e-5 \
  --num_train_epochs=3 \
  --checkpointing_steps=2048 \
  --max_answer_length=40 \
  --seed=123 \
  --weight_decay=0.01 \
  --lr_scheduler_type=cosine \
  --output_dir work/qa1