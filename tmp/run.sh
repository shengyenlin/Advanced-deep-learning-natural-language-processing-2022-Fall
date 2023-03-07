python3.8 run_summarization.py \
  --test_file ${1} \
  --output_dir ./predict_dir \
  --output_file ${2} \
  --text_column maintext \
  --model_name_or_path ./model \
  --do_predict \
  --per_device_eval_batch_size 4 \
  --num_beams 4 \
  --predict_with_generate;
