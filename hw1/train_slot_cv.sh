#10-Fold cv for finetuning hyperparmeters
for f in $(seq 0 9); do
  python3 train_slot.py --data_dir cache/slot/cv$f --ckpt_dir cache/slot/cv$f --device cuda --batch_size 16;
done
# smaller batch size helps

#Evaluation on 10-Fold
for x in cache/slot/cv?/best*; do ls -l "$x"; 
    python3 test_slot.py --test_file data/slot/eval.json --ckpt_path $x; 
done

