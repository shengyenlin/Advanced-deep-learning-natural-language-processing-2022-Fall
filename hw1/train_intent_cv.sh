#10-Fold cv for finetuning hyperparmeters
for f in $(seq 0 9); do
    python3 train_intent.py --data_dir cache/cv$f --ckpt_dir cache/cv$f --device cuda
done

#Evaluation on 10-Fold
for x in cache/slot/cv?/best*; do ls -l "$x"; 
    python3 test_intent.py --test_file data/intent/eval.json --ckpt_path $x; 
done

