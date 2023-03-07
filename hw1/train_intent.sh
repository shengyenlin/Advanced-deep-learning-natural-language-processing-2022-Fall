#Train
python3 train_intent.py --data_dir data/intent/ --ckpt_dir ckpt/intent/ --device cuda;
#Evaluation 
python3 test_intent.py --test_file data/intent/eval.json --ckpt_path ckpt/intent/best.ckpt;