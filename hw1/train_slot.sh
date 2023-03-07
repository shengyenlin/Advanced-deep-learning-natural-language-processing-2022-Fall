#Train
python3 train_slot.py --data_dir data/slot/ --ckpt_dir ckpt/slot/ --device cuda --batch_size 16;
#Evaluation 
python3 test_slot.py --test_file data/slot/eval.json --ckpt_path ckpt/slot/best.ckpt;