# HW3

Refer to `report.pdf` for model performance, experiement results, observation and discussions.

## Problem: Natural language generation  
- Chinese news summarization (title generation)
  - Fine-tune a pre-trained small multilingual T5 model
  - Compare different generation strategies: greedy, beam search, top-k sampling, top-p sampling, temperature, nucleus sampling
- Bonus: applied reinforcement learning on NLG
  - Adding an additional loss component to the supervised training process
  - Adjusting the proportion of using the original loss and the RL loss
  - RL loss is calculated by decoding predicted logits to obtain a complete sentence and then calculating the ROUGE score with the ground truth summary, using an average of rouge-1-f, rouge-2-f, and rouge-L-f as the reward
  - To ensure model stability, rewards are calculated batch-wise
- Metric: [TW Rouge](https://github.com/cccntu/tw_rouge)

--- 

### Set up environment
```shell
conda create -f environment.yml
conda activate adl-hw3
```

### Download model files
```shell
bash download.sh
```

### Predict result with testing data and evaluate
```shell
bash ./run.sh ./data/public.jsonl ./output/public_predict.jsonl
python3.8 eval.py -r ./data/public.jsonl -s ./output/public_predict.jsonl
```

### How to reproduce training result
```shell
bash train.sh
```