# Advanced deep learning, 2022 Fall
This is a graduate-level deep learning course that introduces natural language processing content from the basics to cutting-edge techniques. This repository contains my solutions to the homework assignments as well as the final project.

## Course contents
- NN Basics, Backpropagation
- Word representation
- Recurrent neural network
- Word embedding, ELMo
- Attention, transformer
- BERT
- Reinforcement learning: Value-based RL, policy gradient, actor-critic
- Natural langauge generation, perplexity
- Large language model (LLM) pretraining
- Prompt learning on pretrained LM
- Dual learning, self-supervised learning
- GPTs: GPT-3, InstructGPT, ChatGPT, WebGPT
- Multimodal models: CLIP, DALL·E2

## Homework assignments
- HW1: Slot tagging
  - Intent classification: 40/297 on [Kaggle](https://www.kaggle.com/competitions/slot-tagging-ntu-adl-hw1-fall-2022/leaderboard)
  - Slot tagging: 23/300 on [Kaggle](https://www.kaggle.com/competitions/intent-classification-ntu-adl-hw1-fall-2022/leaderboard)
  - Implement bi-LSTM and multi-layer perceptron in PyTorch
- HW2: BERT
  - Chinese questions answering (QA), 18/276 on [Kaggle](https://www.kaggle.com/competitions/ntu-adl-hw2-fall-2022/leaderboard)
  - Context selection, span selection
  - Bonus: Train a BERT-based model on HW1 dataset 
  - Use several BERT-based models to complete the task and compare their performance
  - Stack Erlangshen-DeBERTa-v2-97M-Chinese, chinese-roberta-wwm-ext, chinese-roberta-wwm-ext models for final testing phase.
- HW3: Natural langauge generation
  - Chinese news summarization (title generation)
  - Fine-tune a pre-trained small multilingual T5 model
  - Compare different generation strategies: greedy, beam search, top-k sampling, top-p sampling, temperature, nucleus sampling
  - Bonus: Reinforcement learning on summarization

You could find more details in the directory of each homework.

## Final projects 
- Rank 1st in the oral presentation
- [Video link](https://www.youtube.com/watch?v=UpOfI-Bp6pc) (in Chinese)
- For codes, please refer to [this repo](https://github.com/shengyenlin/Advanced-deep-learning-final-project-2022-Fall)
- [Collaborative google drive](https://drive.google.com/drive/folders/11ApKnaDlihwn6Iu_QJODIw1F-6jKIGzp?usp=share_link)