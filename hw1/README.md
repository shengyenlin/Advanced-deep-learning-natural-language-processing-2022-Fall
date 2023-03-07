# HW1

Refer to `report.pdf` for model performance, experiement results, observation and discussions.

## Problem 1: Intent classification
- Concatenate GloVe and FastText embedding as word embedding for LSTM
- Implement bi-LSTM and multi-layer perceptron in PyTorch.
- Experiments
  - Use last hidden state as MLP input v.s. concatenate with average hiddent state and concatenate with maximum hidden state
  - Number of layers in LSTM
  - Lower learning rate in embedding layer (since embedding is pretrained) v.s. same learning rate across all network

## Problem 2: Slot tagging
- Concatenate GloVe and FastText embedding as word embedding for LSTM
- Implement bi-LSTM and multi-layer perceptron in PyTorch.
- Experiments
  - Use last hidden state as MLP input v.s. concatenate with average hiddent state and concatenate with maximum hidden state
  - Number of layers in LSTM
  - Lower learning rate in embedding layer (since embedding is pretrained) v.s. same learning rate across all network

---
### Set up environment
```shell
conda create -f environment.yml
make
conda activate adl-hw1
pip install -r requirements.txt
```

### Download cache and ckpt files
```shell
bash download.sh
```

### Directory layout
- The directory should look like this after runnning all the command above
```
ADL21-HW1/ 
┣ cache/ 
┃ ┣ intent/ 
┃ ┃ ┣ embeddings.pt         #word embedding for intent classification
┃ ┃ ┣ intent2idx.json
┃ ┃ ┣ vocab.json                 
┃ ┃ ┗ vocab.pkl             #vocab class for intent classification in utils.py
┃ ┗ slot/
┃   ┣ embeddings.pt         #word embedding for slot tagging
┃   ┣ intent2idx.json
┃   ┣ tag2idx.json
┃   ┗ vocab.pkl
┣ ckpt/
┃ ┣ intent/
┃ ┃ ┗ best_intent.ckpt      #best model of intent classification
┃ ┗ slot/
┃   ┗ best_slot.ckpt        #best model of slot tagging
┣ data/
┃ ┣ intent/
┃ ┃ ┣ eval.json
┃ ┃ ┣ test.json
┃ ┃ ┗ train.json
┃ ┗ slot/
┃   ┣ eval.json
┃   ┣ test.json
┃   ┗ train.json
┣ Makefile
┣ README.md
┣ dataset.py
┣ environment.yml
┣ intent_cls.sh
┣ model.py
┣ nb-intent.ipynb
┣ nb-slot.ipynb
┣ requirements.txt
┣ test_intent.py
┣ test_slot.py
┣ test_slot.sh
┣ train_intent.py
┣ train_slot.py
┣ train_slot.sh
┣ slot_tag.sh
┣ train_intent_cv.sh
┣ train_slot_cv.sh
┗ utils.py
```

### Predict result with testing data and submit
```shell
bash ./intent_cls.sh data/intent/test.json pred_intent.csv
bash ./slot_tag.sh data/slot/test.json pred_slot.csv
kaggle competitions submit -c intent-classification-ntu-adl-hw1-fall-2022 -f pred_intent.csv -m "Message"
kaggle competitions submit -c slot-tagging-ntu-adl-hw1-fall-2022 -f pred_slot.csv -m "Message"
```

### How to reproduce `best_intent.ckpt`
- Run all cells in `nb-intent.ipynb`
    - Split training data into multiple cross-validation files
    - Download and parse pre-trained embeddings
    - Generate vocab and embedding matrix
- run the following command
```shell
bash train_intent.sh
```
- If you want to do cross validation for hyperparameter tuning, run
```shell
bash train_intent_cv.sh
```

### How to reproduce `best_slot.ckpt`
- Run all cells in `nb-slot.ipynb`
    - Split training data into multiple cross-validation files
    - Download and parse pre-trained embeddings
    - Generate vocab and embedding matrix
- run the following command
```shell
bash train_slot.sh
```
- If you want to do cross validation for hyperparameter tuning, run
```shell
bash train_slot_cv.sh
```