# HW2

Refer to `report.pdf` for model performance, experiement results, observation and discussions.

## Problem: Chinese question answering 
- Fine-tune a pre-trained Chinese BERT-based model on the dataset and pass the baselines.
- Use several BERT-based models to complete the task and compare their performance.
- Model used: [hfl/chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext), [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext), [bert-base-chinese]( https://huggingface.co/bert-base-chines), [Erlangshen-DeBERTa-v2-97M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese), [chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base)

### Remark
For the final testing phase, I use the following models.
- Multiple choice: chinese-bert-wwm-ext
- Question answering: ensemble the following three model
    - Erlangshen-DeBERTa-v2-97M-Chinese
    - chinese-roberta-wwm-ext
    - chinese-roberta-wwm-ext

---

### Set up environment
```shell
conda create -n adl-hw2 python=3.9
conda activate adl-hw2
pip install -r requirements.txt
```

### Download cache and ckpt files
```shell
bash download.sh
```

### Predict result with testing data and submit
```shell
bash ./run.sh ./data/context.json ./data/test.json prediction.csv
kaggle competitions submit -c ntu-adl-hw2-fall-2022 -f prediction.csv -m "Message"
```

### How to reproduce training result
- Run all cells in `organize_data.ipynb`
    - organize data into `datasets.load_dataset()` input format
- run the following command
```shell
bash train_qa_model1.sh
bash train_qa_model2.sh
bash train_qa_model3.sh
bash train_swag.sh
```

# Bonus homework

### Download cache and ckpt files
```shell
cd hw1_bert
bash download.sh
```

### Predict result with testing data and submit
```shell
cd hw1_bert
bash organize_data.sh ./data/intent ./data/slot
bash test_intent.sh
bash test_slot.sh
kaggle competitions submit -c intent-classification-ntu-adl-hw1-fall-2022 -f intent.csv -m "Message"
kaggle competitions submit -c slot-tagging-ntu-adl-hw1-fall-2022 -f slot.csv -m "Message"
```
### How to reproduce training result
```shell
cd hw1_bert
bash organize_data.sh ./data/intent ./data/slot
bash train_intent.sh
bash train_slot.sh
```