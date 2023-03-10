import os
import json
import argparse
import logging

from functools import partial

import datasets
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    set_seed,
)

from utils import prepare_features_slot

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--saved_dir", type=str, required=True)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--out_file", type=str, default="./slot.csv")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(os.path.join(args.saved_dir, "args.json"), 'r') as f:
        train_args = json.load(f)
    for k, v in train_args.items():
        if not hasattr(args, k):
            vars(args)[k] = v

# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
# Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    
# Setup logging, we only want one process per machine to log things on the screen.
# accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
# Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.saved_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.saved_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.saved_dir, config=config)

# Load and preprocess the dataset
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    cols = raw_datasets["test"].column_names
    args.token_col, args.tag_col = "tokens", "tags"
    tag2id = config.label2id
    id2tag = config.id2label
    
    test_examples = raw_datasets["test"]
    #test_examples = test_examples.select(range(10))
    prepare_features = partial(prepare_features_slot, args=args, tokenizer=tokenizer, tag2id=tag2id)
    test_dataset = test_examples.map(
        prepare_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )

# Create DataLoaders
    data_collator = DataCollatorForTokenClassification(tokenizer)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.test_batch_size)

# Prepare everything with our accelerator.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

# Test!
    logger.info("\n******** Running predicting ********")
    logger.info(f"Num test examples = {len(test_dataset)}")
    
    model.eval()
    all_predictions = []
    for step, data in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**data)
            predictions = accelerator.gather(outputs.logits.argmax(dim=-1))
            references=accelerator.gather(data["labels"])
            predictions = torch.where(references != -100, predictions, references)
            all_predictions += predictions.cpu().tolist()
    results = {example_id: ' '.join([id2tag[tag_id] for tag_id in pred if tag_id != -100])    \
                            for example_id, pred in zip(test_examples["id"], all_predictions)}
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        f.write("id,tags\n")
        for idx, label in sorted(results.items()):
            f.write("{},{}\n".format(idx, label))