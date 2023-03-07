#!/usr/bin/env python
# End to end evaluation script:
#   - converts original .json to swag/qa input examples
#   - runs swag + qa models in pipeline
#   - runs swag models at two different offsets to better handle long contexts
#   - ensembles multiple swag/qa models

import argparse
import json
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union

import numpy as np
import datasets
import pandas as pd
import torch

from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy, check_min_version, get_full_repo_name

from utils import convert_example_to_squad, convert_example_to_swag, postprocess_qa_predictions
from model import load_swag_model, load_qa_model


def parse_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='Input json file')
    parser.add_argument('-c', '--context-file', default='data/context.json', help='Context json file')
    parser.add_argument('-o', '--output-file', help='Output csv file')
    parser.add_argument('-S', '--swag-model', default=[], action='append',
                        help='Context selection model dir. May specify multiple times to ensemble.')
    parser.add_argument('-Q', '--qa-model', default=[], action='append',
                        help='QA model dir. May specify multiple times to ensemble.')
    parser.add_argument('--maxn', type=int, help='Evalute first n examples only')
    parser.add_argument('--swag-shift', type=int, default=384, help='Evalute swag models a second time at this offset in contexts')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n_best_size', type=int, default=100)
    parser.add_argument('--max_answer_length', type=int, default=40)
    parser.add_argument('--doc_stride', type=int, default=128)
    args = parser.parse_args(**kwargs)
    return args


@dataclass
class SwagDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        if "label" in features[0].keys():
            labels = [feature.pop("label") for feature in features]
        else:
            labels = None
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def swag_preprocess_function(examples, args, tokenizer):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    #padding = "max_length" if args.pad_to_max_length else False
    padding = False

    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"
    label_column_name = "label"

    first_sentences = [[context] * 4 for context in examples[context_name]]
    question_headers = examples[question_header_name]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]
    if label_column_name in examples:
        labels = examples[label_column_name]
    else:
        labels = None

    # Flatten out
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        max_length=args.max_seq_length,
        padding=padding,
        truncation=True,
    )
    # Un-flatten
    tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    if labels is not None:
        tokenized_inputs["label"] = labels
    return tokenized_inputs


def eval_swag_model(path, examples, accelerator, args):
    print('Evaluating %s' % path)

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    model = load_swag_model(path)
    model.eval()

    ds = Dataset.from_list(examples)
    ds = ds.map(
        lambda x: swag_preprocess_function(x, args, tokenizer),
        batched=True,
        remove_columns=ds.column_names
    )
    dl = DataLoader(
        ds,
        shuffle=False,
        collate_fn=SwagDataCollator(tokenizer),
        batch_size=args.batch_size
    )

    device = accelerator.device
    model.to(device)
    model, dl = accelerator.prepare(model, dl)

    logits = []
    for batch in tqdm(dl):
        with torch.no_grad():
            outputs = model(**batch)
        logits.append(outputs.logits.cpu())
    
    return torch.cat(logits)


# Validation preprocessing
def qa_prepare_validation_features(examples, args, tokenizer):
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    pad_on_right = tokenizer.padding_side == "right"
    pad_to_max_length = False

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


def eval_qa_model(path, ds_examples, accelerator, args):
    print('Evaluating %s' % path)

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = load_qa_model(path)
    model.eval()

    # one of more entry per example at different strides for long docs
    ds_features = ds_examples.map(
        lambda x: qa_prepare_validation_features(x, args, tokenizer),
        batched=True,
        remove_columns=ds_examples.column_names
    )

    dl = DataLoader(
        ds_features.remove_columns(["example_id", "offset_mapping"]),
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
        batch_size=args.batch_size,
    )

    model.to(accelerator.device)
    model, dl = accelerator.prepare(model, dl)

    all_start_logits = []
    all_end_logits = []

    for batch in tqdm(dl):
        with torch.no_grad():
            outputs = model(**batch)
        all_start_logits.append(outputs.start_logits.cpu().numpy())
        all_end_logits.append(outputs.end_logits.cpu().numpy())

    # pad to max_seq_length for ease of combining across different models
    start_logits_concat = create_and_fill_np_array(all_start_logits, ds_features, args.max_seq_length)
    end_logits_concat = create_and_fill_np_array(all_end_logits, ds_features, args.max_seq_length)
    return (ds_features, start_logits_concat, end_logits_concat)


def main():
    args = parse_args()

    contexts = json.load(open(args.context_file))  # list of strings
    examples = json.load(open(args.input_file)) # valid.json or text.json
    if args.maxn:
        examples = examples[:args.maxn]
    N = len(examples)

    accelerator = Accelerator()

    # Evaluate SWAG:
    #   - evaluate at two different offsets into contexts to better handle long documents
    #   - ensemble average multiple models
    #   - substitute examples['relevant'] with predictions for next stage

    def eval_accuracy(title, logits1, logits2):
        logits = torch.max(logits1, logits2)
        preds = torch.argmax(logits, dim=-1)
        preds1 = torch.argmax(logits1, dim=-1)
        corr = 0
        corr1 = 0

        for i, example in enumerate(examples):
            if 'relevant' not in example: return # no relevant -> test file
            label = example.get('relevant')
            rel = example['paragraphs'][preds[i].item()]
            rel1 = example['paragraphs'][preds1[i].item()]
            corr += int(rel == label)
            corr1 += int(rel1 == label)

        print('%s: context selection accuracy %.4f (with second eval @%d; one eval %.4f)' \
            % (title, corr/N, args.swag_shift, corr1/N))

    if len(args.swag_model) > 0:
        swag_examples = [convert_example_to_swag(ex, contexts) for ex in examples]
        swag_examples_sh = [convert_example_to_swag(ex, contexts, args.swag_shift) for ex in examples]

        logits1 = torch.zeros(N, 4)
        logits2 = torch.zeros(N, 4)

        for path in args.swag_model:
            l1 = eval_swag_model(path, swag_examples, accelerator, args)
            l2 = eval_swag_model(path, swag_examples_sh, accelerator, args)
            eval_accuracy(path, l1, l2)
            logits1 += l1
            logits2 += l2

        if len(args.swag_model) > 1:
            eval_accuracy('Ensemble', logits1, logits2)

        logits = torch.max(logits1, logits2)
        preds = torch.argmax(logits, dim=-1)

        for i, example in enumerate(examples):
            example['relevant'] = example['paragraphs'][preds[i].item()]

    # Evaluate QA models

    # Aggregate multiple models in an ensemble after prelim_predictions generation during postprocessing.
    # This is easier to implement than averaging start/end logits directly which would have to be
    # first correctly aligned due to different tokenizers.
    def combine_prelim_predictions_for_ensemble(prelim_predictions):
        # Input/output: list of dicts with keys: offsets, start_logit, end_logit, score, model
        # Average scores/logits for the same answer span - note this would also average same model scores across strides
        # Then keep only spans for which we have predictions from all models.
        return pd.DataFrame(prelim_predictions) \
            .groupby(['offsets']) \
            .agg({
                'score': 'mean',
                'start_logit': 'mean',
                'end_logit': 'mean',
                'model': 'nunique',  # count models which picked this span among top n_best_size spans
            }) \
            [lambda X: X['model'] == len(args.qa_model)] \
            .reset_index() \
            .to_dict('records')

    if len(args.qa_model) > 0:
        ds_examples = Dataset.from_list([convert_example_to_squad(ex, contexts) for ex in examples])

        all_features = {}
        all_start_logits = {}
        all_end_logits = {}

        # evaluate each model and ensemble
        for path in args.qa_model + (['ensemble'] if len(args.qa_model) > 1 else []):
            if path != 'ensemble':
                ds_features, start_logits, end_logits = eval_qa_model(path, ds_examples, accelerator, args)
                ds_features = ds_features.add_column('model', [path] * len(ds_features))

                all_features[path] = ds_features
                all_start_logits[path] = start_logits
                all_end_logits[path] = end_logits
                combine_fn = None
            else:
                ds_features = datasets.combine.concatenate_datasets(list(all_features.values()))
                start_logits = np.concatenate(list(all_start_logits.values()))
                end_logits = np.concatenate(list(all_end_logits.values()))
                combine_fn = combine_prelim_predictions_for_ensemble

            predictions = postprocess_qa_predictions(
                examples=ds_examples,
                features=ds_features,
                predictions=(start_logits, end_logits),
                version_2_with_negative=False,
                n_best_size=args.n_best_size,
                max_answer_length=args.max_answer_length,
                null_score_diff_threshold=0,
                output_dir=None,
                combine_prelim_predictions_fn=combine_fn
            )
            # predictions: example id => answer text from top-scoring prediction

            expected_answers = {}
            for example in examples:
                answer = example.get("answer", {}).get("text", None)
                if answer is not None:
                    expected_answers[example["id"]] = answer

            if len(expected_answers) > 0:
                corr = 0
                for example_id, answer in predictions.items():
                    if answer == expected_answers[example_id]:
                        corr += 1
                print('%s: exact match %.4f (%d/%d correct)' % (path, corr*100/N, corr, N))

        # save latest 'predictions' (ensemble or single model) into output csv file
        if args.output_file:
            df = pd.DataFrame({'id': k, 'answer': v} for (k, v) in predictions.items())
            df.to_csv(args.output_file, index=False)
            print('Wrote %s' % args.output_file)


if __name__ == "__main__":
    main()
