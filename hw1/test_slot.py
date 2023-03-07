import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import pandas as pd
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    idx2indent = {i: s for (s, i) in intent2idx.items()}

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, intent2idx, args.max_len)
    data_loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                             collate_fn=dataset.collate_fn,
                             batch_size=args.batch_size)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)

    rows = []
    num_correct = 0


    for batch in data_loader:
        logits = model(batch)    # (bs, seq, class)
        pred = logits.argmax(2)  # (bs, seq) -> top class idx

        #token_ids = batch['token_ids']  # (bs, seq)
        #logits[token_ids == 0] = -torch.inf  # mask out [PAD] class from argmax
        if 'tag_ids' in batch:
            y_true = []
            y_pred = []
            #tags_ids: GT
            tags_ids = batch['tag_ids']  # (bs, seq) -> class idx
            num_correct += (pred == tags_ids).all(dim=1).sum().item()

        if 'tag_ids' in batch:
            for id, tok_ids, pred_ids, tag_ids in zip(batch['ids'], batch['token_ids'], pred.numpy(), batch['tag_ids']):
                tags = dataset.format_output(tok_ids.detach().cpu().numpy(), pred_ids)
                gt = dataset.format_output(tok_ids.detach().cpu().numpy(), tag_ids.detach().cpu().numpy())
                rows.append({
                    'id': id,
                    'tags': tags
                })
                y_pred.append(tags.split(' '))
                y_true.append(gt.split(' '))
        else:
            for id, tok_ids, pred_ids, in zip(batch['ids'], batch['token_ids'], pred.numpy()):
                tags = dataset.format_output(tok_ids.detach().cpu().numpy(), pred_ids)
                rows.append({
                    'id': id,
                    'tags': tags
                })

    if num_correct:
        acc = num_correct / len(dataset)
        print('Accuracy %.6f (%d/%d)' % (acc, num_correct, len(dataset)))
        report = classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
        print(report)

    df = pd.DataFrame(rows)
    df.to_csv(args.pred_file, index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)