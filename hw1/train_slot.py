import json
import pickle
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def to_device(x, device):
    return {k: v.to(device) if type(v) is torch.Tensor else v for k, v in x.items()}


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    data_loaders = {
        split: DataLoader(split_ds, shuffle=True, pin_memory=True,
                          collate_fn=split_ds.collate_fn,
                          batch_size=(args.batch_size if split == TRAIN else 1024))
                          #batch_size=args.batch_size))
        for split, split_ds in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqTagger(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(intent2idx)
    )
    model = model.to(args.device)

    # lower learning rate on embedding layer
    parameters = [
        {'params': [p for p in model.parameters() if p is model.embed.weight],
         'lr': args.lr * 0.1},
        {'params': [p for p in model.parameters() if p is not model.embed.weight],
         'lr': args.lr},
    ]

    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    metrics_hist = []

    for epoch in epoch_pbar:
        metrics = {'epoch': epoch+1, 'lr': optimizer.param_groups[-1]['lr']}
        train_loop(model, data_loaders[TRAIN], optimizer, loss_fn, args, metrics)
        val_loop(model, data_loaders[DEV], loss_fn, args, metrics)
        scheduler.step(metrics['val_loss'])
        print(metrics)
        save_best(model, metrics, metrics_hist)


def train_loop(model, dl, optimizer, loss_fn, args, metrics):
    model.train()
    loss_sum = 0

    for i, batch in enumerate(dl):
        batch = to_device(batch, args.device)
        logits = model(batch)  # (bs, seq, class)
        logits = logits.flatten(start_dim=0, end_dim=1)  # (bs*seq, class)
        tag_ids = batch['tag_ids'].flatten()  # (bs*seq) -> class idx
        loss = loss_fn(logits, tag_ids)
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metrics['train_loss'] = loss_sum / len(dl)


def val_loop(model, dl, loss_fn, args, metrics):
    model.eval()
    loss_sum = 0
    num_correct_sent = 0
    num_correct_tags = 0
    total_tags = 0

    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, args.device)
            tags = batch['tag_ids']  # (bs, seq) -> class idx
            batch_size, batch_len = tags.size()

            logits = model(batch)  # (bs, seq, class)
            logits_flat = logits.flatten(start_dim=0, end_dim=1)  # (bs*seq, class)
            tags_flat = tags.flatten()  # (bs*seq) -> class idx
            loss = loss_fn(logits_flat, tags_flat)
            loss_sum += loss.item()

            pred = logits.argmax(2)  # (bs, seq) -> top class idx
            cmp = (pred == tags)
            num_correct_tags += cmp.sum().item()
            total_tags += batch_len * batch_size
            #if all tags predicted are correct
            cmp_sent = (pred == tags).all(dim=1)  # (bs) => bool(all tags equal)
            num_correct_sent += cmp_sent.sum().item()

    metrics['val_loss'] = loss_sum / len(dl)
    metrics['val_accuracy_tags'] = num_correct_tags / total_tags
    metrics['val_accuracy'] = num_correct_sent / len(dl.dataset)


def save_best(model, metrics, hist, run_id=int(time.time())):
    # save all successive best models with descriptive names, symlink best.ckpt to last one
    best = max([m['val_accuracy'] for m in hist], default=0)
    if metrics['val_accuracy'] > best:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        filename = 'acc%.6f_ep%d_run%d.ckpt' % (metrics['val_accuracy'], metrics['epoch'], run_id)
        path = args.ckpt_dir / filename
        torch.save(model.state_dict(), path)
        metrics['ckpt'] = str(path)
        path = args.ckpt_dir / 'best.ckpt'
        try:
            path.unlink()
        except:
            pass
        path.symlink_to(filename)

    print(metrics)
    hist.append(metrics)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)