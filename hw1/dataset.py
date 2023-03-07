import re
import torch

from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


def basic_tokenizer(text):
    # fasttext style x's => x 's, can't => can 't. 
    # note: glove has n't tokens
    text = text.replace("'", " '")
    text = re.sub(r'([`~!@#$%^&*(){}\[\]":;,.<>/?+’`]+)', r' \1 ', text)
    return text.lower().split()


class SeqClsDataset(Dataset):
    def __init__(
        self,
        #[
        # {"text": ..., "intent":..., "id":...}
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {
            idx: intent for intent, idx in self.label_mapping.items()
            }
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        #[samples[each word]]
        tokens: List[List[str]] = [basic_tokenizer(s['text']) for s in samples]
        #pad sequence length 
        token_ids = self.vocab.encode_batch(tokens)
        res = {
            'ids': [s['id'] for s in samples],
            'token_ids': torch.LongTensor(token_ids),
        }
        #check if 'intent' in the key of dict 
        if samples and 'intent' in samples[0]:
            labels = [s.get('intent') for s in samples]
            label_ids = [self.label2idx(s) if s else -1 for s in labels]
            res['label_ids'] = torch.LongTensor(label_ids)
        return res

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100
    pad_id = 0  # [PAD] tag index

    def collate_fn(self, samples):
        tokens = [s['tokens'] for s in samples]
        token_ids = self.vocab.encode_batch(tokens)
        res = {
            'ids': [s['id'] for s in samples],
            'tokens': tokens,
            'token_ids': torch.LongTensor(token_ids),
        }
        # check if 'tags' in the key of dictionary
        # i.e. if it's "train/eval" or "test"
        if samples and 'tags' in samples[0]:
            tag_ids = [[self.label2idx(t) for t in s['tags']] for s in samples]
            # token_ids[0] = '[PAD]'
            tag_ids = pad_to_len(tag_ids, len(token_ids[0]), self.pad_id)
            res['tag_ids'] = torch.LongTensor(tag_ids)
        return res

    def format_output(self, token_ids, tag_ids):
        # undo padding and convert predicted tag ids back to labels
        res = []
        assert len(token_ids) == len(tag_ids)
        #tag_ids: tag ground truth with '[PAD]'
        #token_ids: word token with '[PAD]'
        for i in range(len(token_ids)):
            if token_ids[i] != self.pad_id:
                #If the model still predict ['PAD'], hard code to 'O'
                if tag_ids[i] == 0:
                    res.append('O')
                else:
                    res.append(self.idx2label(tag_ids[i]))
        return ' '.join(res)

    # def pred_tag_ids_to_tag_batch(self, tok_ids, tag_ids):
    #     gt = []
    #     for i in range(len(tok_ids)):
    #         #print(token_ids[i])
    #         if tok_ids[i] != self.pad_id:
    #             gt.append(self.idx2label(tag_ids[i]))
    #     return gt

    # def gt_tag_ids_to_tag_batch(self, tok_ids, tag_ids):
    #     gt = []
    #     for i in range(len(tok_ids)):
    #         #print(token_ids[i])
    #         if tok_ids[i] != self.pad_id:
    #             gt.append(self.idx2label(tag_ids[i]))
    #     return gt
