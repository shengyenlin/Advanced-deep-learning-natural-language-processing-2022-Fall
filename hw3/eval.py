import json
import argparse
import os
from tw_rouge.tw_rouge import get_rouge

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    refs, preds = {}, {}

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'

    with open(args.submission) as file:
        for line in file:
            line = json.loads(line)
            preds[line['id']] = line['title'].strip() + '\n'

    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]

    print(json.dumps(get_rouge(preds, refs), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-s', '--submission')
    args = parser.parse_args()
    main(args)
