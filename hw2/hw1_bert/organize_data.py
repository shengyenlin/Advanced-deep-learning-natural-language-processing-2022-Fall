import os
import json
import sys

data_path = sys.argv[1]

def main():
    #data_path = './intent/data'
    datas = ['train', 'eval', 'test']
    for data in datas:   
        in_file = os.path.join(data_path, data + ".json")
        out_file = os.path.join(data_path, data + "_organized.json")
        with open(out_file, 'w') as wf:
            with open(in_file, 'r') as rf:
                data = json.load(rf)
            for d in data:
                print(json.dumps(d), file=wf)

if __name__ == '__main__':
    main()