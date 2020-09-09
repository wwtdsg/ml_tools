import os
import random
import argparse
import re
import tqdm
from collections import Counter
import numpy as np


def graph_test():
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]], dtype=float)
    X = np.matrix([
        [i, -i]
        for i in range(A.shape[0])
    ], dtype=float)
    print(A * X)
graph_test()


orig_general_label_file = r'/media/chen2/data/Chinese_check/rowdata/labels/trainset_200701.txt'
orig_poem_label_file = r'/media/chen/wt/data/hw_poem_labels/hw_labels_all_train.txt'
math_tk_file = r'/media/chen/hcn/data/math_tiankong/rec/row_data/text_label.txt'

new_poem_label_file = r'/media/chen/wt/data/hw_poem_data/poem_data_20200622/labels.txt'

dst_poem_label_file = r'/media/chen/wt/data/hw_poem_data/label_20200702.txt'
dst_general_label_file = r'/media/chen/wt/data/hw_poem_data/hw_general_train_data.txt'

def mk_hw_poem_data():
    poem_dict = {}
    poem_sample_num = 200
    generate_poem = []

    with open(orig_poem_label_file) as f:
        orig_labels = f.readlines()
        for line in tqdm.tqdm(orig_labels):
            if "hw_poem_true_negative" in line:
                generate_poem.append(line)
                continue
            path, label = line.split(' ', 1)
            label = label.strip()

            if not os.path.exists(path):
                print(path)
                continue
            if label not in poem_dict:
                poem_dict[label] = [line]
            else:
                poem_dict[label].append(line)

    all_labels = []
    for poem in tqdm.tqdm(poem_dict):
        if len(poem_dict[poem]) > poem_sample_num:
            all_labels.extend(random.sample(poem_dict[poem], poem_sample_num))
        else:
            all_labels.extend(poem_dict[poem])

    with open(new_poem_label_file) as f:
        root_dir = '/media/chen/wt/data/hw_poem_data/poem_data_20200622/'
        new_poem_labels = [root_dir + line for line in f.readlines()]
        for line in tqdm.tqdm(new_poem_labels):
            path, label = line.split(' ', 1)
            if not os.path.exists(path):
                print(path)
                continue
            all_labels.append(line)

    sample_generate_poem = random.sample(generate_poem, len(all_labels) // 20)
    all_labels.extend(sample_generate_poem)

    random.shuffle(all_labels)
    print('done')
    with open(dst_poem_label_file, 'w') as f:
        f.writelines(all_labels)
    pass


def mk_hw_general_data():
    poem_dict = {}
    with open(dst_poem_label_file) as f:
        orig_labels = f.readlines()
        for line in tqdm.tqdm(orig_labels):
            if "hw_poem_true_negative" in line or "gen" in line:
                continue
            path, label = line.split(' ', 1)
            label = label.strip()

            if not os.path.exists(path):
                print(path)
                continue
            if label not in poem_dict:
                poem_dict[label] = [line]
            else:
                poem_dict[label].append(line)



def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "-f",
        "--label_file",
        type=str,
        nargs="?",
        help="",
        default='',
    )
    return parser.parse_args()


def mk_vocb(file_path, dst_file, freq_file):
    dict = Counter()
    with open(file_path) as f:
        lines = f.readlines()
    fw = open(freq_file, "w")
    with open(dst_file, "w") as w:
        for line in lines:
            _, label = line.split(" ", 1)
            for c in label.strip():
                if ord('a') <= ord(c) <= ord('z') or \
                        ord('A') <= ord(c) <= ord('Z') or \
                        ord('0') <= ord(c) <= ord('9'):
                    print(line)
                dict[c] += 1

        sort_dict = sorted(dict.items(), key=lambda kv: kv[1], reverse=True)
        rel = ''
        for kv in sort_dict:
            rel += kv[0]
            fw.write(kv[0] + " " + str(kv[1]) + "\n")
        w.write(rel)
    fw.close()
    return sort_dict


if __name__ == '__main__':
    # file_path = r"/media/chen/wt/data/hw_poem_data/label_20200623.txt"
    # args = parse_arguments()
    path, name = os.path.split(dst_poem_label_file)
    pre, ext = os.path.splitext(name)
    vocab_path = os.path.join(path, pre + "_vocab" + ext)
    freq_path = os.path.join(path, pre + "_freq" + ext)
