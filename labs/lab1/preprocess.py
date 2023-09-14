import json
import numpy as np
import string
from collections import Counter


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('train_data', type=str)
    parser.add_argument('--vocab', type=str, default='vocab.json')
    parser.add_argument('output', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.vocab, encoding='utf-8') as f:
        word2idx = json.load(f)
    vocab_size = len(word2idx)

    data = []
    label_count = [0, 0]
    with open(args.train_data, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # skip header
                continue

            cols = line.rstrip('\n').split('\t')

            if len(cols) != 2:
                continue

            sentence = cols[0]
            label = int(cols[1])

            # Remove punctuation
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))

            # label + bag of words vector
            v = np.zeros(vocab_size + 1, dtype=np.int32)
            v[0] = label

            label_count[label] += 1

            words = sentence.split()
            counter = Counter(words)
            for w in words:
                idx = word2idx[w]
                v[idx + 1] = counter[w]

            data.append(v)

    print(f'Label count: {label_count}')

    np.concatenate(data, axis=0).tofile(args.output)


if __name__ == '__main__':
    main()
