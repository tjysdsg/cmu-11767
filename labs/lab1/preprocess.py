import json
import numpy as np
import string
from collections import Counter


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('--vocab', type=str, default='vocab_14704.json')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('output', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.vocab, encoding='utf-8') as f:
        word2idx = json.load(f)
    vocab_size = len(word2idx)

    data = []
    label_count = [0, 0]
    n_oov = 0
    with open(args.data, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # skip header
                continue

            cols = line.rstrip('\n').split('\t')

            if len(cols) != 2:
                continue

            sentence = cols[0]
            if args.test:
                label = 0
            else:
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
                idx = word2idx.get(w, -1)
                if idx == -1:
                    n_oov += 1
                    continue
                v[idx + 1] = counter[w]

            data.append(v)

    print(f'Label count: {label_count}')
    print(f'OOV: {n_oov}')

    data = np.vstack(data)
    print(f'Shape: {data.shape}')
    np.save(args.output, data)


if __name__ == '__main__':
    main()
