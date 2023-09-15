import json
import string
from collections import Counter


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('train_data', type=str)
    parser.add_argument('--k', type=int, default=-1)
    parser.add_argument('output', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    words = Counter()

    with open(args.train_data, 'r') as f:
        n = 0
        for i, line in enumerate(f):
            if i == 0:  # skip header
                continue

            cols = line.rstrip('\n').split('\t')
            if len(cols) != 2:
                continue

            # Remove punctuation
            sentence = cols[0]
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))

            words.update(sentence.split())
            n += 1

    print(f'Number of lines processed: {n}')
    print(f'Vocabulary size: {len(words)}')

    if args.k != -1:
        print(f'Only keeping top {args.k} words')
        words = [w for w, _ in words.most_common(args.k)]

    words = sorted(list(words))
    word2idx = dict()
    for i, word in enumerate(words):
        word2idx[word] = i

    with open(args.output, 'w') as f:
        json.dump(word2idx, f, indent=2)


if __name__ == '__main__':
    main()
