import os
import json
import torch
from typing import Tuple, List
import time
from torch import optim, nn
import numpy as np
from model import Net
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, asdict


class SST2Data(Dataset):
    def __init__(self, file: str):
        self.data: np.ndarray = np.load(file)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        feats = self.data[idx, 1:]
        label = self.data[idx, 0]
        return dict(feats=torch.as_tensor(feats, dtype=torch.float), label=label)

    @property
    def vocab_size(self):
        return self.data.shape[1] - 1


def train_one_epoch(train_loader: DataLoader, optimizer: optim.Optimizer, model: nn.Module):
    # running_loss = 0
    # last_loss = 0
    for i, batch in enumerate(train_loader):
        x, y = batch['feats'], batch['label']
        y = y.unsqueeze(1).to(dtype=torch.float)

        optimizer.zero_grad()

        y_hat = model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        loss.backward()

        optimizer.step()

        # report
        # running_loss += loss.item()
        # if i % 100 == 99:
        #     last_loss = running_loss / 100  # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     running_loss = 0.


def train(
        train_loader: DataLoader,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        epochs: int,
        trials: int = 5,
) -> Tuple[nn.Module, List[float]]:
    assert trials > 0

    train_time = []

    model = None
    for t in range(trials):
        print(f'Training trial {t + 1}/{trials}')

        model = Net(num_layers, hidden_size, vocab_size, 1)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        start_time = time.time()

        for e in range(epochs):
            train_one_epoch(train_loader, optimizer, model)

        end_time = time.time()
        train_time.append(end_time - start_time)

    print(f'Training time (seconds): {train_time}')
    return model, train_time


def evaluate(
        model: nn.Module,
        test_loader: DataLoader,
):
    y_true = []
    y_pred = []

    model.eval()
    for batch in test_loader:
        x, y = batch['feats'], batch['label']
        assert x.shape[0] == y.shape[0] == 1  # batch=1 during inference
        y_true.append(int(y.squeeze()))

        y_hat = model(x)
        y_hat = int(torch.sigmoid(y_hat).squeeze(0) > 0.5)
        y_pred.append(y_hat)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {acc}')
    return acc


def benchmark_inference(
        model: nn.Module,
        test_loader: DataLoader,
        trials: int = 5,
):
    inference_time = []
    model.eval()

    for t in range(trials):
        start_time = time.time_ns()

        for batch in test_loader:
            x = batch['feats']
            model(x)

        end_time = time.time_ns()
        inference_time.append(end_time - start_time)

    inference_time = np.asarray(inference_time) / (len(test_loader) * 1e6)
    print(f'Inference time (ms): {inference_time}')
    return inference_time.tolist()


@dataclass
class Result:
    acc: float
    train_time: List[float]
    inference_time: List[float]
    num_params: int
    FLOPs: int

    num_layers: int
    hidden_size: int
    vocab_size: int


def experiment(num_layers: int, hidden_size: int, vocab_size: int, train_loader: DataLoader, test_loader: DataLoader):
    # Base
    model, train_time = train(train_loader, num_layers, hidden_size, vocab_size, 2, trials=1)
    acc = evaluate(model, test_loader)
    inference_time = benchmark_inference(model, test_loader, trials=5)

    return Result(
        num_layers=num_layers, hidden_size=hidden_size, vocab_size=vocab_size,
        acc=acc, train_time=train_time, inference_time=inference_time, num_params=model.num_params,
        FLOPs=model.flop,
    )


def main():
    os.makedirs('results', exist_ok=True)

    batch_size = 64
    matrix = dict(
        num_layers=[1, 2, 4],
        hidden_size=[128, 256, 512],
        vocab_size=[1000, 5000, 10000],
    )

    for nl in matrix['num_layers']:
        for hs in matrix['hidden_size']:
            for vocab_size in matrix['vocab_size']:
                # data
                train_data = SST2Data(f'train_bow{vocab_size}.npy')
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                test_data = SST2Data(f'dev_bow{vocab_size}.npy')
                test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

                # prepare
                tag = f'num_layers={nl}_hidden_size={hs}_vocab_size={vocab_size}'
                output_file = f'results/{tag}.json'
                if os.path.exists(output_file):
                    print(f'Skipping experiment {tag}')
                    continue
                print(f'\nRunning experiment {tag}')

                # run
                result = experiment(nl, hs, vocab_size, train_loader, test_loader)

                # save
                result = asdict(result)
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
