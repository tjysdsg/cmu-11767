import torch
import time
from torch import optim, nn
import numpy as np
from model import Net
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


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
    running_loss = 0
    last_loss = 0

    for i, batch in enumerate(train_loader):
        x, y = batch['feats'], batch['label']
        y = y.unsqueeze(1).to(dtype=torch.float)

        optimizer.zero_grad()

        y_hat = model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        loss.backward()

        optimizer.step()

        # report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


def train(
        train_loader: DataLoader,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        epochs: int,
        trials: int = 5,
) -> nn.Module:
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
    return model


def evaluate(
        model: nn.Module,
        test_loader: DataLoader,
):
    y_true = []
    y_pred = []
    inference_time = []

    model.eval()
    for batch in test_loader:
        x, y = batch['feats'], batch['label']
        assert x.shape[0] == y.shape[0] == 1  # batch=1 during inference
        y_true.append(int(y.squeeze()))

        start_time = time.time_ns()
        y_hat = model(x)
        end_time = time.time_ns()
        inference_time.append(end_time - start_time)

        y_hat = int(torch.sigmoid(y_hat).squeeze(0) > 0.5)
        y_pred.append(y_hat)

    from sklearn.metrics import accuracy_score
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')

    print(f'Inference time (ns): {inference_time}')


def main():
    batch_size = 64
    train_data = SST2Data('train_clean.npy')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = SST2Data('dev_clean.npy')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    model = train(train_loader, 2, 256, train_data.vocab_size, 2, trials=5)
    evaluate(model, test_loader)


if __name__ == '__main__':
    main()
