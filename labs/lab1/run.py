import torch
from torch import optim, nn, utils
import numpy as np
import lightning.pytorch as pl
from model import Net
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class SST2Model(pl.LightningModule):
    def __init__(self, num_layers: int, hidden_size: int, vocab_size: int):
        super().__init__()
        self.model = Net(num_layers, hidden_size, vocab_size, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch['feats'], batch['label']
        y = y.unsqueeze(1).to(dtype=torch.float)
        y_hat = self.model(x)

        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


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


def experiment(
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        epochs: int,
):
    model = SST2Model(num_layers, hidden_size, vocab_size)
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model=model, train_dataloaders=train_loader)

    # Evaluation
    y_true = []
    y_pred = []
    for batch in test_loader:
        x, y = batch['feats'], batch['label']
        assert x.shape[0] == y.shape[0] == 1  # batch=1 during inference
        y_true.append(int(y.squeeze()))

        y_hat = model.model(x)
        y_hat = int(torch.sigmoid(y_hat).squeeze(0) > 0.5)
        y_pred.append(y_hat)

    from sklearn.metrics import accuracy_score
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')


def main():
    batch_size = 64
    train_data = SST2Data('train_clean.npy')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = SST2Data('dev_clean.npy')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    experiment(train_loader, test_loader, 2, 256, train_data.vocab_size, 2)


if __name__ == '__main__':
    main()
