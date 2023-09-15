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


def train(
        train_loader: DataLoader,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        epochs: int,
):
    model = SST2Model(num_layers, hidden_size, vocab_size)
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model=model, train_dataloaders=train_loader)


def main():
    batch_size = 64
    train_data = SST2Data('train_clean.npy')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    train(train_loader, 2, 256, train_data.vocab_size, 2)


if __name__ == '__main__':
    main()
