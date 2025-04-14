# Imports (Default Pytorch)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger

# Imports (Pytorch Lightning)
import pytorch_lightning as pl

from callback import MyPrintCallback
# Imports (params)
from config import LEARNING_RATE, INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS, BATCH_SIZE, EPOCHS, DATA_DIR, NUM_WORKERS
from dataset import MnistDataModule

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Network (Pytorch Lightning)
class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        # Torch metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    # Train Network (Pytorch Lightning)
    def training_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        acc = self.accuracy(scores, target)
        f1 = self.f1_score(scores, target)
        self.log_dict({"loss": loss, "acc": acc, "f1": f1}, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss, "acc": acc, "f1": f1}

    def validation_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, target = self._common_step(batch, batch_idx)
        acc = self.accuracy(scores, target)
        f1 = self.f1_score(scores, target)
        self.log_dict({"loss": loss, "acc": acc, "f1": f1}, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss, "acc": acc, "f1": f1}

    def _common_step(self, batch, batch_idx):
        data, target = batch
        data = data.reshape(data.shape[0], -1)
        scores = self.forward(data)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scores, target)
        return loss, scores, target

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def predict_step(self, batch, batch_idx):
        data, target = batch
        data = data.reshape(data.shape[0], -1)
        scores = self.forward(data)
        predicted = torch.argmax(scores, 1)
        return predicted


def main():
    # Initialize Logger
    logger = TensorBoardLogger('lightning_logs', name='MNIST')

    # Initialize Network
    model = NN(INPUT_SIZE, NUM_CLASSES, HIDDEN_LAYERS).to(device)

    # DataLoader setup
    dm = MnistDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Initialize trainer
    trainer = pl.Trainer(min_epochs=1, max_epochs=EPOCHS, accelerator='cpu', callbacks=[MyPrintCallback()], logger=logger)
    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == '__main__':
    main()