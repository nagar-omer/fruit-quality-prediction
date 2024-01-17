from torch import nn, optim
import torch
import lightning as L
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_loader import FruitsDataset, fruit_collate_batch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix



class SelfAttentionOverCNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(SelfAttentionOverCNN, self).__init__()
        self._csl_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # cnn backbone
        self._backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm([32, 112, 112]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm([64, 56, 56]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm([128, 28, 28]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm([256, 14, 14]),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm([512, 7, 7]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # self attention
        self._self_attention = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)

        # MLP for classification
        self._mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    @property
    def optimizer(self):
        return self._optimizer

    def forward(self, x):

        out = []
        for sequence in x:
            # pass through backbone + self attention
            embs = self._backbone(sequence)
            embs = torch.vstack([embs, self._csl_token.squeeze(0)])
            embs, _ = self._self_attention(embs, embs, embs)

            # take cls token only
            out.append(embs[0])

        # pass through MLP
        x = self._mlp(torch.stack(out))
        # softmax
        x = nn.functional.softmax(x, dim=-1)
        return x

    def _set_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# define the LightningModule
class LitClassifier(L.LightningModule):
    def __init__(self, gpu=True):
        super().__init__()
        self._train_loss, self._val_loss = [], []

        self._criteria = nn.CrossEntropyLoss()

        # Choose a specific version of CLIP, e.g., "openai/clip-vit-base-patch32"
        self._classifier = SelfAttentionOverCNN(hidden_size=128, output_size=4)

        mps_available = torch.backends.mps.is_available()
        cuda_available = torch.cuda.is_available()
        self._device_to_use = "cpu"
        if mps_available and gpu:
            self._device_to_use = "mps"
            self._classifier = self._classifier.to("mps")
        elif cuda_available and gpu:
            self._device_to_use = "cuda"
            self._classifier = self._classifier.cuda()
        else:
            self._classifier = self._classifier.cpu()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, y_true = batch

        if self._device_to_use == "mps":
            images = [im.to("mps") for im in images]
        elif self._device_to_use == "cuda":
            images = [im.cuda() for im in images]

        y_hat = self._classifier(images)
        loss = self._criteria(y_hat, torch.max(y_true, 1)[1])

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', self._accuracy(y_hat, y_true), on_epoch=True)

        return loss

    def _accuracy(self, y_hat, y_true):
        return torch.sum(torch.max(y_hat, 1)[1] == torch.max(y_true, 1)[1]).item() / len(y_hat)

    def validation_step(self, batch, batch_idx):
        # Validation step logic here
        images, y_true = batch

        if self._device_to_use == "mps":
            images = [im.to("mps") for im in images]
        elif self._device_to_use == "cuda":
            images = [im.cuda() for im in images]

        y_hat = self._classifier(images)
        loss = self._criteria(y_hat, torch.max(y_true, 1)[1])

        self.log('val_loss', loss, on_epoch=True)  # Log loss for the entire validation set
        self.log('val_accuracy', self._accuracy(y_hat, y_true), on_epoch=True)  # Log accuracy for the entire validation set

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics['train_loss'].item()
        train_accuracy = self.trainer.callback_metrics['train_accuracy'].item()
        print(f'\n\nEpoch [{self.current_epoch + 1}/{self.trainer.max_epochs}] - '
              f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}\n')
        self._train_loss.append(train_loss)

    def on_validation_epoch_end(self):
        # Manually print validation loss and accuracy at the end of each epoch
        val_loss = self.trainer.callback_metrics['val_loss'].item()
        val_accuracy = self.trainer.callback_metrics['val_accuracy'].item()
        print(f'\n\nEpoch [{self.current_epoch + 1}/{self.trainer.max_epochs}] - '
              f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        self._val_loss.append(val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-3)
        return optimizer


def train():
    ds = FruitsDataset(path="/Users/omernagar/Documents/Projects/fruit-quality-prediction/data")

    # split to train and test
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

    dl_train = DataLoader(train_ds, num_workers=4, batch_size=16, shuffle=True, collate_fn=fruit_collate_batch, persistent_workers=True)
    dl_test = DataLoader(test_ds, num_workers=4, batch_size=64, shuffle=False, collate_fn=fruit_collate_batch, persistent_workers=True)

    trainer = L.Trainer(max_epochs=100)
    model = LitClassifier(gpu=False)
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_test)

    # save model
    torch.save(model.state_dict(), "model.pt")

    # plot loss
    plt.plot(model._train_loss, label="train")
    plt.plot(model._val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("CE-Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.show()

    plot_conf(test_ds)


def plot_conf(ds):
    model = LitClassifier(gpu=False)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    dl_test = DataLoader(ds, num_workers=4, batch_size=64, shuffle=False, collate_fn=fruit_collate_batch, persistent_workers=True)

    y_hat = []
    y_true = []
    for x, y in tqdm(dl_test, total=len(dl_test)):
        model._classifier(x)
        y_hat.append(model._classifier(x).detach().numpy())
        y_true.append(y.detach().numpy())

    y_hat = np.vstack(y_hat)
    y_true = np.vstack(y_true)

    print(f"Test accuracy: {(y_hat.argmax(axis=1) == y_true.argmax(axis=1)).sum() / y_true.shape[0]}")

    confusion_matrix_test = confusion_matrix(y_true.argmax(axis=1), y_hat.argmax(axis=1), normalize="true")
    print(confusion_matrix_test)
    plt.imshow(confusion_matrix_test, cmap="Blues")
    plt.xticks(np.arange(4), ["Pasul", "A", "AA", "Muvhar"])
    plt.yticks(np.arange(4), ["Pasul", "A", "AA", "Muvhar"])
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


if __name__ == '__main__':
    train()

