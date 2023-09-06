import os
import pickle

import numpy as np
import torch.nn as nn
import torch
import torch.cuda
from ray import tune
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from model import AENet
class CelebaDataset(Dataset):

    def __init__(self, file_path):
        self.file_path = file_path
        print("loading data...")
        with open(self.file_path, 'rb') as f:
            self.images, self.labels = pickle.load(f)
        print(self.images.shape, self.labels.shape)
        print("data loaded.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = torch.Tensor(image).permute(2, 0, 1)
        label = torch.Tensor(label)
        return image, label


def spoof_condition(values, threshold):
    return (torch.argmax(values, dim=1) == 1) & (values[:, 1] >= threshold)


def train_model(config,  checkpoint_dir="checkpoints"):
    data_dir = config["data_dir"]
    epochs = config["epochs"]
    last_epoch = config["last_epoch"]
    train_files = os.listdir(data_dir)
    device = torch.device("cuda:0")
    model = AENet()
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5, 2]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{last_epoch}.pth")

    if (last_epoch is None) or not os.path.exists(checkpoint_file):
        start_epoch = 0
    else:
        checkpoint_data = torch.load(checkpoint_file)
        start_epoch = checkpoint_data["epoch"] + 1
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    precision_by_epoch = []
    recall_by_epoch = []
    losses_by_epoch = []

    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"EPOCH {epoch}")
        val_loss = 0.0
        val_preds = []
        val_targets = []

        for i, file_name in enumerate(train_files):
            if i == 0:
                print(f"package {i}...")
                dataset = CelebaDataset(os.path.join(train_dir, file_name))
                train_size = int(len(dataset) * 0.8)
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"].sample(), shuffle=True)
                val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"].sample(), shuffle=False)

                model.train()
                for images, labels in tqdm(train_dataloader):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                temp_val_loss = 0.0
                temp_val_preds = []
                temp_val_targets = []

                model.eval()
                with torch.no_grad():
                    for images, labels in tqdm(val_dataloader):
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        temp_val_loss += loss.item()
                        temp_val_preds += list(spoof_condition(outputs, 0.6).detach().cpu().numpy())
                        temp_val_targets += list(torch.argmax(labels, dim=1).detach().cpu().numpy())
                print(f"Package val loss: {temp_val_loss}")
                print(f"Package val precision: {precision_score(temp_val_targets, temp_val_preds, average='weighted')}")
                print(f"Package val recall: {recall_score(temp_val_targets, temp_val_preds, average='weighted')}")
                val_loss += temp_val_loss
                val_preds += temp_val_preds
                val_targets += temp_val_targets

        precision = precision_score(val_targets, val_preds, average='weighted')
        recall = recall_score(val_targets, val_preds, average='weighted')
        losses_by_epoch.append(val_loss)
        recall_by_epoch.append(recall)
        precision_by_epoch.append(precision)
        print(f"Epoch val loss: {val_loss}")
        print(f"Epoch val precision: {precision}")
        print(f"Epoch val recall: {recall}")
        torch.save(model, f'model_epoch_{epoch}.pth')

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_precision": precision,
            "val_recall": recall
        }
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
        torch.save(checkpoint_data, checkpoint_path)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{start_epoch+epochs}.pt")
    # tune.report(val_acc=sum(recall_by_epoch)/len(recall_by_epoch), checkpoint=checkpoint_path)
    # session.report(
    #     {
    #         "loss": sum(losses_by_epoch) / len(losses_by_epoch),
    #         "precision": sum(precision_by_epoch) / len(precision_by_epoch),
    #         "recall": sum(recall_by_epoch) / len(recall_by_epoch)},
    #     checkpoint=checkpoint,
    # )


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    train_dir = "D:/AK/CelebA_Spoof/transformed_dataset/train"
    config = {
        "lr": np.random.uniform(1e-6, 1e-5),
        "batch_size": tune.choice([8, 16, 32]),
        "epochs": 1,
        "last_epoch": None,
        "data_dir": train_dir
    }
    result = tune.run(
        partial(train_model, data_dir=train_dir),
        resources_per_trial={"cpu": 0, "gpu": 1},
        config=config,
        scheduler=ASHAScheduler(),
        num_samples=4)
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
