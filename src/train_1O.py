import os
import torch.nn as nn
import torch
import torch.cuda
import torchvision
from torch.utils.data import DataLoader, random_split
from torch import optim
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
from model import AENet
from celeba_dataset import CelebaDataset



def spoof_condition(values, threshold):
    #return (torch.argmax(values, dim=1) == 1) & (values[:, 1] >= threshold)
    return values >= threshold


def train_model(config, data_dir, epochs, checkpoint_dir="checkpoints", last_epoch=None):

    train_files = os.listdir(data_dir)
    device = torch.device("cuda:0")
    model = AENet()
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{last_epoch}.pt")

    if (last_epoch is None) or not os.path.exists(checkpoint_file):
        print("Last epoch file does not exist!")
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
            if i >= 0:
                print(f"package {i}...")
                dataset = CelebaDataset(os.path.join(train_dir, file_name))
                train_size = int(len(dataset) * 0.8)
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
                val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
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
                        temp_val_preds += list(spoof_condition(outputs, 0.3).detach().cpu().numpy())
                        temp_val_targets += list(labels.detach().cpu().numpy())

                for target, pred in zip(temp_val_targets, temp_val_preds):
                    print(f"{target}:{pred}")

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


if __name__ == '__main__':
    print(torchvision.__version__)
    print(torch.__version__)
    print(torch.version.cuda)
    train_dir = "D:/AK/CelebA_Spoof/transformed_dataset/train"
    config = {
        "lr": 1e-6,
        "batch_size": 16,
        "weight_decay": 1e-1
    }
    train_model(config=config, data_dir=train_dir, epochs=1, last_epoch=None)
