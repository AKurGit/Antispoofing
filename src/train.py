import json
import os
import torch.nn as nn
import torch
import torch.cuda
import torchvision
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from model import AENet
from celeba_dataset import CelebaDataset


def spoof_condition(values, threshold):
    # return (torch.argmax(values, dim=1) == 1) & (values[:, 1] >= threshold)
    return values[:, 1] >= threshold


def train_model(config):
    train_files = os.listdir(config["data_dir"])
    device = torch.device("cuda:0")
    model = AENet()
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    checkpoint_dir = config["checkpoint_path"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_epoch = config["last_epoch"]
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{last_epoch}.pt")

    if (last_epoch < 0 or last_epoch is None) or not os.path.exists(checkpoint_file):
        print("Last epoch file does not exist!")
        start_epoch = 0
    else:
        checkpoint_data = torch.load(checkpoint_file)
        start_epoch = checkpoint_data["epoch"] + 1
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    epochs = config["epochs"]
    for epoch in range(start_epoch, start_epoch + epochs):

        print(f"EPOCH {epoch}")
        model.train()
        for i, file_name in enumerate(train_files):
            print(f"package {i}...")
            dataset = CelebaDataset(os.path.join(train_dir, file_name))
            dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
            for images, labels in tqdm(dataloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                print(f"{outputs} : {labels}")
                loss.backward()
                optimizer.step()
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
        torch.save(checkpoint_data, checkpoint_path)


if __name__ == '__main__':
    print(torchvision.__version__)
    print(torch.__version__)
    print(torch.version.cuda)
    train_dir = "D:/IT/PycharmProjects/Antispoofing/transformed_dataset/train"
    with open('config', 'r') as file:
        config_data = json.load(file)
    print(config_data)
    train_model(config=config_data)
