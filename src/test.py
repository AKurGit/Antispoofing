import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score, precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AENet
from celeba_dataset import CelebaDataset
import torch.nn as nn


def spoof_condition(values, threshold):
    #return (torch.argmax(values, dim=1) == 1) & (values[:, 1] >= threshold)
    value_list = values[:, 1].detach().cpu().numpy().tolist()
    return [int(value >= threshold) for value in value_list]


if __name__ == '__main__':
    test_dir = "../transformed_dataset/test"
    test_files = os.listdir(test_dir)
    checkpoint_dir = "../checkpoints"
    checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda x: int(x.split('_')[1][:-3]))
    checkpoints = checkpoints[0:1]
    print(checkpoints)

    device = torch.device("cuda:0")
    print(torch.version.cuda)

    precisions = []
    recalls = []
    losses = []

    criterion = nn.BCELoss()

    for checkpoint_file in checkpoints:
        print(checkpoint_file)
        model = AENet()
        model = nn.DataParallel(model)
        model.to(device)
        checkpoint_data = torch.load(os.path.join(checkpoint_dir, checkpoint_file))
        model.load_state_dict(checkpoint_data["model_state_dict"])
        preds = []
        targets = []
        loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, file_name in enumerate(test_files):
                print(f"testing file {i}...")
                test_dataset = CelebaDataset(os.path.join(test_dir, file_name))
                test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
                for images, labels in tqdm(test_dataloader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss += criterion(outputs, labels).item()
                    preds += list(spoof_condition(outputs, 0.5))
                    targets += list(torch.argmax(labels, dim=1).detach().cpu().numpy())
        for pred, targ in zip(preds, targets):
            print(f"{pred}:{targ}")
        accuracy = (len(preds) - sum([abs(x - y) for x, y in zip(preds, targets)]))/len(preds)
        print(accuracy)
        precisions.append(precision_score(targets, preds))
        recalls.append(recall_score(targets, preds))
        losses.append(loss)

    plt.subplot(1, 2, 1)
    plt.plot(range(len(checkpoints)), precisions, label='precision')
    plt.plot(range(len(checkpoints)), recalls, label='recall')
    plt.xlabel('"Эпоха"')
    plt.ylabel('Точность')
    plt.title('График точности')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(checkpoints)), losses, label='loss')
    plt.xlabel('"Эпоха"')
    plt.ylabel('Потери')
    plt.title('График потерь')

    print(f"Final precision: {precisions[-1]}")
    print(f"Final recall: {recalls[-1]}")
    print(f"Final loss: {losses[-1]}")
    plt.show()
