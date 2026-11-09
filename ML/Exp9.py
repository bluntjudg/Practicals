import os
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main():
    torch.manual_seed(42)
    dataset_directory = "hymenoptera_data"
    archive_path = "hymenoptera_data.zip"
    dataset_url = (
        "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    )

    if not os.path.isdir(dataset_directory):
        urllib.request.urlretrieve(dataset_url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(".")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_data = datasets.ImageFolder(
        os.path.join(dataset_directory, "train"),
        train_transform,
    )
    test_data = datasets.ImageFolder(
        os.path.join(dataset_directory, "val"),
        test_transform,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    print("Classes:", train_data.classes)
    print("Training images:", len(train_data))
    print("Testing images:", len(test_data))

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
    )
    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(device))
                correct += (outputs.argmax(dim=1).cpu() == labels).sum().item()
        accuracy = 100 * correct / len(test_data)
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Train loss: {total_loss / len(train_loader):.4f} "
            f"Test accuracy: {accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()
