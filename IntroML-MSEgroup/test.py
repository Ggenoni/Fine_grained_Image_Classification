import torch
from tqdm import tqdm

def test_model(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for _, (inputs, labels) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)

            running_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(val_loader)
    return avg_loss, accuracy
