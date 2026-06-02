import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        # logits: [batch, 75, 45]
        # y_batch: [batch, 75]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y_batch.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == y_batch).sum().item()
        total_count += y_batch.numel()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_count

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y_batch.reshape(-1)
            )

            total_loss += loss.item() * X_batch.size(0)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == y_batch).sum().item()
            total_count += y_batch.numel()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_count

    return avg_loss, accuracy