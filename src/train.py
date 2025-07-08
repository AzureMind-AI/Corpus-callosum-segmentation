import torch
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Performs one full training epoch.
    """
    model.train()
    loop = tqdm(loader, leave=True)
    running_loss = 0.0

    for _, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(loader)
    return avg_loss