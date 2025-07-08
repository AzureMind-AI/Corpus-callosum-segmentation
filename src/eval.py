import torch
from tqdm import tqdm

def dice_score(preds, targets, smooth=1e-6):
    """Calculates Dice score for binary segmentation."""
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def evaluate(model, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    val_loss = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        loop = tqdm(loader, leave=True)
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)
            
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()
            
            dice = dice_score(predictions, targets)
            total_dice += dice
            loop.set_postfix(val_loss=loss.item(), dice=dice)

    avg_loss = val_loss / len(loader)
    avg_dice = total_dice / len(loader)
    
    return avg_loss, avg_dice