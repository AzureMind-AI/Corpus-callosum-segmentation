import os
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Import your custom modules
from models import UNet
from dataset import TrainingMonitor
from dataset import Dataset_MRI_2D as Dataset_MRI
from train import train_one_epoch
from eval import evaluate


# --- Configuration ---
CONFIG = {
    "LEARNING_RATE": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 4,
    "NUM_EPOCHS": 30,
    "IMAGE_HEIGHT": 512,
    "IMAGE_WIDTH": 512,
    "IN_CHANNELS": 1,
    "NUM_CLASSES": 1,
    "TRAIN_IMG_DIR": "data/train/images",
    "TRAIN_LBL_DIR": "data/train/labels",
    "TEST_IMG_DIR": "data/test/images",
    "TEST_LBL_DIR":  "data/test/labels",
    # "TRAIN_IMG_DIR": "Train",
    # "TRAIN_LBL_DIR": "Train_Label",
    # "TEST_IMG_DIR": "Test",
    # "TEST_LBL_DIR":  "Test_Label",
    "MODEL_SAVE_PATH": "best_unet_model_nifti.pth",
    "VAL_SPLIT": 0.2,  # 20% of training data for validation
    "PLOT_DIR": "training_plots",
    "USE_MULTI_GPU": True,  # Enable multi-GPU training
    "NUM_WORKERS": 4  # Increased for multi-GPU
}


def setup_multi_gpu():
    """Setup multi-GPU configuration and return device info."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return "cpu", 1, 1
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    if num_gpus > 1 and CONFIG["USE_MULTI_GPU"]:
        print("Multi-GPU training enabled!")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Scale batch size and learning rate for multi-GPU
        gpu_multiplier = num_gpus
        effective_batch_size = CONFIG["BATCH_SIZE"] * gpu_multiplier
        effective_lr = CONFIG["LEARNING_RATE"] * gpu_multiplier
        
        print(f"Scaling batch size from {CONFIG['BATCH_SIZE']} to {effective_batch_size}")
        print(f"Scaling learning rate from {CONFIG['LEARNING_RATE']} to {effective_lr}")
        
        return "cuda", num_gpus, gpu_multiplier
    else:
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        return "cuda", 1, 1


def main(train_=False):
    """Main function to run the training and evaluation pipeline."""
    
    # Setup multi-GPU configuration
    device, num_gpus, gpu_multiplier = setup_multi_gpu()
    
    # Update config based on GPU setup
    effective_batch_size = CONFIG["BATCH_SIZE"] * gpu_multiplier
    effective_lr = CONFIG["LEARNING_RATE"] * gpu_multiplier
    effective_num_workers = CONFIG["NUM_WORKERS"] * gpu_multiplier if num_gpus > 1 else CONFIG["NUM_WORKERS"]
    
    print(f"Using device: {device}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Effective learning rate: {effective_lr}")
    print(f"Number of workers: {effective_num_workers}")

    # Create transforms
    image_transform = transforms.Compose([
        transforms.Resize((CONFIG["IMAGE_HEIGHT"], CONFIG["IMAGE_WIDTH"]), antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((CONFIG["IMAGE_HEIGHT"], CONFIG["IMAGE_WIDTH"]), antialias=True),
    ])

    # --- Data Loading ---
    print("Loading data...")
    # Create main dataset and split into train/val
    master_dataset = Dataset_MRI(
        image_dir=CONFIG["TRAIN_IMG_DIR"], 
        label_dir=CONFIG["TRAIN_LBL_DIR"],
        image_transform=image_transform,
        mask_transform=mask_transform,
        val_split=CONFIG["VAL_SPLIT"],
        verbose=True
    )
    
    # Get train and val datasets
    train_dataset = master_dataset.get_train_dataset()
    val_dataset = master_dataset.get_val_dataset()
    first_train_index_in_master = train_dataset.indices[0]
    train_dataset.dataset.visualize_sample(first_train_index_in_master)
    
    # Create data loaders with scaled batch size and workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=effective_batch_size, 
        shuffle=True, 
        num_workers=effective_num_workers,
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=effective_batch_size, 
        shuffle=False, 
        num_workers=effective_num_workers,
        pin_memory=True if device == "cuda" else False
    )

    # Create test dataset
    test_dataset = Dataset_MRI(
        image_dir=CONFIG["TEST_IMG_DIR"], 
        label_dir=CONFIG["TEST_LBL_DIR"],
        image_transform=image_transform,
        mask_transform=mask_transform,
        verbose=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=effective_batch_size, 
        shuffle=False, 
        num_workers=effective_num_workers,
        pin_memory=True if device == "cuda" else False
    )

    # Visualize a sample from the training set
    print("Visualizing a sample from the dataset...")
    # train_dataset.visualize_sample(5)

    # --- Model, Loss, Optimizer ---
    print("Initializing model...")
    model = UNet(in_channels=CONFIG["IN_CHANNELS"], n_classes=CONFIG["NUM_CLASSES"])
    
    # Move model to GPU before wrapping with DataParallel
    if device == "cuda":
        model = model.cuda()
        
        # Wrap model with DataParallel for multi-GPU training
        if num_gpus > 1 and CONFIG["USE_MULTI_GPU"]:
            print(f"Wrapping model with DataParallel for {num_gpus} GPUs")
            model = torch.nn.DataParallel(model)
            print(f"Model is now using GPUs: {list(range(num_gpus))}")
    
    # Loss function and optimizer with scaled learning rate
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=effective_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

    # Initialize training monitor
    monitor = TrainingMonitor(save_dir=CONFIG["PLOT_DIR"])
    
    if train_:
        # --- Training Loop ---
        best_val_dice = 0.0
        for epoch in range(CONFIG["NUM_EPOCHS"]):
            print(f"\n--- Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} ---")
            
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            print(f"Average Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_dice = evaluate(model, val_loader, loss_fn, device)
            print(f"Average Validation Loss: {val_loss:.4f}, Average Dice Score: {val_dice:.4f}")

            # Update learning rate
            scheduler.step(val_loss)

            # Update monitoring metrics
            monitor.update(train_loss, val_loss, val_dice)
            
            # Save best model (save the underlying model if using DataParallel)
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), CONFIG["MODEL_SAVE_PATH"])
                print(f"Model saved to {CONFIG['MODEL_SAVE_PATH']} (Dice: {val_dice:.4f})")

        # Plot and save training metrics
        monitor.plot_metrics()
        monitor.save_metrics()
        print(f"\nTraining complete. Best validation Dice score: {best_val_dice:.4f}")

    # --- Final Test Evaluation ---
    print("\n--- Final Test Evaluation ---")
    # Load the best model for final evaluation
    print(f"Loading best model from {CONFIG['MODEL_SAVE_PATH']}...")
    
    # Create a clean model for loading (without DataParallel wrapper)
    eval_model = UNet(in_channels=CONFIG["IN_CHANNELS"], n_classes=CONFIG["NUM_CLASSES"])
    eval_model.load_state_dict(torch.load(CONFIG["MODEL_SAVE_PATH"], map_location=device))
    
    if device == "cuda":
        eval_model = eval_model.cuda()
        # Wrap with DataParallel for evaluation if multiple GPUs available
        if num_gpus > 1 and CONFIG["USE_MULTI_GPU"]:
            eval_model = torch.nn.DataParallel(eval_model)
    
    eval_model.eval()  # Set model to evaluation mode
    
    test_loss, test_dice = evaluate(eval_model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Dice Score: {test_dice:.4f}")


if __name__ == "__main__":
    main(train_=False)
                
