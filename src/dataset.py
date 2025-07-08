import os
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import numpy as np
from datetime import datetime
from typing import List, Tuple, Callable, Optional 
from torch.utils.data import Dataset, random_split, Subset
from PIL import Image

class Dataset_MRI(Dataset):
    """Dataset class for handling 4D NIfTI MRI images and their 3D labels."""
    
    def __init__(self, image_dir: str, label_dir: str, image_transform=None, mask_transform=None, 
                 val_split: float = 0.2, verbose: bool = False):
        """
        Initialize MRI dataset.
        
        Args:
            image_dir (str): Directory containing 4D NIfTI images
            label_dir (str): Directory containing 3D NIfTI labels
            image_transform (callable, optional): Transform to be applied on images
            mask_transform (callable, optional): Transform to be applied on masks
            val_split (float): Fraction of data to use for validation (default: 0.2)
            verbose (bool): Whether to print detailed information during initialization
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.verbose = verbose
        self.val_split = val_split
        
        # Verify directories exist
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")
        if not self.label_dir.exists():
            raise ValueError(f"Label directory does not exist: {label_dir}")
            
        if self.verbose:
            print(f"Loading images from: {self.image_dir}")
            print(f"Loading labels from: {self.label_dir}")
        
        self.image_files = self._get_image_files(image_dir)
        if self.verbose:
            print(f"Found {len(self.image_files)} image files")
        
        self.label_files = self._get_label_files(label_dir)
        if self.verbose:
            print(f"Found {len(self.label_files)} label files")
        
        # Verify pairs
        self.valid_pairs = self._verify_pairs()
        if not self.valid_pairs:
            raise ValueError("No valid image-label pairs found. Please check file names and extensions.")
        if self.verbose:
            print(f"Found {len(self.valid_pairs)} valid image-label pairs")
            
        # Create train/val split
        self._create_splits()
        
    def _create_splits(self):
        """Create train/val splits from the dataset."""
        total_size = len(self.valid_pairs)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size
        
        # Create splits
        self.train_pairs, self.val_pairs = random_split(
            self.valid_pairs, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        if self.verbose:
            print(f"Created train/val split: {train_size}/{val_size} samples")
    
    def get_train_dataset(self) -> 'Dataset_MRI':
        """Get training dataset."""
        train_dataset = Dataset_MRI(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
            verbose=self.verbose
        )
        train_dataset.valid_pairs = [self.valid_pairs[i] for i in self.train_pairs.indices]
        return train_dataset
    
    def get_val_dataset(self) -> 'Dataset_MRI':
        """Get validation dataset."""
        val_dataset = Dataset_MRI(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
            verbose=self.verbose
        )
        val_dataset.valid_pairs = [self.valid_pairs[i] for i in self.val_pairs.indices]
        return val_dataset
    
    def _get_image_files(self, path: str) -> List[Path]:
        """Get all NIfTI files in a directory."""
        nifti_extensions = {'.nii', '.nii.gz', '.hdr', '.img'}
        files = []
        for ext in nifti_extensions:
            files.extend(list(Path(path).glob(f'**/*{ext}')))
        if self.verbose:
            print(f"Found image files: {[f.name for f in files]}")
        return files
    
    def _get_label_files(self, path: str) -> List[Path]:
        """Get all NIfTI label files in a directory."""
        nifti_extensions = {'.nii', '.nii.gz', '.hdr', '.img'}
        files = []
        for ext in nifti_extensions:
            files.extend(list(Path(path).glob(f'**/*{ext}')))
        if self.verbose:
            print(f"Found label files: {[f.name for f in files]}")
        return files
    
    def _verify_pairs(self) -> List[Tuple[Path, Path]]:
        """Verify and create image-label pairs."""
        pairs = []
        for img_path in self.image_files:
            for label_path in self.label_files:
                if img_path.stem in label_path.stem:
                    pairs.append((img_path, label_path))
                    if self.verbose:
                        print(f"Matched: {img_path.name} -> {label_path.name}")
                    break
        return pairs
    
    def __len__(self) -> int:
        """Return the number of valid image-label pairs."""
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single image-label pair.
        
        Args:
            idx (int): Index of the pair to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and label tensors
        """
        img_path, label_path = self.valid_pairs[idx]
        
        try:
            # Load image and extract middle slice (12th slice)
            nifti_img = nib.load(img_path)
            img_data = nifti_img.get_fdata()
            if img_data.shape[2] < 13:  # Check if we have enough slices
                raise ValueError(f"Image {img_path.name} has only {img_data.shape[2]} slices, need at least 13")
            try:
                img_slice = img_data[:, :, 10, 0]  # Extract middle slice from 4D volume
            except:
                img_slice = img_data[:, :, 10]     # Extract middle slice from 3D volume
            # Load label and extract middle slice
            nifti_label = nib.load(label_path)
            label_data = nifti_label.get_fdata()
            if label_data.shape[2] < 13:  # Check if we have enough slices
                raise ValueError(f"Label {label_path.name} has only {label_data.shape[2]} slices, need at least 13")
            label_slice = label_data[:, :, 10]  # Extract middle slice from 3D volume
            
            # Convert to torch tensors
            img_tensor = torch.from_numpy(img_slice).float()
            label_tensor = torch.from_numpy(label_slice).float()
            
            # Add channel dimension
            img_tensor = img_tensor.unsqueeze(0)  # [1, 512, 512]
            label_tensor = label_tensor.unsqueeze(0)  # [1, 512, 512]
            
            if self.image_transform:
                img_tensor = self.image_transform(img_tensor)
            if self.mask_transform:
                label_tensor = self.mask_transform(label_tensor)
                
            return img_tensor, label_tensor
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading pair {idx}: {str(e)}")
                print(f"Image path: {img_path}")
                print(f"Label path: {label_path}")
            raise
    
    def visualize_sample(self, idx: int) -> None:
        """
        Visualize a sample image-label pair.
        
        Args:
            idx (int): Index of the pair to visualize
        """
        img_tensor, label_tensor = self[idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(img_tensor.squeeze().numpy(), cmap='gray')
        ax1.set_title('Image')
        ax1.axis('off')
        
        ax2.imshow(label_tensor.squeeze().numpy(), cmap='gray')
        ax2.set_title('Label')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

class Dataset_MRI_2D(Dataset):
    """
    Dataset class for handling 2D MRI images and their corresponding 2D labels,
    with built-in support for creating a train/validation split.
    """
    def __init__(self, image_dir: str, label_dir: str, 
                 image_transform: Optional[Callable] = None, 
                 mask_transform: Optional[Callable] = None,
                 val_split: float = 0.0,
                 verbose: bool = False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.val_split_ratio = val_split
        self.verbose = verbose

        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")
        if not self.label_dir.exists():
            raise ValueError(f"Label directory does not exist: {label_dir}")

        # Find and verify all possible image-label pairs
        self.all_pairs = self._verify_pairs()
        if not self.all_pairs:
            raise ValueError("No valid image-label pairs found. Check file names.")
        if self.verbose:
            print(f"Found {len(self.all_pairs)} total valid image-label pairs.")

        self._train_subset: Optional[Subset] = None
        self._val_subset: Optional[Subset] = None

        # Create splits if requested. This will populate the _train_subset and _val_subset
        if self.val_split_ratio > 0:
            self._create_splits()

    def _verify_pairs(self) -> List[Tuple[Path, Path]]:
        # Using .lower() for suffixes makes matching case-insensitive (e.g., .PNG vs .png)
        image_files = sorted([p for p in self.image_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']])
        label_files = sorted([p for p in self.label_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']])
        
        pairs = []
        
        for img_path in image_files:
            for label_path in label_files:
                if img_path.stem in label_path.stem:
                    pairs.append((img_path, label_path))
        return pairs

    def _create_splits(self):
        total_size = len(self)
        if total_size < 2:
             if self.verbose: print("Warning: Cannot create a validation split with fewer than 2 samples.")
             return

        val_size = int(total_size * self.val_split_ratio)
        if val_size == 0 and self.val_split_ratio > 0:
            val_size = 1 # Ensure val set is not empty if a split is requested
        
        train_size = total_size - val_size
        
        if train_size == 0 or val_size == 0:
             if self.verbose: print(f"Warning: With {total_size} samples, one split is empty. Adjust val_split or add more data.")
             return

        # random_split returns Subset objects
        self._train_subset, self._val_subset = random_split(
            self, # Split the dataset instance itself
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        if self.verbose:
            print(f"Created train/val split: {len(self._train_subset)}/{len(self._val_subset)} samples")

    def get_train_dataset(self) -> Subset:
        """Returns the training subset. Throws error if split wasn't made."""
        if self._train_subset is None:
             raise RuntimeError("get_train_dataset() called but no validation split was created.")
        return self._train_subset

    def get_val_dataset(self) -> Subset:
        """Returns the validation subset. Throws error if split wasn't made."""
        if self._val_subset is None:
             raise RuntimeError("get_val_dataset() called but no validation split was created.")
        return self._val_subset

    def __len__(self) -> int:
        # The length is always the total number of pairs. Subsets handle their own length.
        return len(self.all_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.all_pairs[idx]
        
        image = Image.open(img_path).convert("L")
        label = Image.open(label_path).convert("L")

        img_array = np.array(image, dtype=np.float32) / 255.0
        label_array = np.array(label, dtype=np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        label_tensor = torch.from_numpy(label_array).unsqueeze(0)
        
        if self.image_transform:
            img_tensor = self.image_transform(img_tensor)
        if self.mask_transform:
            label_tensor = self.mask_transform(label_tensor)
            
        return img_tensor, label_tensor

    def visualize_sample(self, idx: int):
        """Visualizes a single sample from the full dataset given its index."""
        if idx >= len(self):
            print(f"Error: Index {idx} is out of range for the master dataset (size: {len(self)}).")
            return

        img_tensor, label_tensor = self[idx]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_tensor.squeeze().cpu().numpy(), cmap='gray')
        ax1.set_title(f'Image (Master Index {idx})')
        ax1.axis('off')
        ax2.imshow(label_tensor.squeeze().cpu().numpy(), cmap='gray')
        ax2.set_title(f'Label (Master Index {idx})')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

class TrainingMonitor:
    """Class to monitor and plot training metrics."""
    
    def __init__(self, save_dir: str = "training_plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': []
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def update(self, train_loss: float, val_loss: float, val_dice: float):
        """Update metrics."""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_dice'].append(val_dice)
    
    def plot_metrics(self):
        """Plot and save training metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.metrics['train_loss'], label='Train Loss')
        ax1.plot(self.metrics['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot dice score
        ax2.plot(self.metrics['val_dice'], label='Val Dice Score', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Validation Dice Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_metrics_{self.timestamp}.png')
        plt.close()
    
    def save_metrics(self):
        """Save metrics to CSV file."""
        import pandas as pd
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.save_dir / f'training_metrics_{self.timestamp}.csv', index=False)

