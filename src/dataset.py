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
from sklearn.model_selection import KFold

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
    """Class to monitor and plot training metrics with comprehensive segmentation metrics."""
    
    def __init__(self, save_dir: str = "training_plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_jaccard': [],
            'val_accuracy': [],
            'val_sensitivity': [],
            'val_specificity': [],
            'val_hausdorff': []
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def update(self, train_loss: float, val_loss: float, val_metrics: dict):
        """Update metrics with comprehensive validation metrics."""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_dice'].append(val_metrics.get('dice', 0.0))
        self.metrics['val_jaccard'].append(val_metrics.get('jaccard', 0.0))
        self.metrics['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
        self.metrics['val_sensitivity'].append(val_metrics.get('sensitivity', 0.0))
        self.metrics['val_specificity'].append(val_metrics.get('specificity', 0.0))
        self.metrics['val_hausdorff'].append(val_metrics.get('hausdorff', 0.0))
    
    def plot_metrics(self):
        """Plot and save comprehensive training metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot losses
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot Dice and Jaccard scores
        axes[0, 1].plot(self.metrics['val_dice'], label='Dice Score', color='green')
        axes[0, 1].plot(self.metrics['val_jaccard'], label='Jaccard Score', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Dice and Jaccard Scores')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot accuracy
        axes[0, 2].plot(self.metrics['val_accuracy'], label='Accuracy', color='purple')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Validation Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Plot sensitivity and specificity
        axes[1, 0].plot(self.metrics['val_sensitivity'], label='Sensitivity', color='cyan')
        axes[1, 0].plot(self.metrics['val_specificity'], label='Specificity', color='magenta')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Sensitivity and Specificity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot Hausdorff distance
        axes[1, 1].plot(self.metrics['val_hausdorff'], label='Hausdorff Distance', color='brown')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].set_title('Hausdorff Distance')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Summary plot with key metrics
        axes[1, 2].plot(self.metrics['val_dice'], label='Dice', color='green', linewidth=2)
        axes[1, 2].plot(self.metrics['val_jaccard'], label='Jaccard', color='orange', linewidth=2)
        axes[1, 2].plot(self.metrics['val_accuracy'], label='Accuracy', color='purple', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Key Metrics Summary')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_metrics_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics(self):
        """Save metrics to CSV file."""
        import pandas as pd
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.save_dir / f'training_metrics_{self.timestamp}.csv', index=False)
    
    def get_best_metrics(self):
        """Get best validation metrics across all epochs."""
        best_metrics = {}
        if self.metrics['val_dice']:
            best_metrics['best_dice'] = max(self.metrics['val_dice'])
            best_metrics['best_jaccard'] = max(self.metrics['val_jaccard'])
            best_metrics['best_accuracy'] = max(self.metrics['val_accuracy'])
            best_metrics['best_sensitivity'] = max(self.metrics['val_sensitivity'])
            best_metrics['best_specificity'] = max(self.metrics['val_specificity'])
            # For Hausdorff distance, lower is better
            hausdorff_values = [h for h in self.metrics['val_hausdorff'] if not (np.isnan(h) or np.isinf(h))]
            if hausdorff_values:
                best_metrics['best_hausdorff'] = min(hausdorff_values)
            else:
                best_metrics['best_hausdorff'] = float('inf')
        return best_metrics

class Dataset_MRI_2D_Fold(Dataset):
    """
    Dataset class for k-fold cross-validation that combines train and test directories
    and provides functionality to get different folds for training and testing.
    """
    def __init__(self, train_image_dir: str, train_label_dir: str,
                 test_image_dir: str, test_label_dir: str,
                 image_transform: Optional[Callable] = None, 
                 mask_transform: Optional[Callable] = None,
                 k_folds: int = 10,
                 verbose: bool = False):
        self.train_image_dir = Path(train_image_dir)
        self.train_label_dir = Path(train_label_dir)
        self.test_image_dir = Path(test_image_dir)
        self.test_label_dir = Path(test_label_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.k_folds = k_folds
        self.verbose = verbose

        # Verify directories exist
        for dir_path, name in [(self.train_image_dir, "Train image"), 
                              (self.train_label_dir, "Train label"),
                              (self.test_image_dir, "Test image"), 
                              (self.test_label_dir, "Test label")]:
            if not dir_path.exists():
                raise ValueError(f"{name} directory does not exist: {dir_path}")

        # Combine all pairs from both train and test directories
        self.all_pairs = self._get_all_pairs()
        if not self.all_pairs:
            raise ValueError("No valid image-label pairs found in train or test directories.")
        
        if self.verbose:
            print(f"Found {len(self.all_pairs)} total valid image-label pairs for k-fold CV.")
        
        # Create k-fold indices
        self.fold_indices = self._create_kfold_indices()

    def _verify_pairs_in_dir(self, image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
        """Find and verify image-label pairs in a specific directory."""
        image_files = sorted([p for p in image_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']])
        label_files = sorted([p for p in label_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']])
        
        pairs = []
        for img_path in image_files:
            for label_path in label_files:
                if img_path.stem in label_path.stem:
                    pairs.append((img_path, label_path))
                    # if self.verbose:
                    #     print(f"Found pair: {img_path.name} -> {label_path.name}")
                    break
        return pairs

    def _get_all_pairs(self) -> List[Tuple[Path, Path]]:
        """Get all image-label pairs from both train and test directories."""
        train_pairs = self._verify_pairs_in_dir(self.train_image_dir, self.train_label_dir)
        test_pairs = self._verify_pairs_in_dir(self.test_image_dir, self.test_label_dir)
        all_pairs = train_pairs + test_pairs
        if self.verbose:
            print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
            print(f"Total pairs: {len(all_pairs)}")
        
        return all_pairs

    def _create_kfold_indices(self) -> List[Tuple[List[int], List[int]]]:
        """Create k-fold train/test indices."""
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        indices = list(range(len(self.all_pairs)))
        fold_indices = []
        for train_idx, test_idx in kfold.split(indices):
            fold_indices.append((train_idx.tolist(), test_idx.tolist()))
        if self.verbose:
            for i, (train_idx, test_idx) in enumerate(fold_indices):
                print(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
        
        return fold_indices

    def get_fold_datasets(self, fold: int) -> Tuple['Dataset_MRI_2D_Fold_Subset', 'Dataset_MRI_2D_Fold_Subset']:
        """
        Get train and test datasets for a specific fold.
        
        Args:
            fold (int): Fold number (0-based)
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if fold >= self.k_folds or fold < 0:
            raise ValueError(f"Fold must be between 0 and {self.k_folds-1}")
        
        train_indices, test_indices = self.fold_indices[fold]
        
        train_dataset = Dataset_MRI_2D_Fold_Subset(self, train_indices)
        test_dataset = Dataset_MRI_2D_Fold_Subset(self, test_indices)
        
        return train_dataset, test_dataset

    def __len__(self) -> int:
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
            print(f"Error: Index {idx} is out of range for the dataset (size: {len(self)}).")
            return

        img_tensor, label_tensor = self[idx]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_tensor.squeeze().cpu().numpy(), cmap='gray')
        ax1.set_title(f'Image (Index {idx})')
        ax1.axis('off')
        ax2.imshow(label_tensor.squeeze().cpu().numpy(), cmap='gray')
        ax2.set_title(f'Label (Index {idx})')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

class Dataset_MRI_2D_Fold_Subset(Dataset):
    """Subset class for k-fold cross-validation."""
    
    def __init__(self, parent_dataset: Dataset_MRI_2D_Fold, indices: List[int]):
        self.parent_dataset = parent_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        parent_idx = self.indices[idx]
        return self.parent_dataset[parent_idx]

