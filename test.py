import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import cv2
import os
import torchvision
from tqdm import tqdm


plt.switch_backend('Agg')


# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

# Hyperparameters
batch_size = 10
w = int(((960) // 14) * 14)   # Full resolution (no /2)
h = int(((540) // 14) * 14)   # Full resolution (no /2)
lr = 3e-4   # Updated for AdamW
n_epochs = 10

# Transforms
transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((h, w), interpolation=InterpolationMode.NEAREST),
    transforms.PILToTensor()   # keeps raw uint8 values — no scaling
])

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask).long()

        return image, mask, data_id

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class), dice_per_class

def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []

    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    with torch.no_grad():
        for batch in loader:
            # Handle both 2-tuple and 3-tuple (with data_ids)
            imgs, labels = batch[0], batch[1]
            imgs, labels = imgs.to(device), labels.to(device)

            output = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            iou, _ = compute_iou(outputs, labels, num_classes=num_classes)
            dice, _ = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)

    model.train()
    return np.nanmean(iou_scores), np.nanmean(dice_scores), np.mean(pixel_accuracies)

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of input, ground truth, and prediction."""
    # Denormalize image
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # Convert masks to color
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: Combined metrics plot
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train')
    plt.plot(history['val_iou'], label='val')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train')
    plt.plot(history['val_dice'], label='val')
    plt.title('Dice Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics_train_val.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i]
            ))

    print(f"Saved evaluation metrics to {filepath}")


def save_metrics_summary(results, output_dir):
    """Save metrics summary to a text file and create bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    # Save text summary
    filepath = os.path.join(output_dir, 'evaluation_metrics_test.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for i, (name, iou) in enumerate(zip(class_names, results['class_iou'])):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")

    print(f"\nSaved evaluation metrics to {filepath}")

    # Create bar chart for per-class IoU
    fig, ax = plt.subplots(figsize=(10, 6))

    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    ax.bar(range(n_classes), valid_iou, color=[color_palette[i] / 255 for i in range(n_classes)],
           edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', label='Mean')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")

def dice_loss(pred, target, num_classes=10, smooth=1e-6):
    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()


def compute_class_weights(data_loader, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, labels, _ in tqdm(data_loader, desc="Computing class weights", leave=False):
        labels = labels.view(-1)
        counts += torch.bincount(labels, minlength=num_classes)
    # Inverse frequency weighting
    freq = counts.float() / counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * num_classes  # normalize so mean weight = 1
    print("Class counts:", counts.tolist())
    print("Class weights:", weights.tolist())
    return weights.to(device)

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load DINOv2 backbone
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    class SegmentationHeadConvNeXt(nn.Module):
        def __init__(self, in_channels, out_channels, tokenW, tokenH):
            super().__init__()
            self.H, self.W = tokenH, tokenW

            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
                nn.GELU()
            )

            # FIX: Deeper block with bottleneck for more capacity
            self.block = nn.Sequential(
                nn.Conv2d(128, 128, 7, padding=3, groups=128),
                nn.GELU(),
                nn.Conv2d(128, 256, 1),
                nn.GELU(),
                nn.Conv2d(256, 128, 1),
                nn.GELU(),
            )

            self.classifier = nn.Conv2d(128, out_channels, 1)

        def forward(self, x):
            B, N, C = x.shape # (batch size, number of patch tokens, embedding dimension) --------> (B,10, H, W)
            x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
            x = self.stem(x)
            x = self.block(x)
            return self.classifier(x)
      

    base_dir = "/kaggle/working"
    # Create your custom output folder
    output_dir_test = os.path.join(base_dir, "test_stats")
    # Create the directory if it doesn't exist
    os.makedirs(output_dir_test, exist_ok=True)

    print("Output directory:", output_dir_test)

    test_dir = '/kaggle/input/datasets/noobcoder27/offroad-segmentation-testimages/Offroad_Segmentation_testImages';

    batch_size = 2
    num_samples = 5

    # Load the already downloaded Segmentation_head.pth file from /kaggle/output -> /kaggle/input folder and copy the relative file path 
    model_path = '/kaggle/input/models/noobcoder27/dinov2-segmentation-head/pytorch/default/1/segmentation_head.pth'

    # Create dataset
    print(f"Loading dataset from {test_dir}...")
    valset = MaskDataset(data_dir=test_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    print(f"Loaded {len(valset)} samples")

    # Load classifier
    print(f"Loading model from {model_path}...")

    # Get embedding dimension from backbone
    imgs, _, _ = next(iter(val_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Patch tokens shape: {output.shape}")
            
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier = classifier.to(device) 

    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier_model = classifier.to(device)
    classifier_model.eval()
    print("Model loaded successfully!")

    masks_dir = os.path.join(output_dir_test, 'masks')
    masks_color_dir = os.path.join(output_dir_test, 'masks_color')
    comparisons_dir = os.path.join(output_dir_test, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Run evaluation and save predictions for ALL images
    print(f"\nRunning evaluation and saving predictions for all {len(valset)} images...")

    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_iou = []
    all_class_dice = []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Processing", unit="batch")
        for batch_idx, (imgs, labels, data_ids) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass
            output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier_model(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels_squeezed = labels.squeeze(dim=1).long()
            predicted_masks = torch.argmax(outputs, dim=1)

            # Calculate metrics
            iou, class_iou = compute_iou(outputs, labels_squeezed, num_classes=n_classes)
            dice, class_dice = compute_dice(outputs, labels_squeezed, num_classes=n_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels_squeezed)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_iou.append(class_iou)
            all_class_dice.append(class_dice)

            # Save predictions for every image
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                # Save raw prediction mask (class IDs 0-9)
                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                pred_img = Image.fromarray(pred_mask)
                pred_img.save(os.path.join(masks_dir, f'{base_name}_pred.png'))

                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Save comparison visualization for first N samples
                if sample_count < num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_squeezed[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count}_comparison.png'),
                        data_id
                    )

                sample_count += 1

            # Update progress bar with metrics
            pbar.set_postfix(iou=f"{iou:.3f}")

    # Aggregate results
    mean_iou = np.nanmean(iou_scores)
    mean_dice = np.nanmean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)

    # Average per-class metrics
    avg_class_iou = np.nanmean(all_class_iou, axis=0)
    avg_class_dice = np.nanmean(all_class_dice, axis=0)

    results = {
        'mean_iou': mean_iou,
        'class_iou': avg_class_iou
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU:          {mean_iou:.4f}")
    print(f"Mean Dice:         {mean_dice:.4f}")
    print(f"Mean Pixel Acc:    {mean_pixel_acc:.4f}")
    print("=" * 50)

    # FIX 9: Print per-class IoU
    print("\nPer-Class IoU:")
    print("-" * 40)
    for i, (name, iou_val) in enumerate(zip(class_names, avg_class_iou)):
        iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "N/A"
        print(f"  {name:<20}: {iou_str}")
    print("-" * 40)

    # Save all results
    save_metrics_summary(results, output_dir_test)

    print(f"\nPrediction complete! Processed {len(valset)} images.")
    print(f"\nOutputs saved to {output_dir_test}/")
    print(f"  - masks/           : Raw prediction masks (class IDs 0-9)")
    print(f"  - masks_color/     : Colored prediction masks (RGB)")
    print(f"  - comparisons/     : Side-by-side comparison images ({num_samples} samples)")
    print(f"  - evaluation_metrics.txt")
    print(f"  - per_class_metrics.png")



if __name__ == "__main__":
    main()