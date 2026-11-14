from torchvision import transforms
from dataloader import DeepfakeDataset
from pipeline import DeepfakeDetectionPipeline

TRAIN_DIR = "./dataset/train"
TEST_DIR = "./dataset/test"
BATCH_SIZE = 32
VAL_SPLIT = 0.2
NUM_WORKERS = 4  # parallel image decoding + augmentation; 0 = a single CPU core (slow)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    ], p=0.5),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
    ], p=0.2),
    # Degradations present in the test set but rare in train: pixelation, greyscale, posterise, false colour, block occlusion
    transforms.RandomApply([transforms.RandomChoice([
        transforms.Compose([transforms.Resize(s), transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST)])
        for s in (32, 48, 64)
    ])], p=0.15),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomPosterize(bits=2, p=0.1),
    transforms.RandomSolarize(threshold=128, p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), value="random"),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# The guard is needed for num_workers > 0: on Windows/macOS each data-loading worker re-imports this file,
# and without it every worker would start training again.
if __name__ == "__main__":
    train_dataset = DeepfakeDataset(
        TRAIN_DIR,
        isTrain=True,
        transform=train_transforms,
        two_views=True
    )

    test_dataset = DeepfakeDataset(
        TEST_DIR,
        isTrain=False,
        transform=eval_transforms
    )

    train_loader, val_loader = DeepfakeDataset.load_dataloader(
        train_dataset, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, val_transform=eval_transforms,
        num_workers=NUM_WORKERS
    )
    test_loader = DeepfakeDataset.load_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Three backbones, same recipe: LoRA fine-tuning, BCE + consistency (1.0) + supervised contrastive (0.1).
    # consistency_weight: both augmented views of an image must get the same fake-probability
    # supcon_weight: supervised contrastive loss on a projection head (set either to 0 to ablate)
    # Each run saves ./models/{arch}_best.pt and the merged inference weights ./models/{arch}_weights.pt;
    # inference_script.py averages the three models.
    ARCHS = ["efficientnet_b4", "convnext_tiny", "swin_t"]
    TRAIN_KWARGS = dict(lr=3e-4, use_lora=True, consistency_weight=1.0, supcon_weight=0.1)

    for arch in ARCHS:
        pipeline = DeepfakeDetectionPipeline("./dataset", arch=arch, **TRAIN_KWARGS)
        pipeline.fit(
            train_loader, val_loader, epochs=20, save_path=f"./models/{arch}_best.pt"
        )
        pipeline.load(f"./models/{arch}_best.pt")
        pipeline.model.export_merged(f"./models/{arch}_weights.pt")
        del pipeline
