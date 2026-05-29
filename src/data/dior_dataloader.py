import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt',
    'bridge', 'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
    'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DIOR_CLASSES)}

class DIORDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_size=64):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        if self.split == 'train':
            img_dir = os.path.join(self.root_dir, 'JPEGImages-trainval')
        else:
            img_dir = os.path.join(self.root_dir, 'JPEGImages-test')

        ann_dir = os.path.join(self.root_dir, 'Annotations', 'Horizontal Bounding Boxes')

        if not os.path.exists(img_dir):
            print(f"Image dir not found: {img_dir}")
            return
        if not os.path.exists(ann_dir):
            print(f"Annotation dir not found: {ann_dir}")
            return

        ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
        print(f"Found {len(ann_files)} annotation files")

        loaded = 0
        skipped = 0

        for ann_file in ann_files:
            ann_path = os.path.join(ann_dir, ann_file)
            img_path = os.path.join(img_dir, ann_file.replace('.xml', '.jpg'))

            if not os.path.exists(img_path):
                skipped += 1
                continue

            try:
                tree = ET.parse(ann_path)
                root = tree.getroot()

                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is None:
                        continue
                    cls_name = name_elem.text.strip()
                    if cls_name not in CLASS_TO_IDX:
                        continue

                    bbox = obj.find('bndbox')
                    if bbox is None:
                        continue

                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    self.samples.append({
                        'img_path': img_path,
                        'label': CLASS_TO_IDX[cls_name],
                        'bbox': (xmin, ymin, xmax, ymax),
                        'class_name': cls_name
                    })
                    loaded += 1

            except Exception:
                skipped += 1
                continue

        print(f"DIOR {self.split}: {loaded} samples loaded ({skipped} skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            img = Image.open(sample['img_path']).convert('RGB')
            xmin, ymin, xmax, ymax = sample['bbox']
            img = img.crop((xmin, ymin, xmax, ymax))
            if img.size[0] < 4 or img.size[1] < 4:
                img = Image.new('RGB', (64, 64), (0, 0, 0))
        except Exception:
            img = Image.new('RGB', (64, 64), (0, 0, 0))

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img)

        return img, sample['label']


def get_dior_loaders(root_dir, batch_size=32, img_size=64, num_workers=2):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DIORDataset(root_dir, split='train', transform=transform_train, img_size=img_size)
    test_dataset = DIORDataset(root_dir, split='test', transform=transform_test, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


if __name__ == "__main__":
    print("Testing DIOR dataloader...")
    print(f"Total classes: {len(DIOR_CLASSES)}")

    train_loader, test_loader = get_dior_loaders(root_dir='data/DIOR', batch_size=4, img_size=64, num_workers=0)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    if len(train_loader) > 0:
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Classes: {[DIOR_CLASSES[l] for l in labels]}")
        print("DIOR dataloader working correctly")
