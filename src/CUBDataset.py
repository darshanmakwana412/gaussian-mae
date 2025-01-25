import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class CUBDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        super().__init__()
        self.root_dir = root_dir

        self.transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        self.classes = sorted(os.listdir(self.root_dir))
        
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append(
                        (os.path.join(class_folder, fname), class_idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label

if __name__ == "__main__":
    dataset = CUBDataset(root_dir="datasets/CUB_200_2011/images", img_size=256)
    print("Total samples:", len(dataset))

    img, label = dataset[0]
    print("Image shape:", img.shape)  # [3, 224, 224]
    print("Pixel range ~:", (img.min().item(), img.max().item()))  # Should be around [-1, 1]
    print("Label:", label)