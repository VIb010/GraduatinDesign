import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class YOLODetectionDataset(Dataset):
    """
    将每张原图和对应的 YOLO .txt（每行 'class x_center y_center w h'）组合，
    裁剪出每个检测框的小图，并返回 (crop_tensor, class_id)。
    """
    def __init__(self, images_dir, labels_dir, transform=None):
        self.transform = transform
        self.samples = []

        # 遍历所有图片文件
        for img_path in sorted(glob.glob(os.path.join(images_dir, '*.jpg'))):
            base = os.path.basename(img_path).rsplit('.', 1)[0]
            lbl_path = os.path.join(labels_dir, base + '.txt')
            if not os.path.isfile(lbl_path):
                continue
            # 读取 YOLO 格式标注：每行 5 个字段
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid, x_c, y_c, w, h = parts
                    cid = int(cid)
                    x_c, y_c, w, h = map(float, (x_c, y_c, w, h))
                    # 存储 (图路径, 类别id, 归一化框)
                    self.samples.append((img_path, cid, (x_c, y_c, w, h)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cid, (x_c, y_c, w, h) = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        # 归一化坐标 → 像素坐标
        x1 = (x_c - w/2) * W
        y1 = (y_c - h/2) * H
        x2 = (x_c + w/2) * W
        y2 = (y_c + h/2) * H

        # 裁剪并预处理
        crop = img.crop((x1, y1, x2, y2))
        if self.transform:
            crop = self.transform(crop)

        return crop, cid
