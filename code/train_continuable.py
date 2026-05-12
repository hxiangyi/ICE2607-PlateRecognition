import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os

class LicensePlateDataset(Dataset):
    def __init__(self, dataset_path, txt_file, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.types = []  # 新增类型记录
        self.max_length = 8  # 最大标签长度为 8

        # Load data from filtered txt file
        with open(txt_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                if len(parts) == 3:
                    img_path, label, plate_type = parts
                    if all(char in "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in label):
                        self.image_paths.append(os.path.join(self.dataset_path, img_path))
                        self.labels.append(label)
                        self.types.append(plate_type)  # 记录车牌类型

        self.char_dict = {char: idx for idx, char in enumerate(
            list("京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        )}
        self.type_dict = {"普通蓝牌": 0, "新能源小型车": 1}  # 新增类型字典

    def __len__(self):
        return len(self.image_paths) * 3  # 每张图片生成三种样本：原图和两种增强图

    def __getitem__(self, idx):
        original_idx = idx // 3
        augmentation_type = idx % 3

        img_path = self.image_paths[original_idx]
        label = self.labels[original_idx]
        plate_type = self.types[original_idx]

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if augmentation_type == 1:  # 横向压缩
            img = self.augment_horizontal_compress(img)
        elif augmentation_type == 2:  # 纵向压缩
            img = self.augment_vertical_compress(img)

        img = cv2.resize(img, (128, 48))  # 调整图像大小到 128x48
        label_indices = [self.char_dict[char] for char in label]  # 假定所有字符已合法
        label_indices += [-1] * (self.max_length - len(label_indices))  # 使用 -1 作为填充值
        label_indices = torch.tensor(label_indices, dtype=torch.long)

        type_label = torch.tensor(self.type_dict[plate_type], dtype=torch.long)  # 类型标签

        if self.transform:
            img = self.transform(img)

        return img, label_indices, type_label  # 返回填充后的标签和类型标签

    def augment_horizontal_compress(self, img):
        h, w = img.shape[:2]
        new_w = int(w * 0.8)
        compressed = cv2.resize(img, (new_w, h))
        result = np.full((h, w, 3), 128, dtype=np.uint8)  # 灰色填充
        result[:, (w - new_w) // 2:(w - new_w) // 2 + new_w] = compressed
        return result

    def augment_vertical_compress(self, img):
        h, w = img.shape[:2]
        new_h = int(h * 0.8)
        compressed = cv2.resize(img, (w, new_h))
        result = np.full((h, w, 3), 128, dtype=np.uint8)  # 灰色填充
        result[(h - new_h) // 2:(h - new_h) // 2 + new_h, :] = compressed
        return result

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.type_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 16, 128),  # 调整全连接层输入大小为实际大小
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 两种类型：普通蓝牌、新能源小型车
        )
        self.classifiers_7 = nn.ModuleList([nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 65)
        ) for _ in range(7)])

        self.classifiers_8 = nn.ModuleList([nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 65)
        ) for _ in range(8)])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        plate_type_output = self.type_classifier(x)  # 车牌类型预测
        outputs_7 = [classifier(x) for classifier in self.classifiers_7]  # 普通蓝牌
        outputs_8 = [classifier(x) for classifier in self.classifiers_8]  # 新能源小型车
        return plate_type_output, outputs_7, outputs_8

if __name__ == "__main__":
    dataset_path = "./DataSet"
    txt_file = "./DataSet/filtered_train.txt"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = LicensePlateDataset(dataset_path, txt_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    type_loss_weight = 2.0  # 提高类型损失的权重

    # 尝试加载已保存的 checkpoint
    checkpoint_path = "train_model_aug/cnn_epoch_10.pth"  # 替换为实际路径
    start_epoch = 0  # 默认从头开始
    if os.path.exists(checkpoint_path):
        print(f"加载 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']  # 恢复起始 epoch
        print(f"从 epoch {start_epoch} 恢复训练")

    epochs = 35
    os.makedirs("train_model_aug", exist_ok=True)
    for epoch in range(start_epoch, epochs):  # 从恢复点继续训练
        model.train()
        total_loss = 0
        for images, labels, type_labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            type_labels = type_labels.to(device)

            optimizer.zero_grad()
            plate_type_output, outputs_7, outputs_8 = model(images)

            # 类型分类损失
            type_loss = criterion(plate_type_output, type_labels) * type_loss_weight

            # 车牌类型的索引
            blue_indices = (type_labels == 0).nonzero(as_tuple=True)[0]
            green_indices = (type_labels == 1).nonzero(as_tuple=True)[0]

            # 普通蓝牌字符损失
            blue_char_loss = 0
            if len(blue_indices) > 0:
                valid_mask_blue = (labels[blue_indices, :] >= 0) & (labels[blue_indices, :] < 65)
                for j in range(7):
                    valid_indices = valid_mask_blue[:, j]
                    if valid_indices.sum().item() > 0:
                        blue_char_loss += criterion(outputs_7[j][blue_indices][valid_indices],
                                                    labels[blue_indices, j][valid_indices])

            # 新能源小型车字符损失
            green_char_loss = 0
            if len(green_indices) > 0:
                valid_mask_green = (labels[green_indices, :] >= 0) & (labels[green_indices, :] < 65)
                for j in range(8):
                    valid_indices = valid_mask_green[:, j]
                    if valid_indices.sum().item() > 0:
                        green_char_loss += criterion(outputs_8[j][green_indices][valid_indices],
                                                     labels[green_indices, j][valid_indices])

            # 总字符损失
            char_loss = (blue_char_loss + green_char_loss) / len(images)

            # 总损失
            loss = type_loss + char_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}")

        # Save model and loss after each epoch
        model_save_path = os.path.join("train_model_aug", f"cnn_epoch_{epoch+1}.pth")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss
        }
        torch.save(checkpoint, model_save_path)

        with open(os.path.join("train_model_aug", "loss_log.txt"), "a") as log_file:
            log_file.write(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}\n")

    print("Training complete. Models and logs are saved in 'train_model' directory.")
