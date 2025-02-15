import matplotlib.pyplot as plt

# 폰트 설정을 위한 matplotlib의 rcParams 사용
plt.rcParams['font.family'] = 'Arial'  # Arial 글씨체 설정

import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# 2. CityscapesDataset 클래스 정의
class CityscapesDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # 이미지 및 라벨 파일 목록 생성
        self.image_files = sorted(os.listdir(images_dir))
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])  # color.png만 가져오기

        # 이미지와 라벨 수 확인
        if len(self.image_files) != len(self.label_files):
            raise ValueError("이미지와 라벨의 수가 일치하지 않습니다.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 로드
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 라벨 로드
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)  # 색상 이미지로 로드

        # 마스크로 변환 (특정 색상 값만 1로 설정, 나머지는 0으로 설정)
        target_color = np.array([128, 64, 128])
        mask = np.all(label == target_color, axis=-1)  # (H, W, 3)에서 특정 색상 검출
        label = np.where(mask, 1, 0).astype(np.uint8)  # target_color에 해당하면 1, 아니면 0

        # 변환 적용
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        return image, label.clone().detach().long()

# 3. 데이터셋 경로 설정
val_images_dir = "content/test/images"
val_labels_dir = "content/test/labels"

# 4. 데이터셋 변환 및 데이터 로더 설정
train_transform = A.Compose([
    A.Resize(height=256, width=512),  # 이미지 크기 조정
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),  # albumentations의 ToTensorV2 사용
])

val_transform = train_transform
val_dataset = CityscapesDataset(val_images_dir, val_labels_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# 5. U-Net 모델 정의
# Residual Layer
def residual_layer(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim)
    )
    return model

# Max Pooling
def max_pooling():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

# Convolution Block for Decoder
def convolution_block_decoder(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn
    )
    return model

# Convolution Block with Residual
def convolution_block_residual(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        residual_layer(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0),  # 1x1 conv for residual connection
        nn.BatchNorm2d(out_dim)
    )
    return model

# UnetGenerator Class
class UnetGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, num_filter):
        super(UnetGenerator, self).__init__()
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        # Encoder
        self.encoder_1 = convolution_block_residual(in_dim, num_filter, act_fn)
        self.max_pool_1 = max_pooling()
        self.encoder_2 = convolution_block_residual(num_filter, num_filter * 2, act_fn)
        self.max_pool_2 = max_pooling()
        self.encoder_3 = convolution_block_residual(num_filter * 2, num_filter * 4, act_fn)
        self.max_pool_3 = max_pooling()
        self.encoder_4 = convolution_block_residual(num_filter * 4, num_filter * 8, act_fn)
        self.max_pool_4 = max_pooling()

        # Bridge
        self.bridge_residual = convolution_block_residual(num_filter * 8, num_filter * 16, act_fn)

        # Decoder
        self.decoder_1 = convolution_block_decoder(num_filter * 16, num_filter * 8, act_fn)
        self.residual_1 = convolution_block_residual(num_filter * 16, num_filter * 8, act_fn)  # 수정됨
        self.decoder_2 = convolution_block_decoder(num_filter * 8, num_filter * 4, act_fn)
        self.residual_2 = convolution_block_residual(num_filter * 8, num_filter * 4, act_fn)
        self.decoder_3 = convolution_block_decoder(num_filter * 4, num_filter * 2, act_fn)
        self.residual_3 = convolution_block_residual(num_filter * 4, num_filter * 2, act_fn)
        self.decoder_4 = convolution_block_decoder(num_filter * 2, num_filter, act_fn)
        self.residual_4 = convolution_block_residual(num_filter * 2, num_filter, act_fn)

        # Output layer
        self.out = nn.Sequential(
            nn.Conv2d(num_filter, out_dim, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Encoder
        encoder_1 = self.encoder_1(x)
        pool_1 = self.max_pool_1(encoder_1)
        
        encoder_2 = self.encoder_2(pool_1)
        pool_2 = self.max_pool_2(encoder_2)
        
        encoder_3 = self.encoder_3(pool_2)
        pool_3 = self.max_pool_3(encoder_3)
        
        encoder_4 = self.encoder_4(pool_3)
        pool_4 = self.max_pool_4(encoder_4)

        bridge = self.bridge_residual(pool_4)
        
        # Decoder with skip connections
        decoder_1 = self.decoder_1(bridge)
        concat_1 = torch.cat([decoder_1, encoder_4], dim=1)
        residual_1 = self.residual_1(concat_1)

        decoder_2 = self.decoder_2(residual_1)
        concat_2 = torch.cat([decoder_2, encoder_3], dim=1)
        residual_2 = self.residual_2(concat_2)

        decoder_3 = self.decoder_3(residual_2)
        concat_3 = torch.cat([decoder_3, encoder_2], dim=1)
        residual_3 = self.residual_3(concat_3)

        decoder_4 = self.decoder_4(residual_3)
        concat_4 = torch.cat([decoder_4, encoder_1], dim=1)
        residual_4 = self.residual_4(concat_4)

        # Output
        out = self.out(residual_4)
        return out

# 6. 모델 초기화
n_classes = 2  # Cityscapes의 클래스 수
in_channels = 3  # 입력 이미지의 채널 수 (RGB)
num_filter = 64  # 필터 수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnetGenerator(in_channels, n_classes, num_filter).to(device)

# 7. 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. 모델 로드 함수 추가
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 9. 시각화 함수 정의
def denormalize(image):
    """이미지를 정규화 해제하여 원래 범위로 복원."""
    image = image * torch.Tensor([0.229, 0.224, 0.225])[:, None, None] + torch.Tensor([0.485, 0.456, 0.406])[:, None, None]
    return image.permute(1, 2, 0).clamp(0, 1).numpy()

def plot_image(image, label, pred):
    """이미지, 라벨, 예측 마스크를 시각화하는 함수"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 원본 이미지
    axes[0].imshow(denormalize(image))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 실제 라벨
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # 예측 결과
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.show()

# 10. 성능 평가 함수 정의
def calculate_metrics(preds, labels):
    # Pixel Accuracy 계산
    correct = (preds == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total

    # IoU 계산
    intersection = (preds & labels).sum().item()
    union = (preds | labels).sum().item()
    iou = intersection / union if union != 0 else 0

    # Dice 유사도 계산
    dice = 2 * intersection / (preds.sum().item() + labels.sum().item()) if (preds.sum().item() + labels.sum().item()) != 0 else 0

    return accuracy, iou, dice

# 11. 모델 평가 함수 정의
def evaluate_model(model, val_loader, device):
    model.eval()
    total_accuracy = 0
    total_iou = 0
    total_dice = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # 모델 예측
            outputs = model(images)

            # 가장 높은 확률을 가진 클래스 선택
            _, preds = torch.max(outputs, 1)

            # 성능 지표 계산
            accuracy, iou, dice = calculate_metrics(preds, labels)
            total_accuracy += accuracy
            total_iou += iou
            total_dice += dice
            total_samples += 1

    # 평균 지표 계산
    avg_accuracy = total_accuracy / total_samples
    avg_iou = total_iou / total_samples
    avg_dice = total_dice / total_samples

    return avg_accuracy, avg_iou, avg_dice

# 12. 모델 평가 실행
avg_accuracy, avg_iou, avg_dice = evaluate_model(model, val_loader, device)

# 13. 평가 결과 출력 - 전체 테스트 데이터에 대한 평균을 의미함 (val 참고 위치 확인, val이지만 테스트 데이터 가져옴)
print(f"평균 Pixel Accuracy: {avg_accuracy:.4f}")
print(f"평균 IoU: {avg_iou:.4f}")
print(f"평균 Dice Similarity Coefficient: {avg_dice:.4f}")
