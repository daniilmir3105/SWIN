# -*- coding: utf-8 -*-
import os
# 1) Включаем синхронный режим ошибок CUDA для точной отладки
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import time
from torch.autograd import Variable
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # прогресс-бар
import torch.cuda.amp as amp  # для смешанной точности

from model.SCRN import SCRN
from util.Datasets2 import MyDataset
from util.My_tool1 import save_csv, produce_csv

# 2) Аргументы
parser = argparse.ArgumentParser(description='PyTorch SCRN')
parser.add_argument('--train_data_dir', default='./patches', type=str, help='path to .npy patches')
parser.add_argument('--epoch',          default=4,       type=int,   help='number of epochs')
parser.add_argument('--lr',             default=1e-3,     type=float, help='learning rate')
parser.add_argument('--batch_size',     default=32,       type=int,   help='batch size')
parser.add_argument('--num_workers',    default=4,        type=int,   help='DataLoader workers')
args = parser.parse_args()

if __name__ == '__main__':
    # 3) Подготовка
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('trained_model', exist_ok=True)
    produce_csv('SCRN_loss.csv')

    # 4) Модель, оптимизатор и AMP
    print('====> Building SCRN model')
    model = SCRN().to(device)
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.2)
    scaler = amp.GradScaler()  # масштабировщик для mixed precision

    # 5) Даталоадер
    transform = transforms.ToTensor()
    dataset = MyDataset(root_dir=args.train_data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # 6) Тренировочный цикл с tqdm и AMP
    step = 0
    for epoch in range(1, args.epoch + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epoch}", unit="batch")
        for X, Y in loop:
            # переносим данные и гарантируем contiguous / non_blocking
            X = X.to(device, dtype=torch.float32, non_blocking=True).contiguous()
            Y = Y.to(device, dtype=torch.float32, non_blocking=True).contiguous()

            optimizer.zero_grad()

            # forward + backward в mixed precision
            with amp.autocast():
                pred = model(X)
                assert pred.shape == Y.shape, f"Shape mismatch: {pred.shape} vs {Y.shape}"
                loss = criterion(pred, Y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            step += X.size(0)

            loop.set_postfix(batch_loss=loss.item())

        # средний loss за эпоху
        epoch_loss = running_loss / len(dataloader)
        elapsed = time.time() - start_time

        print(f'\nEpoch {epoch}/{args.epoch}  Avg Loss: {epoch_loss:.4f}  Steps: {step}  Time: {elapsed:.1f}s')

        scheduler.step()
        # save_csv('SCRN_loss.csv', epoch, epoch_loss, 0)
        torch.save(model, f'trained_model/model_{epoch:02d}.pth')
