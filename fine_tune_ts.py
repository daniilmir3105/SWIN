# fine_tune_kd.py
# -*- coding: utf-8 -*-

import os
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model.SCRN import SCRN
from util.Datasets2 import MyDataset
from util.My_tool1 import save_csv, produce_csv

def parse_args():
    p = argparse.ArgumentParser(description='Fine-tune SCRN with Knowledge Distillation')
    p.add_argument('--teacher_path', default='trained_model/model.pth', type=str,
                   help='путь к предобученной teacher-модели')
    p.add_argument('--train_data_dir', default='./patches', type=str,
                   help='директория с .npy патчами')
    p.add_argument('--epochs', default=100, type=int, help='число эпох дообучения')
    p.add_argument('--lr', default=1e-4, type=float, help='learning rate для fine-tuning')
    p.add_argument('--batch_size', default=16, type=int, help='batch size')
    p.add_argument('--num_workers', default=4, type=int, help='DataLoader workers')
    p.add_argument('--alpha', default=0.5, type=float,
                   help='коэффициент между reconstruction и distillation loss')
    return p.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('trained_model_ft', exist_ok=True)
    produce_csv('SCRN_kd_loss.csv')

    # --- 1) Загрузка teacher ---
    print(f"====> Loading teacher model from {args.teacher_path}")
    teacher = SCRN()
    ckpt = torch.load(args.teacher_path, map_location='cpu')
    if isinstance(ckpt, dict):
        sd = ckpt.get('state_dict', ckpt)
        teacher.load_state_dict(sd)
    else:
        teacher = ckpt
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- 2) Инициализация student (копия teacher) ---
    student = SCRN().to(device)
    student.load_state_dict(teacher.state_dict())

    # --- 3) Определяем loss, optimizer, scheduler, amp ---
    mse_loss = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer,
                            milestones=[int(args.epochs*0.3), int(args.epochs*0.6)],
                            gamma=0.5)
    scaler = amp.GradScaler()

    # --- 4) Даталоадер ---
    transform = transforms.ToTensor()
    dataset = MyDataset(root_dir=args.train_data_dir, transform=transform)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        persistent_workers=True)

    # --- 5) Fine-tuning loop с Distillation ---
    global_step = 0
    for epoch in range(1, args.epochs+1):
        student.train()
        epoch_loss = 0.0
        t0 = time.time()
        loop = tqdm(loader, desc=f"[KD] Epoch {epoch}/{args.epochs}", unit='batch')

        for X, Y in loop:
            X = X.to(device, dtype=torch.float32, non_blocking=True).contiguous()
            Y = Y.to(device, dtype=torch.float32, non_blocking=True).contiguous()

            optimizer.zero_grad()
            with amp.autocast():
                S_out = student(X)
                with torch.no_grad():
                    T_out = teacher(X)

                loss_rec  = mse_loss(S_out, Y)
                loss_dist = mse_loss(S_out, T_out)
                loss = args.alpha * loss_rec + (1 - args.alpha) * loss_dist

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += X.size(0)
            loop.set_postfix(batch_loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        print(f"\n[KD] Epoch {epoch}/{args.epochs}  Avg Loss: {avg_loss:.6f}  Steps: {global_step}  Time: {elapsed:.1f}s")

        scheduler.step()
        # save_csv('SCRN_kd_loss.csv', epoch, avg_loss, 0)
        torch.save(student.state_dict(), f'trained_model_ft/model_kd_{epoch:03d}.pth')

    print("Fine-tuning with KD completed.")

if __name__ == '__main__':
    main()
