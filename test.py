# -*- coding: utf-8 -*-
from util.My_tool1 import *
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    model = torch.load('trained_model\\model.pth', map_location='cuda:0')

    model.eval()  # evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()

    x = np.load('test_data/clear.npy')
    x = x.astype(np.float64)

    y = np.load('test_data/noise_and_miss.npy')
    y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

    torch.cuda.synchronize()
    start_time = time.time()
    y_ = y_.type(torch.float32)
    y_ = y_.cuda()

    x_ = model(y_)  # inferences
    x_ = x_.view(y.shape[0], y.shape[1])
    x_ = x_.cpu()
    x_ = x_.detach().numpy().astype(np.float64)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    pre_snr = snr_(y, x)
    print("before：snr" + str(pre_snr))
    snr = snr_(x_, x)
    print("After：snr" + str(snr))

    # Исправленный вызов ssim_
    pre_ssim = ssim_(y, x, data_range=1.0)  # Указываем data_range
    print("before：ssim" + str(pre_ssim))
    ssim = ssim_(x_, x, data_range=1.0)  # Указываем data_range
    print("After：ssim" + str(ssim))

    # Calculate MSE between original and denoised data
    mse = np.mean((x - x_) ** 2)
    print("MSE between original and denoised data: ", mse)

    # Вывод графиков в сером цвете
    plt.imshow(x, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    plt.title("Исходные данные")
    plt.xlabel('Расстояние от источника, м')
    plt.ylabel('Время свободного пробега, мс')
    plt.show()

    plt.imshow(y, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    plt.title("Прореженные данные")
    plt.xlabel('Расстояние от источника, м')
    plt.ylabel('Время свободного пробега, мс')
    plt.show()

    plt.imshow(x_, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    plt.title("Результат работы нейросети")
    plt.xlabel('Расстояние от источника, м')
    plt.ylabel('Время свободного пробега, мс')
    plt.show()

    plt.imshow(x-x_, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    plt.title("Разница")
    plt.xlabel('Расстояние от источника, м')
    plt.ylabel('Время свободного пробега, мс')
    plt.show()
