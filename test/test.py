# -*- coding: utf-8 -*-
from util.My_tool1 import *
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def preprocess_seismic_data(seismic_data, nsub, noise_level=0.1):
    """
    Preprocess seismic data by adding noise and subsampling traces.

    Parameters:
        seismic_data (np.ndarray): Original seismic data array.
        nsub (int): Subsampling factor. From every 'nsub' traces, only one is kept.
        noise_level (float): Standard deviation of the Gaussian noise to be added.

    Returns:
        np.ndarray: Preprocessed seismic data.
    """
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level * np.std(seismic_data), seismic_data.shape)
    noisy_data = seismic_data + noise

    # Subsample traces by zeroing out
    # Determine which traces to keep
    kept_traces = np.arange(0, noisy_data.shape[1], nsub)
    # Zero out traces that are not in kept_traces
    subsampled_data = noisy_data.copy()
    for trace_idx in range(subsampled_data.shape[1]):
        if trace_idx not in kept_traces:
            subsampled_data[:, trace_idx] = 0

    return subsampled_data


# Load clear seismic data
x = np.load('test_data/clear.npy')
x = x.astype(np.float64)

# Define preprocessing parameters
nsub = 2  # From every 4 traces, keep 1
noise_level = 0  # 10% of data's standard deviation

# Preprocess the seismic data
y = preprocess_seismic_data(x, nsub, noise_level)

# Save the preprocessed data if needed
np.save('test_data/new_noise_and_miss.npy', y)

if __name__ == '__main__':

    # model = torch.load('trained_model\\model.pth')
    model = torch.load('trained_model\\model.pth', weights_only=False)

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

    # pre_snr = snr_(y, x)
    # print("before：snr" + str(pre_snr))
    # snr = snr_(x_, x)
    # print("After：snr" + str(snr))

    # # Исправленный вызов ssim_
    # pre_ssim = ssim_(y, x)  # Указываем data_range
    # print("before：ssim" + str(pre_ssim))
    # ssim = ssim_(x_, x, dat)  # Указываем data_range
    # print("After：ssim" + str(ssim))

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

    plt.imshow(x - x_, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    plt.title("Разница")
    plt.xlabel('Расстояние от источника, м')
    plt.ylabel('Время свободного пробега, мс')
    plt.show()
