#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRN — аффинная калибровка вывода без изменения исходных весов.
y_hat = alpha * y + beta
Сохраняет новую модель, в которой калибратор добавлен как 1×1 depthwise Conv2d.
"""
import argparse, torch, pathlib

def load_model(path, device):
    model = torch.load(path, map_location=device)
    model.eval()
    return model

def add_affine_calibrator(model, alpha=1.0, beta=0.0):
    """Вставляем Conv2d(1×1, groups=C) => по-канальный y·α + β."""
    if abs(alpha - 1.0) < 1e-8 and abs(beta) < 1e-8:
        return model                                  # калибровка не нужна

    C = model.m_tail[0].out_channels                 # число каналов выхода
    calib = torch.nn.Conv2d(
        in_channels=C, out_channels=C,
        kernel_size=1, stride=1, padding=0,
        groups=C, bias=True
    )
    with torch.no_grad():
        calib.weight.fill_(alpha)                    # y * α
        calib.bias.fill_(beta)                       # + β
    # m_tail: [Conv(3×3), ...] →  превратим в nn.Sequential
    model.m_tail = torch.nn.Sequential(model.m_tail[0], calib)
    return model

def main():
    p = argparse.ArgumentParser("SCRN cosmetic calibration (non-destructive)")
    p.add_argument("-i", "--input",  required=True, help="path to model.pth")
    p.add_argument("-o", "--output", required=True, help="where to save calibrated model")
    p.add_argument("--alpha", type=float, default=1.0, help="multiplicative factor α")
    p.add_argument("--beta",  type=float, default=0.0, help="additive shift β")
    p.add_argument("--device", default="cpu", help="cpu | cuda:0 …")
    args = p.parse_args()

    model = load_model(args.input, args.device)
    model = add_affine_calibrator(model, args.alpha, args.beta)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)                   # сохраняем *новый* .pth
    print(f"✔ Calibrated wrapper saved to {args.output}\n"
          f"  (оригинальные веса не изменены)")

if __name__ == "__main__":
    main()
