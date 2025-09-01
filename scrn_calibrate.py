#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Линейная калибровка вывода SCRN:
y_hat = alpha * y + beta
"""
import argparse, torch, pathlib, sys

def load_model(path, device):
    model = torch.load(path, map_location=device)
    model.eval()
    return model

def apply_alpha(model, alpha):
    conv = model.m_tail[0]                        # Conv2d, bias=False
    with torch.no_grad():
        conv.weight.mul_(alpha)
    return model

def add_beta(model, beta):
    if abs(beta) < 1e-8:
        return model                              # смещение не нужно
    bias = torch.nn.Conv2d(                       # 1×1 "калибратор"
        in_channels=model.m_tail[0].out_channels,
        out_channels=model.m_tail[0].out_channels,
        kernel_size=1,
        bias=True
    )
    torch.nn.init.constant_(bias.bias, beta)
    bias.weight.data.zero_()                      # чтобы был только сдвиг
    model.m_tail = torch.nn.Sequential(model.m_tail[0], bias)
    return model

def main():
    p = argparse.ArgumentParser(description="SCRN cosmetic calibration")
    p.add_argument("-i", "--input",  required=True, help="model.pth to load")
    p.add_argument("-o", "--output", required=True, help="where to save new model")
    p.add_argument("--alpha", type=float, default=1.0, help="scale factor")
    p.add_argument("--beta",  type=float, default=0.0, help="additive shift")
    p.add_argument("--device", default="cpu", help="cpu | cuda:0 ...")
    args = p.parse_args()

    model = load_model(args.input, args.device)
    model = apply_alpha(model, args.alpha)
    model = add_beta(model,  args.beta)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)
    print(f"✔ Calibrated model saved to {args.output}")

if __name__ == "__main__":
    main()


