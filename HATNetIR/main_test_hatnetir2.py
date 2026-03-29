import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import math
from functools import reduce

# --------------------------------------------------------------------------------
# Import Models & Utils
# --------------------------------------------------------------------------------
from models.network_hatnetir import HATNetIR as net
from utils import utils_image as util


# --------------------------------------------------------------------------------
# HATNetIR Helper Functions
# --------------------------------------------------------------------------------
def get_lcm(numbers):
    def lcm(a, b): return abs(a * b) // math.gcd(a, b)

    return reduce(lcm, numbers)


def create_tile_weight(size, overlap):
    weight_x = torch.ones(size)
    weight_y = torch.ones(size)

    if overlap > 0:
        fade = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, overlap)))

        weight_x[:overlap] = fade
        weight_x[-overlap:] = fade.flip(0)

        weight_y[:overlap] = fade
        weight_y[-overlap:] = fade.flip(0)

    weight = weight_x.view(1, 1, -1, 1) * weight_y.view(1, 1, 1, -1)
    return weight


# --------------------------------------------------------------------------------
# Inference Function
# --------------------------------------------------------------------------------
def test(img_lq, model, args, max_grid):
    if args.tile is None:
        return model(img_lq)
    else:
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)

        tile = (tile // max_grid) * max_grid
        tile_overlap = (args.tile_overlap // max_grid) * max_grid
        sf = args.scale

        stride = tile - tile_overlap

        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
        W = torch.zeros_like(E)

        mask = create_tile_weight(tile * sf, tile_overlap * sf).type_as(img_lq)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]

                with torch.no_grad():
                    out_patch = model(in_patch)

                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch * mask)
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(mask)

        return E.div_(W + 1e-8)


# --------------------------------------------------------------------------------
# Model Definition
# --------------------------------------------------------------------------------
def define_model(args):
    print(f"Defining HATNetIR (Scale={args.scale})...")
    model = net(upscale=args.scale,
                in_chans=3,
                grid_sizes=[2, 3, 4, 6, 8, 12],
                img_range=1.0,
                depths=[3, 3, 3, 3, 3, 3],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv')

    if os.path.exists(args.model_path):
        print(f'Loading model from {args.model_path}')
        pretrained_model = torch.load(args.model_path)

        if 'params' in pretrained_model.keys():
            model.load_state_dict(pretrained_model['params'], strict=True)
        elif 'params_ema' in pretrained_model.keys():
            model.load_state_dict(pretrained_model['params_ema'], strict=True)
        else:
            model.load_state_dict(pretrained_model, strict=True)
    else:
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    return model


# --------------------------------------------------------------------------------
# Setup & Image Processing
# --------------------------------------------------------------------------------
def setup(args):
    save_dir = f'results/hatnetir_{args.task}_x{args.scale}'
    folder = args.folder_gt
    border = args.scale
    return folder, save_dir, border


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    lq_path = os.path.join(args.folder_lq, f"{imgname}x{args.scale}{imgext}")
    img_lq = cv2.imread(lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    if img_lq is None:
        raise FileNotFoundError(f"Low quality image not found at: {lq_path}")

    return imgname, img_lq, img_gt


# --------------------------------------------------------------------------------
# Main Entry
# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 2, 4')
    parser.add_argument('--model_path', type=str, required=True, help='path to .pth model')
    parser.add_argument('--folder_lq', type=str, required=True, help='input low-quality folder')
    parser.add_argument('--folder_gt', type=str, required=True, help='input ground-truth folder')
    parser.add_argument('--tile', type=int, default=120, help='Tile size (must be multiple of 24)')
    parser.add_argument('--tile_overlap', type=int, default=24, help='Overlap size (must be multiple of 24)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = define_model(args)
    model.eval()
    model = model.to(device)

    folder, save_dir, border = setup(args)
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(model, 'grid_sizes'):
        max_grid = get_lcm(model.grid_sizes)
    else:
        max_grid = 24
    print(f"HATNetIR Grid Alignment (LCM): {max_grid}")

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    print(f"\nTesting on {os.path.basename(args.folder_gt)} ...")

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        imgname, img_lq, img_gt = get_image_pair(args, path)

        img_lq_tensor = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        img_lq_tensor = torch.from_numpy(img_lq_tensor).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = test(img_lq_tensor, model, args, max_grid)

        output = output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        cv2.imwrite(f'{save_dir}/{imgname}_HATNetIR.png', output)

        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)
            h_old, w_old = img_lq.shape[0], img_lq.shape[1]
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]

            psnr = util.calculate_psnr(output, img_gt, border=border)
            try:
                ssim = util.calculate_ssim(output, img_gt, border=border)
            except:
                ssim = 0.0

            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)

            if img_gt.ndim == 3:
                output_y = util.bgr2ycbcr(output.astype(np.float32) / 255., only_y=True) * 255.
                img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255., only_y=True) * 255.
                psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border)
                try:
                    ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)
                except:
                    ssim_y = 0.0

                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            else:
                psnr_y, ssim_y = psnr, ssim

            print(f"[{idx + 1:02d}] {imgname:20s} - PSNR_Y: {psnr_y:.2f}dB, SSIM_Y: {ssim_y:.4f}")

    if test_results['psnr']:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print(f"\nResults Saved in: {save_dir}")
        print(f"Average PSNR/SSIM(RGB): {ave_psnr:.2f} dB; {ave_ssim:.4f}")

        if test_results['psnr_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print(f"Average PSNR_Y/SSIM_Y: {ave_psnr_y:.2f} dB; {ave_ssim_y:.4f}")


if __name__ == '__main__':
    main()