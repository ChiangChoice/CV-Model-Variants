import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import torch.nn.functional as F

import math
from functools import reduce

def get_lcm(numbers):
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    return reduce(lcm, numbers)

def test_with_tile(model, img_lq, opt):
    b, c, h, w = img_lq.size()

    tile_size = opt['train'].get('tile_size', 256)
    tile_overlap = opt['train'].get('tile_overlap', 32)

    if hasattr(model.netG, 'grid_sizes'):
        max_grid = get_lcm(model.netG.grid_sizes)
    else:
        max_grid = 4

    tile_size = min(tile_size, h, w)
    tile_size = (tile_size // max_grid) * max_grid
    if tile_size < max_grid:
        tile_size = max_grid

    tile_overlap = (tile_overlap // max_grid) * max_grid

    stride = tile_size - tile_overlap
    sf = opt['scale']

    h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]

    E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_lq[..., h_idx:h_idx + tile_size, w_idx:w_idx + tile_size]

            _, _, patch_h, patch_w = in_patch.shape
            need_pad_h = (max_grid - patch_h % max_grid) % max_grid
            need_pad_w = (max_grid - patch_w % max_grid) % max_grid

            if need_pad_h > 0 or need_pad_w > 0:
                in_patch = F.pad(in_patch, (0, need_pad_w, 0, need_pad_h), 'reflect')

            with torch.no_grad():
                model.netG.eval()
                out_patch = model.netG(in_patch)

            if need_pad_h > 0 or need_pad_w > 0:
                out_patch = out_patch[..., :patch_h * sf, :patch_w * sf]

            out_patch_mask = create_tile_weight(tile_size * sf, tile_overlap * sf).type_as(img_lq)

            E[..., h_idx * sf:(h_idx + tile_size) * sf, w_idx * sf:(w_idx + tile_size) * sf].add_(
                out_patch * out_patch_mask)
            W[..., h_idx * sf:(h_idx + tile_size) * sf, w_idx * sf:(w_idx + tile_size) * sf].add_(out_patch_mask)

    return E.div_(W + 1e-8)

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


def main(json_path='options/train_hatnetir_sr_classical.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    if opt['dist']:
        init_dist('pytorch')

    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E

    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
                                                                             net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG

    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']

    if opt['rank'] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))

            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))

            if opt['dist']:
                train_sampler = DistributedSampler(train_set,
                                                   shuffle=dataset_opt['dataloader_shuffle'],
                                                   drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    model = define_Model(opt)
    model.init_train()

    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    for epoch in range(1000000):
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)

        for i, train_data in enumerate(train_loader):
            current_step += 1

            model.update_learning_rate(current_step)
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                avg_psnr = 0.0
                idx = 0

                use_tile = opt['train'].get('use_tile_test', False)

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)

                    if use_tile:
                        with torch.no_grad():
                            E_tensor = test_with_tile(model, test_data['L'].to(model.device), opt)
                        model.E = E_tensor
                    else:
                        model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(
                    epoch, current_step, avg_psnr))


if __name__ == '__main__':
    main()