"""
Name    : trainer_seg_hist_fuse.py (Pure DRBN Baseline Version)
Author  : ChatGPT
Desc    : Pure DRBN Training Loop without HRNet / SKF / Semantic / GAN / Histogram / Portrait Loss
"""

import os
from decimal import Decimal

import torch
import torch.nn as nn
from tqdm import tqdm
import pytorch_ssim

import utility


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, adv=False):

        self.args = args
        self.scale = args.scale
        self.ckp = ckp

        self.loader_train = loader.loader_train
        self.loader_test  = loader.loader_test

        # ---------------------------------------------------------
        # 使用传入的 DRBN 模型，不含语义、GAN 等
        # ---------------------------------------------------------
        self.model = my_model

        # ---------------------------------------------------------
        # baseline 加载逻辑（可保留）
        # ---------------------------------------------------------
        # baseline_path = './baseline_model.pt'

        # if os.path.exists(baseline_path):
        #     self.ckp.write_log(f'Loading Baseline model from {baseline_path}...')

        #     try:
        #         raw = torch.load(baseline_path, map_location='cpu')

        #         # 如果 baseline 为 {"state_dict": ...}
        #         if isinstance(raw, dict) and 'state_dict' in raw:
        #             state_dict = raw['state_dict']
        #         else:
        #             state_dict = raw

        #         fixed_state = {}

        #         # 给所有权重加 "model." 前缀（如果没有）
        #         for k, v in state_dict.items():
        #             new_key = k if k.startswith("model.") else ("model." + k)
        #             fixed_state[new_key] = v

        #         self.model.load_state_dict(fixed_state, strict=True)
        #         self.ckp.write_log(f'>>> SUCCESS: Baseline loaded ({len(fixed_state)} params).')

        #     except Exception as e:
        #         self.ckp.write_log(f'>>> Error loading baseline: {e}')
        # else:
        #     self.ckp.write_log(f'>>> Warning: Baseline {baseline_path} not found. Training from scratch.')

        # ---------------------------------------------------------
        # 无语义、无 GAN 模式
        # ---------------------------------------------------------
        self.adv = False   # 强制关闭对抗

        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        self.error_last = 1e8


    # ============================================================
    # 训练
    # ============================================================
    def train(self):

        self.scheduler.step()
        self.loss.step()

        epoch = self.scheduler.last_epoch + 1
        lr    = self.scheduler.get_lr()[0]

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(
            epoch, Decimal(lr)
        ))
        self.loss.start_log()

        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        # SSIM 损失
        criterion_ssim = pytorch_ssim.SSIM(window_size=11)

        for batch, (lr, hr, _, _) in enumerate(self.loader_train):

            lr, hr = self.prepare(lr, hr)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            lr = lr / 255.0
            hr = hr / 255.0

            # 多尺度 GT
            hr1 = hr
            hr2 = hr[:, :, 0::2, 0::2]
            hr4 = hr[:, :, 0::4, 0::4]

            # =======================
            # DRBN 第一阶段
            # =======================
            res_g3_s1, res_g3_s2, res_g3_s4, \
            feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.model.forward_1(lr, 3)

            # =======================
            # DRBN 第二阶段（无语义）
            # =======================
            phr1, phr2, phr4 = self.model.forward_2(
                lr, res_g3_s1, res_g3_s2, res_g3_s4,
                feat_g3_s1, feat_g3_s2, feat_g3_s4,
                None, None     # ← 语义输入取消
            )

            # =======================
            # 唯一损失：rect_loss（多层 SSIM）
            # =======================
            rect_loss = (
                criterion_ssim(phr1, hr1) +
                criterion_ssim(phr2, hr2) +
                criterion_ssim(phr4, hr4)
            )

            full_loss = rect_loss
            full_loss.backward()
            self.optimizer.step()

            timer_model.hold()

            if batch % 24 == 0:
                self.ckp.write_log(
                    '[{}/{}]\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch+1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        full_loss.item(),
                        rect_loss.item(),
                        timer_model.release(),
                        timer_data.release()
                    )
                )

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]


    # ============================================================
    # 测试（只调用 DRBN，不用语义）
    # ============================================================
    def test(self):

        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))

        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():

            for idx_scale, scale in enumerate(self.scale):

                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)

                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):

                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)

                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        (lr,) = self.prepare(lr)

                    lr = lr / 255.0
                    hr = hr / 255.0

                    # DRBN stage-1
                    res_g3_s1, res_g3_s2, res_g3_s4, \
                    feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.model.forward_1(lr, 3)

                    # stage-2 (无语义)
                    phr1, phr2, phr4 = self.model.forward_2(
                        lr, res_g3_s1, res_g3_s2, res_g3_s4,
                        feat_g3_s1, feat_g3_s2, feat_g3_s4,
                        None, None
                    )

                    phr = utility.quantize(phr1 * 255, self.args.rgb_range)
                    lr  = utility.quantize(lr  * 255, self.args.rgb_range)
                    hr  = utility.quantize(hr  * 255, self.args.rgb_range)

                    save_list = [hr, lr, phr, lr]

                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            phr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale, epoch)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)

                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))


    # ============================================================
    # utils
    # ============================================================
    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
