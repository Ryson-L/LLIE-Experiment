import os
import glob

from data import common
import pickle
import numpy as np
import imageio

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self._set_filesystem(args.dir_data)

        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()

        if args.ext.find('bin') >= 0:
            list_hr, list_lr = self._scan()

            print('...check and load hr...')
            self.images_hr = self._check_and_load(
                args.ext, list_hr, self._name_hrbin()
            )
            print('...check and load lr...')
            self.images_lr = [
                self._check_and_load(args.ext, l, self._name_lrbin(s)) \
                for s, l in zip(self.scale, list_lr)
            ]

        else:
            if args.ext.find('img') >= 0 or benchmark:
                self.images_hr, self.images_lr = list_hr, list_lr

            elif args.ext.find('sep') >= 0:

                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
              
                os.makedirs(
                    self.dir_lr.replace(self.apath, path_bin),
                    exist_ok=True
                )
 
                self.images_hr, self.images_lr = [], []

                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr.append(b)

        #            print(b)

                    self._check_and_load(
                        args.ext, [h], b, verbose=True, load=False
                    )

                for l in list_lr:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr.append(b)

        #            print(b)

                    self._check_and_load(
                        args.ext, [l], b,  verbose=True, load=False
                    )
 
        if train:
            # self.repeat \
            #     = args.test_every // (len(self.images_hr) // args.batch_size)
            self.repeat = 3

    # Below functions as used to prepare images
    def _scan(self):
        # 1. 读取所有文件路径
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])))
        names_lr_all = sorted(glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1])))

        # 2. 建立 Low 文件夹的索引字典 {文件名: 完整路径}
        # 这样我们就可以通过名字瞬间找到路径
        lr_map = {}
        for lr_path in names_lr_all:
            # 提取文件名，例如 'C:/.../low/1.png' -> '1.png'
            file_name = os.path.basename(lr_path)
            lr_map[file_name] = lr_path

        # 3. 重新构建一一对应的列表
        names_hr_filtered = []
        names_lr_filtered = []

        print(f"Matching images for {self.name}...")
        
        for hr_path in names_hr:
            file_name = os.path.basename(hr_path)
            
            # 关键逻辑：只有当 Low 里也有这个名字时，才加入列表
            if file_name in lr_map:
                names_hr_filtered.append(hr_path)
                names_lr_filtered.append(lr_map[file_name])
            else:
                # 如果是 eval15，这里其实是正常的，不需要报警，但如果是 train 就不对
                pass 

        print(f"Final dataset size: {len(names_hr_filtered)} pairs.")
        
        # 确保两个列表长度一致且顺序一致
        return names_hr_filtered, names_lr_filtered

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)

        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR.pt'.format(self.split)
        )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f: ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))

            b = []
            for _l in l:
                if _l.find('Our_low')==-1 and _l.find('Our_low_test')==-1:
                    tmp_name = os.path.splitext(os.path.basename(_l))[0]
                    tmp_image = _l
    #                print(tmp_name)
    #                print(tmp_image)
                else:
                    tmp_name = 'low'+os.path.splitext(os.path.basename(_l))[0][6:]
                    tmp_image = 'low'.join(_l.split('normal'))

    #                print(tmp_name)
    #                print(tmp_image)                    

                tmp = {
                'name': tmp_name,
                'image': imageio.imread(tmp_image)
                }

                b.append(tmp)

            with open(f, 'wb') as _f: pickle.dump(b, _f) 
            return b

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr,  hr  = self.get_patch(lr,  hr)

        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)

        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr, rgb_range=self.args.rgb_range
        )

        return lr_tensor, hr_tensor, filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        if self.args.ext.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.args.ext == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
            elif self.args.ext.find('sep') >= 0:
                with open(f_hr, 'rb') as _f: hr = np.load(_f, allow_pickle=True)[0]['image']
                with open(f_lr, 'rb') as _f: lr = np.load(_f, allow_pickle=True)[0]['image']

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih, 0:iw]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

