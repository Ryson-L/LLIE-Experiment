import torch
from torch.utils.data import DataLoader
from torch.utils.data import _utils


class MSDataLoader(DataLoader):
    """
    兼容 PyTorch 2.x 的简化版 MSDataLoader

    - 仍然接受 (args, dataset, batch_size, shuffle, ... 等参数)
    - 内部直接调用标准 DataLoader 的实现
    - 在 __iter__ 中为每个 batch 追加一个 idx_scale，保持原来
      for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
      这种解包方式不变
    """

    def __init__(
        self,
        args,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        collate_fn=_utils.collate.default_collate,
        pin_memory=True,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        # 直接用官方 DataLoader 的构造函数
        super(MSDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=args.n_threads,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if args.n_threads > 0 else False,
            prefetch_factor=2 if args.n_threads > 0 else None,
        )

        # 保留原来的 scale 信息（通常在 option.py 里解析成 list）
        self.scale = getattr(args, "scale", [1])

    def __iter__(self):
        """
        使用 PyTorch 自带迭代逻辑拿到一个 batch，然后在最后补一个 idx_scale。

        注意：
        - 这里 idx_scale 目前固定为 0，相当于只用 args.scale[0]
        - 你的 DRBN-SKF 任务本身不是多尺度 SR，不会受影响
        - 如果以后真要做多尺度，可以在这里改成随机或轮询的 idx_scale
        """

        for batch in super(MSDataLoader, self).__iter__():
            # 简单处理不同 batch 类型，默认 dataset 返回的是 tuple/list
            idx_scale = 0

            if isinstance(batch, list):
                batch = list(batch)
                batch.append(idx_scale)
                # Trainer 那边是按 tuple 解包，这里转成 tuple 比较稳
                yield tuple(batch)

            elif isinstance(batch, tuple):
                # 直接在末尾加一个 idx_scale
                yield (*batch, idx_scale)

            elif isinstance(batch, dict):
                # 极少见的情况：dataset 返回 dict，这里就 (dict, idx_scale)
                yield (batch, idx_scale)

            else:
                # dataset 返回单个张量等情况，也强行 (data, idx_scale)
                yield (batch, idx_scale)
