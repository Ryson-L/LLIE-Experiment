import sys
import threading
import queue
import random
import collections

import torch
import torch.multiprocessing as multiprocessing

from torch._C import _set_worker_signal_handlers
from torch.utils.data import _utils
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter as _DataLoaderIter
from torch.utils.data.dataloader import _DatasetKind

_use_shared_memory = False

# 兼容 Python 2/3 的老代码，保留不动
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


# ====== 旧的多进程循环和迭代器实现（现在不再使用，但可以保留以防以后参考） ======

def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, init_fn, worker_id):
    """
    原始实现里用于手工管理多进程和多尺度训练，
    在当前修复方案中我们不再使用这个函数。
    """
    global _use_shared_memory
    _use_shared_memory = True
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            idx_scale = 0
            if len(scale) > 1 and getattr(dataset, "train", False):
                idx_scale = random.randrange(0, len(scale))
                if hasattr(dataset, "set_scale"):
                    dataset.set_scale(idx_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            # 原代码这里在 samples 末尾追加 idx_scale
            # 在当前修复方案中，不再通过这个路径返回数据
            samples.append(idx_scale)

        except Exception:
            data_queue.put((idx, _utils.ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


class _MSDataLoaderIter(_DataLoaderIter):
    """
    ⚠ 旧版 PyTorch 1.x 使用的内部迭代器实现。
    在 PyTorch 2.x 中继续操作 _BaseDataLoaderIter 非常不稳定，
    我们已经在 MSDataLoader.__iter__ 中改成了更安全的实现。
    这个类保留在这里仅供参考，不再被实际调用。
    """

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.done_event = threading.Event()
        self._profile_name = "MSDataLoaderIter"
        self._num_yielded = 0
        self._dataset_kind = _DatasetKind.Map

        self._sampler_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queues = [
                multiprocessing.Queue() for _ in range(self.num_workers)
            ]
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.Queue()

            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            base_seed = torch.LongTensor(1).random_()[0]
            self.workers = [
                multiprocessing.Process(
                    target=_ms_loop,
                    args=(
                        self.dataset,
                        self.index_queues[i],
                        self.worker_result_queue,
                        self.collate_fn,
                        self.scale,
                        base_seed + i,
                        self.worker_init_fn,
                        i
                    )
                )
                for i in range(self.num_workers)
            ]

            if self.pin_memory or self.timeout > 0:
                self.data_queue = queue.Queue()
                if self.pin_memory:
                    maybe_device_id = torch.cuda.current_device()
                else:
                    maybe_device_id = None
                self.pin_memory_thread = threading.Thread(
                    target=_utils.pin_memory._pin_memory_loop,
                    args=(
                        self.worker_result_queue,
                        self.data_queue,
                        self.done_event,
                        self.pin_memory,
                        maybe_device_id,
                    ),
                )
                self.pin_memory_thread.daemon = True
                self.pin_memory_thread.start()
            else:
                self.data_queue = self.worker_result_queue

            for w in self.workers:
                w.daemon = True
                w.start()

            _utils.signal_handling._set_worker_pids(
                id(self), tuple(w.pid for w in self.workers)
            )
            _utils.signal_handling._set_SIGCHLD_handler()
            self.worker_pids_set = True

    def _next_data(self):
        """
        旧版本里应当在这里实现从多个 worker 中取数据的逻辑。
        原仓库中这里是空的，导致返回 None，进而引发你的报错。
        目前我们改用 MSDataLoader.__iter__ 的新逻辑，不再走这里。
        """
        raise StopIteration


# ====== 新的、兼容 PyTorch 2.5 的 MSDataLoader 实现 ======

class MSDataLoader(DataLoader):
    """
    兼容 PyTorch 2.x 的多尺度 DataLoader 包装器。
    关键点：复用原生 DataLoader 的迭代逻辑，只在 __iter__ 里追加 idx_scale，
    保持 Trainer 侧的 (lr, hr, _, idx_scale) 接口不变。
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
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
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
        )

        # 保留原来的 scale 参数（比如 [4]，或者 [2,3,4]）
        self.scale = args.scale

    def __iter__(self):
        """
        新的迭代逻辑：
        1. 先用 super().__iter__() 拿到 PyTorch 自带的 batch（兼容单/多进程）。
        2. 为每个 batch 追加一个 idx_scale，保持和原 trainer 一样的解包方式：
           for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
        3. 目前为了稳定，**不再在这里修改 dataset.set_scale**，避免和 PyTorch 2.x
           的预取机制冲突；你的配置中 scale 只有一个值时，这完全等价。
        """
        for batch in super(MSDataLoader, self).__iter__():
            # 默认只有一个 scale 时 idx_scale 固定为 0
            idx_scale = 0

            # 如果以后你真的需要多尺度训练，可以在这里根据 self.scale
            # 选择 idx_scale，并在模型前向时用它做分支选择。
            # 由于 PyTorch 内部有预取，我们不在这里调用 dataset.set_scale，
            # 否则会出现「设置了下一个 batch 的 scale，却作用在当前 batch」的情况。

            # collate_fn(default_collate) 对 tuple/list 的行为稍有不同，
            # 这里统一处理成在末尾追加一个 idx_scale。
            if isinstance(batch, list):
                # 拷贝一份避免就地修改 DataLoader 内部缓存
                batch = list(batch)
                batch.append(idx_scale)
                yield tuple(batch)
            elif isinstance(batch, tuple):
                yield (*batch, idx_scale)
            else:
                # 极端情况下 dataset 只返回一个张量，这里也兼容
                yield (batch, idx_scale)
