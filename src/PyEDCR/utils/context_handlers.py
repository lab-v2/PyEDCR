import abc
import torch
import matplotlib.pyplot as plt
import time
from PyEDCR.PyEDCR import utils

if utils.is_local():
    if torch.backends.mps.is_available():
        from torch import mps
        import tqdm


class Context(abc.ABC):
    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ClearCache(Context):
    def __init__(self,
                 device: torch.device):

        self.device_backend = {'cuda': torch.cuda,
                               'mps': mps if utils.is_local() and torch.backends.mps.is_available() else None,
                               'cpu': None}[device.type]

    def __enter__(self):
        if self.device_backend:
            self.device_backend.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device_backend:
            self.device_backend.empty_cache()


class Plot(Context):
    def __init__(self,
                 fig_sizes: tuple = None):
        if fig_sizes:
            plt.figure(figsize=fig_sizes)

    def __enter__(self):
        plt.cla()
        plt.clf()

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show()
        plt.cla()
        plt.clf()


class WrapTQDM(Context):
    def __init__(self,
                 total: int = None):
        self.tqdm = tqdm.tqdm(total=total) if utils.is_local() else None

    def __enter__(self):
        if self.tqdm is not None:
            return self.tqdm.__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.tqdm is not None:
            return self.tqdm.__exit__(exc_type, exc_value, exc_tb)

    def update(self,
               n: int = 1):
        if self.tqdm is not None:
            return self.tqdm.update(n)


class TimeWrapper(Context):
    def __init__(self,
                 s: str = None):
        self.s = s

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.s is not None:
            print(self.s)

        print(f'Total time: {utils.format_seconds(int(time.time() - self.start))}')
