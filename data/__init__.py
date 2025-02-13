import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def load_dataset_module(dataset_name):
    """动态加载指定的 dataset 模块。

    将加载 "data/[dataset_name]_dataset.py" 文件。
    在该文件中，应该存在一个继承自 BaseDataset 的类。
    """
    dataset_module_name = f"data.{dataset_name}_dataset"
    dataset_module = importlib.import_module(dataset_module_name)

    target_class_name = dataset_name.replace('_', '') + 'dataset'
    target_class = None
    for class_name, cls in dataset_module.__dict__.items():
        if class_name.lower() == target_class_name.lower() and issubclass(cls, BaseDataset):
            target_class = cls
            break

    if target_class is None:
        raise NotImplementedError(
            f"In {dataset_module_name}.py, a subclass of BaseDataset matching class name {target_class_name.lower()} should be implemented."
        )

    return target_class


def get_dataset_options(dataset_name):
    """获取指定数据集类的 modify_commandline_options 静态方法。"""
    dataset_class = load_dataset_module(dataset_name)
    return dataset_class.modify_commandline_options


def create_data_loader(opt):
    """根据选项创建数据集实例。

    该函数会创建一个数据加载器实例，负责数据集的加载过程。

    示例：
        >>> from data import create_data_loader
        >>> data_loader = create_data_loader(opt)
    """
    loader = DatasetLoader(opt)
    return loader.load()


class DatasetLoader:
    """封装数据集类的多线程数据加载器。"""

    def __init__(self, opt):
        """初始化类，进行数据集创建和数据加载器的配置。

        1. 创建数据集实例
        2. 创建多线程数据加载器
        """
        self.opt = opt
        dataset_class = load_dataset_module(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print(f"Dataset [{type(self.dataset).__name__}] created successfully.")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads)
        )

    def load(self):
        """返回当前数据加载器实例。"""
        return self

    def __len__(self):
        """返回数据集中的样本数量，考虑最大数据集大小选项。"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """按批次迭代数据集并返回数据。"""
        for idx, data_batch in enumerate(self.dataloader):
            if idx * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data_batch
