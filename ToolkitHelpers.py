"""
辅助工具类
"""


import torch
import numpy as np


from Constants import logger


class ToolkitHelpers:
    """
    工具类
    """
    def __init__(self):
        pass

    @staticmethod
    def get_unique_random(min_val:int, max_val:int, num:int, seed_value:int = 42) -> list:
        """
        获取不重复的随机数, 包括最小值和最大值
        :param seed_value: 随机种子
        :param min_val: 最小值
        :param max_val: 最大值
        :param num: 随机数个数
        :return: 随机数列表
        """
        np.random.seed(seed_value)
        return np.random.choice(np.arange(min_val, max_val + 1), num, replace=False).tolist()

    @staticmethod
    def random_split_data_tensor(
        features: torch.Tensor,
        labels: torch.Tensor,
        split_ratio: tuple = (8, 2),
        seed: int = 42
    ) -> tuple:
        """
        随机切分数据集（特征和标签），支持动态比例和随机种子控制[1,7]

        :param features: 特征张量（形状：[样本数, 特征维度])
        :param labels: 标签张量（形状：[样本数])
        :param split_ratio: 分割比例元组（例如(80,20)表示训练集80%、测试集20%)
                        支持多比例分割（如(60,20,20))
        :param seed: 随机种子，确保可重复性
        :return: 分割后的特征和标签元组(按split_ratio顺序返回)
        """
        # 参数校验
        if len(split_ratio) < 2:
            raise ValueError("split_ratio至少需要两个元素")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(f"特征样本数({features.shape[0]})与标签样本数({labels.shape[0]})不一致")

        # 计算分割长度（兼容任意比例，不强制要求和为100）
        total_samples = features.shape[0]
        split_sizes = [int(total_samples * ratio / sum(split_ratio)) for ratio in split_ratio]

        # 处理余数分配问题
        remainder = total_samples - sum(split_sizes)
        split_sizes[-1] += remainder  # 余数加到最后一部分

        # 使用PyTorch的随机分割函数[1,7]
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_samples, generator=generator)

        # 按分割尺寸划分索引
        split_indices = []
        start_idx = 0
        for size in split_sizes:
            end_idx = start_idx + size
            split_indices.append(indices[start_idx:end_idx])
            start_idx = end_idx

        # 返回分割结果
        result = []
        for idx in split_indices:
            result.append(features[idx])
            result.append(labels[idx])
        return tuple(result)


def test_random_split_data_tensor():
    import pytest
    # 生成固定10样本数据
    features = torch.arange(10).reshape(10, 1).float()
    labels = torch.tensor([9,8,7,6,5,4,3,2,1,0], dtype=torch.int8)

    logger.info(f"\n原始特征数据：\n{features.squeeze().numpy()}")
    logger.info(f"原始标签数据：\n{labels.numpy()}\n")

    # 测试1：基本分割
    splits = ToolkitHelpers.random_split_data_tensor(
        features, labels, (8, 2), seed=42
    )
    logger.info("=== 测试1 分割结果 ===")
    logger.info(f"训练特征样本索引：{splits[0].squeeze().numpy().astype(int)}")
    logger.info(f"训练标签对应数据：{splits[1].numpy()}")
    logger.info(f"测试特征样本索引：{splits[2].squeeze().numpy().astype(int)}")
    logger.info(f"测试标签对应数据：{splits[3].numpy()}\n")

    assert len(splits) == 4, "输出元组长度应为split_ratio数量*2"
    assert splits[0].shape[0] == 8, "训练集特征数量错误"
    assert splits[3].shape[0] == 2, "测试集标签数量错误"

    # 测试2：余数处理（样本总数非整除数）
    splits = ToolkitHelpers.random_split_data_tensor(
        features, labels, (7, 0), seed=42
    )
    logger.info("=== 测试2 分割结果 ===")
    logger.info(f"训练特征：{splits[0].squeeze().numpy().astype(int)}")
    logger.info(f"剩余特征：{splits[2].squeeze().numpy().astype(int)}\n")

    assert sum([splits[i].shape[0] for i in [0,2]]) == 10, "总样本数丢失"

    # 测试3：随机种子可重复性
    splits1 = ToolkitHelpers.random_split_data_tensor(
        features, labels, (8,2), seed=42
    )
    splits2 = ToolkitHelpers.random_split_data_tensor(
        features, labels, (8,2), seed=42
    )
    logger.info("=== 测试3 可重复性验证 ===")
    logger.info(f"第一次分割测试标签：{splits1[3].numpy()}")
    logger.info(f"第二次分割测试标签：{splits2[3].numpy()}\n")

    assert torch.equal(splits1[0], splits2[0]), "相同种子应输出相同结果"

    # 测试4：多比例分割（训练/验证/测试）
    splits = ToolkitHelpers.random_split_data_tensor(
        features, labels, (6,2,2), seed=42
    )
    logger.info("=== 测试4 三组分割 ===")
    logger.info(f"训练集索引：{splits[0].squeeze().numpy().astype(int)}")
    logger.info(f"验证集索引：{splits[2].squeeze().numpy().astype(int)}")
    logger.info(f"测试集索引：{splits[4].squeeze().numpy().astype(int)}\n")

    assert splits[0].shape[0] == 6, "三部分分割训练集错误"
    assert splits[5].shape[0] == 2, "三部分分割测试集错误"

    # 测试5：异常处理
    with pytest.raises(ValueError):
        ToolkitHelpers.random_split_data_tensor(features, labels[:5], (8,2))
    with pytest.raises(ValueError):
        ToolkitHelpers.random_split_data_tensor(features, labels, (10,))

if __name__ == '__main__':
    logger.info("工具类测试启动")  # 使用自定义logger[4,6]
    logger.enable_console()  # 激活控制台输出
    test_random_split_data_tensor()
    logger.disable_console()
