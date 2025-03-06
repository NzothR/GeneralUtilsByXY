"""
通用工具模块
"""


import os
import cv2
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from typing import Callable, Set, Optional, Tuple, List, Dict

# 自定义模块
from Constants import logger, Metrics, AverageType
from MetricsUtils import MetricsUtils


class FileUtils:
    def __init__(self):
        pass

    @staticmethod
    def process_files_with_filter(directory: str, operation: Callable[[str], None], 
                                 filter_ext: Optional[Set[str]] = None, recursive: bool = False) -> None:
        """
        遍历指定文件夹下的每个文件, 根据文件类型限定参数过滤文件,
        并对每个过滤后的文件应用提供的操作函数。

        :param directory: 要遍历的文件夹路径
        :param operation: 应用于每个文件的操作函数
        :param filter_ext: 用来过滤文件的扩展名集合(如 {'.txt', '.csv'})
        :param recursive: 是否递归遍历子文件夹，默认为False
        """
        # 判断目录是否存在
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录 {directory} 不存在")

        # 定义内部递归函数来处理文件夹
        def process_directory(dir_path):
            # 遍历目录下的所有文件和文件夹
            for entry in os.listdir(dir_path):
                # 构建完整的文件路径
                file_path = os.path.join(dir_path, entry)
                # 检查是否为文件并且是指定文件
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(entry)[1].lower()
                    if filter_ext is None or file_ext in filter_ext:
                        operation(file_path)
                elif os.path.isdir(file_path) and recursive:
                    # 如果是目录且设置了递归，则递归调用自身
                    process_directory(file_path)

        # 开始处理根目录
        process_directory(directory)


class TrainUtils:
    """训练工具类"""
    def __init__(self):
        # 初始化指标计算工具
        self.metrics_utils = MetricsUtils()

    @staticmethod
    def is_dataloader_empty(data_loader: DataLoader) -> bool:
        """
        判断DataLoader是否为空
        :param data_loader: DataLoader对象
        :return: 如果DataLoader为空, 返回True, 否则返回False
        """
        return len(data_loader.dataset) == 0
    @staticmethod
    def get_num_classes(data_loaders: List[DataLoader]) -> int:
        """
        从数据加载器中推导出类别数量
        :param data_loaders: 数据加载器列表
        :return: 类别数量
        """
        all_labels = set()

        for data_loader in data_loaders:
            for _, labels in data_loader:
                all_labels.update(labels.numpy())

        return len(all_labels)

    def calculate_metric(self, metric: Metrics, y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int, average_type: AverageType = AverageType.MACRO, device: str = 'cpu') -> float:
        """
        根据预测值和真实值计算对应的指标
        :param metric: 指标类型
        :param y_pred: 预测值
        :param y_true: 真实值
        :param num_classes: 类别数量
        :param average_type: 平均类型, 可选值: MACRO, MICRO
        :param device: 计算设备
        :return: 指标计算结果
        """
        # 获取预测值
        predictions = torch.argmax(y_pred, dim=1)
        # 计算指标
        if metric == Metrics.ACCURACY:
            return self.metrics_utils.calculate_accuracy(predictions, y_true, num_classes, device)
        if metric == Metrics.PRECISION:
            return self.metrics_utils.calculate_precision(predictions, y_true, num_classes, average_type, device)
        if metric == Metrics.RECALL:
            return self.metrics_utils.calculate_recall(predictions, y_true, num_classes, average_type, device)
        if metric == Metrics.F1_SCORE:
            return self.metrics_utils.calculate_f1_score(predictions, y_true, num_classes, average_type, device)

    def general_train(self,
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
        net: nn.Module,
        criterion: Optional[Callable],
        num_classes: int = None,
        optimizer: Optional[optim.Optimizer] = None,
        epochs: int = 20,
        learning_rate: float = 0.1,
        device: str = "cpu",
        lr_scheduler: Optional[_LRScheduler] = None,
        metrics: Optional[List[Metrics]] = None,
        log_interval: int = 10,
        patience: Optional[int] = None,
        init_weights: Optional[Callable] = None,
        checkpoint_path: Optional[str] = None,
        save_every_n_epochs: int = 10
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        通用训练函数
        :param train_data_loader: 训练数据集的DataLoader
        :param test_data_loader: 测试数据集的DataLoader
        :param net: 待训练的网络模型
        :param epochs: 训练周期数, 默认为20
        :param learning_rate: 学习率, 默认为0.1
        :param device: 计算设备, 例如"cpu"或"cuda", 默认为"cpu"
        :param criterion: 损失函数
        :param num_classes: 类别数量, 默认为None(自动根据数据集获取)
        :param optimizer: 优化器实例, 默认为None(内部将创建一个SGD优化器)
        :param lr_scheduler: 学习率调度器, 默认为None, 即不使用学习率调度器
        :param metrics: 评估指标列表, 默认为None, 默认使用准确率评估
        :param log_interval: 日志记录间隔(每多少个批次打印一次信息), 默认为10
        :param patience: 早停机制的容忍度, 默认为None(不启用早停)
        :param init_weights: 权重初始化函数, 默认为None
        :param checkpoint_path: 检查点保存路径, 默认为None
        :param save_every_n_epochs: 每隔多少轮保存一次模型, 默认为10
        :return: 训练好的网络模型和训练过程记录
        """

        # 初始化类别数量
        if num_classes is None:
            num_classes = self.get_num_classes([train_data_loader, test_data_loader])

        # 设置优化器
        if optimizer is None:
            # 默认使用SGD优化器
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        # 设置评价指标
        if metrics is None:
            metrics = [Metrics.ACCURACY]

        # 存储训练过程
        history = {"Loss": []}
        for metric in metrics:
            history[metric.value] = []

        # 将网络模型移动到指定设备
        net.to(device)

        # 基础准确率
        best_accuracy = 0.0
        # 开始训练
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            # 构建tqdm进度条, 用于监控训练过程
            process_bar = tqdm(train_data_loader, unit='step')

            # 存储每个批次的对应指标
            metric_list = {}
            # 打开训练模式
            net.train()

            # 对DataLoader中的每个批次进行训练
            for step, (train_images, labels) in enumerate(process_bar):
                # 将数据移动到指定设备
                train_images = train_images.to(device)
                labels = labels.to(device)

                # 清空梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = net(train_images)

                # 计算损失
                loss = criterion(outputs, labels)

                # 计算指标
                for metric in metrics:
                    metric_list[metric.value] = self.calculate_metric(metric = metric, y_pred = outputs, y_true = labels, num_classes = num_classes, device = device)

                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

                # 构建描述字符串
                description = f"Epoch: {epoch} "
                for key, value in metric_list.items():
                    description += f"{key}: {value:.4f} "

                # 设置进度条的描述
                process_bar.set_description(description)

            # 在每个epoch结束后进行测试集评估
            net.eval()
            with torch.no_grad():
                all_test_metrics = {metric.value: 0.0 for metric in metrics}
                total_loss = 0.0

                for test_imgs, labels in test_data_loader:
                    test_imgs = test_imgs.to(device)
                    labels = labels.to(device)
                    outputs = net(test_imgs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                    # 计算并存储所有指标
                    for metric in metrics:
                        metric_value = self.calculate_metric(metric=metric, y_pred=outputs, y_true=labels, num_classes=num_classes, device=device)
                        all_test_metrics[metric.value] += metric_value

                # 计算平均损失
                test_loss = total_loss / len(test_data_loader)
                history['Loss'].append(test_loss)

                # 计算并存储平均指标
                for metric in metrics:
                    all_test_metrics[metric.value] = all_test_metrics[metric.value] / len(test_data_loader)
                    history[metric.value].append(all_test_metrics[metric.value])


                # 打印测试结果
                print(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}")
                for metric in metrics:
                    print(f"Epoch: {epoch}, Test {metric.value}: {all_test_metrics[metric.value]:.4f}")

            # 学习率调度
            if lr_scheduler is not None:
                lr_scheduler.step(test_loss)

            # 关闭进度条
            process_bar.close()
        # 结束时间记录
        end_time = time.time()
        print("训练结束, 耗时：%.2f秒" % (end_time - start_time))
        return net, history

    @staticmethod
    def plot_history(history, save_path=None) -> None:
        """
        绘制训练过程记录中的损失曲线和指标曲线，每个指标一个图表
        :param history: 训练过程记录, 包含损失和指标等数据
        :param save_path: 图表保存路径前缀，如果给出，则会保存图表
        """
        # 检查history是否为空
        if not history:
            print("History is empty.")
            return
        # 确保目录存在
        try:
            os.makedirs(save_path, exist_ok=True)
        except OSError as e:
            logger.error(f"目录 {save_path} 创建失败: {e}")
            raise FileNotFoundError(f"无法创建目录: {save_path}") from e

        epochs = np.arange(1, len(next(iter(history.values()))) + 1)  # 获取epochs数量并生成从1开始的序列

        # 创建一个新的图形窗口
        for metric, values in history.items():
            # 创建一个单独的子图用于当前指标
            fig, ax = plt.subplots(figsize=(10, 5))

            # 绘制曲线
            ax.plot(epochs, values, label=f'{metric}')  # 使用epochs作为x轴
            ax.set_title(f'Training {metric} Over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()

            # 设置x轴的刻度为每个epoch出现一次
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))

            plt.tight_layout()

            # 保存图表
            if save_path:
                plt.savefig(f'{save_path}_{metric}.png')

            # 显示图表
            plt.show()
