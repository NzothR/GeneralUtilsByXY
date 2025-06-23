# TrainUtils.py



"""
训练工具类
"""
import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from typing import Callable, Optional, Tuple, List, Dict

# 自定义模块
from .Constants import logger, Metrics, AverageType
from .MetricsUtils import MetricsUtils

class TrainUtils:
    """
    训练工具类
    """
    def __init__(self):
        # 初始化指标计算工具
        self.metrics_utils = MetricsUtils()

    @staticmethod
    def is_dataloader_empty(data_loader: DataLoader) -> bool:
        """
        判断DataLoader是否为空
        Args:
            data_loader: DataLoader对象
        Returns:
            如果DataLoader为空, 返回True, 否则返回False
        """
        try:
            return len(data_loader.dataset) == 0
        except (TypeError, NotImplementedError):
            logger.error("无法获取DataLoader的dataset长度, 可能数据集未实现__len__方法")
            raise ValueError("无法获取DataLoader的dataset长度, 可能数据集未实现__len__方法")
    @staticmethod
    def get_num_classes(data_loaders: List[DataLoader]) -> int:
        """
        从数据加载器中推导出类别数量
        Args:
            data_loaders: 数据加载器列表
        Returns:
            类别数量
        """
        all_labels = set()

        for data_loader in data_loaders:
            for _, labels in data_loader:
                # 确保标签转换为CPU后处理，兼容不同设备的数据
                all_labels.update(labels.cpu().numpy())
        if len(all_labels) == 0:
            logger.error("数据加载器中未发现有效标签，请检查数据集！")
            raise ValueError("数据加载器中未发现有效标签，请检查数据集！")

        return len(all_labels)

    def calculate_metric(self, metric: Metrics, y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int, average_type: AverageType = AverageType.MACRO, device: str = 'cpu') -> float:
        """
        根据预测值和真实值计算对应的指标
        Args:
            metric: 指标类型
            y_pred: 预测值
            y_true: 真实值
            num_classes: 类别数量
            average_type: 平均类型, 可选值: MACRO, MICRO
            device: 计算设备
        Returns:
            指标计算结果
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

    def general_classification_train(self,
            train_data_loader: DataLoader,
            val_data_loader: DataLoader,
            model: nn.Module,
            model_name: str,
            criterion: Optional[Callable],
            num_classes: int = None,
            optimizer: Optional[optim.Optimizer] = None,
            epochs: int = 20,
            learning_rate: float = 0.1,
            device: str = "cpu",
            lr_scheduler: Optional[_LRScheduler] = None,
            metrics: Optional[List[Metrics]] = None,
            checkpoint_path: Optional[str] = None,
            checkpoint_interval: int = 5,
            use_amp: bool = False
        ) -> Tuple[nn.Module, Dict[str, List[float]]]:
            """
            通用分类模型训练函数
            Args:
                train_data_loader: 训练数据集的DataLoader
                val_data_loader: 验证数据集的DataLoader
                model: 待训练的网络模型
                model_name: 模型名称
                criterion: 损失函数
                num_classes: 类别数量,默认为None(自动根据数据集获取)
                optimizer: 优化器实例, 默认为None(内部将创建一个SGD优化器)
                epochs: 训练周期数, 默认为20
                learning_rate: 学习率, 默认为0.1
                device: 计算设备, 例如"cpu"或"cuda", 默认为"cpu"
                lr_scheduler: 学习率调度器, 默认为None, 即不使用学习率调度器
                metrics: 评估指标列表, 默认为None, 默认使用准确率评估
                checkpoint_path: 检查点保存路径, 默认为None
                checkpoint_interval: 每隔多少轮保存一次模型, 默认为5
                use_amp: 是否启用自动混合精度训练, 默认为False
            Returns:
                训练好的网络模型和训练过程记录
            """
            # 确保检查点保存路径存在（当提供时）
            if checkpoint_path is not None:
                try:
                    os.makedirs(checkpoint_path, exist_ok=True)
                except OSError as e:
                    logger.error(f"目录 {checkpoint_path} 创建失败: {e}")
                    raise FileNotFoundError(f"无法创建目录: {checkpoint_path}") from e

            # 初始化类别数量
            if num_classes is None:
                num_classes = self.get_num_classes([train_data_loader, val_data_loader])

            # 设置优化器
            if optimizer is None:
                # 默认使用SGD优化器
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

            # 设置评价指标
            if metrics is None:
                metrics = [Metrics.ACCURACY]

            # 存储训练过程（同时记录训练集和验证集指标）
            history = {"Train Loss": [], "Val Loss": []}
            for metric in metrics:
                history[f"Train {metric.value}"] = []
                history[f"Val {metric.value}"] = []

            # 将网络模型移动到指定设备
            model.to(device)

            # 初始化最佳验证准确度
            best_val_acc = 0

            # 根据use_amp参数初始化自动混合精度（AMP）
            scaler = torch.amp.GradScaler(
                    device='cuda',
                    enabled=use_amp and (device == "cuda"),
                    init_scale=2.**16
                )

            # 开始训练
            start_time = time.time()
            for epoch in range(1, epochs + 1):
                # ===================== 训练阶段 =====================
                model.train()
                total_train_loss = 0.0
                train_metric_sums = {metric.value: 0.0 for metric in metrics}

                # 构建tqdm进度条, 显示训练过程
                process_bar = tqdm(train_data_loader, desc=f"Epoch {epoch}/{epochs}", unit="step")

                for step, (train_images, labels) in enumerate(process_bar):
                    train_images = train_images.to(device)
                    labels = labels.to(device)

                    # 梯度清零
                    optimizer.zero_grad()

                    # 前向传播（根据use_amp启用混合精度）
                    with torch.amp.autocast(
                        device_type='cuda',
                        enabled=use_amp and (device == "cuda"),
                        dtype=torch.float16  # 显式指定精度类型
                    ):
                        outputs = model(train_images)
                        loss = criterion(outputs, labels)

                    # 反向传播与参数更新（自动缩放损失）
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # 累加损失和指标
                    total_train_loss += loss.item()
                    for metric in metrics:
                        metric_value = self.calculate_metric(
                            metric=metric,
                            y_pred=outputs,
                            y_true=labels,
                            num_classes=num_classes,
                            device=device
                        )
                        train_metric_sums[metric.value] += metric_value

                    # 更新进度条显示（显示当前平均指标）
                    avg_loss = total_train_loss / (step + 1)
                    desc = f"Train Loss: {avg_loss:.4f}"
                    for metric in metrics:
                        avg_metric = train_metric_sums[metric.value] / (step + 1)
                        desc += f" | Train {metric.value}: {avg_metric:.4f}"
                    process_bar.set_description(desc)

                # 计算训练集平均指标
                avg_train_loss = total_train_loss / len(train_data_loader)
                history["Train Loss"].append(avg_train_loss)
                for metric in metrics:
                    avg_metric = train_metric_sums[metric.value] / len(train_data_loader)
                    history[f"Train {metric.value}"].append(avg_metric)

                # ===================== 验证阶段 =====================
                model.eval()
                total_val_loss = 0.0
                val_metric_sums = {metric.value: 0.0 for metric in metrics}

                with torch.no_grad():
                    for val_images, labels in val_data_loader:
                        val_images = val_images.to(device)
                        labels = labels.to(device)

                        # 前向传播（验证阶段默认不启用混合精度）
                        outputs = model(val_images)
                        loss = criterion(outputs, labels)

                        total_val_loss += loss.item()
                        for metric in metrics:
                            metric_value = self.calculate_metric(
                                metric=metric,
                                y_pred=outputs,
                                y_true=labels,
                                num_classes=num_classes,
                                device=device
                            )
                            val_metric_sums[metric.value] += metric_value

                # 计算验证集平均指标
                avg_val_loss = total_val_loss / len(val_data_loader)
                history["Val Loss"].append(avg_val_loss)
                for metric in metrics:
                    avg_metric = val_metric_sums[metric.value] / len(val_data_loader)
                    history[f"Val {metric.value}"].append(avg_metric)
                # 计算准确率
                val_acc = self.calculate_metric(
                                metric=Metrics.ACCURACY,
                                y_pred=outputs,
                                y_true=labels,
                                num_classes=num_classes,
                                device=device
                            )

                # ===================== 日志记录 =====================
                logger.info(f"\nEpoch: {epoch}/{epochs}")
                logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                for metric in metrics:
                    logger.info(f"Train {metric.value}: {history[f'Train {metric.value}'][-1]:.4f} | "
                                f"Val {metric.value}: {history[f'Val {metric.value}'][-1]:.4f}")

                # ===================== 学习率调整 =====================
                if lr_scheduler is not None:
                    # 根据调度器类型调整学习率
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        lr_scheduler.step(avg_val_loss)
                    else:
                        lr_scheduler.step()

                # ===================== 模型保存 =====================
                if checkpoint_path is not None:
                    # 保存定期检查点
                    if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                        checkpoint_file = os.path.join(
                            checkpoint_path,
                            f"{model_name}_epoch_{epoch}.pth"
                        )
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                            'train_loss': avg_train_loss,
                            'val_loss': avg_val_loss,
                        }, checkpoint_file)
                        logger.info(f"检查点已保存至 {checkpoint_file}")

                    # 保存最佳模型（根据验证准确率）
                    if best_val_acc < val_acc:
                        best_val_acc = val_acc
                        best_model_path = os.path.join(checkpoint_path, f"{model_name}_best.pth")
                        torch.save(model.state_dict(), best_model_path)
                        logger.info(f"最佳模型已保存至 {best_model_path} (Val Acc: {val_acc:.4f})")

            # ===================== 最终处理 =====================
            end_time = time.time()

            # 保存最终模型
            if checkpoint_path is not None:
                final_model_path = os.path.join(checkpoint_path, f"{model_name}_final.pth")
                torch.save(model.state_dict(), final_model_path)
                logger.info(f"训练完成, 最终模型已保存至 {final_model_path}")

            logger.info(f"总训练时间: {end_time - start_time:.2f}秒")
            return model, history


    @staticmethod
    def plot_history(history, save_path=None) -> None:
        """
        绘制训练过程记录中的损失曲线和指标曲线, 每个指标一个图表
        Args:
            history: 训练过程记录, 包含损失和指标等数据
            save_path: 图表保存路径前缀, 如果给出, 则会保存图表
        """
        # 检查history是否为空
        if not history:
            logger.info("History is empty.")
            return
        # 确保目录存在
        try:
            if save_path is not None:
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
            if save_path is not None:
                save_metric_path = os.path.normpath(save_path)
                # 去除空格
                save_name = metric.replace(' ', '')
                save_metric_path = os.path.join(save_metric_path, f'{save_name}.png')
                plt.savefig(save_metric_path)
                logger.info(f"{save_name}.png'图表已保存至 {save_metric_path}")

            # 显示图表
            plt.show()
