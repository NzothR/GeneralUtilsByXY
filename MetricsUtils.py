import torch

from Constants import AverageType



class MetricsUtils:
    """指标计算工具类"""
    def __init__(self):
        # 缓存混淆矩阵
        self.confusion_matrix_cache = None
        # 缓存预测值
        self.pred_cache = None
        # 缓存真实值
        self.true_cache = None
        # 缓存TP, FN, FP
        self.tp_cache = None
        self.fn_cache = None
        self.fp_cache = None

    def calculate_confusion_matrix(self, y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int, device: str = 'cpu') -> torch.Tensor:
        """
        计算多分类的混淆矩阵
        :param y_pred: 预测值，形状为 (N,)
        :param y_true: 真实值，形状为 (N,)
        :param num_classes: 类别数量
        :param device: 指定张量计算设备
        :return: 混淆矩阵
        """
        # 移动预测值和真实值到指定设备
        y_pred, y_true = y_pred.to(device), y_true.to(device)
        # 检查缓存
        if (self.confusion_matrix_cache is not None and
            torch.equal(y_pred, self.pred_cache) and 
            torch.equal(y_true, self.true_cache)):
            return self.confusion_matrix_cache


        # 初始化混淆矩阵
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

        # 构建混淆矩阵
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        # 更新缓存
        self.confusion_matrix_cache = confusion_matrix.clone()  # 使用 clone 来避免意外修改
        self.pred_cache = y_pred.clone()
        self.true_cache = y_true.clone()

        # 计算各个类别的TP，FP，FN，并缓存
        self.tp_cache = confusion_matrix.diag().clone()
        self.fp_cache = (confusion_matrix.sum(dim=0) - self.tp_cache).clone()
        self.fn_cache = (confusion_matrix.sum(dim=1) - self.tp_cache).clone()

        return confusion_matrix

    def calculate_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor, num_classes: int, device:str = 'cpu') -> float:
        """
        计算准确率
        :param predictions: 预测值，形状为 (N,)
        :param labels: 真实值，形状为 (N,)
        :param num_classes: 类别数量
        :param device: 指定张量计算设备
        :return: 准确率
        """
        # 计算混淆矩阵
        confusion_matrix = self.calculate_confusion_matrix(predictions, labels, num_classes, device)
        accuracy = (confusion_matrix.diag().sum() / confusion_matrix.sum()).item()
        return accuracy

        # 新增辅助方法
    def _calculate_macro_precision(self, num_classes: int, device: str = 'cpu') -> torch.Tensor:
        """
        辅助方法，计算宏精确率
        :param num_classes: 类别数量
        :param device: 指定张量计算设备
        :return: 宏精确率向量
        """
        precision_classes = torch.zeros(num_classes, device=device)
        for i in range(num_classes):
            tp = self.tp_cache[i]
            fp = self.fp_cache[i]
            # 避免除0
            if tp + fp == 0:
                precision_classes[i] = 0
            else:
                precision_classes[i] = tp / (tp + fp)
        return precision_classes

    def calculate_precision(self, predictions: torch.Tensor, labels: torch.Tensor, num_classes: int, average_type: AverageType = AverageType.MACRO, device: str = 'cpu') -> float:
        """
        计算精确率
        :param predictions: 预测值，形状为 (N,)
        :param labels: 真实值，形状为 (N,)
        :param num_classes: 类别数量
        :param average_type: 平均方式，默认为宏平均
        :param device: 指定张量计算设备
        :return: 精确率
        """
        # 计算混淆矩阵, 并缓存
        confusion_matrix = self.calculate_confusion_matrix(predictions, labels, num_classes, device)
        # 计算宏/微精确率
        if average_type == AverageType.MACRO:
            precision_classes = self._calculate_macro_precision(num_classes, device)
            return torch.mean(precision_classes).item()
        elif average_type == AverageType.MICRO:
            tp_sum = self.tp_cache.to(device).sum()
            fp_sum = self.fp_cache.to(device).sum()
            if tp_sum + fp_sum == 0:
                return 0
            micro_precision = (tp_sum / (tp_sum + fp_sum)).item()
            return micro_precision
        else:
            raise ValueError("Invalid precision average type")

    def _calculate_macro_recall(self, num_classes: int, device:str = 'cpu') -> torch.Tensor:
        """
        辅助方法，计算宏召回率
        :param num_classes: 类别数量
        :param device: 指定张量计算设备
        :return: 宏召回率向量
        """
        recall_classes = torch.zeros(num_classes, device=device)
        for i in range(num_classes):
            tp = self.tp_cache[i]
            fn = self.fn_cache[i]
            if tp + fn == 0:
                recall_classes[i] = 0
            else:
                recall_classes[i] = tp / (tp + fn)
        return recall_classes
    def calculate_recall(self, predictions: torch.Tensor, labels: torch.Tensor, num_classes: int, average_type: AverageType = AverageType.MACRO, device: str = 'cpu') -> float:
        """
        计算召回率
        :param predictions: 预测值，形状为 (N,)
        :param labels: 真实值，形状为 (N,)
        :param num_classes: 类别数量
        :param average_type: 平均方式，默认为宏平均
        :param device: 指定张量计算设备
        :return: 召回率
        """
        # 计算混淆矩阵, 并缓存
        confusion_matrix = self.calculate_confusion_matrix(predictions, labels, num_classes, device)
        # 计算宏/微召回率
        if average_type == AverageType.MACRO:
            recall_classes = self._calculate_macro_recall(num_classes, device)
            return torch.mean(recall_classes).item()
        elif average_type == AverageType.MICRO:
            tp_sum = self.tp_cache.to(device).sum()
            fn_sum = self.fn_cache.to(device).sum()
            if tp_sum + fn_sum == 0:
                return 0
            micro_recall = (tp_sum / (tp_sum + fn_sum)).item()
            return micro_recall
        else:
            raise ValueError("Invalid recall average type")

    def calculate_f1_score(self, predictions: torch.Tensor, labels: torch.Tensor, num_classes: int, average_type: AverageType = AverageType.MACRO, device: str = 'cpu') -> float:
        """
        计算F1分数
        :param predictions: 预测值，形状为 (N,)
        :param labels: 真实值，形状为 (N,)
        :param num_classes: 类别数量
        :param average_type: 平均方式，默认为宏平均
        :param device: 指定张量计算设备
        :return: F1分数
        """
        # 计算混淆矩阵, 并缓存
        confusion_matrix = self.calculate_confusion_matrix(predictions, labels, num_classes, device)

        # 计算宏/微F1分数
        if average_type == AverageType.MACRO:
            # 获取宏精确率
            precision_classes = self._calculate_macro_precision( num_classes, device)
            # 获取宏召回率
            recall_classes = self._calculate_macro_recall(num_classes, device)
            f1_classes = 2 * precision_classes * recall_classes / (precision_classes + recall_classes + 0.0000001)  # 避免除以0
            macro_f1 = torch.mean(f1_classes).item()
            return macro_f1

        elif average_type == AverageType.MICRO:
            micro_precision = self.calculate_precision(predictions, labels, num_classes, AverageType.MICRO, device)
            micro_recall = self.calculate_recall(predictions, labels, num_classes, AverageType.MICRO, device)
            if micro_precision + micro_recall == 0:
                return 0
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            return micro_f1
        else:
            raise ValueError("Invalid F1 score average type")
