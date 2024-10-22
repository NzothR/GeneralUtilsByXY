from enum import Enum
from Logger import Logger

"""
这里是一些常用的常量集合
"""

# 日志实例
logger = Logger()


class Metrics(Enum):
    """
    常用评估指标
    """
    ACCURACY = "accuracy"                   # 准确率, 正确预测的比例
    PRECISION = "precision"                 # 精确率，真正例(True Positives, TP)占所有被预测为正例的比例(TP / (TP + FP))
    RECALL = "recall"                       # 召回率，真正例(True Positives, TP)占所有正例的比例(TP / (TP + FN))
    F1_SCORE = "f1_score"                   # F1分数, Precision 和 Recall 的调和平均数(2 * precision * recall) / (precision + recall)，适用于不平衡数据集
    AUC_ROC = "auc_roc"                     # ROC 曲线下面积，衡量二分类模型的性能
    CONFUSION_MATRIX = "confusion_matrix"   # 混淆矩阵
    COHEN_KAPPA = "cohen_kappa"             # Cohen Kappa值(科恩卡帕系数)，用于评估分类器的一致性，适用于平衡数据集