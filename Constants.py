"""
常量定义集合
"""


from enum import Enum
from Logger import Logger



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
    MAE = "mae"                             # 平均绝对误差, 预测值和真实值之间的平均绝对差值
    MSE = "mse"                             # 均方误差, 预测值和真实值之间的平方差
    RMSE = "rmse"                           # 均方根误差, 预测值和真实值之间的平方差开根号
    R2 = "r2"                               # R2 值, 预测值和真实值之间的相关性平方值
    MCC = "mcc"                             # Matthews Correlation Coefficient, 预测值和真实值之间的相关性平方值
    MAP = "mAP"                             # mean Average Precision, 平均精度均值, 在不同类别上计算的 AP 的平均值
    IOU = "iou"                             # Intersection over Union, 交并比, 检测框与真实框的重叠程度
    NMS = "nms"                             # Non-Maximum Suppression, 非极大抑制, 用于去除冗余的检测框
    LOG_LOSS= "log_loss"                    # Log Loss, 对数损失, 也称为交叉熵损失，衡量概率估计的准确性


class AverageType(Enum):
    """
    多分类问题计算平均值的类型
    """
    MACRO = 'macro' # 宏平均，即计算每个类的平均值，然后取平均值
    MICRO = 'micro' # 微平均，即计算所有类的平均值