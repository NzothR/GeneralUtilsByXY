import torch
import time
from pprint import pprint

from MetricsUtils import MetricsUtils
from Constants import Metrics, AverageType

# 假设有 3 个类别
num_classes = 3
N = 10  # 样本数量

# 真实标签和预测标签
y_true = torch.tensor([1, 2, 2, 2, 0, 1, 2, 1, 2, 2])
y_pred = torch.tensor([1, 1, 1, 0, 1, 1, 0, 1, 0, 0])

print("真实值:", y_true)
print("预测值:", y_pred)

# 初始化评估工具
metrics_utils = MetricsUtils()
# 张量计算设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 计算混淆矩阵
confusion_matrix = metrics_utils.calculate_confusion_matrix(y_pred, y_true, num_classes, device)
print("混淆矩阵:\n", confusion_matrix)

start_time = time.time()
accuracy = (confusion_matrix.diag().sum() / confusion_matrix.sum()).item()
end_time = time.time()
print(f"使用原始方法计算的准确率:{accuracy}, 耗时:{end_time - start_time}s")

start_time = time.time()
accuracy = metrics_utils.calculate_accuracy(y_pred, y_true, num_classes, device)
end_time = time.time()
print(f"使用MetricsUtils计算准确率:{accuracy}, 耗时:{end_time - start_time}s")



start_time = time.time()
precision_classes = torch.zeros(num_classes)
for i in range(num_classes):
    tp = confusion_matrix[i, i]
    fp = confusion_matrix[:, i].sum() - tp
    if tp + fp == 0:
        precision_classes[i] = 0
    else:
        precision_classes[i] = tp / (tp + fp)
macro_precision =  torch.mean(precision_classes).item()
end_time = time.time()
print(f"使用原始方法计算的宏观精确率:{macro_precision}, 耗时:{end_time - start_time}s")

start_time = time.time()
macro_precision = metrics_utils.calculate_precision(y_pred, y_true, num_classes, AverageType.MACRO, device)
end_time = time.time()
print(f"使用MetricsUtils计算宏精确率:{macro_precision}, 耗时:{end_time - start_time}s")


start_time = time.time()
tp = confusion_matrix.diag()
fp = confusion_matrix.sum(dim=0) - tp
tp_sum = tp.sum()
fp_sum = fp.sum()
micro_precision = (tp_sum / (tp_sum + fp_sum)).item()
end_time = time.time()
print(f"使用原始方法计算的微观精确率:{micro_precision}, 耗时:{end_time - start_time}s")

start_time = time.time()
micro_precision = metrics_utils.calculate_precision(y_pred, y_true, num_classes, AverageType.MICRO, device)
end_time = time.time()
print(f"使用MetricsUtils计算微精确率: {micro_precision}, 耗时:{end_time - start_time}s")



start_time = time.time()
recall_classes = torch.zeros(num_classes)
for i in range(num_classes):
    tp = confusion_matrix[i, i]
    fn = confusion_matrix[i, :].sum() - tp
    if tp + fn == 0:
        recall_classes[i] = 0
    else:
        recall_classes[i] = tp / (tp + fn)
macro_recall = torch.mean(recall_classes).item()
end_time = time.time()
print(f"使用原始方法计算的宏观召回率: {macro_recall}, 耗时:{end_time - start_time}s")

start_time = time.time()
macro_recall = metrics_utils.calculate_recall(y_pred, y_true, num_classes, AverageType.MACRO, device)
end_time = time.time()
print(f"使用MetricsUtils计算宏召回率:{macro_recall}, 耗时:{end_time - start_time}s")



start_time = time.time()
tp = confusion_matrix.diag()
fn = confusion_matrix.sum(dim=1) - tp
tp_sum = tp.sum()
fn_sum = fn.sum()
micro_recall = (tp_sum / (tp_sum + fn_sum)).item()
end_time = time.time()
print(f"使用原始方法计算的微观召回率:{micro_recall}, 耗时:{end_time - start_time}s")


start_time = time.time()
micro_recall = metrics_utils.calculate_recall(y_pred, y_true, num_classes, AverageType.MICRO, device)
end_time = time.time()
print(f"使用MetricsUtils计算微召回率:{micro_recall}, 耗时:{end_time - start_time}s")


start_time = time.time()
f1_classes = torch.zeros(num_classes)
for i in range(num_classes):
    tp = confusion_matrix[i, i]
    fn = confusion_matrix[i, :].sum() - tp
    fp = confusion_matrix[:, i].sum() - tp
    if tp + fn == 0 or tp + fp == 0:
        f1_classes[i] = 0
    else:
        f1_classes[i] = 2 * tp / (2 * tp + fn + fp)
macro_f1 = torch.mean(f1_classes).item()
end_time = time.time()
print("使用原始方法计算的宏观F1:", macro_f1)

start_time = time.time()
macro_f1 = metrics_utils.calculate_f1_score(y_pred, y_true, num_classes, AverageType.MACRO, device)
end_time = time.time()
print(f"使用MetricsUtils计算宏F1:{macro_f1}, 耗时:{end_time - start_time}s")


start_time = time.time()
tp = confusion_matrix.diag()
fn = confusion_matrix.sum(dim=1) - tp
fp = confusion_matrix.sum(dim=0) - tp
tp_sum = tp.sum()
fn_sum = fn.sum()
fp_sum = fp.sum()
micro_f1 = (2 * tp_sum / (2 * tp_sum + fn_sum + fp_sum)).item()
end_time = time.time()
print("使用原始方法计算的微观F1:", micro_f1)

start_time = time.time()
micro_f1 = metrics_utils.calculate_f1_score(y_pred, y_true, num_classes, AverageType.MICRO, device)
end_time = time.time()
print(f"使用MetricsUtils计算微F1:{micro_f1}, 耗时:{end_time - start_time}s")