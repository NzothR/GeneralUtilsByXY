import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Constants import logger
from GeneralUtils import TrainUtils
import os

# 配置日志记录
logger.enable_console()


def test_general_train():
    """测试通用训练函数"""
    # 配置参数
    config = {
        "batch_size": 64,
        "num_epochs": 2,
        "learning_rate": 0.001,
        "num_classes": 10,
        "checkpoint_path": "./test_checkpoints",
        "use_amp": True         # 测试混合精度可以改为True（需要GPU支持）
    }

    # 设备检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device.upper()}")

    # 构建数据管道
    def prepare_dataloaders():
        """准备MNIST数据集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        test_set = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

        train_loader = DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0  # 为避免多进程问题，测试时设为0
        )
        test_loader = DataLoader(
            test_set,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0
        )
        return train_loader, test_loader

    # 定义简单CNN模型
    class TestCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(TestCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3),  # 输入通道1，输出32
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 5 * 5, 128),  # 计算特征图尺寸 (28-2)/2=13 → (13-2)/2=5.5 → floor为5
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    # 准备数据
    train_loader, test_loader = prepare_dataloaders()
    logger.info(f"训练样本数: {len(train_loader.dataset)} | 测试样本数: {len(test_loader.dataset)}")

    # 初始化模型
    model = TestCNN(num_classes=config["num_classes"])
    logger.info("模型结构:\n%s", model)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 执行训练
    trained_model, history = TrainUtils().general_classification_train(
        train_data_loader=train_loader,
        val_data_loader=test_loader,
        model=model,
        model_name="test_cnn",
        criterion=criterion,
        num_classes=config["num_classes"],
        optimizer=optim.Adam(model.parameters(), lr=config["learning_rate"]),
        epochs=config["num_epochs"],
        device=device,
        checkpoint_path=config["checkpoint_path"],
        use_amp=config["use_amp"]
    )

    # 验证训练结果
    assert len(history["Train Loss"]) == config["num_epochs"], "训练轮数不匹配"
    assert len(history["Val Loss"]) == config["num_epochs"], "验证轮数不匹配"
    logger.info("训练损失记录: %s", history["Train Loss"])
    logger.info("验证损失记录: %s", history["Val Loss"])

    # 最终模型测试
    trained_model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = trained_model(images)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

    accuracy = 100 * correct / len(test_loader.dataset)
    logger.info(f"最终测试准确率: {accuracy:.2f}%")
    assert accuracy > 90, "模型准确率未达到预期"  # 正常训练2个epoch应能达到约97%+

    TrainUtils.plot_history(history, "./test")

    # 清理测试检查点
    if os.path.exists(config["checkpoint_path"]):
        for f in os.listdir(config["checkpoint_path"]):
            os.remove(os.path.join(config["checkpoint_path"], f))
        os.rmdir(config["checkpoint_path"])
        logger.info("已清理测试检查点")

if __name__ == "__main__":
    test_general_train()