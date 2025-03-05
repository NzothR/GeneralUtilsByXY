import os
import sys
import logging
from datetime import date, datetime

from Decorators import singleton

"""
一个简单的全局单一日志记录器，用于记录日志到文件，控制台
"""

class CustomFormatter(logging.Formatter):
    """
    自定义时间格式化器，将日期和时间分开，方便后续处理
    """
    def formatTime(self, record, datefmt=None):
        created_time = datetime.fromtimestamp(record.created)
        if datefmt:
            return created_time.strftime(datefmt)
        else:
            return created_time.strftime('%Y-%m-%d %H:%M:%S')

    def format(self, record):
        # 获取完整的日期和时间
        full_datetime = self.formatTime(record, self.datefmt)

        # 分离日期和时间
        date, time = full_datetime.split(' ')

        # 替换记录中的 asctime 字段
        record.date = date
        record.time = time

        # 使用自定义格式字符串
        self._style._fmt = '[%(date)s][%(time)s]%(levelname)s: %(message)s'

        return super().format(record)


@singleton
class Logger:
    """
    日志记录器，支持动态控制控制台输出及日志级别
    """
    def __init__(
        self,
        name: str = "log",
        console_enabled: bool = True,  # 初始是否启用控制台输出
        console_level: int = logging.DEBUG  # 控制台默认日志级别
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # 全局最低级别

        # 日志文件路径
        date_log = date.today().strftime("%Y-%m-%d") + ".log"
        latest_log = "latest.log"
        self.date_log_path = os.path.join("logs", date_log)
        self.latest_log_path = os.path.join("logs", latest_log)

        if not os.path.exists("logs"):
            os.mkdir("logs")

        # 清理旧Handler（防止单例重复初始化问题）
        self.logger.handlers.clear()

        # 初始化文件Handler（始终存在）
        self._init_file_handlers()

        # 初始化控制台Handler（动态管理）
        self.console_handler = None
        if console_enabled:
            self.enable_console(console_level)

    def _init_file_handlers(self):
        """初始化固定文件Handler"""
        # 按日期保存的日志
        date_file_handler = logging.FileHandler(self.date_log_path)
        date_file_handler.setLevel(logging.DEBUG)

        # 当前日志
        latest_file_handler = logging.FileHandler(self.latest_log_path)
        latest_file_handler.setLevel(logging.DEBUG)

        # 统一格式化
        formatter = CustomFormatter(
            '[%(date)s][%(time)s]%(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        for handler in [date_file_handler, latest_file_handler]:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def enable_console(self, level: int = logging.INFO):
        """启用控制台输出（或更新级别）"""
        if not self.console_handler:
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setFormatter(CustomFormatter(
                '[%(date)s][%(time)s]%(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(self.console_handler)
        self.console_handler.setLevel(level)

    def disable_console(self):
        """禁用控制台输出"""
        if self.console_handler:
            self.logger.removeHandler(self.console_handler)
            self.console_handler = None

    def clear(self):
        """清空当前日志文件"""
        with open(self.latest_log_path, 'w') as f:
            f.write("")

    # 代理Logger方法
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)


if __name__ == "__main__":
    logger = Logger()

    logger.info("这条日志会输出到控制台和文件")

    # 动态调整控制台级别为DEBUG
    logger.enable_console(logging.DEBUG)
    logger.debug("Debug信息现在会显示在控制台")

    # 关闭控制台输出
    logger.disable_console()
    logger.warning("这条警告只会记录到文件")

    # 重新开启控制台
    logger.enable_console()
    # 动态调整控制台级别为ERROR
    logger.enable_console(logging.ERROR)
    logger.warning("这条警告只会记录到文件")