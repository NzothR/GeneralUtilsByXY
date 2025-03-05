import logging
import os
from datetime import date, datetime
from Decorators import singleton
from datetime import datetime

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
    日志记录器，用于记录日志到文件，控制台, 采用单例模式, 避免重复创建日志记录器
    """
    def __init__(self, name: str = "log"):
        # 创建一个日志器
        self.logger = logging.getLogger(name)
        # 设置日志级别
        self.logger.setLevel(logging.DEBUG)
        # 按日期保存的日志
        date_log = date.today().strftime("%Y-%m-%d") + ".log"
        # 当前日志
        latest_log = "latest.log"
        self.date_log_path = os.path.join("logs", date_log)
        self.latest_log_path = os.path.join("logs", latest_log)

        if not os.path.exists("logs"):
            os.mkdir("logs")


        # 创建两个文件 handler 并设置其日志级别为 DEBUG
        date_file_handler = logging.FileHandler(self.date_log_path)
        date_file_handler.setLevel(logging.DEBUG)
        latest_file_handler = logging.FileHandler(self.latest_log_path)
        latest_file_handler.setLevel(logging.DEBUG)

        # 创建一个控制台 handler 并设置其日志级别为 WARNING
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # 创建一个 formatter 并将其添加到 handlers 中
        formatter = CustomFormatter(
                                    '[%(date)s][%(time)s]%(levelname)s: %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S'
                                    )
        date_file_handler.setFormatter(formatter)
        latest_file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将 handlers 添加到 logger
        self.logger.addHandler(date_file_handler)
        self.logger.addHandler(latest_file_handler)
        self.logger.addHandler(console_handler)
    def clear(self):
        """
        清空当前日志文件
        """
        with open(self.latest_log_path, 'w') as f:
            f.write("")
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


