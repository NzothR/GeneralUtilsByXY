from typing import Any, Dict
import threading


"""
一些自定义装饰器，具有多线程安全
"""

# 单例装饰器
def singleton(cls):
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance