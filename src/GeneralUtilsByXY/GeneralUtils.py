# GeneralUtils.py



"""
通用工具模块
"""

import os
from typing import Callable, Set



class FileUtils:
    def __init__(self):
        pass

    @staticmethod
    def process_files_with_filter(
                    directory: str,
                    operation: Callable[[str], None],
                    filter_ext: Set[str] = None,
                    recursive: bool = False
                    ) -> None:
        """
        遍历指定文件夹下的每个文件, 根据文件类型限定参数过滤文件,
        并对每个过滤后的文件应用提供的操作函数。
        Args:
            directory: 要遍历的文件夹路径
            operation: 应用于每个文件的操作函数
            filter_ext: 用来过滤文件的扩展名集合(如 {'.txt', '.csv'})
            recursive: 是否递归遍历子文件夹, 默认为False
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
                    # 如果是目录且设置了递归, 则递归调用自身
                    process_directory(file_path)

        # 开始处理根目录
        process_directory(directory)

    @staticmethod
    def ensure_absolute_path(data_root_dir:str) -> str:
        """
        将相对路径转换为绝对路径
        Args:
            data_root_dir: 输入的相对路径
        Returns:
            绝对路径
        """
        if not os.path.isabs(data_root_dir):
            data_root_dir = os.path.abspath(data_root_dir)
        return data_root_dir



