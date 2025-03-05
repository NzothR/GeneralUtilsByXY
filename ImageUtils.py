"""
图像处理工具模块
"""



import os
import cv2
import time
import numpy as np
from typing import Callable, Set, Optional, Tuple, List, Dict

from Constants import logger
from GeneralUtils import FileUtils

class ImageUtils:
    def __init__(self, image_ext: Optional[Set[set]] = None):
        """
        :param image_ext: 允许处理的图像文件拓展名, 默认为{'.jpg', '.jpeg', '.png', '.bmp', '.gif'}"
        """
        # 有效的图像拓展名
        if image_ext is None:
            self.valid_image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        else:
            self.valid_image_ext = set(image_ext)

    def resize_and_pad(self, image_directory: str, target_size: Optional[Tuple[int, int]],
                        fill_color: Tuple[int, int, int] = (0, 0, 0), save_directory: str = None, recursive: bool = False) -> None:
        """
        缩放与填充图像至统一尺寸
        :param image_directory: 包含需要处理的图像目录
        :param target_size: 目标尺寸 (width, height)
        :param fill_color: 填充颜色 (B, G, R), 默认为(0, 0, 0), 即黑色
        :param save_directory: 保存目录, 默认为上级目录的process文件夹
        :param recursive: 是否递归遍历子文件夹, 默认为False
        :return: None
        """
        # 参数校验
        if target_size is None or target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError("target_size 必须为正整数元组")
        if any(c < 0 or c > 255 for c in fill_color):
            raise ValueError("填充颜色值必须在 0-255 范围内")

        # 检查目录是否存在
        if not os.path.exists(image_directory):
            logger.error(f"目录 {image_directory} 不存在!")
            raise FileNotFoundError("Directory does not exist")

        if save_directory is None:
            # 规范化路径并获取上级目录
            normalized_image_dir = os.path.normpath(image_directory)
            parent_dir = os.path.dirname(normalized_image_dir)
            save_directory = os.path.join(parent_dir, "process")

        # 创建目录（兼容已存在的情况）
        try:
            os.makedirs(save_directory, exist_ok=True)
        except OSError as e:
            logger.error(f"目录 {save_directory} 创建失败: {e}")
            raise FileNotFoundError(f"无法创建目录: {save_directory}") from e

        # 计数
        count = 0
        # 错误文件计数
        error_count = 0

        # 处理函数
        def process(image_file_path):
            # 导入外部变量
            nonlocal save_directory, error_count, count, fill_color, target_size
            # 获取相对路径并构建保存路径
            relative_path = os.path.relpath(image_file_path, image_directory)
            saved_image_path = os.path.join(save_directory, relative_path)
            os.makedirs(os.path.dirname(saved_image_path), exist_ok=True)

            try:
                # 读取图像
                image = cv2.imread(image_file_path)
                if image is None:
                    logger.error(f"无法读取图像: {image_file_path}")
                    error_count += 1
                    return

                # 获取图像尺寸
                height, width = image.shape[:2]
                target_width, target_height = target_size

                # 如果图像尺寸与目标尺寸相同，则直接保存
                if height == target_height and width == target_width:
                    cv2.imwrite(saved_image_path, image)
                    count += 1
                    return

                # 获取缩放比例
                target_ratio = target_width / target_height
                original_ratio = width / height

                # 计算缩放后的尺寸
                if original_ratio > target_ratio:
                    new_width = target_width
                    new_height = int(target_width / original_ratio)
                else:
                    new_height = target_height
                    new_width = int(target_height * original_ratio)

                # 缩放图像
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                top, bottom, left, right = 0, 0, 0, 0
                # 计算填充尺寸
                if new_width < target_width:
                    difference = target_width - new_width
                    left, right = int(difference / 2), int(difference / 2)
                    if difference % 2 != 0:
                        left += 1
                elif new_height < target_height:
                    difference = target_height - new_height
                    top, bottom = int(difference / 2), int(difference / 2)
                    if difference % 2 != 0:
                        top += 1
                # 填充图像
                padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)

                # 保存图像
                cv2.imwrite(saved_image_path, padded_image)
                count += 1
            except Exception as e:
                logger.error(f"处理图像时发生错误: {image_file_path}, 错误信息: {e}")
                error_count += 1

        start = time.time()
        logger.info(f"开始处理目录{image_directory}下的图片, 目标尺寸: {target_size[0]}x{target_size[1]}, 填充颜色: {fill_color}, 保存目录: {save_directory}")
        # 处理文件
        FileUtils.process_files_with_filter(image_directory, process, self.valid_image_ext, recursive)
        end = time.time()
        # 计算处理耗时并四舍五入到小数点后六位
        processing_time = end - start
        logger.info(f"处理完成, 耗时{processing_time:.6f}秒, 处理成功{count}张图像, 处理失败{error_count}张图像")

    def is_image_valid(self, directory: str, recursive: bool = False) -> Tuple[List[str], List[str]]:
        """
        检查指定目录下的图片是否能够正确读取
        :param directory: 要检查的文件夹路径
        :param recursive: 是否递归遍历子文件夹, 默认为False
        """
        # 读取失败的图片
        error_load_img_paths = []
        # 无法打开的图片
        error_open_img_paths = []
        def operation(file_path):
            nonlocal error_load_img_paths, error_open_img_paths
            try:
                # 读取图片
                img = cv2.imread(file_path)
                # 检查是否成功读取了图片
                if img is None:
                    error_load_img_paths.append(file_path)
            except (IOError, FileNotFoundError) as e:
                error_open_img_paths.append(file_path)
            except Exception as e:
                error_open_img_paths.append(file_path)

        FileUtils.process_files_with_filter(directory, operation, self.valid_image_ext, recursive)
        return error_load_img_paths, error_open_img_paths

    def get_mm_dimensions(self, directory: str, recursive: bool = False) -> Tuple[int, int, int, int]:
        """获取目录中所有图像的最大最小宽度和高度
        :param directory: 目录路径
        :param recursive: 是否递归遍历子文件夹, 默认为False
        """
        max_width = 0
        max_height = 0
        min_width = np.inf
        min_height =  np.inf

        def operation(file_path):
            nonlocal max_width, max_height, min_width, min_height
            # 读取图像
            image = cv2.imread(file_path)

            # 获取图像的尺寸
            if image is not None:
                height, width, _ = image.shape
                if width > max_width:
                    max_width = width
                elif width < min_width:
                    min_width = width
                if height > max_height:
                    max_height = height
                elif height < min_height:
                    min_height = height
        FileUtils.process_files_with_filter(directory, operation, self.valid_image_ext, recursive)
        return max_width, max_height, min_width, min_height
