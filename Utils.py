import cv2
import os
import time
import numpy as np
import sys
from typing import Callable, Set, Optional, Tuple, List

from Logger import logger

"""一些辅助函数"""

class FileUtils:
    def __init__(self):
        pass

    @staticmethod
    def process_files_with_filter(directory : str, operation : Callable[[str], None], filter_ext: Optional[Set[str]] = None) -> None:
        """
        遍历指定文件夹下的每个文件, 根据文件类型限定参数过滤文件, 
        并对每个过滤后的文件应用提供的操作函数。

        :param directory: 要遍历的文件夹路径
        :param operation: 应用于每个文件的操作函数
        :param filter_ext: 用来过滤文件的扩展名(如 '.txt')
        """
        # 判断目录是否存在
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录 {directory} 不存在")
        # 遍历目录下的所有文件和文件夹
        for entry in os.listdir(directory):
            # 构建完整的文件路径
            file_path = os.path.join(directory, entry)
            # 检查是否为文件并且是指定文件
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(entry)[1].lower()
                if filter_ext is None or file_ext in filter_ext:
                    operation(file_path)


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

    def resize_and_pad(self, image_directory: str, target_size: Optional[tuple[int, int]], fill_color: tuple = (0, 0, 0), save_directory: str = None):
        """
        缩放与填充图像至统一尺寸
        :param image_directory: 包含需要处理的图像目录
        :param target_width: 目标宽度
        :param target_height: 目标高度
        :param save_directory: 保存目录, 默认为None
        """
        # 检查目录是否存在
        if not os.path.exists(image_directory):
            logger.error(f"目录 {image_directory} 不存在!")
            raise FileNotFoundError("Directory does not exist")
        # 指定保存目录
        if save_directory is None:
            last_directory = os.path.basename(os.path.normpath(image_directory))
            save_directory = os.path.join('process', last_directory)
        # 创建保存目录
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            # 检查保存目录是否创建成功
            if not os.path.exists(save_directory):
                logger.error(f"目录 {save_directory} 创建失败!")
                raise FileNotFoundError("Failed to create save directory")

        # 计数
        count = 0
        # 错误文件计数
        error_count = 0
        def process(image_file_path):
            nonlocal save_directory, error_count, count, fill_color, target_size
            # 构建保存路径
            saved_image_path = os.path.join(save_directory, os.path.basename(image_file_path))
            # 检查文件是否存在
            if os.path.exists(saved_image_path):
                return
            try:
                # 读取图像
                image = cv2.imread(image_file_path)
                if image is None:
                    logger.error(f"无法读取图像: {image_file_path}")
                    error_count += 1
                    return

                # 获取图像尺寸
                height, width = image.shape[:2]

                # 初始化边距
                top, bottom, left, right = 0, 0, 0, 0

                # 确定需要填充的像素
                if width > height:
                    difference = width - height
                    top, bottom = int(difference / 2), int(difference / 2)
                    # 如果 difference 是奇数, 顶部多一个像素
                    if difference % 2 != 0:
                        top += 1

                else:
                    difference = height - width
                    left, right = int(difference / 2), int(difference / 2)
                    # 如果 difference 是奇数, 左侧多一个像素
                    if difference % 2 != 0:
                        left += 1


                border_type = cv2.BORDER_CONSTANT  # 边界类型为常数值

                # 填充图像
                process_image = cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=fill_color)


                # 如果填充后图像高度大于目标高度
                target_width = target_size[0]
                target_height = target_size[1]
                if process_image.shape[0] > target_height:
                    # 使用像素区域关系进行重采样进行缩小
                    cur_interpolation = cv2.INTER_AREA
                else:
                    # 使用2x2像素邻域内的双线性插值进行放大
                    cur_interpolation = cv2.INTER_LINEAR

                # 缩放图像
                process_image = cv2.resize(process_image, (target_width, target_height), interpolation=cur_interpolation)


                # 保存处理后的图像
                cv2.imwrite(saved_image_path, process_image)
                count += 1
            except Exception as e:
                logger.error(f"处理图像时发生错误: {image_file_path}, 错误信息: {e}")
                error_count += 1

        start = time.time()
        logger.info(f"开始处理目录{image_directory}下的图片, 目标尺寸: {target_size[0]}x{target_size[1]}, 填充颜色: {fill_color}, 保存目录: {save_directory}")
        # 处理文件
        FileUtils.process_files_with_filter(image_directory, process, self.valid_image_ext)
        end = time.time()
        # 计算处理耗时并四舍五入到小数点后六位
        processing_time = end - start
        logger.info(f"处理完成, 耗时{processing_time:.6f}秒, 处理成功{count}张图像, 处理失败{error_count}张图像")

    def is_image_valid(self, directory: str) -> Tuple[List[str], List[str]]:
        """
        检查指定目录下的图片是否能够正确读取
        :param directory: 要检查的文件夹路径
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

        FileUtils.process_files_with_filter(directory, operation, self.valid_image_ext)
        return error_load_img_paths, error_open_img_paths

    def get_mm_dimensions(self, directory: str) -> Tuple[int, int, int, int]:
        """获取目录中所有图像的最大最小宽度和高度
        :param directory: 目录路径
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
        FileUtils.process_files_with_filter(directory, operation, self.valid_image_ext)
        return max_width, max_height, min_width, min_height
