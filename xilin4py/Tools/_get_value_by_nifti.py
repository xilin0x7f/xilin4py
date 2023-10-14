# Author: 赩林, xilin0x7f@163.com
import numpy as np
import nibabel as nib

def extract_value(file_path, mask_path):
    """
    从给定的文件路径和掩膜路径中提取指定掩膜下的数据值。

    参数：
    file_path (str): 要加载的文件的路径。
    mask_path (str): 掩膜文件的路径。

    返回值：
    numpy.ndarray: 提取的数据值。

    """
    file_image = nib.load(file_path)  # 加载文件
    mask_image = nib.load(mask_path)  # 加载掩膜文件
    origin_data = file_image.get_fdata()  # 获取文件数据
    mask_data = mask_image.get_fdata()  # 获取掩膜数据
    return origin_data[mask_data == 1]  # 返回在掩膜为1的位置上的数据值


def restore_image(values, mask_path, save_path):
    """
    使用给定的数据值将恢复的图像保存为 NIfTI 格式文件。

    参数：
    values (numpy.ndarray): 用于恢复图像的数据值。
    mask_path (str): 用作掩膜的文件路径。
    save_path (str): 保存恢复图像的文件路径。

    返回值：
    numpy.ndarray: 恢复的图像数据。

    """
    mask_image = nib.load(mask_path)  # 加载掩膜文件
    mask_data = mask_image.get_fdata()  # 获取掩膜数据
    restored_data = np.zeros_like(mask_data)  # 创建与掩膜相同形状的全零数组
    restored_data[mask_data == 1] = values  # 将数据值填充到掩膜为1的位置上
    header = mask_image.header  # 获取掩膜文件的头信息
    nifti_image = nib.Nifti1Image(restored_data, affine=header.get_best_affine(), header=header)  # 创建 NIfTI 图像
    nib.save(nifti_image, save_path)  # 保存恢复图像到文件
    return restored_data  # 返回恢复的图像数据

