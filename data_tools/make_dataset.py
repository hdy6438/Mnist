from __future__ import division
import os
import time
import json
import numpy as np
import struct
from PIL import Image

# 训练集文件
train_images_idx3_ubyte_file = "../data/origin/train-images.idx3-ubyte"
# 训练集标签文件
train_labels_idx1_ubyte_file = "../data/origin/train-labels.idx1-ubyte"
# 测试集文件
test_images_idx3_ubyte_file = "../data/origin/t10k-images.idx3-ubyte"
# 测试集标签文件
test_labels_idx1_ubyte_file = "../data/origin/t10k-labels.idx1-ubyte"


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, "rb").read()
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = ">iiii"
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset
    )
    print(
        "魔数:%d, 图片数量: %d张, 图片大小: %d*%d" % (magic_number, num_images, num_rows, num_cols)
    )
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = ">" + str(image_size) + "B"
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print("已解析 %d" % (i + 1) + "张")
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape(
            (num_rows, num_cols)
        )
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, "rb").read()
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = ">ii"
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print("魔数:%d, 图片数量: %d张" % (magic_number, num_images))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = ">B"
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print("已解析 %d" % (i + 1) + "张")
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def parseMnist2Img(resDir="binary-mnist/"):
    """
    转化为图像数据
    """
    train_images = load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file)
    train_labels = load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file)
    test_images = load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file)
    test_labels = load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file)
    # 解析训练集
    trainDir = resDir + "train/"
    print("train_images_nums: ", len(train_images))
    for i in range(len(train_images)):
        one_label = train_labels[i]
        print("one_label: ", one_label)
        one_img = train_images[i]
        oneDir = trainDir + str(int(one_label)) + "/"
        if not os.path.exists(oneDir):
            os.makedirs(oneDir)
        print("one_img_shape: ", one_img.shape)
        onePic = Image.fromarray(np.uint8(one_img))
        one_path = oneDir + str(len(os.listdir(oneDir))) + ".jpg"
        onePic.save(one_path)
    # 解析测试集
    testDir = resDir + "test/"
    print("test_images_nums: ", len(test_images))
    for i in range(len(test_images)):
        one_label = test_labels[i]
        print("one_label: ", one_label)
        one_img = test_images[i]
        oneDir = testDir + str(int(one_label)) + "/"
        if not os.path.exists(oneDir):
            os.makedirs(oneDir)
        print("one_img_shape: ", one_img.shape)
        onePic = Image.fromarray(np.uint8(one_img))
        one_path = oneDir + str(len(os.listdir(oneDir))) + ".jpg"
        onePic.save(one_path)


if __name__ == "__main__":
    print(
        "=========================================Loading binaryHandle==========================================="
    )

    parseMnist2Img(resDir="binary-mnist/")

