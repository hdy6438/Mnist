import os

from keras.preprocessing.image import ImageDataGenerator
from setting import *

def load_data(dataset_root):
    train_datagen = ImageDataGenerator(
        rotation_range=20,  # 随机旋转度数
        rescale=1 / 255,  # 数据归一化
        fill_mode='nearest',  # 填充方式,
    )

    train_dataset = train_datagen.flow_from_directory(
        os.path.join(dataset_root, "train"),
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )

    test_datagen = ImageDataGenerator(
        rescale=1 / 255,  # 数据归一化
        fill_mode='nearest',  # 填充方式,
    )

    test_dataset = test_datagen.flow_from_directory(
        os.path.join(dataset_root, "test"),
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )

    return train_dataset,test_dataset