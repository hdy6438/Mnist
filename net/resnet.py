from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.optimizers import Adam
from keras.applications import ResNet101
from setting import *


def net():
    resnet_model = ResNet101(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size[0], input_size[1], 3)
    )

    resnet_model.summary()

    top_model = Sequential(name="top_model")
    top_model.add(Flatten(input_shape=resnet_model.output_shape[1:]))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dense(10, activation="softmax"))
    top_model.summary()  # 查看网络

    model = Sequential()
    model.add(resnet_model)
    model.add(top_model)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["acc"]
    )

    model.summary()

    return model,resnet_model,top_model