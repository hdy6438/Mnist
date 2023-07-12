from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.optimizers import Adam
from keras.applications import VGG19
from setting import *


def net():
    vgg19_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size[0], input_size[1], 3)
    )

    vgg19_model.summary()

    top_model = Sequential(name="top_model")
    top_model.add(Flatten(input_shape=vgg19_model.output_shape[1:]))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dense(10, activation="softmax"))
    top_model.summary()  # 查看网络

    model = Sequential()
    model.add(vgg19_model)
    model.add(top_model)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["acc"]
    )

    model.summary()

    return model,vgg19_model,top_model