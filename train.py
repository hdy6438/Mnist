import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from data_tools.load_data import load_data
from net.vgg import net
from setting import *

train_dataset,test_dataset = load_data(dataset)

model,_,_ = net()

history = model.fit_generator(train_dataset,validation_data=test_dataset,epochs=epochs,callbacks=[
    ModelCheckpoint(
        filepath='model/my_model.h5',
        monitor='val_acc',
        save_best_only=True,
        verbose=1,
        period=1,
    ),
])

np.save("model/history",history.history)


