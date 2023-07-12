import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from data_tools.load_data import load_data
from setting import *
from tqdm import trange


print("loading model")
model = load_model("../model/my_model.h5")
print("loading dataset")
_,dataset = load_data(dataset)

print("running")
y_predict = []
y_true = []
for i in trange(dataset.samples//batch_size +1):
    y_predict.extend(np.argmax(model.predict(dataset[i][0]), axis=1))
    y_true.extend(np.argmax(dataset[i][1], axis=1))

print('Classification Report')
with open("../model/report/report.txt", "w") as f:
    report = classification_report(y_true, y_predict, target_names=dataset.class_indices.keys())
    print(report)
    f.write(report)
    f.close()

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_predict), display_labels=dataset.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.savefig("model/report/confusion_matrix.png")
plt.show()


