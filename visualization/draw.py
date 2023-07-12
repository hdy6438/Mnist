import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from setting import *

history = np.load("../model/history.npy", allow_pickle=True).item()


fig = plt.figure()  # 新建画布
plt.title("Loss")  # 设置标题
ax=plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(2))
plt.xlim(0,epochs)
x = list(range(1,epochs+1))
plt.plot(x,history["loss"], label="train")
plt.plot(x,history["val_loss"], label="val")
plt.xlabel("Epoch")  # 设置x坐标标签
plt.ylabel("Loss")  # 设置y坐标标签
plt.legend(loc="lower left")  # 显示图例
plt.show()  # 显示图像
fig.savefig("model/report/loss.png")  # 保存图像
plt.close(fig)  # 清理内存


fig = plt.figure()  # 新建画布
plt.title("Accuracy")  # 设置标题
ax=plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(2))
plt.xlim(0,epochs)
x = list(range(1,epochs+1))
plt.plot(x,history["acc"], label="train")
plt.plot(x,history["val_acc"], label="val")
plt.xlabel("Epoch")  # 设置x坐标标签
plt.ylabel("Accuracy")  # 设置y坐标标签
plt.legend(loc="lower left")  # 显示图例
plt.show()  # 显示图像
fig.savefig("model/report/acc.png")  # 保存图像
plt.close(fig)  # 清理内存