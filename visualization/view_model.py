from keras.utils.vis_utils import plot_model

from net.vgg import net

model,vgg19_model,top_model = net()

plot_model(model, to_file='../model/diagram/model.png', show_shapes=True)
plot_model(vgg19_model, to_file='../model/diagram/vgg19_model.png', show_shapes=True)
plot_model(top_model, to_file='../model/diagram/top_model.png', show_shapes=True)

