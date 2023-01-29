import cv2
from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
from keras.applications import NASNetLarge
import keras.backend as K
from keras.models import Model
import tensorflow as tf
# tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="0"
model=load_model("resnet_4.h5")

# zerolist = os.listdir('/home/zhl/DL_Classification_ZK/DLFUNDUS/0-1/test1/1')
# for item in zerolist:
#     img = cv2.imread(os.path.join('/home/zhl/DL_Classification_ZK/DLFUNDUS/0-1/test1/1', item))
#     img = cv2.resize(img, (256, 256))
#     img = img / 255
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     print(model.predict_on_batch(img))

rootdir='/home/zhl/copydata_validation1_2'
savedir='/home/zhl/heatmap_copydata_validation1_2/binary1'
imglist=os.listdir(rootdir)
for item in imglist:
    img = cv2.imread(os.path.join('/home/zhl/copydata_validation1_2', item))
    img = cv2.resize(img, (256, 256))
#    img = img / 255
    img=img_to_array(img)
    img=np.expand_dims(img,axis=0)
    output=model.predict(img)
    #obtain the output of the last convolutional layer

    # layer_model = Model(inputs=model.input, outputs=model.layers[779].output)#6 is temporary and replace it when specified

    # feature=layer_model.predict(img)
    print("=======================================")
    print(len(model.layers))
    grads=K.gradients(model.output[0,0],model.layers[174].output)
    outputsize=model.layers[174].output.shape
    # print(grads.shape)
    coeff=[]
    # grads=grads[0,:,:,:]
    print(type(grads))
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    grads_np=sess.run(grads,feed_dict={model.input:img})
    outputfeature=np.squeeze(sess.run(model.layers[174].output,feed_dict={model.input:img}))
    # with sess.as_default():
    #     grads=grads.eval()
    grads_np=np.squeeze(np.array(grads_np))

    cam=np.zeros((8,8))
    print(grads_np.shape[2])
    for index in range(grads_np.shape[2]):
        cam=cam+outputfeature[:,:,index]*np.mean(grads_np[:,:,index])
    cam = cv2.resize(cam, (256, 256))
    heatmap = cam / np.max(cam)
    print(np.max(heatmap))
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(savedir,"heatmap_"+item), cam)
    cam = np.float32(cam) + np.float32(img)*255
    cam = 255 * cam / np.max(cam)
    print(np.max(cam))
    cv2.imwrite(os.path.join(savedir,"heatmap_original_"+item), np.squeeze(cam))
