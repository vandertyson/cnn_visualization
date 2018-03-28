from keras.models import Model
from keras.layers import Input, Activation, Reshape, Conv2D, Lambda, Add
import tensorflow as tf
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.layers import merge as merge_l
from keras.layers import (
    Input, Convolution2D, MaxPooling2D, UpSampling2D,
    Reshape, core, Dropout,
    Activation, BatchNormalization)
import rasterio
import numpy as np

K.set_image_dim_ordering("th")

def get_unet(channel=3):
    conv_params = dict(activation='relu', border_mode='same')
    merge_params = dict(mode='concat', concat_axis=1)
    inputs = Input((channel, 256, 256))
    conv1 = Convolution2D(32, 3, 3, **conv_params)(inputs)
    conv1 = Convolution2D(32, 3, 3, **conv_params)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, **conv_params)(pool1)
    conv2 = Convolution2D(64, 3, 3, **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, **conv_params)(pool2)
    conv3 = Convolution2D(128, 3, 3, **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, **conv_params)(pool3)
    conv4 = Convolution2D(256, 3, 3, **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, **conv_params)(pool4)
    conv5 = Convolution2D(512, 3, 3, **conv_params)(conv5)

    up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], **merge_params)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(up6)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(conv6)

    up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3], **merge_params)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(up7)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(conv7)

    up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2], **merge_params)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(up8)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(conv8)

    up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1], **merge_params)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(up9)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    adam = Adam()

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy', jaccard_coef, jaccard_coef_int])
    return model

def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

BAND_CUT_3 = {
    0: {'max': 224.0, 'min': 44.0}
    , 1: {'max': 217.0, 'min': 64.0}
    , 2: {'max': 204.0, 'min': 67.0}
}

BAND_CUT_8 = {
    0: {'max': 224.0, 'min': 44.0}
    , 1: {'max': 217.0, 'min': 64.0}
    , 2: {'max': 204.0, 'min': 67.0}
    , 3: {'max': 171.0, 'min': 5.0}
    , 4: {'max': 100.0, 'min': 4.0}
    , 5: {'max': 100.0, 'min': 4.0}
    , 6: {'max': 71.0, 'min': 15.0}
    , 7: {'max': 150.0, 'min': 3.0}
}


def preprocess(img_path, channel):
    X_val = []
    x_mean=np.load('xmean_%d.npy'%channel)    
    if channel == 3:
        bandcut = BAND_CUT_3
    else:
        bandcut = BAND_CUT_8        
    with rasterio.open(img_path, 'r') as f:            
        values = f.read().astype(np.float32)            
        for chan_i in range(channel):
            min_val = bandcut[chan_i]['min']
            max_val = bandcut[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
        X_val.append(values)
    return np.array(X_val) - x_mean
