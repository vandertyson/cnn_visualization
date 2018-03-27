import numpy as np
import cv2
from keras import backend as K
from utils import *
from model import *
from unet import get_unet
import argparse
import rasterio
import os
#Define regularizations:
def blur_regularization(img, grads, size = (3, 3)):
    return cv2.blur(img, size)

def decay_regularization(img, grads, decay = 0.8):
    return decay * img

def clip_weak_pixel_regularization(img, grads, percentile = 1):
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped

def gradient_ascent_iteration(loss_function, img):
    print(img)
    loss_value, grads_value = loss_function([img])    
    gradient_ascent_step = img + grads_value * 0.9

    #Convert to row major format for using opencv routines
    grads_row_major = np.transpose(grads_value[0, :], (1, 2, 0))
    img_row_major = np.transpose(gradient_ascent_step[0, :], (1, 2, 0))

    #List of regularization functions to use
    regularizations = [blur_regularization, decay_regularization, clip_weak_pixel_regularization]

    #The reguarlization weights
    weights = np.float32([3, 3, 1])
    weights /= np.sum(weights)

    images = [reg_func(img_row_major, grads_row_major) for reg_func in regularizations]
    weighted_images = np.float32([w * image for w, image in zip(weights, images)])
    img = np.sum(weighted_images, axis = 0)

    #Convert image back to 1 x 3 x height x width
    img = np.float32([np.transpose(img, (2, 0, 1))])

    return img

def visualize_filter(input_img, filter_index, img_placeholder, number_of_iterations = 20):
    loss = K.mean(layer[:, filter_index, :, :])
    grads = K.gradients(loss, img_placeholder)[0]
    grads = normalize(grads)
    # this function returns the loss and grads given the input picture
    iterate = K.function([img_placeholder], [loss, grads])

    img = input_img * 1

    # we run gradient ascent for 20 steps
    for i in range(number_of_iterations):
        img = gradient_ascent_iteration(iterate, img)

    # decode the resulting input image
    img = deprocess_image(img[0])
    print("Done with filter", filter_index)
    return img

def layer_to_visualize(img_to_visualize, layer, save_path, size):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)

    # n = convolutions.shape[0]
    # n = int(np.ceil(np.sqrt(n)))
    directory = './%s'%save_path
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(len(convolutions)):
        cv2.imwrite(os.path.join(directory, 'conv_%d.png' % i), convolutions[i])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type = int, default = 20, help = 'Number of gradient ascent iterations')
    parser.add_argument("--img", type = str, help = \
        'Path to image to project filter on, like in google dream. If not specified, uses a random init')
    parser.add_argument("--weights_path", type = str, default = 'vgg16_weights.h5', help = 'Path to network weights file')
    parser.add_argument("--layer", type = str, default = 'conv5_1', help = 'Name of layer to use. Uses layer names in model.py')
    parser.add_argument("--num_filters", type = int, default = 16, help = 'Number of filters to vizualize, starting from filter number 0.')
    parser.add_argument("--size", type = int, default = 256, help = 'Image width and height')
    parser.add_argument("--channel", type = int, default = 3, help = 'Image channel')
    parser.add_argument("--conv_size", type = int, default = 256, help = 'Filter Image size ')
    parser.add_argument("--conv_path", type = str, default = './', help = 'Filter Image directory ')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    print(args)

    #Configuration:
    img_width, img_height = args.size, args.size
    filter_indexes = range(0, args.num_filters)

    input_placeholder = K.placeholder((1, args.channel, img_width, img_height))
    first_layer = ZeroPadding2D((1, 1), input_shape=(args.channel, img_width, img_height))
    # first_layer.input = input_placeholder


    # model = get_model(first_layer)
    model = get_unet(args.channel)
    # model = load_model_weights(model, args.weights_path)
    model.load_weights(args.weights_path)

    layer_name = args.layer
    if not args.layer:
        all_layers = [l.name for l in model.layers]
        print(all_layers)
        layer_name= input('Select layer: ')
    
    layer = get_output_layer(model, layer_name)


    

    if args.img is None:
        # we start from a gray image with some random noise
        init_img = np.random.random((1, args.channel, img_width, img_height)) * 20 + 128.
    else:
        with rasterio.open(args.img, 'r') as f:
            values = f.read().astype(np.float32)            
            init_img = [values]

    layer_to_visualize(init_img, layer, layer_name, args.conv_size)

    # vizualizations = [None] * len(filter_indexes)
    # for i, index in enumerate(filter_indexes):
    #     vizualizations[i] = visualize_filter(init_img, index, input_placeholder, args.iterations)
    #     #Save the visualizations see the progress made so far
    #     save_filters(vizualizations, img_width, img_height)
