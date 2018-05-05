import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
from scipy.io import loadmat

class FastGraph:
    def __init__(self, width, height, alpha, beta, gamma):
        self.sess = tf.InteractiveSession()
        self.width = width
        self.height = height
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
        self.batch_size = 1
        self.channels = 3
        self.load_model()

    def get_weights(self, parameters, layer):
        return(parameters[0][layer][0][0][0][0][0])

    def get_bias(self, parameters, layer):
        return(parameters[0][layer][0][0][0][0][1])

    def conv2d_relu(self, parameters, previous_layer, current_layer):
        weights = tf.constant(self.get_weights(parameters, current_layer))
        bias = tf.constant(np.reshape(self.get_bias(parameters, current_layer), (self.get_bias(parameters, current_layer).size)))
        return(tf.nn.relu(tf.nn.conv2d(previous_layer, filter=weights, strides=[1, 1, 1, 1], padding='SAME') + bias))

    def load_model(self):
        parameters = loadmat('model/vgg19.model')['layers']
        self.inputs = tf.Variable(np.zeros((self.batch_size, self.height, self.width, self.channels)), dtype = 'float32')
        # Block 1
        self.conv1_1 = self.conv2d_relu(parameters, self.inputs, 0)
        self.conv1_2 = self.conv2d_relu(parameters, self.conv1_1, 2)
        self.avgpool1 = tf.nn.avg_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Block 2
        self.conv2_1 = self.conv2d_relu(parameters, self.avgpool1, 5)
        self.conv2_2 = self.conv2d_relu(parameters, self.conv2_1, 7)
        self.avgpool2 = tf.nn.avg_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Block 3
        self.conv3_1 = self.conv2d_relu(parameters, self.avgpool2, 10)
        self.conv3_2 = self.conv2d_relu(parameters, self.conv3_1, 12)
        self.conv3_3 = self.conv2d_relu(parameters, self.conv3_2, 14)
        self.conv3_4 = self.conv2d_relu(parameters, self.conv3_3, 16)
        self.avgpool3 = tf.nn.avg_pool(self.conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Block 4
        self.conv4_1 = self.conv2d_relu(parameters, self.avgpool3, 19)
        self.conv4_2 = self.conv2d_relu(parameters, self.conv4_1, 21)
        self.conv4_3 = self.conv2d_relu(parameters, self.conv4_2, 23)
        self.conv4_4 = self.conv2d_relu(parameters, self.conv4_3, 25)
        self.avgpool4 = tf.nn.avg_pool(self.conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Block 5
        self.conv5_1 = self.conv2d_relu(parameters, self.avgpool4, 28)
        self.conv5_2 = self.conv2d_relu(parameters, self.conv5_1, 30)
        self.conv5_3 = self.conv2d_relu(parameters, self.conv5_2, 32)
        self.conv5_4 = self.conv2d_relu(parameters, self.conv5_3, 34)
        self.avgpool5 = tf.nn.avg_pool(self.conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    def content_loss(self):
        c_mix = self.sess.run(self.conv2_2)
        c_cont = self.conv2_2
        const = 4 * c_mix.shape[3] * c_mix.shape[2] * c_mix.shape[1]
        return(tf.reduce_sum(tf.pow(c_mix - c_cont, 2)) / const)

    def gram_matrix(self, volume, area, depth):
        V = tf.reshape(volume, (area, depth))
        return(tf.matmul(tf.transpose(V), V))

    def style_loss_over_layer(self, layer):
        s_mix = self.sess.run(self.get_layer(layer))
        s_styl = self.get_layer(layer)
        area, depth = s_mix.shape[1] * s_mix.shape[2], s_mix.shape[3]
        const = 4 * depth**2 * area**2
        return(tf.reduce_sum(tf.pow(self.gram_matrix(s_mix, area, depth) - self.gram_matrix(s_styl, area, depth), 2)) / const)

    def style_loss(self):
        layers = ['conv1_2', 'conv2_2', 'conv3_3' ,'conv4_3', 'conv5_3']
        loss_in_style = 0
        for layer in layers:
            loss_in_style += (self.beta / len(layers)) * self.style_loss_over_layer(layer)
        return(loss_in_style)

    def get_layer(self, layer):
        if layer is 'input':
            return self.inputs
        elif layer is 'conv1_2':
            return self.conv1_2
        elif layer is 'conv2_2':
            return self.conv2_2
        elif layer is 'conv3_3':
            return self.conv3_3
        elif layer is 'conv4_3':
            return self.conv4_3
        elif layer is 'conv5_3':
            return self.conv5_3

    def preprocess(self, path):
        temp = imresize(imread(path), (self.height, self.width))
        temp = np.reshape(temp, (self.batch_size, self.height, self.width, self.channels))
        return(temp - self.mean)

