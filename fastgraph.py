import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from PIL import Image

class FastGraph:
    def __init__(self, width, height, alpha, beta, gamma):
        self.width = width
        self.height = height
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.content_layer = ['conv1_1']
        self.style_layers = ['conv1_2', 'conv2_2', 'conv3_4' ,'conv4_4', 'conv5_4']
        self.initialize_model()

    def conv2d_relu(self, parameters, previous_layer, current_layer):
        weights = tf.constant(parameters[0][current_layer][0][0][0][0][0])
        bias = tf.constant(np.reshape(parameters[0][current_layer][0][0][0][0][1], (parameters[0][current_layer][0][0][0][0][1]).size))
        return(tf.nn.relu(tf.nn.conv2d(previous_layer, filter=weights, strides=[1, 1, 1, 1], padding='SAME') + bias))

    def initialize_model(self):
        self.sess = tf.InteractiveSession()
        self.batch_size = 1
        self.channels = 3
        self.inputs = tf.Variable(np.zeros((self.batch_size, self.height, self.width, self.channels)), dtype = 'float32')
        self.mean = np.array([123.68, 116.779, 103.939])
        parameters = loadmat('model/vgg19.model')['layers']
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
        c_cont = self.sess.run(self.get_layer(self.content_layer[0]))
        c_mix = self.get_layer(self.content_layer[0])
        const = 4 * c_cont.shape[3] * c_cont.shape[2] * c_cont.shape[1]
        return(self.alpha * tf.reduce_sum(tf.pow(c_mix - c_cont, 2)) / const)

    def gram_matrix(self, volume, area, depth):
        V = tf.reshape(volume, (area, depth))
        return(tf.matmul(tf.transpose(V), V))

    def style_loss_over_layer(self, layer):
        s_styl = self.sess.run(self.get_layer(layer))
        s_mix = self.get_layer(layer)
        area, depth = s_styl.shape[1] * s_styl.shape[2], s_styl.shape[3]
        const = 4 * depth**2 * area**2
        return(tf.reduce_sum(tf.pow(self.gram_matrix(s_mix, area, depth) - self.gram_matrix(s_styl, area, depth), 2)) / const)

    def style_loss(self):
        loss_in_style = 0
        for layer in self.style_layers:
            loss_in_style += (self.beta / len(self.style_layers)) * self.style_loss_over_layer(layer)
        return(self.beta * loss_in_style)

    def variation_loss(self):
        x = self.inputs
        a = tf.pow((x[:, :self.height-1, :self.width-1, :] - x[:, 1:, :self.width-1, :]), 2)
        b = tf.pow((x[:, :self.height-1, :self.width-1, :] - x[:, :self.height-1, 1:, :]), 2)
        return(self.gamma * tf.reduce_sum(tf.pow(a + b, 1.25)))

    def get_layer(self, layer):
        # Make it a dictionary, function is bit different for both models.
        if layer is 'input':
            return self.inputs
        elif layer is 'conv1_1':
            return self.conv1_1
        elif layer is 'conv1_2':
            return self.conv1_2
        elif layer is 'avgpool1':
            return self.avgpool1
        elif layer is 'conv2_1':
            return self.conv2_1
        elif layer is 'conv2_2':
            return self.conv2_2
        elif layer is 'avgpool2':
            return self.avgpool2
        elif layer is 'conv3_1':
            return self.conv3_1
        elif layer is 'conv3_2':
            return self.conv3_2
        elif layer is 'conv3_3':
            return self.conv3_3
        elif layer is 'conv3_4':
            return self.conv3_4
        elif layer is 'avgpool3':
            return self.avgpool3
        elif layer is 'conv4_1':
            return self.conv4_1
        elif layer is 'conv4_2':
            return self.conv4_2
        elif layer is 'conv4_3':
            return self.conv4_3
        elif layer is 'conv4_4':
            return self.conv4_4
        elif layer is 'avgpool4':
            return self.avgpool4
        elif layer is 'conv5_1':
            return self.conv5_1
        elif layer is 'conv5_2':
            return self.conv5_2
        elif layer is 'conv5_3':
            return self.conv5_3
        elif layer is 'conv5_4':
            return self.conv5_4
        elif layer is 'avgpool5':
            return self.avgpool5

    def preprocess(self, path):
        temp = Image.open(path).resize((self.width, self.height))
        temp = np.asarray(temp, dtype='float32')
        temp -= self.mean
        temp = np.expand_dims(temp, axis=0)
        return(temp[:, :, :, ::-1])

