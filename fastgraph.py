from basegraph import BaseGraph
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.io import loadmat

class FastGraph(BaseGraph):
    def __init__(self, content_images_list, style_images_list, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta):
        super().__init__(content_images_list, style_images_list, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta)
        self.initialize_model()
        self.define_final_loss()

    def conv2d_relu(self, parameters, previous_layer, current_layer):
        weights = tf.constant(parameters[0][current_layer][0][0][0][0][0])
        bias = tf.constant(np.reshape(parameters[0][current_layer][0][0][0][0][1], (parameters[0][current_layer][0][0][0][0][1]).size))
        return(tf.nn.relu(tf.nn.conv2d(previous_layer, filter=weights, strides=[1, 1, 1, 1], padding='SAME') + bias))

    def initialize_model(self):
        parameters = loadmat('model/fast.model')['layers']
        # Block 1
        self.conv1_1 = self.conv2d_relu(parameters, self.inputs, 0)
        self.conv1_2 = self.conv2d_relu(parameters, self.conv1_1, 2)
        self.pool1 = tf.nn.avg_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Block 2
        self.conv2_1 = self.conv2d_relu(parameters, self.pool1, 5)
        self.conv2_2 = self.conv2d_relu(parameters, self.conv2_1, 7)
        self.pool2 = tf.nn.avg_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Block 3
        self.conv3_1 = self.conv2d_relu(parameters, self.pool2, 10)
        self.conv3_2 = self.conv2d_relu(parameters, self.conv3_1, 12)
        self.conv3_3 = self.conv2d_relu(parameters, self.conv3_2, 14)
        self.conv3_4 = self.conv2d_relu(parameters, self.conv3_3, 16)
        self.pool3 = tf.nn.avg_pool(self.conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Block 4
        self.conv4_1 = self.conv2d_relu(parameters, self.pool3, 19)
        self.conv4_2 = self.conv2d_relu(parameters, self.conv4_1, 21)
        self.conv4_3 = self.conv2d_relu(parameters, self.conv4_2, 23)
        self.conv4_4 = self.conv2d_relu(parameters, self.conv4_3, 25)
        self.pool4 = tf.nn.avg_pool(self.conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Block 5
        self.conv5_1 = self.conv2d_relu(parameters, self.pool4, 28)
        self.conv5_2 = self.conv2d_relu(parameters, self.conv5_1, 30)
        self.conv5_3 = self.conv2d_relu(parameters, self.conv5_2, 32)
        self.conv5_4 = self.conv2d_relu(parameters, self.conv5_3, 34)
        self.pool5 = tf.nn.avg_pool(self.conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Adding layers to layer_func
        self.layer_func['input'] = self.inputs
        self.layer_func['conv1_1'] = self.conv1_1
        self.layer_func['conv1_2'] = self.conv1_2
        self.layer_func['pool1'] = self.pool1
        self.layer_func['conv2_1'] = self.conv2_1
        self.layer_func['conv2_2'] = self.conv2_2
        self.layer_func['pool2'] = self.pool2
        self.layer_func['conv3_1'] = self.conv3_1
        self.layer_func['conv3_2'] = self.conv3_2
        self.layer_func['conv3_3'] = self.conv3_3
        self.layer_func['conv3_4'] = self.conv3_4
        self.layer_func['pool3'] = self.pool3
        self.layer_func['conv4_1'] = self.conv4_1
        self.layer_func['conv4_2'] = self.conv4_2
        self.layer_func['conv4_3'] = self.conv4_3
        self.layer_func['conv4_4'] = self.conv4_4
        self.layer_func['pool4'] = self.pool4
        self.layer_func['conv5_1'] = self.conv5_1
        self.layer_func['conv5_2'] = self.conv5_2
        self.layer_func['conv5_3'] = self.conv5_3
        self.layer_func['conv5_4'] = self.conv5_4
        self.layer_func['pool5'] = self.pool5 

    def preprocess_content_layers(self):
        for content_image in self.content_images_list:
            cont_layers = []
            if content_image.content_conv1_1_check:
                cont_layers.append(('conv1_1', content_image.content_conv1_1_weight))
            if content_image.content_conv1_2_check:
                cont_layers.append(('conv1_2', content_image.content_conv1_2_weight))
            if content_image.content_pool1_check:
                cont_layers.append(('pool1', content_image.content_pool1_weight))
            if content_image.content_conv2_1_check:
                cont_layers.append(('conv2_1', content_image.content_conv2_1_weight))
            if content_image.content_conv2_2_check:
                cont_layers.append(('conv2_2', content_image.content_conv2_2_weight))
            if content_image.content_pool2_check:
                cont_layers.append(('pool2', content_image.content_pool2_weight))
            if content_image.content_conv3_1_check:
                cont_layers.append(('conv3_1', content_image.content_conv3_1_weight))
            if content_image.content_conv3_2_check:
                cont_layers.append(('conv3_2', content_image.content_conv3_2_weight))
            if content_image.content_conv3_3_check:
                cont_layers.append(('conv3_3', content_image.content_conv3_3_weight))
            if content_image.content_conv3_4_check:
                cont_layers.append(('conv3_4', content_image.content_conv3_4_weight))
            if content_image.content_pool3_check:
                cont_layers.append(('pool3', content_image.content_pool3_weight))
            if content_image.content_conv4_1_check:
                cont_layers.append(('conv4_1', content_image.content_conv4_1_weight))
            if content_image.content_conv4_2_check:
                cont_layers.append(('conv4_2', content_image.content_conv4_2_weight))
            if content_image.content_conv4_3_check:
                cont_layers.append(('conv4_3', content_image.content_conv4_3_weight))
            if content_image.content_conv4_4_check:
                cont_layers.append(('conv4_4', content_image.content_conv4_4_weight))
            if content_image.content_pool4_check:
                cont_layers.append(('pool4', content_image.content_pool4_weight))
            if content_image.content_conv5_1_check:
                cont_layers.append(('conv5_1', content_image.content_conv5_1_weight))
            if content_image.content_conv5_2_check:
                cont_layers.append(('conv5_2', content_image.content_conv5_2_weight))
            if content_image.content_conv5_3_check:
                cont_layers.append(('conv5_3', content_image.content_conv5_3_weight))
            if content_image.content_conv5_4_check:
                cont_layers.append(('conv5_4', content_image.content_conv5_4_weight))
            if content_image.content_pool5_check:
                cont_layers.append(('pool5', content_image.content_pool5_weight))
            self.content_layers.append(cont_layers)

    def preprocess_style_layers(self):
        for style_image in self.style_images_list:
            styl_layers = []
            if style_image.style_conv1_1_check:
                styl_layers.append(('conv1_1', style_image.style_conv1_1_weight))
            if style_image.style_conv1_2_check:
                styl_layers.append(('conv1_2', style_image.style_conv1_2_weight))
            if style_image.style_pool1_check:
                styl_layers.append(('pool1', style_image.style_pool1_weight))
            if style_image.style_conv2_1_check:
                styl_layers.append(('conv2_1', style_image.style_conv2_1_weight))
            if style_image.style_conv2_2_check:
                styl_layers.append(('conv2_2', style_image.style_conv2_2_weight))
            if style_image.style_pool2_check:
                styl_layers.append(('pool2', style_image.style_pool2_weight))
            if style_image.style_conv3_1_check:
                styl_layers.append(('conv3_1', style_image.style_conv3_1_weight))
            if style_image.style_conv3_2_check:
                styl_layers.append(('conv3_2', style_image.style_conv3_2_weight))
            if style_image.style_conv3_3_check:
                styl_layers.append(('conv3_3', style_image.style_conv3_3_weight))
            if style_image.style_conv3_4_check:
                styl_layers.append(('conv3_4', style_image.style_conv3_4_weight))
            if style_image.style_pool3_check:
                styl_layers.append(('pool3', style_image.style_pool3_weight))
            if style_image.style_conv4_1_check:
                styl_layers.append(('conv4_1', style_image.style_conv4_1_weight))
            if style_image.style_conv4_2_check:
                styl_layers.append(('conv4_2', style_image.style_conv4_2_weight))
            if style_image.style_conv4_3_check:
                styl_layers.append(('conv4_3', style_image.style_conv4_3_weight))
            if style_image.style_conv4_4_check:
                styl_layers.append(('conv4_4', style_image.style_conv4_4_weight))
            if style_image.style_pool4_check:
                styl_layers.append(('pool4', style_image.style_pool4_weight))
            if style_image.style_conv5_1_check:
                styl_layers.append(('conv5_1', style_image.style_conv5_1_weight))
            if style_image.style_conv5_2_check:
                styl_layers.append(('conv5_2', style_image.style_conv5_2_weight))
            if style_image.style_conv5_3_check:
                styl_layers.append(('conv5_3', style_image.style_conv5_3_weight))
            if style_image.style_conv5_4_check:
                styl_layers.append(('conv5_4', style_image.style_conv5_4_weight))
            if style_image.style_pool5_check:
                styl_layers.append(('pool5', style_image.style_pool5_weight))
            self.style_layers.append(styl_layers)

