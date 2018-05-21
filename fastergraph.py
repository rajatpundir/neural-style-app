from basegraph import BaseGraph
import numpy as np
import tensorflow as tf
from PIL import Image

class FasterGraph(BaseGraph):
    def __init__(self, content_images_list, style_images_list, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta):
        super().__init__(content_images_list, style_images_list, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta)
        self.initialize_model()
        self.define_final_loss()

    def initialize_model(self):
        parameters = np.load('model/faster.model')
        keys = sorted(parameters.keys())
        # Block 1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.constant(parameters[keys[0]], shape=[3, 3, 3, 64], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[1]], shape=[64], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.constant(parameters[keys[2]], shape=[3, 3, 64, 64], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[3]], shape=[64], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
        self.pool1 = tf.nn.avg_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        # Block 2
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.constant(parameters[keys[4]], shape=[3, 3, 64, 128], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[5]], shape=[128], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.constant(parameters[keys[6]], shape=[3, 3, 128, 128], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[7]], shape=[128], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
        self.pool2 = tf.nn.avg_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        # Block 3
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.constant(parameters[keys[8]], shape=[3, 3, 128, 256], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[9]], shape=[256], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.constant(parameters[keys[10]], shape=[3, 3, 256, 256], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[11]], shape=[256], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.constant(parameters[keys[12]], shape=[3, 3, 256, 256], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[13]], shape=[256], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
        self.pool3 = tf.nn.avg_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        # Block 4
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.constant(parameters[keys[14]], shape=[3, 3, 256, 512], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[15]], shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.constant(parameters[keys[16]], shape=[3, 3, 512, 512], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[17]], shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.constant(parameters[keys[18]], shape=[3, 3, 512, 512], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[19]], shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
        self.pool4 = tf.nn.avg_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        # Block 5
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.constant(parameters[keys[20]], shape=[3, 3, 512, 512], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[21]], shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.constant(parameters[keys[22]], shape=[3, 3, 512, 512], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[23]], shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.constant(parameters[keys[24]], shape=[3, 3, 512, 512], dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(parameters[keys[25]], shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
        self.pool5 = tf.nn.avg_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
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
        self.layer_func['pool3'] = self.pool3
        self.layer_func['conv4_1'] = self.conv4_1
        self.layer_func['conv4_2'] = self.conv4_2
        self.layer_func['conv4_3'] = self.conv4_3
        self.layer_func['pool4'] = self.pool4
        self.layer_func['conv5_1'] = self.conv5_1
        self.layer_func['conv5_2'] = self.conv5_2
        self.layer_func['conv5_3'] = self.conv5_3
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
            if content_image.content_pool3_check:
                cont_layers.append(('pool3', content_image.content_pool3_weight))
            if content_image.content_conv4_1_check:
                cont_layers.append(('conv4_1', content_image.content_conv4_1_weight))
            if content_image.content_conv4_2_check:
                cont_layers.append(('conv4_2', content_image.content_conv4_2_weight))
            if content_image.content_conv4_3_check:
                cont_layers.append(('conv4_3', content_image.content_conv4_3_weight))
            if content_image.content_pool4_check:
                cont_layers.append(('pool4', content_image.content_pool4_weight))
            if content_image.content_conv5_1_check:
                cont_layers.append(('conv5_1', content_image.content_conv5_1_weight))
            if content_image.content_conv5_2_check:
                cont_layers.append(('conv5_2', content_image.content_conv5_2_weight))
            if content_image.content_conv5_3_check:
                cont_layers.append(('conv5_3', content_image.content_conv5_3_weight))
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
            if style_image.style_pool3_check:
                styl_layers.append(('pool3', style_image.style_pool3_weight))
            if style_image.style_conv4_1_check:
                styl_layers.append(('conv4_1', style_image.style_conv4_1_weight))
            if style_image.style_conv4_2_check:
                styl_layers.append(('conv4_2', style_image.style_conv4_2_weight))
            if style_image.style_conv4_3_check:
                styl_layers.append(('conv4_3', style_image.style_conv4_3_weight))
            if style_image.style_pool4_check:
                styl_layers.append(('pool4', style_image.style_pool4_weight))
            if style_image.style_conv5_1_check:
                styl_layers.append(('conv5_1', style_image.style_conv5_1_weight))
            if style_image.style_conv5_2_check:
                styl_layers.append(('conv5_2', style_image.style_conv5_2_weight))
            if style_image.style_conv5_3_check:
                styl_layers.append(('conv5_3', style_image.style_conv5_3_weight))
            if style_image.style_pool5_check:
                styl_layers.append(('pool5', style_image.style_pool5_weight))
            self.style_layers.append(styl_layers)

