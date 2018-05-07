import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from PIL import Image

class FasterGraph:
    def __init__(self, width, height, alpha, beta, gamma):
        self.width = width
        self.height = height
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.content_layer = ['conv2_2']
        self.style_layers = ['conv1_2', 'conv2_2', 'conv3_3' ,'conv4_3', 'conv5_3']
        self.initialize_model()

    def initialize_model(self):
        self.sess = tf.Session()
        self.batch_size = 1
        self.channels = 3
        self.inputs = tf.Variable(np.zeros((self.batch_size, self.height, self.width, self.channels)), dtype = 'float32', name='input')
        self.mean = np.array([123.68, 116.779, 103.939])
        parameters = np.load('model/vgg16.model')
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
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
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
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
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
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
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
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
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
        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        if True:
            return
        # Block FC
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.constant(parameters[keys[26]], shape=[shape, 4096], dtype=tf.float32, name='weights')
            fc1b = tf.constant(parameters[keys[27]], shape=[4096], dtype=tf.float32, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
        with tf.name_scope('fc2') as scope:
            fc2w = tf.constant(parameters[keys[28]], shape=[4096, 4096], dtype=tf.float32, name='weights')
            fc2b = tf.constant(parameters[keys[29]], shape=[4096], dtype=tf.float32, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
        with tf.name_scope('fc3') as scope:
            fc3w = tf.constant(parameters[keys[30]], shape=[4096, 1000], dtype=tf.float32, name='weights')
            fc3b = tf.constant(parameters[keys[31]], shape=[1000], dtype=tf.float32, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)

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
        elif layer is 'pool1':
            return self.pool1
        elif layer is 'conv2_1':
            return self.conv2_1
        elif layer is 'conv2_2':
            return self.conv2_2
        elif layer is 'pool2':
            return self.pool2
        elif layer is 'conv3_1':
            return self.conv3_1
        elif layer is 'conv3_2':
            return self.conv3_2
        elif layer is 'conv3_3':
            return self.conv3_3
        elif layer is 'pool3':
            return self.pool3
        elif layer is 'conv4_1':
            return self.conv4_1
        elif layer is 'conv4_2':
            return self.conv4_2
        elif layer is 'conv4_3':
            return self.conv4_3
        elif layer is 'pool4':
            return self.pool4
        elif layer is 'conv5_1':
            return self.conv5_1
        elif layer is 'conv5_2':
            return self.conv5_2
        elif layer is 'conv5_3':
            return self.conv5_3
        elif layer is 'pool5':
            return self.pool5

    def preprocess(self, path):
        temp = Image.open(path).resize((self.width, self.height))
        temp = np.asarray(temp, dtype='float32')
        temp -= self.mean
        temp = np.expand_dims(temp, axis=0)
        return(temp[:, :, :, ::-1])

    def predict(self):
        # preprocess images first.
        pass

