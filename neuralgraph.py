import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf


class NeuralGraph:
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height

    def get_weights(self, model_matrix, c):
        # Extracting weights for current layer c in the matrix of the model.
        return(model_matrix[0][c][0][0][0][0][0])

    def get_biases(self, model_matrix, c):
        # Extracting biases for current layer c in the matrix of the model.
        return(model_matrix[0][c][0][0][0][0][1])

    def perform_convolution(self, model_matrix, p, c):
        # Function to perform convolution over output of layer p, to get output of layer c.
        weights_for_layer = tf.constant(self.get_weights(model_matrix, c))
        biases_for_layer = tf.constant(np.reshape(self.get_biases(model_matrix, c), (self.get_biases(model_matrix, c).size)))
        return tf.nn.conv2d(p, filter=weights_for_layer, strides=[1, 1, 1, 1], padding='SAME') + biases_for_layer

    def perform_convolution_and_relu(self, model_matrix, p, c):
        # Function to perform both convolution and relu operations.
        return tf.nn.relu(self.perform_convolution(model_matrix, p, c))

    def perform_average_pooling(self, model_matrix, p):
        # Function to perform avergae pooling over output of layer p.
        return tf.nn.avg_pool(p, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def construct_model(self):
        # Function to construct the computation graph of the tensorlfow model.
        try:
            model_matrix = scipy.io.loadmat('model/vgg19.model')['layers']
            model = {}
            # Creating input layer with shape of (1, 600, 800, 3), 600 = height, 800 = width, 3 = channels(RGB)
            model['input']    = tf.Variable(np.zeros((1, self.height, self.width, 3)), dtype = 'float32')
            # Creating other layers in the computation graph of tensorflow model.
            # Convolution layers are at 0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32 and 34.
            # Block 1
            model['conv1_1']  = self.perform_convolution_and_relu(model_matrix, model['input'], 0)
            model['conv1_2']  = self.perform_convolution_and_relu(model_matrix, model['conv1_1'], 2)
            model['avgpool1'] = self.perform_average_pooling(model_matrix, model['conv1_2'])
            # Block 2
            model['conv2_1']  = self.perform_convolution_and_relu(model_matrix, model['avgpool1'], 5)
            model['conv2_2']  = self.perform_convolution_and_relu(model_matrix, model['conv2_1'], 7)
            model['avgpool2'] = self.perform_average_pooling(model_matrix, model['conv2_2'])
            # Block 3
            model['conv3_1']  = self.perform_convolution_and_relu(model_matrix, model['avgpool2'], 10)
            model['conv3_2']  = self.perform_convolution_and_relu(model_matrix, model['conv3_1'], 12)
            model['conv3_3']  = self.perform_convolution_and_relu(model_matrix, model['conv3_2'], 14)
            model['conv3_4']  = self.perform_convolution_and_relu(model_matrix, model['conv3_3'], 16)
            model['avgpool3'] = self.perform_average_pooling(model_matrix, model['conv3_4'])
            # Block 4
            model['conv4_1']  = self.perform_convolution_and_relu(model_matrix, model['avgpool3'], 19)
            model['conv4_2']  = self.perform_convolution_and_relu(model_matrix, model['conv4_1'], 21)
            model['conv4_3']  = self.perform_convolution_and_relu(model_matrix, model['conv4_2'], 23)
            model['conv4_4']  = self.perform_convolution_and_relu(model_matrix, model['conv4_3'], 25)
            model['avgpool4'] = self.perform_average_pooling(model_matrix, model['conv4_4'])
            # Block 5
            model['conv5_1']  = self.perform_convolution_and_relu(model_matrix, model['avgpool4'], 28)
            model['conv5_2']  = self.perform_convolution_and_relu(model_matrix, model['conv5_1'], 30)
            model['conv5_3']  = self.perform_convolution_and_relu(model_matrix, model['conv5_2'], 32)
            model['conv5_4']  = self.perform_convolution_and_relu(model_matrix, model['conv5_3'], 34)
            model['avgpool5'] = self.perform_average_pooling(model_matrix, model['conv5_4'])
            return(model)
        except:
            print('Exception: model not found inside model directory')
            return({})
        

    def calulate_loss_over_content(self, sess, model):
        p = sess.run(model['conv2_2'])
        x = model['conv2_2']
        depth = p.shape[3]
        area = p.shape[1] * p.shape[2]
        # Constant value is not required but can speed up training process a bit.
        cons = 1 / (4 * depth * area)
        return(cons * tf.reduce_sum(tf.pow(x - p, 2)))

    def calculate_gram_matrix(self, volume, depth, area):
        # Function to calculate gram matrix of a layer.
        # Reshape 3D volume into a 2D volume with value of area as its rows, and depth as its columns.
        V = tf.reshape(volume, (area, depth))
        # Calculating gram matrix of [depth x depth] where each value in the matrix describes covariance between depth fibers.
        return(tf.matmul(tf.transpose(V), V))

    def calculate_loss_in_style_over_layer(self, sess, model, x, y):
        # Function to calculate loss in style of a particular layer.
        depth = x.shape[3]
        area = x.shape[1] * x.shape[2]
        # Calculate gram matrix for the layer over image being constructed.
        loss_over_image_constructed = self.calculate_gram_matrix(x, depth, area)
        # Calculate gram matrix for the layer over style image.
        loss_over_image_of_style = self.calculate_gram_matrix(y, depth, area)
        cons = 1 / (4 * depth**2 * area**2)
        result = cons * tf.reduce_sum(tf.pow(loss_over_image_of_style - loss_over_image_constructed, 2))
        return result

    def calulate_loss_over_style(self, sess, model):
        # Function to calculate loss of style over constructed image.
        # Specifying layers to consider while calculating loss over style
        layers_to_consider = ['conv1_2', 'conv2_2', 'conv3_3' ,'conv4_3', 'conv5_3']
        # Specifying weightage of layers while calculating loss over style
        # weightage_for_layer = [0.5, 1.0, 1.5, 3.0, 4.0]
        weightage_for_layer = [0.2, 0.2, 0.2, 0.2, 0.2]
        # Calculating loss over each layer in the list layers_to_consider
        loss_over_layer = []
        for layer in layers_to_consider:
            loss_over_layer.append(self.calculate_loss_in_style_over_layer(sess, model, sess.run(model[layer]), model[layer]))
        # Calculating final loss over style with losses of layers multiplied with their weightage
        loss_over_style = 0
        for i in range(len(layers_to_consider)):
                # loss_over_style += (weightage_for_layer[i]) * loss_over_layer[i]
                loss_over_style += (0.025 / len(layers_to_consider)) * loss_over_layer[i]
        return(loss_over_style)

    def calculate_total_variation_loss(self, sess, model):
        x = sess.run(model['input'])
        a = tf.pow((x[:, :self.height-1, :self.width-1, :] - x[:, 1:, :self.width-1, :]), 2)
        b = tf.pow((x[:, :self.height-1, :self.width-1, :] - x[:, :self.height-1, 1:, :]), 2)
        return(tf.reduce_sum(tf.pow(a + b, 1.25)))

