import numpy as np
from keras import backend

class DeepGraph:
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        self.channels = 3
    def calulate_loss_over_content(self, content, combination):
        return backend.sum(backend.square(combination - content))
    def calculate_gram_matrix(self, x):
        features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
        gram = backend.dot(features, backend.transpose(features))
        return gram
    def calculate_loss_in_style_over_layer(self, style, combination):
        S = self.calculate_gram_matrix(style)
        C = self.calculate_gram_matrix(combination)
        return backend.sum(backend.square(S - C)) / (4. * (self.channels ** 2) * (self.height * self.width ** 2))
    def calculate_total_variation_loss(self, x):
        a = backend.square(x[:, :self.height-1, :self.width-1, :] - x[:, 1:, :self.width-1, :])
        b = backend.square(x[:, :self.height-1, :self.width-1, :] - x[:, :self.height-1, 1:, :])
        return backend.sum(backend.pow(a + b, 1.25))


class Evaluator(object):
    def __init__(self, width, height, final_outputs):
        self.loss_value = None
        self.grads_values = None
        self.width = width
        self.height = height
        self.final_outputs = final_outputs
    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
    def eval_loss_and_grads(self, x):
        x = x.reshape((1, self.height, self.width, 3))
        outs = self.final_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        return loss_value, grad_values

