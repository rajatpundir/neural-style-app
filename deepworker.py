from deepgraph import DeepGraph, Evaluator
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from PIL import Image
from keras import backend
# from keras.models import Model
from keras.applications.vgg16 import VGG16
import time


class DeepWorker:
    def __init__(self, result_queue, command_queue, response_queue, content_path_list, style_path_list, width=512, height=512, use_meta=False, save_meta=False, use_lbfgs=True, max_iterations=20, noise_ratio=0.4, alpha=0.025, beta=5.0, gamma=1.0):
        # for intialization
        self.result_queue = result_queue
        self.command_queue = command_queue
        self.response_queue = response_queue
        self.content_path_list = content_path_list
        self.style_path_list = style_path_list
        self.width = width
        self.height = width
        self.use_meta = use_meta
        self.save_meta = save_meta
        self.use_lbfgs = use_lbfgs
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # some constants
        self.mean_of_images = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
        self.batch_size = 1
        self.channels = 3
        # other variables
        self.content_list = []
        self.style_list = []
        self.input_to_model = None
        self.computation_graph = None
        self.sess = None
        self.model = None
        self.train_step = None
        self.image_counter = 0
        self.iterations_counter = 0
        #########
        self.layers = None
        self.evaluator = None
        self.combination = None
        self.initialize_deep_worker()


    def prepare_content_list_from_content_source_list(self):
        for path in self.content_path_list:
            temp_content = Image.open(path).resize((self.width, self.height))
            temp_content = np.asarray(temp_content, dtype='float32')
            temp_content = np.expand_dims(temp_content, axis=0)
            temp_content[:, :, :, 0] -= 103.939
            temp_content[:, :, :, 1] -= 116.779
            temp_content[:, :, :, 2] -= 123.68
            temp_content = temp_content[:, :, :, ::-1]
            self.content_list.append(temp_content)

    def prepare_style_list_from_style_source_list(self):
        for path in self.style_path_list:
            temp_style = Image.open(path).resize((self.width, self.height))
            temp_style = np.asarray(temp_style, dtype='float32')
            temp_style = np.expand_dims(temp_style, axis=0)
            temp_style[:, :, :, 0] -= 103.939
            temp_style[:, :, :, 1] -= 116.779
            temp_style[:, :, :, 2] -= 123.68
            temp_style = temp_style[:, :, :, ::-1]
            self.style_list.append(temp_style)

    def initialize_session_and_model(self):
        self.computation_graph = DeepGraph(self.width, self.height)
        content = backend.variable(self.content_list[0])
        style = backend.variable(self.style_list[0])
        self.combination = backend.placeholder((self.batch_size, self.height, self.width, self.channels))
        input_tensor = backend.concatenate([content, style, self.combination], axis=0)
        model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
        self.layers = dict([(layer.name, layer.output) for layer in model.layers])

    def prepare_input_to_model(self):
        self.input_to_model = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128.

    def construct_output_image(self, x):
        self.response_queue.put('Constructing image %d...' % self.image_counter)
        constructed_image = x.reshape((self.height, self.width, 3))
        constructed_image = constructed_image[:, :, ::-1]
        constructed_image[:, :, 0] += 103.939
        constructed_image[:, :, 1] += 116.779
        constructed_image[:, :, 2] += 123.68
        constructed_image = np.clip(constructed_image, 0, 255).astype('uint8')
        Image.fromarray(constructed_image).save('out/%d.png' % self.image_counter,'PNG')
        self.result_queue.put('out/%d.png' % self.image_counter)
        self.image_counter += 1

    def prepare_session(self):
        loss_final = backend.variable(0.)
        # Calculate content loss
        layer_features = self.layers['block2_conv2']
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss_final += self.alpha * self.computation_graph.calulate_loss_over_content(content_image_features, combination_features)
        # Calculate style loss
        feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
        for layer_name in feature_layers:
            layer_features = self.layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = self.computation_graph.calculate_loss_in_style_over_layer(style_features, combination_features)
            loss_final += (self.beta / len(feature_layers)) * sl
        # Calculate total variation loss
        loss_final += self.gamma * self.computation_graph.calculate_total_variation_loss(self.combination)  
        outputs = [loss_final]
        outputs += backend.gradients(loss_final, self.combination)
        final_outputs = backend.function([self.combination], outputs)
        self.evaluator = Evaluator(self.width, self.height, final_outputs)

    def initialize_deep_worker(self):
        self.prepare_content_list_from_content_source_list()
        self.prepare_style_list_from_style_source_list()
        self.prepare_input_to_model()
        self.initialize_session_and_model()
        self.prepare_session()

    def train(self):
        stopped = False
        while True:
            while not self.command_queue.empty():
                command = str(self.command_queue.get())
                if command is 'pause':
                    while True:
                        self.response_queue.put('paused')
                        if not self.command_queue.empty():
                            command = str(self.command_queue.get())
                            if command is 'resume':
                                self.response_queue.put('resumed')
                                break
                            elif command is 'stop':
                                stopped = True
                                break
                        time.sleep(.5)
                elif command is 'stop':
                    stopped = True
                    break
            #########
            if stopped:
                self.response_queue.put('stopped')
                print('Thread Stopped.')
                break
            #########
            self.input_to_model, min_val, info = fmin_l_bfgs_b(self.evaluator.loss, self.input_to_model.flatten(), fprime=self.evaluator.grads, maxfun=self.max_iterations)
            self.construct_output_image(self.input_to_model)

