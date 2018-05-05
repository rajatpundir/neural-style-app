from neuralgraph import NeuralGraph
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
import time


class NeuralWorker:
    def __init__(self, result_queue, command_queue, response_queue, content_path_list, style_path_list, width=800, height=600, use_meta=False, save_meta=False, use_lbfgs=True, max_iterations=10, noise_ratio=0.4, alpha=50, beta=100, gamma=10):
        # for intialization
        self.result_queue = result_queue
        self.command_queue = command_queue
        self.response_queue = response_queue
        self.content_path_list = content_path_list
        self.style_path_list = style_path_list
        self.width = width
        self.height = height
        self.use_meta = use_meta
        self.save_meta = save_meta
        self.use_lbfgs = use_lbfgs
        self.max_iterations = max_iterations
        self.noise_ratio = noise_ratio
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
        self.initialize_neural_worker()


    def initialize_session_and_model(self):
        self.sess = tf.InteractiveSession()
        self.computation_graph = NeuralGraph(self.width, self.height)
        self.model = self.computation_graph.construct_model()
        print(self.model)


    def prepare_content_list_from_content_source_list(self):
        for path in self.content_path_list:
            temp_content = scipy.misc.imread(path)
            temp_content = scipy.misc.imresize(temp_content, (self.height, self.width))
            temp_content = np.reshape(temp_content, (self.batch_size, self.height, self.width, self.channels))
            temp_content = temp_content - self.mean_of_images
            self.content_list.append(temp_content)


    def prepare_style_list_from_style_source_list(self):
        for path in self.style_path_list:
            temp_style = scipy.misc.imread(path)
            temp_style = scipy.misc.imresize(temp_style, (self.height, self.width))
            temp_style = np.reshape(temp_style, (self.batch_size, self.height, self.width, self.channels))
            temp_style = temp_style - self.mean_of_images
            self.style_list.append(temp_style)


    def prepare_input_to_model(self):
        if not self.use_meta:
            noise = np.random.uniform(-20, 20, (self.batch_size, self.height, self.width, self.channels)).astype('float32')
            self.input_to_model = noise * self.noise_ratio
            for content in self.content_list:
                self.input_to_model += content * ((1 - self.noise_ratio) / len(self.content_list))
        else:
            try:
                temp_input_to_model = scipy.misc.imread('meta/meta.png')
                temp_input_to_model = scipy.misc.imresize(temp_input_to_model, (self.height, self.width))
                temp_input_to_model = np.reshape(temp_input_to_model, (self.batch_size, self.height, self.width, self.channels))
                self.input_to_model = temp_input_to_model - self.mean_of_images
            except:
                noise = np.random.uniform(-20, 20, (self.batch_size, self.height, self.width, self.channels)).astype('float32')
                self.input_to_model = noise * self.noise_ratio
                for content in self.content_list:
                    self.input_to_model += content * ((1 - self.noise_ratio) / len(self.content_list))


    def construct_output_image(self):
        self.response_queue.put('Constructing image %d...' % self.image_counter)
        constructed_image = self.sess.run(self.model['input'])
        scipy.misc.imsave('out/%d.png' % self.image_counter, np.clip((constructed_image + self.mean_of_images)[0], 0, 255).astype('uint8'))
        if self.save_meta:
            scipy.misc.imsave('meta/meta.png', np.clip((constructed_image + self.mean_of_images)[0], 0, 255).astype('uint8'))
        self.result_queue.put('out/%d.png' % self.image_counter)
        self.image_counter += 1


    def prepare_session(self):
        loss_over_content = None
        self.sess.run(tf.global_variables_initializer())
        for content in self.content_list:
            self.sess.run(self.model['input'].assign(content))
            temp_loss_over_content = self.computation_graph.calulate_loss_over_content(self.sess, self.model)
            if loss_over_content is None:
                loss_over_content = temp_loss_over_content
            else:
                loss_over_content += temp_loss_over_content
        #########
        loss_over_style = None
        for style in self.style_list:
            self.sess.run(self.model['input'].assign(style))
            temp_loss_over_style = self.computation_graph.calulate_loss_over_style(self.sess, self.model)
            if loss_over_style is None:
                loss_over_style = temp_loss_over_style
            else:
                loss_over_style += temp_loss_over_style
        #########
        self.sess.run(self.model['input'].assign(self.input_to_model))
        loss_variational = 1.0 * self.computation_graph.calculate_total_variation_loss(self.sess, self.model)
        #########
        loss_final = self.alpha * loss_over_content + self.beta * loss_over_style + self.gamma * loss_variational
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.model['input'].assign(self.input_to_model))
        self.construct_output_image()
        #########
        if self.use_lbfgs:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.model['input'].assign(self.input_to_model))
            self.train_step = tf.contrib.opt.ScipyOptimizerInterface(loss_final, options={'maxiter': self.max_iterations - 1})
        else:
            optimizer = tf.train.AdamOptimizer(2.0)
            self.train_step = optimizer.minimize(loss_final)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.model['input'].assign(self.input_to_model))


    def initialize_neural_worker(self):
        self.initialize_session_and_model()
        self.prepare_content_list_from_content_source_list()
        self.prepare_style_list_from_style_source_list()
        self.prepare_input_to_model()
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
                self.sess.close()
                self.response_queue.put('stopped')
                print('Thread Stopped.')
                break
            #########
            if self.use_lbfgs:
                self.train_step.minimize(self.sess)
                self.construct_output_image()
            else:
                while True:
                    self.sess.run(self.train_step)
                    if self.iterations_counter % self.max_iterations == 0:
                        self.construct_output_image()
                        break
                    self.iterations_counter += 1

