from fastgraph import FastGraph
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
from scipy.io import loadmat
import time


class FastWorker:
    def __init__(self, result_queue, command_queue, response_queue, content_path_list, style_path_list, width=224, height=224, use_meta=False, save_meta=False, use_lbfgs=True, max_iterations=10, noise_ratio=0.4, alpha=0.025, beta=5.0, gamma=1.0):
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
        # other variables
        self.content_list = []
        self.style_list = []
        self.mix_image = None
        self.model = None
        self.train_step = None
        self.image_counter = 0
        self.iterations_counter = 0
        self.intialize_fast_worker()


    def intialize_fast_worker(self):
        self.model = FastGraph(self.width, self.height, self.alpha, self.beta, self.gamma)
        self.prepare_content_list()
        self.prepare_style_list()
        self.prepare_mix_image()
        self.prepare_session()


    def prepare_content_list(self):
        for path in self.content_path_list:
            self.content_list.append(self.model.preprocess(path))


    def prepare_style_list(self):
        for path in self.style_path_list:
            self.style_list.append(self.model.preprocess(path))


    def prepare_mix_image(self):
        if not self.use_meta:
            noise = np.random.uniform(-20, 20, (1, self.height, self.width, 3)).astype('float32')
            self.mix_image = noise * self.noise_ratio
            for content in self.content_list:
                self.mix_image += content * ((1 - self.noise_ratio) / len(self.content_list))
        else:
            try:
                self.mix_image = model.preprocess('meta/meta.png')
            except:
                self.use_meta = False
                self.prepare_mix_image()


    def save_mix_image(self):
        self.response_queue.put('Constructing image %d...' % self.image_counter)
        mix_image = self.model.sess.run(self.model.inputs)
        imsave('out/%d.png' % self.image_counter, np.clip((mix_image + np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)))[0], 0, 255).astype('uint8'))
        if self.save_meta:
            imsave('meta/meta.png', np.clip((mix_image + np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)))[0], 0, 255).astype('uint8'))
        self.result_queue.put('out/%d.png' % self.image_counter)
        self.image_counter += 1


    def prepare_session(self):
        total_content_loss = 0
        self.model.sess.run(tf.global_variables_initializer())
        for content in self.content_list:
            self.model.sess.run(self.model.inputs.assign(content))
            total_content_loss += self.model.content_loss()
        #########
        total_style_loss = 0
        for style in self.style_list:
            self.model.sess.run(self.model.inputs.assign(style))
            total_style_loss += self.model.style_loss()
        #########
        self.model.sess.run(self.model.inputs.assign(self.mix_image))
        #########
        final_loss = self.alpha * total_content_loss + self.beta * total_style_loss
        self.model.sess.run(tf.global_variables_initializer())
        self.model.sess.run(self.model.inputs.assign(self.mix_image))
        self.save_mix_image()
        #########
        if self.use_lbfgs:
            self.model.sess.run(tf.global_variables_initializer())
            self.model.sess.run(self.model.inputs.assign(self.mix_image))
            self.train_step = tf.contrib.opt.ScipyOptimizerInterface(final_loss, options={'maxiter': self.max_iterations - 1})
        else:
            optimizer = tf.train.AdamOptimizer(2.0)
            self.train_step = optimizer.minimize(final_loss)
            self.model.sess.run(tf.global_variables_initializer())
            self.model.sess.run(self.model.inputs.assign(self.mix_image))


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
                self.model.sess.close()
                self.response_queue.put('stopped')
                print('Thread Stopped.')
                break
            #########
            if self.use_lbfgs:
                self.train_step.minimize(self.model.sess)
                self.save_mix_image()
            else:
                while True:
                    self.model.sess.run(self.train_step)
                    if self.iterations_counter % self.max_iterations == 0:
                        self.save_mix_image()
                        break
                    self.iterations_counter += 1

