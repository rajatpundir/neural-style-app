from fastgraph import FastGraph
from fastergraph import FasterGraph
import time
import numpy as np
import tensorflow as tf
from PIL import Image


class FastWorker:
    def __init__(self, result_queue, command_queue, response_queue, content_path_list, style_path_list, width, height, use_meta, save_meta, use_lbfgs, max_iterations, noise_ratio, alpha, beta, gamma):
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
        self.model = FasterGraph(self.width, self.height, self.alpha, self.beta, self.gamma)
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
            noise = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128.0
            self.mix_image = noise * self.noise_ratio
            for content in self.content_list:
                self.mix_image += content * ((1 - self.noise_ratio) / len(self.content_list))
        else:
            try:
                self.mix_image = self.model.preprocess('meta/meta.png')
            except:
                self.use_meta = False
                self.prepare_mix_image()


    def save_mix_image(self):
        self.response_queue.put('Constructing image %d...' % self.image_counter)
        mix_image = self.model.sess.run(self.model.inputs)
        mix_image = mix_image.reshape((self.model.height, self.model.width, 3))
        mix_image = mix_image[:, :, ::-1]
        mix_image += self.model.mean
        mix_image = np.clip(mix_image, 0, 255).astype('uint8')
        Image.fromarray(mix_image).save('out/%d.png' % self.image_counter,'PNG')
        if self.save_meta:
            Image.fromarray(mix_image).save('meta/meta.png','PNG')
        self.result_queue.put('out/%d.png' % self.image_counter)
        self.image_counter += 1


    def prepare_session(self):
        self.model.sess.run(tf.global_variables_initializer())
        total_content_loss = 0
        for content in self.content_list:
            self.model.sess.run(self.model.inputs.assign(content))
            total_content_loss += self.model.content_loss()
        #########
        total_style_loss = 0
        for style in self.style_list:
            self.model.sess.run(self.model.inputs.assign(style))
            total_style_loss += self.model.style_loss()
        #########
        total_variation_loss = 0
        total_variation_loss = self.model.variation_loss()
        self.model.sess.run(self.model.inputs.assign(self.mix_image))
        #########
        final_loss = total_content_loss + total_style_loss + total_variation_loss
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
                        time.sleep(.2)
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

