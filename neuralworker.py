from fastgraph import FastGraph
from fastergraph import FasterGraph
import tensorflow as tf
import time

class NeuralWorker:
    def __init__(self, result_queue, command_queue, response_queue, faster_content_list, faster_style_list, fast_content_list, fast_style_list, use_faster_graph, use_lbfgs, max_iterations, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta):
        self.result_queue = result_queue
        self.command_queue = command_queue
        self.response_queue = response_queue
        #################
        self.use_lbfgs = use_lbfgs
        self.max_iterations = max_iterations
        #################
        self.train_step = None
        self.iterations_counter = 0
        self.model = None
        if use_faster_graph:
            self.model = FasterGraph(faster_content_list, faster_style_list, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta)
        else:
            self.model = FastGraph(fast_content_list, fast_style_list, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta)
        self.initialize_neural_worker()

    def save_image(self):
        image_counter = self.model.image_counter
        self.response_queue.put('Constructing image %d...' % image_counter)
        self.model.save_mix_image()
        self.result_queue.put('out/%d.png' % image_counter)
        self.response_queue.put('Loss @ ' +  str(self.model.compute_final_loss()))

    def initialize_neural_worker(self):
        self.model.sess.run(tf.global_variables_initializer())
        self.model.sess.run(self.model.inputs.assign(self.model.mix_image))
        self.save_image()
        #################
        if self.use_lbfgs:
            self.model.sess.run(tf.global_variables_initializer())
            self.model.sess.run(self.model.inputs.assign(self.model.mix_image))
            self.train_step = tf.contrib.opt.ScipyOptimizerInterface(self.model.final_loss, method='L-BFGS-B', options={'maxiter': self.max_iterations - 1})
        else:
            optimizer = tf.train.AdamOptimizer(2.0)
            self.train_step = optimizer.minimize(self.model.final_loss)
            self.model.sess.run(tf.global_variables_initializer())
            self.model.sess.run(self.model.inputs.assign(self.model.mix_image))

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
                self.save_image()
            else:
                while True:
                    self.model.sess.run(self.train_step)
                    if self.iterations_counter % self.max_iterations == 0:
                        self.save_image()
                        break
                    self.iterations_counter += 1

