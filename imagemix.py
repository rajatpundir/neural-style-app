from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager

from outputscreen import OutputScreen
from configscreen import ConfigScreen
from contentscreen import ContentScreen
from stylescreen import StyleScreen
from fastgraphconfigscreen import FastGraphConfigScreen
from fastergraphconfigscreen import FasterGraphConfigScreen

from contentfast import FastContent
from stylefast import FastStyle
from contentfaster import FasterContent
from stylefaster import FasterStyle

from kivyqueue import KivyQueue
from neuralworker import NeuralWorker

from functools import partial
import threading
import os, glob, sys


class ImageMixController(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.content_path_list = []
        self.content_path_list_counter = -1
        self.style_path_list = []
        self.style_path_list_counter = -1
        self.output_path_list = []
        self.output_path_list_counter = -1
        self.latest_output = True
        self.fast_content_list = []
        self.fast_content_list_counter = -1
        self.fast_style_list = []
        self.fast_style_list_counter = -1
        self.faster_content_list = []
        self.faster_content_list_counter = -1
        self.faster_style_list = []
        self.faster_style_list_counter = -1
        self.state = None
        self.worker = None
        self.result_queue = KivyQueue(self.result_queue_callback)
        self.command_queue = KivyQueue(self.command_queue_callback)
        self.response_queue = KivyQueue(self.reponse_queue_callback)
        self.output_screen.train_button.bind(on_press = self.set_config_screen_on_train_button)
        self.cleanup()
        # prfloat("FINISHED:", sys._getframe().f_code.co_name)

    def clear(self):
        if self.worker is not None:
            self.command_queue.put('stop')
            self.output_screen.clear_button.text = 'Wait...'
            self.output_screen.logs.text = 'Please wait...'
        else:
            self.content_path_list = []
            self.content_path_list_counter = -1
            self.output_screen.content.source = ''
            self.style_path_list = []
            self.style_path_list_counter = -1
            self.output_screen.style.source = ''
            self.output_path_list = []
            self.output_path_list_counter = -1
            self.output_screen.output.source = ''
            self.latest_output = True
            self.fast_content_list = []
            self.fast_content_list_counter = -1
            self.fast_style_list = []
            self.fast_style_list_counter = -1
            self.faster_content_list = []
            self.faster_content_list_counter = -1
            self.faster_style_list = []
            self.faster_style_list_counter = -1
            self.state = None
            self.worker = None
            self.empty_all_queue()
            self.cleanup()
            self.output_screen.train_button.text = 'Train'
            self.output_screen.clear_button.text = 'Clear'
            self.output_screen.logs.text = 'Cleared...'

    def cleanup(self):
        if not os.path.exists('./meta'):
            os.makedirs('./meta')
        if not os.path.exists('./out'):
            os.makedirs('./out')
        if os.path.exists('./out'):
            filelist = glob.glob(os.path.join('./out', "*.png"))
            for f in filelist:
                os.remove(f)

    def empty_all_queue(self):
        while not self.result_queue.empty():
            self.result_queue.get()
        while not self.command_queue.empty():
            self.command_queue.get()
        while not self.response_queue.empty():
            self.response_queue.get()

    def set_image(self, filename, target):
        try:
            if target is 'content' and self.worker is None:
                self.output_screen.content.source = filename[0]
                if filename[0] not in self.content_path_list:
                    self.content_path_list.append(filename[0])
                    self.content_path_list_counter = len(self.content_path_list) - 1
                    self.fast_content_list.append(FastContent(filename[0]))
                    self.fast_content_list_counter = len(self.fast_content_list) - 1
                    self.fast_content_screen_update()
                    self.faster_content_list.append(FasterContent(filename[0]))
                    self.faster_content_list_counter = len(self.faster_content_list) - 1
                    self.faster_content_screen_update()
            elif target is 'style' and self.worker is None:
                self.output_screen.style.source = filename[0]
                if filename[0] not in self.style_path_list:
                    self.style_path_list.append(filename[0])
                    self.style_path_list_counter = len(self.style_path_list) - 1
                    self.fast_style_list.append(FastStyle(filename[0]))
                    self.fast_style_list_counter = len(self.fast_style_list) - 1
                    self.fast_style_screen_update()
                    self.faster_style_list.append(FasterStyle(filename[0]))
                    self.faster_style_list_counter = len(self.faster_style_list) - 1
                    self.faster_style_screen_update()
            elif target is 'output':
                self.output_screen.output.source = filename[0]
        except:
            pass

    def cycle_image(self, target):
        try:
            if target is 'content':
                if self.content_path_list_counter != -1:
                    self.content_path_list_counter += 1
                    if self.content_path_list_counter == len(self.content_path_list):
                        self.content_path_list_counter = 0
                    self.output_screen.content.source = self.content_path_list[self.content_path_list_counter]
            elif target is 'style':
                if self.style_path_list_counter != -1:
                    self.style_path_list_counter += 1
                    if self.style_path_list_counter == len(self.style_path_list):
                        self.style_path_list_counter = 0
                    self.output_screen.style.source = self.style_path_list[self.style_path_list_counter]
            elif target is 'output_forward':
                if self.output_path_list_counter  + 1 != len(self.output_path_list):
                    self.output_screen.output.source = self.output_path_list[self.output_path_list_counter + 1]
                    self.output_path_list_counter += 1
                    self.latest_output = False
                    if self.output_path_list_counter == len(self.output_path_list) - 1:
                        self.latest_output = True
            elif target is 'output_backward':
                if self.output_path_list_counter  > 0:
                    self.output_screen.output.source = self.output_path_list[self.output_path_list_counter - 1]
                    self.output_path_list_counter -= 1
                    self.latest_output = False
                    if self.output_path_list_counter == len(self.output_path_list) - 1:
                        self.latest_output = True
            elif target is 'output_latest':
                if self.output_screen.output.source is not self.output_path_list[-1]:
                    self.output_screen.output.source = self.output_path_list[-1]
                    self.output_path_list_counter = len(self.output_path_list) - 1
                else:
                    self.output_screen.output.source = self.output_screen.content.source
                    self.output_path_list_counter = -1
                self.latest_output = True
        except:
            pass

    def result_queue_callback_logic(self, dt):
        if not self.result_queue.empty():
            path = str(self.result_queue.get())
            self.output_path_list.append(path)
            if self.latest_output:
                self.cycle_image('output_latest')

    def result_queue_callback(self):
        # Trigger created can be called wherever, not necessary immediately.
        # Maybe a good way to schedule things as even main thread may be frozen.
        event = Clock.create_trigger(self.result_queue_callback_logic)
        event()

    def command_queue_callback_logic(self, dt):
        # self.command_queue.put(command)
        pass

    def command_queue_callback(self):
        # event = Clock.create_trigger(self.command_queue_callback_logic)
        # event()
        pass

    def reponse_queue_callback_logic(self, dt):
        if not self.response_queue.empty():
            response = str(self.response_queue.get())
            if response is 'paused':
                self.state = 'paused'
                self.output_screen.train_button.text = 'Resume'
                self.output_screen.logs.text = 'Training has been paused...'
            elif response is 'resumed':
                self.state = 'resumed'
                self.output_screen.train_button.text = 'Pause'
                self.output_screen.logs.text = 'Resuming to train...'
            elif response is 'stopped':
                self.content_path_list = []
                self.content_path_list_counter = -1
                self.output_screen.content.source = ''
                self.style_path_list = []
                self.style_path_list_counter = -1
                self.output_screen.style.source = ''
                self.output_path_list = []
                self.output_path_list_counter = -1
                self.output_screen.output.source = ''
                self.latest_output = True
                self.fast_content_list = []
                self.fast_content_list_counter = -1
                self.fast_style_list = []
                self.fast_style_list_counter = -1
                self.faster_content_list = []
                self.faster_content_list_counter = -1
                self.faster_style_list = []
                self.faster_style_list_counter = -1
                self.state = None
                self.worker = None
                self.empty_all_queue()
                self.cleanup()
                self.output_screen.train_button.text = 'Train'
                self.output_screen.clear_button.text = 'Clear'
                self.output_screen.logs.text = 'Cleared...'
                #########
                self.output_screen.train_button.unbind(on_press = self.pause_or_resume_button)
                self.output_screen.train_button.bind(on_press = self.set_config_screen_on_train_button)
            else:
                self.output_screen.logs.text = response

    def reponse_queue_callback(self):
        event = Clock.create_trigger(self.reponse_queue_callback_logic)
        event()

    def train_worker(self, use_faster_graph, use_lbfgs, max_iterations, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta):
        # Threads share data.
        worker = NeuralWorker(self.result_queue, self.command_queue, self.response_queue, self.faster_content_list, self.faster_style_list, self.fast_content_list, self.fast_style_list, use_faster_graph, use_lbfgs, max_iterations, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta)
        worker.train()

    def pause_or_resume_button(self, *args):
        if self.worker is not None:
            if self.state is 'resumed':
                self.command_queue.put('pause')
                self.state = 'waiting'
                self.output_screen.train_button.text = 'Wait...'
            elif self.state is 'paused':
                self.command_queue.put('resume')
                self.state = 'waiting'
                self.output_screen.train_button.text = 'Wait...'
            elif self.state is 'waiting':
                self.output_screen.train_button.text = 'Please wait...?'
            self.output_screen.logs.text = 'Please wait while the background thread is busy...'

    def set_config_screen_on_train_button(self, *args):
        if len(self.content_path_list) == 0 or len(self.style_path_list) == 0:
            return
        self.current = 'config_screen' 

    def start_button(self, width, height, use_meta, save_meta, use_lbfgs, max_iterations, noise_ratio, alpha, beta, gamma, use_faster_graph):
        try:
            use_faster_graph = bool(use_faster_graph)
            use_lbfgs = bool(use_lbfgs)
            max_iterations = int(max_iterations)
            width = int(width)
            height = int(height)
            alpha = float(alpha)
            beta = float(beta)
            gamma = float(gamma)
            noise_ratio = float(noise_ratio)
            use_meta = bool(use_meta)
            save_meta = bool(save_meta)
            if self.worker is None:
                if len(self.content_path_list) == 0 or len(self.style_path_list) == 0:
                    return
                self.worker = threading.Thread(target = self.train_worker, args=(use_faster_graph, use_lbfgs, max_iterations, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta))
                self.worker.daemon = True
                self.worker.start()
                self.state = 'resumed'
                self.output_screen.logs.text = 'Starting to train...'
                self.output_screen.train_button.text = 'Pause'
                self.output_screen.train_button.unbind(on_press = self.set_config_screen_on_train_button)
                self.output_screen.train_button.bind(on_press = self.pause_or_resume_button)
        except:
            pass

    def set_config_screen(self):
        if self.config_screen.use_faster_graph.active:
            self.current = 'faster_graph_config_screen'
        else:
            self.current = 'fast_graph_config_screen'

    def cycle_config(self, target, *args):
        try:
            if target is 'faster_content_next':
                if self.faster_content_list_counter != -1:
                    self.faster_content_save(*args)
                    self.faster_content_list_counter += 1
                    if self.faster_content_list_counter == len(self.faster_content_list):
                        self.faster_content_list_counter = 0
                    self.faster_content_screen_update()
            elif target is 'faster_content_prev':
                if self.faster_content_list_counter != -1:
                    self.faster_content_save(*args)
                    self.faster_content_list_counter -= 1
                    if self.faster_content_list_counter == -1:
                        self.faster_content_list_counter = len(self.faster_content_list) - 1
                    self.faster_content_screen_update()
            elif target is 'faster_style_next':
                if self.faster_style_list_counter != -1:
                    self.faster_style_save(*args)
                    self.faster_style_list_counter += 1
                    if self.faster_style_list_counter == len(self.faster_style_list):
                        self.faster_style_list_counter = 0
                    self.faster_style_screen_update()
            elif target is 'faster_style_prev':
                if self.faster_style_list_counter != -1:
                    self.faster_style_save(*args)
                    self.faster_style_list_counter -= 1
                    if self.faster_style_list_counter == -1:
                        self.faster_style_list_counter = len(self.faster_style_list) - 1
                    self.faster_style_screen_update()
            elif target is 'fast_content_next':
                if self.fast_content_list_counter != -1:
                    self.fast_content_save(*args)
                    self.fast_content_list_counter += 1
                    if self.fast_content_list_counter == len(self.fast_content_list):
                        self.fast_content_list_counter = 0
                    self.fast_content_screen_update()
            elif target is 'fast_content_prev':
                if self.fast_content_list_counter != -1:
                    self.fast_content_save(*args)
                    self.fast_content_list_counter -= 1
                    if self.fast_content_list_counter == -1:
                        self.fast_content_list_counter = len(self.fast_content_list) - 1
                    self.fast_content_screen_update()
            elif target is 'fast_style_next':
                if self.fast_style_list_counter != -1:
                    self.fast_style_save(*args)
                    self.fast_style_list_counter += 1
                    if self.fast_style_list_counter == len(self.fast_style_list):
                        self.fast_style_list_counter = 0
                    self.fast_style_screen_update()
            elif target is 'fast_style_prev':
                if self.fast_style_list_counter != -1:
                    self.fast_style_save(*args)
                    self.fast_style_list_counter -= 1
                    if self.fast_style_list_counter == -1:
                        self.fast_style_list_counter = len(self.fast_style_list) - 1
                    self.fast_style_screen_update()
        except:
            pass

    def accept_faster_config_button(self, alpha, content_conv1_1_check, content_conv1_1_weight, content_conv1_2_check, content_conv1_2_weight, content_pool1_check, content_pool1_weight, content_conv2_1_check, content_conv2_1_weight, content_conv2_2_check, content_conv2_2_weight, content_pool2_check, content_pool2_weight, content_conv3_1_check, content_conv3_1_weight, content_conv3_2_check, content_conv3_2_weight, content_conv3_3_check, content_conv3_3_weight, content_pool3_check, content_pool3_weight, content_conv4_1_check, content_conv4_1_weight, content_conv4_2_check, content_conv4_2_weight, content_conv4_3_check, content_conv4_3_weight, content_pool4_check, content_pool4_weight, content_conv5_1_check, content_conv5_1_weight, content_conv5_2_check, content_conv5_2_weight, content_conv5_3_check, content_conv5_3_weight, content_pool5_check, content_pool5_weight, beta, style_conv1_1_check, style_conv1_1_weight, style_conv1_2_check, style_conv1_2_weight, style_pool1_check, style_pool1_weight, style_conv2_1_check, style_conv2_1_weight, style_conv2_2_check, style_conv2_2_weight, style_pool2_check, style_pool2_weight, style_conv3_1_check, style_conv3_1_weight, style_conv3_2_check, style_conv3_2_weight, style_conv3_3_check, style_conv3_3_weight, style_pool3_check, style_pool3_weight, style_conv4_1_check, style_conv4_1_weight, style_conv4_2_check, style_conv4_2_weight, style_conv4_3_check, style_conv4_3_weight, style_pool4_check, style_pool4_weight, style_conv5_1_check, style_conv5_1_weight, style_conv5_2_check, style_conv5_2_weight, style_conv5_3_check, style_conv5_3_weight, style_pool5_check, style_pool5_weight):
        self.faster_content_save(alpha, content_conv1_1_check, content_conv1_1_weight, content_conv1_2_check, content_conv1_2_weight, content_pool1_check, content_pool1_weight, content_conv2_1_check, content_conv2_1_weight, content_conv2_2_check, content_conv2_2_weight, content_pool2_check, content_pool2_weight, content_conv3_1_check, content_conv3_1_weight, content_conv3_2_check, content_conv3_2_weight, content_conv3_3_check, content_conv3_3_weight, content_pool3_check, content_pool3_weight, content_conv4_1_check, content_conv4_1_weight, content_conv4_2_check, content_conv4_2_weight, content_conv4_3_check, content_conv4_3_weight, content_pool4_check, content_pool4_weight, content_conv5_1_check, content_conv5_1_weight, content_conv5_2_check, content_conv5_2_weight, content_conv5_3_check, content_conv5_3_weight, content_pool5_check, content_pool5_weight)
        self.faster_style_save(beta, style_conv1_1_check, style_conv1_1_weight, style_conv1_2_check, style_conv1_2_weight, style_pool1_check, style_pool1_weight, style_conv2_1_check, style_conv2_1_weight, style_conv2_2_check, style_conv2_2_weight, style_pool2_check, style_pool2_weight, style_conv3_1_check, style_conv3_1_weight, style_conv3_2_check, style_conv3_2_weight, style_conv3_3_check, style_conv3_3_weight, style_pool3_check, style_pool3_weight, style_conv4_1_check, style_conv4_1_weight, style_conv4_2_check, style_conv4_2_weight, style_conv4_3_check, style_conv4_3_weight, style_pool4_check, style_pool4_weight, style_conv5_1_check, style_conv5_1_weight, style_conv5_2_check, style_conv5_2_weight, style_conv5_3_check, style_conv5_3_weight, style_pool5_check, style_pool5_weight)

    def accept_fast_config_button(self, alpha, content_conv1_1_check, content_conv1_1_weight, content_conv1_2_check, content_conv1_2_weight, content_pool1_check, content_pool1_weight, content_conv2_1_check, content_conv2_1_weight, content_conv2_2_check, content_conv2_2_weight, content_pool2_check, content_pool2_weight, content_conv3_1_check, content_conv3_1_weight, content_conv3_2_check, content_conv3_2_weight, content_conv3_3_check, content_conv3_3_weight, content_conv3_4_check, content_conv3_4_weight, content_pool3_check, content_pool3_weight, content_conv4_1_check, content_conv4_1_weight, content_conv4_2_check, content_conv4_2_weight, content_conv4_3_check, content_conv4_3_weight, content_conv4_4_check, content_conv4_4_weight, content_pool4_check, content_pool4_weight, content_conv5_1_check, content_conv5_1_weight, content_conv5_2_check, content_conv5_2_weight, content_conv5_3_check, content_conv5_3_weight, content_conv5_4_check, content_conv5_4_weight, content_pool5_check, content_pool5_weight, beta, style_conv1_1_check, style_conv1_1_weight, style_conv1_2_check, style_conv1_2_weight, style_pool1_check, style_pool1_weight, style_conv2_1_check, style_conv2_1_weight, style_conv2_2_check, style_conv2_2_weight, style_pool2_check, style_pool2_weight, style_conv3_1_check, style_conv3_1_weight, style_conv3_2_check, style_conv3_2_weight, style_conv3_3_check, style_conv3_3_weight, style_conv3_4_check, style_conv3_4_weight, style_pool3_check, style_pool3_weight, style_conv4_1_check, style_conv4_1_weight, style_conv4_2_check, style_conv4_2_weight, style_conv4_3_check, style_conv4_3_weight, style_conv4_4_check, style_conv4_4_weight, style_pool4_check, style_pool4_weight, style_conv5_1_check, style_conv5_1_weight, style_conv5_2_check, style_conv5_2_weight, style_conv5_3_check, style_conv5_3_weight, style_conv5_4_check, style_conv5_4_weight, style_pool5_check, style_pool5_weight):
        self.fast_content_save(alpha, content_conv1_1_check, content_conv1_1_weight, content_conv1_2_check, content_conv1_2_weight, content_pool1_check, content_pool1_weight, content_conv2_1_check, content_conv2_1_weight, content_conv2_2_check, content_conv2_2_weight, content_pool2_check, content_pool2_weight, content_conv3_1_check, content_conv3_1_weight, content_conv3_2_check, content_conv3_2_weight, content_conv3_3_check, content_conv3_3_weight, content_conv3_4_check, content_conv3_4_weight, content_pool3_check, content_pool3_weight, content_conv4_1_check, content_conv4_1_weight, content_conv4_2_check, content_conv4_2_weight, content_conv4_3_check, content_conv4_3_weight, content_conv4_4_check, content_conv4_4_weight, content_pool4_check, content_pool4_weight, content_conv5_1_check, content_conv5_1_weight, content_conv5_2_check, content_conv5_2_weight, content_conv5_3_check, content_conv5_3_weight, content_conv5_4_check, content_conv5_4_weight, content_pool5_check, content_pool5_weight)
        self.fast_style_save(beta, style_conv1_1_check, style_conv1_1_weight, style_conv1_2_check, style_conv1_2_weight, style_pool1_check, style_pool1_weight, style_conv2_1_check, style_conv2_1_weight, style_conv2_2_check, style_conv2_2_weight, style_pool2_check, style_pool2_weight, style_conv3_1_check, style_conv3_1_weight, style_conv3_2_check, style_conv3_2_weight, style_conv3_3_check, style_conv3_3_weight, style_conv3_4_check, style_conv3_4_weight, style_pool3_check, style_pool3_weight, style_conv4_1_check, style_conv4_1_weight, style_conv4_2_check, style_conv4_2_weight, style_conv4_3_check, style_conv4_3_weight, style_conv4_4_check, style_conv4_4_weight, style_pool4_check, style_pool4_weight, style_conv5_1_check, style_conv5_1_weight, style_conv5_2_check, style_conv5_2_weight, style_conv5_3_check, style_conv5_3_weight, style_conv5_4_check, style_conv5_4_weight, style_pool5_check, style_pool5_weight)

    def faster_content_screen_update(self):
        current_faster_content = self.faster_content_list[self.faster_content_list_counter]
        self.faster_graph_config_screen.content_file.text = current_faster_content.path.split('\\')[-1]
        self.faster_graph_config_screen.alpha.text = str(current_faster_content.alpha)
        self.faster_graph_config_screen.content_conv1_1_check.active = current_faster_content.content_conv1_1_check
        self.faster_graph_config_screen.content_conv1_1_weight.text = str(current_faster_content.content_conv1_1_weight)
        self.faster_graph_config_screen.content_conv1_2_check.active = current_faster_content.content_conv1_2_check
        self.faster_graph_config_screen.content_conv1_2_weight.text = str(current_faster_content.content_conv1_2_weight)
        self.faster_graph_config_screen.content_pool1_check.active = current_faster_content.content_pool1_check
        self.faster_graph_config_screen.content_pool1_weight.text = str(current_faster_content.content_pool1_weight)
        self.faster_graph_config_screen.content_conv2_1_check.active = current_faster_content.content_conv2_1_check
        self.faster_graph_config_screen.content_conv2_1_weight.text = str(current_faster_content.content_conv2_1_weight)
        self.faster_graph_config_screen.content_conv2_2_check.active = current_faster_content.content_conv2_2_check
        self.faster_graph_config_screen.content_conv2_2_weight.text = str(current_faster_content.content_conv2_2_weight)
        self.faster_graph_config_screen.content_pool2_check.active = current_faster_content.content_pool2_check
        self.faster_graph_config_screen.content_pool2_weight.text = str(current_faster_content.content_pool2_weight)
        self.faster_graph_config_screen.content_conv3_1_check.active = current_faster_content.content_conv3_1_check
        self.faster_graph_config_screen.content_conv3_1_weight.text = str(current_faster_content.content_conv3_1_weight)
        self.faster_graph_config_screen.content_conv3_2_check.active = current_faster_content.content_conv3_2_check
        self.faster_graph_config_screen.content_conv3_2_weight.text = str(current_faster_content.content_conv3_2_weight)
        self.faster_graph_config_screen.content_conv3_3_check.active = current_faster_content.content_conv3_3_check
        self.faster_graph_config_screen.content_conv3_3_weight.text = str(current_faster_content.content_conv3_3_weight)
        self.faster_graph_config_screen.content_pool3_check.active = current_faster_content.content_pool3_check
        self.faster_graph_config_screen.content_pool3_weight.text = str(current_faster_content.content_pool3_weight)
        self.faster_graph_config_screen.content_conv4_1_check.active = current_faster_content.content_conv4_1_check
        self.faster_graph_config_screen.content_conv4_1_weight.text = str(current_faster_content.content_conv4_1_weight)
        self.faster_graph_config_screen.content_conv4_2_check.active = current_faster_content.content_conv4_2_check
        self.faster_graph_config_screen.content_conv4_2_weight.text = str(current_faster_content.content_conv4_2_weight)
        self.faster_graph_config_screen.content_conv4_3_check.active = current_faster_content.content_conv4_3_check
        self.faster_graph_config_screen.content_conv4_3_weight.text = str(current_faster_content.content_conv4_3_weight)
        self.faster_graph_config_screen.content_pool4_check.active = current_faster_content.content_pool4_check
        self.faster_graph_config_screen.content_pool4_weight.text = str(current_faster_content.content_pool4_weight)
        self.faster_graph_config_screen.content_conv5_1_check.active = current_faster_content.content_conv5_1_check
        self.faster_graph_config_screen.content_conv5_1_weight.text = str(current_faster_content.content_conv5_1_weight)
        self.faster_graph_config_screen.content_conv5_2_check.active = current_faster_content.content_conv5_2_check
        self.faster_graph_config_screen.content_conv5_2_weight.text = str(current_faster_content.content_conv5_2_weight)
        self.faster_graph_config_screen.content_conv5_3_check.active = current_faster_content.content_conv5_3_check
        self.faster_graph_config_screen.content_conv5_3_weight.text = str(current_faster_content.content_conv5_3_weight)
        self.faster_graph_config_screen.content_pool5_check.active = current_faster_content.content_pool5_check
        self.faster_graph_config_screen.content_pool5_weight.text = str(current_faster_content.content_pool5_weight)

    def faster_style_screen_update(self):
        current_faster_style = self.faster_style_list[self.faster_style_list_counter]
        self.faster_graph_config_screen.style_file.text = current_faster_style.path.split('\\')[-1]
        self.faster_graph_config_screen.beta.text = str(current_faster_style.beta)
        self.faster_graph_config_screen.style_conv1_1_check.active = current_faster_style.style_conv1_1_check
        self.faster_graph_config_screen.style_conv1_1_weight.text = str(current_faster_style.style_conv1_1_weight)
        self.faster_graph_config_screen.style_conv1_2_check.active = current_faster_style.style_conv1_2_check
        self.faster_graph_config_screen.style_conv1_2_weight.text = str(current_faster_style.style_conv1_2_weight)
        self.faster_graph_config_screen.style_pool1_check.active = current_faster_style.style_pool1_check
        self.faster_graph_config_screen.style_pool1_weight.text = str(current_faster_style.style_pool1_weight)
        self.faster_graph_config_screen.style_conv2_1_check.active = current_faster_style.style_conv2_1_check
        self.faster_graph_config_screen.style_conv2_1_weight.text = str(current_faster_style.style_conv2_1_weight)
        self.faster_graph_config_screen.style_conv2_2_check.active = current_faster_style.style_conv2_2_check
        self.faster_graph_config_screen.style_conv2_2_weight.text = str(current_faster_style.style_conv2_2_weight)
        self.faster_graph_config_screen.style_pool2_check.active = current_faster_style.style_pool2_check
        self.faster_graph_config_screen.style_pool2_weight.text = str(current_faster_style.style_pool2_weight)
        self.faster_graph_config_screen.style_conv3_1_check.active = current_faster_style.style_conv3_1_check
        self.faster_graph_config_screen.style_conv3_1_weight.text = str(current_faster_style.style_conv3_1_weight)
        self.faster_graph_config_screen.style_conv3_2_check.active = current_faster_style.style_conv3_2_check
        self.faster_graph_config_screen.style_conv3_2_weight.text = str(current_faster_style.style_conv3_2_weight)
        self.faster_graph_config_screen.style_conv3_3_check.active = current_faster_style.style_conv3_3_check
        self.faster_graph_config_screen.style_conv3_3_weight.text = str(current_faster_style.style_conv3_3_weight)
        self.faster_graph_config_screen.style_pool3_check.active = current_faster_style.style_pool3_check
        self.faster_graph_config_screen.style_pool3_weight.text = str(current_faster_style.style_pool3_weight)
        self.faster_graph_config_screen.style_conv4_1_check.active = current_faster_style.style_conv4_1_check
        self.faster_graph_config_screen.style_conv4_1_weight.text = str(current_faster_style.style_conv4_1_weight)
        self.faster_graph_config_screen.style_conv4_2_check.active = current_faster_style.style_conv4_2_check
        self.faster_graph_config_screen.style_conv4_2_weight.text = str(current_faster_style.style_conv4_2_weight)
        self.faster_graph_config_screen.style_conv4_3_check.active = current_faster_style.style_conv4_3_check
        self.faster_graph_config_screen.style_conv4_3_weight.text = str(current_faster_style.style_conv4_3_weight)
        self.faster_graph_config_screen.style_pool4_check.active = current_faster_style.style_pool4_check
        self.faster_graph_config_screen.style_pool4_weight.text = str(current_faster_style.style_pool4_weight)
        self.faster_graph_config_screen.style_conv5_1_check.active = current_faster_style.style_conv5_1_check
        self.faster_graph_config_screen.style_conv5_1_weight.text = str(current_faster_style.style_conv5_1_weight)
        self.faster_graph_config_screen.style_conv5_2_check.active = current_faster_style.style_conv5_2_check
        self.faster_graph_config_screen.style_conv5_2_weight.text = str(current_faster_style.style_conv5_2_weight)
        self.faster_graph_config_screen.style_conv5_3_check.active = current_faster_style.style_conv5_3_check
        self.faster_graph_config_screen.style_conv5_3_weight.text = str(current_faster_style.style_conv5_3_weight)
        self.faster_graph_config_screen.style_pool5_check.active = current_faster_style.style_pool5_check
        self.faster_graph_config_screen.style_pool5_weight.text = str(current_faster_style.style_pool5_weight)

    def fast_content_screen_update(self):
        current_fast_content = self.fast_content_list[self.fast_content_list_counter]
        self.fast_graph_config_screen.content_file.text = current_fast_content.path.split('\\')[-1]
        self.fast_graph_config_screen.alpha.text = str(current_fast_content.alpha)
        self.fast_graph_config_screen.content_conv1_1_check.active = current_fast_content.content_conv1_1_check
        self.fast_graph_config_screen.content_conv1_1_weight.text = str(current_fast_content.content_conv1_1_weight)
        self.fast_graph_config_screen.content_conv1_2_check.active = current_fast_content.content_conv1_2_check
        self.fast_graph_config_screen.content_conv1_2_weight.text = str(current_fast_content.content_conv1_2_weight)
        self.fast_graph_config_screen.content_pool1_check.active = current_fast_content.content_pool1_check
        self.fast_graph_config_screen.content_pool1_weight.text = str(current_fast_content.content_pool1_weight)
        self.fast_graph_config_screen.content_conv2_1_check.active = current_fast_content.content_conv2_1_check
        self.fast_graph_config_screen.content_conv2_1_weight.text = str(current_fast_content.content_conv2_1_weight)
        self.fast_graph_config_screen.content_conv2_2_check.active = current_fast_content.content_conv2_2_check
        self.fast_graph_config_screen.content_conv2_2_weight.text = str(current_fast_content.content_conv2_2_weight)
        self.fast_graph_config_screen.content_pool2_check.active = current_fast_content.content_pool2_check
        self.fast_graph_config_screen.content_pool2_weight.text = str(current_fast_content.content_pool2_weight)
        self.fast_graph_config_screen.content_conv3_1_check.active = current_fast_content.content_conv3_1_check
        self.fast_graph_config_screen.content_conv3_1_weight.text = str(current_fast_content.content_conv3_1_weight)
        self.fast_graph_config_screen.content_conv3_2_check.active = current_fast_content.content_conv3_2_check
        self.fast_graph_config_screen.content_conv3_2_weight.text = str(current_fast_content.content_conv3_2_weight)
        self.fast_graph_config_screen.content_conv3_3_check.active = current_fast_content.content_conv3_3_check
        self.fast_graph_config_screen.content_conv3_3_weight.text = str(current_fast_content.content_conv3_3_weight)
        self.fast_graph_config_screen.content_conv3_4_check.active = current_fast_content.content_conv3_4_check
        self.fast_graph_config_screen.content_conv3_4_weight.text = str(current_fast_content.content_conv3_4_weight)
        self.fast_graph_config_screen.content_pool3_check.active = current_fast_content.content_pool3_check
        self.fast_graph_config_screen.content_pool3_weight.text = str(current_fast_content.content_pool3_weight)
        self.fast_graph_config_screen.content_conv4_1_check.active = current_fast_content.content_conv4_1_check
        self.fast_graph_config_screen.content_conv4_1_weight.text = str(current_fast_content.content_conv4_1_weight)
        self.fast_graph_config_screen.content_conv4_2_check.active = current_fast_content.content_conv4_2_check
        self.fast_graph_config_screen.content_conv4_2_weight.text = str(current_fast_content.content_conv4_2_weight)
        self.fast_graph_config_screen.content_conv4_3_check.active = current_fast_content.content_conv4_3_check
        self.fast_graph_config_screen.content_conv4_3_weight.text = str(current_fast_content.content_conv4_3_weight)
        self.fast_graph_config_screen.content_conv4_4_check.active = current_fast_content.content_conv4_4_check
        self.fast_graph_config_screen.content_conv4_4_weight.text = str(current_fast_content.content_conv4_4_weight)
        self.fast_graph_config_screen.content_pool4_check.active = current_fast_content.content_pool4_check
        self.fast_graph_config_screen.content_pool4_weight.text = str(current_fast_content.content_pool4_weight)
        self.fast_graph_config_screen.content_conv5_1_check.active = current_fast_content.content_conv5_1_check
        self.fast_graph_config_screen.content_conv5_1_weight.text = str(current_fast_content.content_conv5_1_weight)
        self.fast_graph_config_screen.content_conv5_2_check.active = current_fast_content.content_conv5_2_check
        self.fast_graph_config_screen.content_conv5_2_weight.text = str(current_fast_content.content_conv5_2_weight)
        self.fast_graph_config_screen.content_conv5_3_check.active = current_fast_content.content_conv5_3_check
        self.fast_graph_config_screen.content_conv5_3_weight.text = str(current_fast_content.content_conv5_3_weight)
        self.fast_graph_config_screen.content_conv5_4_check.active = current_fast_content.content_conv5_4_check
        self.fast_graph_config_screen.content_conv5_4_weight.text = str(current_fast_content.content_conv5_4_weight)
        self.fast_graph_config_screen.content_pool5_check.active = current_fast_content.content_pool5_check
        self.fast_graph_config_screen.content_pool5_weight.text = str(current_fast_content.content_pool5_weight)

    def fast_style_screen_update(self):
        current_fast_style = self.fast_style_list[self.fast_style_list_counter]
        self.fast_graph_config_screen.style_file.text = current_fast_style.path.split('\\')[-1]
        self.fast_graph_config_screen.beta.text = str(current_fast_style.beta)
        self.fast_graph_config_screen.style_conv1_1_check.active = current_fast_style.style_conv1_1_check
        self.fast_graph_config_screen.style_conv1_1_weight.text = str(current_fast_style.style_conv1_1_weight)
        self.fast_graph_config_screen.style_conv1_2_check.active = current_fast_style.style_conv1_2_check
        self.fast_graph_config_screen.style_conv1_2_weight.text = str(current_fast_style.style_conv1_2_weight)
        self.fast_graph_config_screen.style_pool1_check.active = current_fast_style.style_pool1_check
        self.fast_graph_config_screen.style_pool1_weight.text = str(current_fast_style.style_pool1_weight)
        self.fast_graph_config_screen.style_conv2_1_check.active = current_fast_style.style_conv2_1_check
        self.fast_graph_config_screen.style_conv2_1_weight.text = str(current_fast_style.style_conv2_1_weight)
        self.fast_graph_config_screen.style_conv2_2_check.active = current_fast_style.style_conv2_2_check
        self.fast_graph_config_screen.style_conv2_2_weight.text = str(current_fast_style.style_conv2_2_weight)
        self.fast_graph_config_screen.style_pool2_check.active = current_fast_style.style_pool2_check
        self.fast_graph_config_screen.style_pool2_weight.text = str(current_fast_style.style_pool2_weight)
        self.fast_graph_config_screen.style_conv3_1_check.active = current_fast_style.style_conv3_1_check
        self.fast_graph_config_screen.style_conv3_1_weight.text = str(current_fast_style.style_conv3_1_weight)
        self.fast_graph_config_screen.style_conv3_2_check.active = current_fast_style.style_conv3_2_check
        self.fast_graph_config_screen.style_conv3_2_weight.text = str(current_fast_style.style_conv3_2_weight)
        self.fast_graph_config_screen.style_conv3_3_check.active = current_fast_style.style_conv3_3_check
        self.fast_graph_config_screen.style_conv3_3_weight.text = str(current_fast_style.style_conv3_3_weight)
        self.fast_graph_config_screen.style_conv3_4_check.active = current_fast_style.style_conv3_4_check
        self.fast_graph_config_screen.style_conv3_4_weight.text = str(current_fast_style.style_conv3_4_weight)
        self.fast_graph_config_screen.style_pool3_check.active = current_fast_style.style_pool3_check
        self.fast_graph_config_screen.style_pool3_weight.text = str(current_fast_style.style_pool3_weight)
        self.fast_graph_config_screen.style_conv4_1_check.active = current_fast_style.style_conv4_1_check
        self.fast_graph_config_screen.style_conv4_1_weight.text = str(current_fast_style.style_conv4_1_weight)
        self.fast_graph_config_screen.style_conv4_2_check.active = current_fast_style.style_conv4_2_check
        self.fast_graph_config_screen.style_conv4_2_weight.text = str(current_fast_style.style_conv4_2_weight)
        self.fast_graph_config_screen.style_conv4_3_check.active = current_fast_style.style_conv4_3_check
        self.fast_graph_config_screen.style_conv4_3_weight.text = str(current_fast_style.style_conv4_3_weight)
        self.fast_graph_config_screen.style_conv4_4_check.active = current_fast_style.style_conv4_4_check
        self.fast_graph_config_screen.style_conv4_4_weight.text = str(current_fast_style.style_conv4_4_weight)
        self.fast_graph_config_screen.style_pool4_check.active = current_fast_style.style_pool4_check
        self.fast_graph_config_screen.style_pool4_weight.text = str(current_fast_style.style_pool4_weight)
        self.fast_graph_config_screen.style_conv5_1_check.active = current_fast_style.style_conv5_1_check
        self.fast_graph_config_screen.style_conv5_1_weight.text = str(current_fast_style.style_conv5_1_weight)
        self.fast_graph_config_screen.style_conv5_2_check.active = current_fast_style.style_conv5_2_check
        self.fast_graph_config_screen.style_conv5_2_weight.text = str(current_fast_style.style_conv5_2_weight)
        self.fast_graph_config_screen.style_conv5_3_check.active = current_fast_style.style_conv5_3_check
        self.fast_graph_config_screen.style_conv5_3_weight.text = str(current_fast_style.style_conv5_3_weight)
        self.fast_graph_config_screen.style_conv5_4_check.active = current_fast_style.style_conv5_4_check
        self.fast_graph_config_screen.style_conv5_4_weight.text = str(current_fast_style.style_conv5_4_weight)
        self.fast_graph_config_screen.style_pool5_check.active = current_fast_style.style_pool5_check
        self.fast_graph_config_screen.style_pool5_weight.text = str(current_fast_style.style_pool5_weight)

    def faster_content_save(self, alpha, content_conv1_1_check, content_conv1_1_weight, content_conv1_2_check, content_conv1_2_weight, content_pool1_check, content_pool1_weight, content_conv2_1_check, content_conv2_1_weight, content_conv2_2_check, content_conv2_2_weight, content_pool2_check, content_pool2_weight, content_conv3_1_check, content_conv3_1_weight, content_conv3_2_check, content_conv3_2_weight, content_conv3_3_check, content_conv3_3_weight, content_pool3_check, content_pool3_weight, content_conv4_1_check, content_conv4_1_weight, content_conv4_2_check, content_conv4_2_weight, content_conv4_3_check, content_conv4_3_weight, content_pool4_check, content_pool4_weight, content_conv5_1_check, content_conv5_1_weight, content_conv5_2_check, content_conv5_2_weight, content_conv5_3_check, content_conv5_3_weight, content_pool5_check, content_pool5_weight):
        try:
            alpha = float(alpha)
            content_conv1_1_check = bool(content_conv1_1_check)
            content_conv1_1_weight = float(content_conv1_1_weight)
            content_conv1_2_check = bool(content_conv1_2_check)
            content_conv1_2_weight = float(content_conv1_2_weight)
            content_pool1_check = bool(content_pool1_check)
            content_pool1_weight = float(content_pool1_weight)
            content_conv2_1_check = bool(content_conv2_1_check)
            content_conv2_1_weight = float(content_conv2_1_weight)
            content_conv2_2_check = bool(content_conv2_2_check)
            content_conv2_2_weight = float(content_conv2_2_weight)
            content_pool2_check = bool(content_pool2_check)
            content_pool2_weight = float(content_pool2_weight)
            content_conv3_1_check = bool(content_conv3_1_check)
            content_conv3_1_weight = float(content_conv3_1_weight)
            content_conv3_2_check = bool(content_conv3_2_check)
            content_conv3_2_weight = float(content_conv3_2_weight)
            content_conv3_3_check = bool(content_conv3_3_check)
            content_conv3_3_weight = float(content_conv3_3_weight)
            content_pool3_check = bool(content_pool3_check)
            content_pool3_weight = float(content_pool3_weight)
            content_conv4_1_check = bool(content_conv4_1_check)
            content_conv4_1_weight = float(content_conv4_1_weight)
            content_conv4_2_check = bool(content_conv4_2_check)
            content_conv4_2_weight = float(content_conv4_2_weight)
            content_conv4_3_check = bool(content_conv4_3_check)
            content_conv4_3_weight = float(content_conv4_3_weight)
            content_pool4_check = bool(content_pool4_check)
            content_pool4_weight = float(content_pool4_weight)
            content_conv5_1_check = bool(content_conv5_1_check)
            content_conv5_1_weight = float(content_conv5_1_weight)
            content_conv5_2_check = bool(content_conv5_2_check)
            content_conv5_2_weight = float(content_conv5_2_weight)
            content_conv5_3_check = bool(content_conv5_3_check)
            content_conv5_3_weight = float(content_conv5_3_weight)
            content_pool5_check = bool(content_pool5_check)
            content_pool5_weight = float(content_pool5_weight)
            ########
            current_faster_content = self.faster_content_list[self.faster_content_list_counter]
            current_faster_content.alpha = alpha
            current_faster_content.content_conv1_1_check = content_conv1_1_check
            current_faster_content.content_conv1_1_weight = content_conv1_1_weight
            current_faster_content.content_conv1_2_check = content_conv1_2_check
            current_faster_content.content_conv1_2_weight = content_conv1_2_weight
            current_faster_content.content_pool1_check = content_pool1_check
            current_faster_content.content_pool1_weight = content_pool1_weight
            current_faster_content.content_conv2_1_check = content_conv2_1_check
            current_faster_content.content_conv2_1_weight = content_conv2_1_weight
            current_faster_content.content_conv2_2_check = content_conv2_2_check
            current_faster_content.content_conv2_2_weight = content_conv2_2_weight
            current_faster_content.content_pool2_check = content_pool2_check
            current_faster_content.content_pool2_weight = content_pool2_weight
            current_faster_content.content_conv3_1_check = content_conv3_1_check
            current_faster_content.content_conv3_1_weight = content_conv3_1_weight
            current_faster_content.content_conv3_2_check = content_conv3_2_check
            current_faster_content.content_conv3_2_weight = content_conv3_2_weight
            current_faster_content.content_conv3_3_check = content_conv3_3_check
            current_faster_content.content_conv3_3_weight = content_conv3_3_weight
            current_faster_content.content_pool3_check = content_pool3_check
            current_faster_content.content_pool3_weight = content_pool3_weight
            current_faster_content.content_conv4_1_check = content_conv4_1_check
            current_faster_content.content_conv4_1_weight = content_conv4_1_weight
            current_faster_content.content_conv4_2_check = content_conv4_2_check
            current_faster_content.content_conv4_2_weight = content_conv4_2_weight
            current_faster_content.content_conv4_3_check = content_conv4_3_check
            current_faster_content.content_conv4_3_weight = content_conv4_3_weight
            current_faster_content.content_pool4_check = content_pool4_check
            current_faster_content.content_pool4_weight = content_pool4_weight
            current_faster_content.content_conv5_1_check = content_conv5_1_check
            current_faster_content.content_conv5_1_weight = content_conv5_1_weight
            current_faster_content.content_conv5_2_check = content_conv5_2_check
            current_faster_content.content_conv5_2_weight = content_conv5_2_weight
            current_faster_content.content_conv5_3_check = content_conv5_3_check
            current_faster_content.content_conv5_3_weight = content_conv5_3_weight
            current_faster_content.content_pool5_check = content_pool5_check
            current_faster_content.content_pool5_weight = content_pool5_weight
        except:
            pass

    def faster_style_save(self, beta, style_conv1_1_check, style_conv1_1_weight, style_conv1_2_check, style_conv1_2_weight, style_pool1_check, style_pool1_weight, style_conv2_1_check, style_conv2_1_weight, style_conv2_2_check, style_conv2_2_weight, style_pool2_check, style_pool2_weight, style_conv3_1_check, style_conv3_1_weight, style_conv3_2_check, style_conv3_2_weight, style_conv3_3_check, style_conv3_3_weight, style_pool3_check, style_pool3_weight, style_conv4_1_check, style_conv4_1_weight, style_conv4_2_check, style_conv4_2_weight, style_conv4_3_check, style_conv4_3_weight, style_pool4_check, style_pool4_weight, style_conv5_1_check, style_conv5_1_weight, style_conv5_2_check, style_conv5_2_weight, style_conv5_3_check, style_conv5_3_weight, style_pool5_check, style_pool5_weight):
        try:
            beta = float(beta)
            style_conv1_1_check = bool(style_conv1_1_check)
            style_conv1_1_weight = float(style_conv1_1_weight)
            style_conv1_2_check = bool(style_conv1_2_check)
            style_conv1_2_weight = float(style_conv1_2_weight)
            style_pool1_check = bool(style_pool1_check)
            style_pool1_weight = float(style_pool1_weight)
            style_conv2_1_check = bool(style_conv2_1_check)
            style_conv2_1_weight = float(style_conv2_1_weight)
            style_conv2_2_check = bool(style_conv2_2_check)
            style_conv2_2_weight = float(style_conv2_2_weight)
            style_pool2_check = bool(style_pool2_check)
            style_pool2_weight = float(style_pool2_weight)
            style_conv3_1_check = bool(style_conv3_1_check)
            style_conv3_1_weight = float(style_conv3_1_weight)
            style_conv3_2_check = bool(style_conv3_2_check)
            style_conv3_2_weight = float(style_conv3_2_weight)
            style_conv3_3_check = bool(style_conv3_3_check)
            style_conv3_3_weight = float(style_conv3_3_weight)
            style_pool3_check = bool(style_pool3_check)
            style_pool3_weight = float(style_pool3_weight)
            style_conv4_1_check = bool(style_conv4_1_check)
            style_conv4_1_weight = float(style_conv4_1_weight)
            style_conv4_2_check = bool(style_conv4_2_check)
            style_conv4_2_weight = float(style_conv4_2_weight)
            style_conv4_3_check = bool(style_conv4_3_check)
            style_conv4_3_weight = float(style_conv4_3_weight)
            style_pool4_check = bool(style_pool4_check)
            style_pool4_weight = float(style_pool4_weight)
            style_conv5_1_check = bool(style_conv5_1_check)
            style_conv5_1_weight = float(style_conv5_1_weight)
            style_conv5_2_check = bool(style_conv5_2_check)
            style_conv5_2_weight = float(style_conv5_2_weight)
            style_conv5_3_check = bool(style_conv5_3_check)
            style_conv5_3_weight = float(style_conv5_3_weight)
            style_pool5_check = bool(style_pool5_check)
            style_pool5_weight = float(style_pool5_weight)
            ######
            current_faster_style = self.faster_style_list[self.faster_style_list_counter]
            current_faster_style.beta = beta
            current_faster_style.style_conv1_1_check = style_conv1_1_check
            current_faster_style.style_conv1_1_weight = style_conv1_1_weight
            current_faster_style.style_conv1_2_check = style_conv1_2_check
            current_faster_style.style_conv1_2_weight = style_conv1_2_weight
            current_faster_style.style_pool1_check = style_pool1_check
            current_faster_style.style_pool1_weight = style_pool1_weight
            current_faster_style.style_conv2_1_check = style_conv2_1_check
            current_faster_style.style_conv2_1_weight = style_conv2_1_weight
            current_faster_style.style_conv2_2_check = style_conv2_2_check
            current_faster_style.style_conv2_2_weight = style_conv2_2_weight
            current_faster_style.style_pool2_check = style_pool2_check
            current_faster_style.style_pool2_weight = style_pool2_weight
            current_faster_style.style_conv3_1_check = style_conv3_1_check
            current_faster_style.style_conv3_1_weight = style_conv3_1_weight
            current_faster_style.style_conv3_2_check = style_conv3_2_check
            current_faster_style.style_conv3_2_weight = style_conv3_2_weight
            current_faster_style.style_conv3_3_check = style_conv3_3_check
            current_faster_style.style_conv3_3_weight = style_conv3_3_weight
            current_faster_style.style_pool3_check = style_pool3_check
            current_faster_style.style_pool3_weight = style_pool3_weight
            current_faster_style.style_conv4_1_check = style_conv4_1_check
            current_faster_style.style_conv4_1_weight = style_conv4_1_weight
            current_faster_style.style_conv4_2_check = style_conv4_2_check
            current_faster_style.style_conv4_2_weight = style_conv4_2_weight
            current_faster_style.style_conv4_3_check = style_conv4_3_check
            current_faster_style.style_conv4_3_weight = style_conv4_3_weight
            current_faster_style.style_pool4_check = style_pool4_check
            current_faster_style.style_pool4_weight = style_pool4_weight
            current_faster_style.style_conv5_1_check = style_conv5_1_check
            current_faster_style.style_conv5_1_weight = style_conv5_1_weight
            current_faster_style.style_conv5_2_check = style_conv5_2_check
            current_faster_style.style_conv5_2_weight = style_conv5_2_weight
            current_faster_style.style_conv5_3_check = style_conv5_3_check
            current_faster_style.style_conv5_3_weight = style_conv5_3_weight
            current_faster_style.style_pool5_check = style_pool5_check
            current_faster_style.style_pool5_weight = style_pool5_weight
        except:
            pass

    def fast_content_save(self, alpha, content_conv1_1_check, content_conv1_1_weight, content_conv1_2_check, content_conv1_2_weight, content_pool1_check, content_pool1_weight, content_conv2_1_check, content_conv2_1_weight, content_conv2_2_check, content_conv2_2_weight, content_pool2_check, content_pool2_weight, content_conv3_1_check, content_conv3_1_weight, content_conv3_2_check, content_conv3_2_weight, content_conv3_3_check, content_conv3_3_weight, content_conv3_4_check, content_conv3_4_weight, content_pool3_check, content_pool3_weight, content_conv4_1_check, content_conv4_1_weight, content_conv4_2_check, content_conv4_2_weight, content_conv4_3_check, content_conv4_3_weight, content_conv4_4_check, content_conv4_4_weight, content_pool4_check, content_pool4_weight, content_conv5_1_check, content_conv5_1_weight, content_conv5_2_check, content_conv5_2_weight, content_conv5_3_check, content_conv5_3_weight, content_conv5_4_check, content_conv5_4_weight, content_pool5_check, content_pool5_weight):
        try:
            alpha = float(alpha)
            content_conv1_1_check = bool(content_conv1_1_check)
            content_conv1_1_weight = float(content_conv1_1_weight)
            content_conv1_2_check = bool(content_conv1_2_check)
            content_conv1_2_weight = float(content_conv1_2_weight)
            content_pool1_check = bool(content_pool1_check)
            content_pool1_weight = float(content_pool1_weight)
            content_conv2_1_check = bool(content_conv2_1_check)
            content_conv2_1_weight = float(content_conv2_1_weight)
            content_conv2_2_check = bool(content_conv2_2_check)
            content_conv2_2_weight = float(content_conv2_2_weight)
            content_pool2_check = bool(content_pool2_check)
            content_pool2_weight = float(content_pool2_weight)
            content_conv3_1_check = bool(content_conv3_1_check)
            content_conv3_1_weight = float(content_conv3_1_weight)
            content_conv3_2_check = bool(content_conv3_2_check)
            content_conv3_2_weight = float(content_conv3_2_weight)
            content_conv3_3_check = bool(content_conv3_3_check)
            content_conv3_3_weight = float(content_conv3_3_weight)
            content_conv3_4_check = bool(content_conv3_4_check)
            content_conv3_4_weight = float(content_conv3_4_weight)
            content_pool3_check = bool(content_pool3_check)
            content_pool3_weight = float(content_pool3_weight)
            content_conv4_1_check = bool(content_conv4_1_check)
            content_conv4_1_weight = float(content_conv4_1_weight)
            content_conv4_2_check = bool(content_conv4_2_check)
            content_conv4_2_weight = float(content_conv4_2_weight)
            content_conv4_3_check = bool(content_conv4_3_check)
            content_conv4_3_weight = float(content_conv4_3_weight)
            content_conv4_4_check = bool(content_conv4_4_check)
            content_conv4_4_weight = float(content_conv4_4_weight)
            content_pool4_check = bool(content_pool4_check)
            content_pool4_weight = float(content_pool4_weight)
            content_conv5_1_check = bool(content_conv5_1_check)
            content_conv5_1_weight = float(content_conv5_1_weight)
            content_conv5_2_check = bool(content_conv5_2_check)
            content_conv5_2_weight = float(content_conv5_2_weight)
            content_conv5_3_check = bool(content_conv5_3_check)
            content_conv5_3_weight = float(content_conv5_3_weight)
            content_conv5_4_check = bool(content_conv5_4_check)
            content_conv5_4_weight = float(content_conv5_4_weight)
            content_pool5_check = bool(content_pool5_check)
            content_pool5_weight = float(content_pool5_weight)
            ########
            current_fast_content = self.fast_content_list[self.fast_content_list_counter]
            current_fast_content.alpha = alpha
            current_fast_content.content_conv1_1_check = content_conv1_1_check
            current_fast_content.content_conv1_1_weight = content_conv1_1_weight
            current_fast_content.content_conv1_2_check = content_conv1_2_check
            current_fast_content.content_conv1_2_weight = content_conv1_2_weight
            current_fast_content.content_pool1_check = content_pool1_check
            current_fast_content.content_pool1_weight = content_pool1_weight
            current_fast_content.content_conv2_1_check = content_conv2_1_check
            current_fast_content.content_conv2_1_weight = content_conv2_1_weight
            current_fast_content.content_conv2_2_check = content_conv2_2_check
            current_fast_content.content_conv2_2_weight = content_conv2_2_weight
            current_fast_content.content_pool2_check = content_pool2_check
            current_fast_content.content_pool2_weight = content_pool2_weight
            current_fast_content.content_conv3_1_check = content_conv3_1_check
            current_fast_content.content_conv3_1_weight = content_conv3_1_weight
            current_fast_content.content_conv3_2_check = content_conv3_2_check
            current_fast_content.content_conv3_2_weight = content_conv3_2_weight
            current_fast_content.content_conv3_3_check = content_conv3_3_check
            current_fast_content.content_conv3_3_weight = content_conv3_3_weight
            current_fast_content.content_conv3_4_check = content_conv3_4_check
            current_fast_content.content_conv3_4_weight = content_conv3_4_weight
            current_fast_content.content_pool3_check = content_pool3_check
            current_fast_content.content_pool3_weight = content_pool3_weight
            current_fast_content.content_conv4_1_check = content_conv4_1_check
            current_fast_content.content_conv4_1_weight = content_conv4_1_weight
            current_fast_content.content_conv4_2_check = content_conv4_2_check
            current_fast_content.content_conv4_2_weight = content_conv4_2_weight
            current_fast_content.content_conv4_3_check = content_conv4_3_check
            current_fast_content.content_conv4_3_weight = content_conv4_3_weight
            current_fast_content.content_conv4_4_check = content_conv4_4_check
            current_fast_content.content_conv4_4_weight = content_conv4_4_weight
            current_fast_content.content_pool4_check = content_pool4_check
            current_fast_content.content_pool4_weight = content_pool4_weight
            current_fast_content.content_conv5_1_check = content_conv5_1_check
            current_fast_content.content_conv5_1_weight = content_conv5_1_weight
            current_fast_content.content_conv5_2_check = content_conv5_2_check
            current_fast_content.content_conv5_2_weight = content_conv5_2_weight
            current_fast_content.content_conv5_3_check = content_conv5_3_check
            current_fast_content.content_conv5_3_weight = content_conv5_3_weight
            current_fast_content.content_conv5_4_check = content_conv5_4_check
            current_fast_content.content_conv5_4_weight = content_conv5_4_weight
            current_fast_content.content_pool5_check = content_pool5_check
            current_fast_content.content_pool5_weight = content_pool5_weight
        except:
            pass

    def fast_style_save(self, beta, style_conv1_1_check, style_conv1_1_weight, style_conv1_2_check, style_conv1_2_weight, style_pool1_check, style_pool1_weight, style_conv2_1_check, style_conv2_1_weight, style_conv2_2_check, style_conv2_2_weight, style_pool2_check, style_pool2_weight, style_conv3_1_check, style_conv3_1_weight, style_conv3_2_check, style_conv3_2_weight, style_conv3_3_check, style_conv3_3_weight, style_conv3_4_check, style_conv3_4_weight, style_pool3_check, style_pool3_weight, style_conv4_1_check, style_conv4_1_weight, style_conv4_2_check, style_conv4_2_weight, style_conv4_3_check, style_conv4_3_weight, style_conv4_4_check, style_conv4_4_weight, style_pool4_check, style_pool4_weight, style_conv5_1_check, style_conv5_1_weight, style_conv5_2_check, style_conv5_2_weight, style_conv5_3_check, style_conv5_3_weight, style_conv5_4_check, style_conv5_4_weight, style_pool5_check, style_pool5_weight):
        try:
            beta = float(beta)
            style_conv1_1_check = bool(style_conv1_1_check)
            style_conv1_1_weight = float(style_conv1_1_weight)
            style_conv1_2_check = bool(style_conv1_2_check)
            style_conv1_2_weight = float(style_conv1_2_weight)
            style_pool1_check = bool(style_pool1_check)
            style_pool1_weight = float(style_pool1_weight)
            style_conv2_1_check = bool(style_conv2_1_check)
            style_conv2_1_weight = float(style_conv2_1_weight)
            style_conv2_2_check = bool(style_conv2_2_check)
            style_conv2_2_weight = float(style_conv2_2_weight)
            style_pool2_check = bool(style_pool2_check)
            style_pool2_weight = float(style_pool2_weight)
            style_conv3_1_check = bool(style_conv3_1_check)
            style_conv3_1_weight = float(style_conv3_1_weight)
            style_conv3_2_check = bool(style_conv3_2_check)
            style_conv3_2_weight = float(style_conv3_2_weight)
            style_conv3_3_check = bool(style_conv3_3_check)
            style_conv3_3_weight = float(style_conv3_3_weight)
            style_conv3_4_check = bool(style_conv3_4_check)
            style_conv3_4_weight = float(style_conv3_4_weight)
            style_pool3_check = bool(style_pool3_check)
            style_pool3_weight = float(style_pool3_weight)
            style_conv4_1_check = bool(style_conv4_1_check)
            style_conv4_1_weight = float(style_conv4_1_weight)
            style_conv4_2_check = bool(style_conv4_2_check)
            style_conv4_2_weight = float(style_conv4_2_weight)
            style_conv4_3_check = bool(style_conv4_3_check)
            style_conv4_3_weight = float(style_conv4_3_weight)
            style_conv4_4_check = bool(style_conv4_4_check)
            style_conv4_4_weight = float(style_conv4_4_weight)
            style_pool4_check = bool(style_pool4_check)
            style_pool4_weight = float(style_pool4_weight)
            style_conv5_1_check = bool(style_conv5_1_check)
            style_conv5_1_weight = float(style_conv5_1_weight)
            style_conv5_2_check = bool(style_conv5_2_check)
            style_conv5_2_weight = float(style_conv5_2_weight)
            style_conv5_3_check = bool(style_conv5_3_check)
            style_conv5_3_weight = float(style_conv5_3_weight)
            style_conv5_4_check = bool(style_conv5_4_check)
            style_conv5_4_weight = float(style_conv5_4_weight)
            style_pool5_check = bool(style_pool5_check)
            style_pool5_weight = float(style_pool5_weight)
            ######
            current_fast_style = self.fast_style_list[self.fast_style_list_counter]
            current_fast_style.beta = beta
            current_fast_style.style_conv1_1_check = style_conv1_1_check
            current_fast_style.style_conv1_1_weight = style_conv1_1_weight
            current_fast_style.style_conv1_2_check = style_conv1_2_check
            current_fast_style.style_conv1_2_weight = style_conv1_2_weight
            current_fast_style.style_pool1_check = style_pool1_check
            current_fast_style.style_pool1_weight = style_pool1_weight
            current_fast_style.style_conv2_1_check = style_conv2_1_check
            current_fast_style.style_conv2_1_weight = style_conv2_1_weight
            current_fast_style.style_conv2_2_check = style_conv2_2_check
            current_fast_style.style_conv2_2_weight = style_conv2_2_weight
            current_fast_style.style_pool2_check = style_pool2_check
            current_fast_style.style_pool2_weight = style_pool2_weight
            current_fast_style.style_conv3_1_check = style_conv3_1_check
            current_fast_style.style_conv3_1_weight = style_conv3_1_weight
            current_fast_style.style_conv3_2_check = style_conv3_2_check
            current_fast_style.style_conv3_2_weight = style_conv3_2_weight
            current_fast_style.style_conv3_3_check = style_conv3_3_check
            current_fast_style.style_conv3_3_weight = style_conv3_3_weight
            current_fast_style.style_conv3_4_check = style_conv3_4_check
            current_fast_style.style_conv3_4_weight = style_conv3_4_weight
            current_fast_style.style_pool3_check = style_pool3_check
            current_fast_style.style_pool3_weight = style_pool3_weight
            current_fast_style.style_conv4_1_check = style_conv4_1_check
            current_fast_style.style_conv4_1_weight = style_conv4_1_weight
            current_fast_style.style_conv4_2_check = style_conv4_2_check
            current_fast_style.style_conv4_2_weight = style_conv4_2_weight
            current_fast_style.style_conv4_3_check = style_conv4_3_check
            current_fast_style.style_conv4_3_weight = style_conv4_3_weight
            current_fast_style.style_conv4_4_check = style_conv4_4_check
            current_fast_style.style_conv4_4_weight = style_conv4_4_weight
            current_fast_style.style_pool4_check = style_pool4_check
            current_fast_style.style_pool4_weight = style_pool4_weight
            current_fast_style.style_conv5_1_check = style_conv5_1_check
            current_fast_style.style_conv5_1_weight = style_conv5_1_weight
            current_fast_style.style_conv5_2_check = style_conv5_2_check
            current_fast_style.style_conv5_2_weight = style_conv5_2_weight
            current_fast_style.style_conv5_3_check = style_conv5_3_check
            current_fast_style.style_conv5_3_weight = style_conv5_3_weight
            current_fast_style.style_conv5_4_check = style_conv5_4_check
            current_fast_style.style_conv5_4_weight = style_conv5_4_weight
            current_fast_style.style_pool5_check = style_pool5_check
            current_fast_style.style_pool5_weight = style_pool5_weight
        except:
            pass

