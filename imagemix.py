from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager

from outputscreen import OutputScreen
from configscreen import ConfigScreen
from contentscreen import ContentScreen
from stylescreen import StyleScreen
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
        self.state = None
        self.worker = None
        self.result_queue = KivyQueue(self.result_queue_callback)
        self.command_queue = KivyQueue(self.command_queue_callback)
        self.response_queue = KivyQueue(self.reponse_queue_callback)
        self.output_screen.train_button.bind(on_press = self.set_config_screen_on_train_button)
        self.cleanup()
        # print("FINISHED:", sys._getframe().f_code.co_name)


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
            elif target is 'style' and self.worker is None:
                self.output_screen.style.source = filename[0]
                if filename[0] not in self.style_path_list:
                    self.style_path_list.append(filename[0])
                    self.style_path_list_counter = len(self.style_path_list) - 1
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


    def train_worker(self, width, height, use_meta, save_meta, use_lbfgs, max_iterations, noise_ratio, alpha, beta):
        # Threads share data.
        neural_worker = NeuralWorker(self.result_queue, self.command_queue, self.response_queue, self.content_path_list, self.style_path_list, width, height, use_meta, save_meta, use_lbfgs, max_iterations, noise_ratio, alpha, beta)
        neural_worker.train()


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


    def start_button(self, width, height, use_meta, save_meta, use_lbfgs, max_iterations, noise_ratio, alpha, beta):
        try:
            width = int(width)
            height = int(height)
            use_meta = bool(use_meta)
            save_meta = bool(save_meta)
            use_lbfgs = bool(use_lbfgs)
            max_iterations = int(max_iterations)
            noise_ratio = float(noise_ratio)
            alpha = int(alpha)
            beta = int(beta)
            if self.worker is None:
                if len(self.content_path_list) == 0 or len(self.style_path_list) == 0:
                    return
                self.worker = threading.Thread(target = self.train_worker, args=(width, height, use_meta, save_meta, use_lbfgs, max_iterations, noise_ratio, alpha, beta))
                self.worker.daemon = True
                self.worker.start()
                self.state = 'resumed'
                self.output_screen.logs.text = 'Starting to train...'
                self.output_screen.train_button.text = 'Pause'
                self.output_screen.train_button.unbind(on_press = self.set_config_screen_on_train_button)
                self.output_screen.train_button.bind(on_press = self.pause_or_resume_button)
        except:
            pass

