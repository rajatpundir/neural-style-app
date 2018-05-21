import numpy as np
import tensorflow as tf
from PIL import Image

class BaseGraph:
    def __init__(self, content_images_list, style_images_list, width, height, alpha, beta, gamma, noise_ratio, use_meta, save_meta):
        self.content_images_list = content_images_list
        self.style_images_list = style_images_list
        self.width = width
        self.height = height
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.noise_ratio = noise_ratio
        self.use_meta = use_meta
        self.save_meta = save_meta
        # some constants
        self.batch_size = 1
        self.channels = 3
        self.mean = np.array([123.68, 116.779, 103.939])
        self.sess = tf.Session()
        self.inputs = tf.Variable(np.zeros((self.batch_size, self.height, self.width, self.channels)), dtype = 'float32', name='input')
        # some other variables
        self.layer_func = {}
        self.content_list = []
        self.style_list = []
        self.preprocess_content()
        self.preprocess_style()
        self.preprocess_mix_image()
        self.content_layers = []
        self.style_layers = []
        self.preprocess_content_layers()
        self.preprocess_style_layers()
        self.preprocess_mix_image()
        self.final_loss = 0
        self.image_counter = 0

    def preprocess(self, path):
        temp = Image.open(path).resize((self.width, self.height))
        temp = np.asarray(temp, dtype='float32')
        temp -= self.mean
        temp = np.expand_dims(temp, axis=0)
        return(temp[:, :, :, ::-1])

    def preprocess_content(self):
        for content in self.content_images_list:
            self.content_list.append(self.preprocess(content.path))

    def preprocess_style(self):
        for style in self.style_images_list:
            self.style_list.append(self.preprocess(style.path))

    def preprocess_mix_image(self):
        if not self.use_meta:
            noise = np.random.uniform(0, 255, (self.batch_size, self.height, self.width, self.channels)) - 128.0
            self.mix_image = noise * self.noise_ratio
            for content in self.content_list:
                self.mix_image += content * ((1 - self.noise_ratio) / len(self.content_list))
        else:
            try:
                self.mix_image = self.preprocess('meta/meta.png')
            except:
                self.use_meta = False
                self.prepare_mix_image()

    def save_mix_image(self):
        mix_image = self.sess.run(self.inputs)
        mix_image = mix_image.reshape((self.height, self.width, self.channels))
        mix_image = mix_image[:, :, ::-1]
        mix_image += self.mean
        mix_image = np.clip(mix_image, 0, 255).astype('uint8')
        Image.fromarray(mix_image).save('out/%d.png' % self.image_counter,'PNG')
        if self.save_meta:
            Image.fromarray(mix_image).save('meta/meta.png','PNG')
        self.image_counter += 1

    def content_loss(self, index):
        loss_in_content = 0
        for layer, weight in self.content_layers[index]:
            c_cont = self.sess.run(self.layer_func[layer])
            c_mix = self.layer_func[layer]
            const = 2 * ((c_cont.shape[1] * c_cont.shape[2]) ** 0.5) * (c_cont.shape[3] ** 0.5)
            loss_in_content += (weight * tf.reduce_sum(tf.pow(c_mix - c_cont, 2)) / const)
        return(self.content_images_list[index].alpha * loss_in_content)

    def gram_matrix(self, volume, area, depth):
        V = tf.reshape(volume, (area, depth))
        return(tf.matmul(tf.transpose(V), V))

    def style_loss_over_layer(self, layer):
        s_styl = self.sess.run(self.layer_func[layer])
        s_mix = self.layer_func[layer]
        area, depth = s_styl.shape[1] * s_styl.shape[2], s_styl.shape[3]
        const = 4 * (depth**2) * (area**2)
        return(tf.reduce_sum(tf.pow(self.gram_matrix(s_mix, area, depth) - self.gram_matrix(s_styl, area, depth), 2)) / const)

    def style_loss(self, index):
        loss_in_style = 0
        for layer, weight in self.style_layers[index]:
            loss_in_style += (weight * self.beta / len(self.style_layers[index])) * self.style_loss_over_layer(layer)
        return(self.style_images_list[index].beta * loss_in_style)

    def variation_loss(self):
        x = self.inputs
        a = tf.pow((x[:, :self.height-1, :self.width-1, :] - x[:, 1:, :self.width-1, :]), 2)
        b = tf.pow((x[:, :self.height-1, :self.width-1, :] - x[:, :self.height-1, 1:, :]), 2)
        return(self.gamma * tf.reduce_sum(tf.pow(a + b, 1.25)))

    def final_content_loss(self):
        loss = 0
        for index, content in enumerate(self.content_list):
            self.sess.run(self.inputs.assign(content))
            loss += self.content_loss(index)
        return(self.alpha * loss)
    
    def final_style_loss(self):
        loss = 0
        for index, style in enumerate(self.style_list):
            self.sess.run(self.inputs.assign(style))
            loss += self.style_loss(index)
        return(self.beta * loss)

    def define_final_loss(self):
        self.sess.run(tf.global_variables_initializer())
        self.final_loss = self.final_content_loss() + self.final_style_loss() + self.variation_loss()

    def compute_final_loss(self):
        return(self.sess.run(self.final_loss))

