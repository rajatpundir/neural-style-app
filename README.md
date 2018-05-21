# Neural-Style-App
This is a GUI application for Neural Style Transfer, written in Python using Kivy framework. It can do style transfer over multiple content and style images.

What do you need to do to run the app?
1. Install Tensorflow with GPU spport, follow the installation guide at https://www.tensorflow.org.
2. Install some other dependencies: scipy and pillow using pip.
3. And, then install Kivy by following the installation guide at https://kivy.org.
4. Run the app by typing 'python main.py' in the terminal.
5. Neural network models are to be placed inside the 'models' directory(if not present), if you wish to download the models, follow instruction #6 and #7.
6. Download pre-trained VGG-19 model from 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat' and place it inside 'models' directory, renaming it as 'fast.model'.
7. Download pre-trained VGG-16 model from 'https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz' and place it inside 'models' directory, renaming it as 'faster.model'.
