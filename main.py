from kivy.app import App
from imagemix import ImageMixController

class ImageMixApp(App):
    def build(self):
        return ImageMixController()


if __name__ == "__main__":
    ImageMixApp().run()
