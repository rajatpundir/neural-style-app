from kivy.lang import Builder
from kivy.uix.image import Image

Builder.load_file('styleimage.kv')

class StyleImage(Image):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def on_touch_down(self, touch):
		if self.collide_point(*touch.pos):
			self.output_screen.parent.cycle_image('style')

