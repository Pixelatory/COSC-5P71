import numpy as np
from PIL import Image, ImageFilter
from deap import base

from A2.util import Tuple

edges = Image.open('images/standard.png').convert(mode="RGB").filter(ImageFilter.EMBOSS)
#edges.show()
#edges.convert(mode="L").show()
print(edges.getpixel((100, 100)))

a = Tuple(3)
b = Tuple([2, 3, 4])

print(a.items, a.capacity)
print(b.items, b.capacity)
