from PIL import Image
from deap import base

im = Image.open("images/standard.png")
im = im.convert(mode="L")
im.close()

im = Image.open("images/classification.png")
x = 4


def wow(a, b, c, d):
    return x


toolbox = base.Toolbox()
toolbox.register('wow', wow, a=4, b=2, c=1, d=5)

print(toolbox.wow())
x = 5
print(toolbox.wow())
