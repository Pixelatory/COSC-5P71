from PIL import Image
from deap import base

im = Image.open("images/standard.png")
im = im.convert(mode="L")

classPos = []
nonClassPos = []
classIm = Image.open("images/classification.png")
for x in range(classIm.size[0]):
    for y in range(classIm.size[1]):
        if sum(classIm.getpixel((x, y))) > 0:
            classPos.append((x, y))
        else:
            nonClassPos.append((x, y))

print(classPos)
print(nonClassPos)

def wow(a, b, c, d):
    return x


toolbox = base.Toolbox()
toolbox.register('wow', wow, a=4, b=2, c=1, d=5)

print(toolbox.wow())
x = 5
print(toolbox.wow())
