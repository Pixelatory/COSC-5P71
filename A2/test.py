from PIL import Image, ImageFilter

im = Image.open("images/standard.png")
im = im.convert(mode="L")


for i in range(1, 12):
    try:
        maxFilter = im.filter(ImageFilter.MaxFilter(i))
        print(maxFilter.getpixel((0,0)))
        minFilter = im.filter(ImageFilter.MinFilter(i))
        meanFilter = im.filter(ImageFilter.BoxBlur(i))
        maxFilter.save("./images/filtered/Max-" + str(i) + "-gray.png")
        minFilter.save("./images/filtered/Min-" + str(i) + "-gray.png")
        meanFilter.save("./images/filtered/Mean-" + str(i) + "-gray.png")
        print(i)
    except Exception:
        pass


tmp = im.filter(ImageFilter.EDGE_ENHANCE)
tmp.save("./images/filtered/EdgeEnhance-gray.png")
tmp = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
tmp.save("./images/filtered/EdgeEnhanceMore-gray.png")
