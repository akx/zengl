import zengl

ctx = zengl.context(zengl.loader(headless=True))

img = ctx.image((32, 32), 'rgba8unorm', array=3)

print(img)

face = img.face(layer=1, level=1)

print(face.image, face.size, face.layer, face.level)

