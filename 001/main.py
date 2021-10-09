import colorsys
from PIL import Image


def hsv2rgb(h, s, v):
    return tuple(round(_ * 255) for _ in colorsys.hsv_to_rgb(h / 255, s / 255, v / 255))


if __name__ == "__main__":
    w, h, zoom = 800, 600, 1
    bitmap = Image.new("RGB", (w, h), "white")
    pix = bitmap.load()

    cX, cY = -0.7, 0.27015
    moveX, moveY = 0.0, 0.0
    maxIter = 300

    for x in range(w):
        for y in range(h):
            zx = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX
            zy = 1.0 * (y - h / 2) / (0.5 * zoom * h) + moveY
            i = maxIter

            while zx * zx + zy * zy < 4 and i > 0:
                tmp = zx * zx - zy * zy + cX
                zy, zx = 2.0 * zx * zy + cY, tmp
                i -= 1

            b, g, r = hsv2rgb(i / maxIter * 255, 255, 255 if i > 0 else 0)
            # r, g, b = (i % 256, i % 256, (i * 8) % 256)
            pix[x, y] = (r, g, b)

    bitmap.show()
