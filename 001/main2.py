import taichi as ti
ti.init(arch=ti.gpu)

n = 320
pixels = ti.Vector.field(3, dtype=float, shape=(n * 2, n * 2))


@ti.func
def hsv_to_rgb(h, s, v):
    r = 0.0
    g = 0.0
    b = 0.0
    if s == 0.0:
        r = v
        g = v
        b = v
    else:
        i = int(h * 6.0)  # XXX assume int() truncates!
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            r = v
            g = t
            b = p
        if i == 1:
            r = q
            g = v
            b = p
        if i == 2:
            r = p
            g = v
            b = t
        if i == 3:
            r = p
            g = q
            b = v
        if i == 4:
            r = t
            g = p
            b = v
        if i == 5:
            r = v
            g = p
            b = q
    return ti.Vector([r, g, b])


@ti.func
def hsv2rgb(h, s, v):
    t = hsv_to_rgb(h, s, v)
    return ti.Vector([t[2], t[1], t[0]])


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])


@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallized over all pixels
        c = 0.7885 * ti.Vector([ti.cos(t), ti.sin(t)])
        z = ti.Vector([i / n - 1, j / n - 1]) * 2
        iterations = 300
        while z.norm() < 20 and iterations > 0:
            z = complex_sqr(z) + c
            iterations -= 1
        pixels[i, j] = hsv2rgb(iterations / 300, 1, 1 if iterations > 0 else 0)
        # pixels[i, j] = ti.Vector([iterations % 256 / 255.0, iterations % 256 / 255.0, (iterations * 8) % 256 / 255.0])


with ti.GUI("Julia Set", res=(n * 2, n * 2)) as gui:
    for i in range(1000000):
        paint(i * 0.03)
        gui.set_image(pixels)
        gui.show()
