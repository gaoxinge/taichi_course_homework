import taichi as ti
ti.init(arch=ti.cuda)

# control
paused = True
save_images = False

# problem setting
n = 64
scatter = 8
res = n * scatter

# physical parameters
# time-integration related
h = 1e-3  # time-step size
substep = 1  # number of substeps
dx = 1  # finite difference step size (in space)
# heat-source related
t_max = 300  # max temperature (in Celsius)
t_min = 0  # min temperature
heat_center = (n // 2, n // 2)
heat_radius = 5.1
k = 100.0  # rate of heat diffusion

# temperature now and temperature next_time
t_n = ti.field(ti.f32, shape=(n, n))
t_np1 = ti.field(ti.f32, shape=(n, n))
# visualization
pixels = ti.Vector.field(3, ti.f32, shape=(res, res))


@ti.kernel
def init():
    for i, j in t_n:
        if (float(i) - heat_center[0]) ** 2 + (float(j) - heat_center[1]) ** 2 <= heat_radius ** 2:
            t_n[i, j] = t_max  # source
            t_np1[i, j] = t_max  # source
        else:
            t_n[i, j] = t_min
            t_np1[i, j] = t_min


@ti.kernel
def diffuse(dt: ti.f32):
    c = dt * k / dx ** 2
    for i, j in t_n:
        t_np1[i, j] = t_n[i, j]
        if i - 1 >= 0:
            t_np1[i, j] += c * (t_n[i - 1, j] - t_n[i, j])
        if i + 1 < n:
            t_np1[i, j] += c * (t_n[i + 1, j] - t_n[i, j])
        if j - 1 >= 0:
            t_np1[i, j] += c * (t_n[i, j - 1] - t_n[i, j])
        if j + 1 < n:
            t_np1[i, j] += c * (t_n[i, j + 1] - t_n[i, j])


@ti.kernel
def update_source():
    for i, j in t_n:
        if (float(i) - heat_center[0]) ** 2 + (float(j) - heat_center[1]) ** 2 <= heat_radius ** 2:
            t_np1[i, j] = t_max


@ti.func
def restrict(v):
    if v < t_min:
        v = t_min
    if v > t_max:
        v = t_max
    return v


@ti.kernel
def commit():
    for i, j in t_n:
        t_n[i, j] = restrict(t_np1[i, j])


@ti.func
def get_color(v):
    c = ti.Vector([1.0, 1.0, 1.0])  # white
    dv = t_max - t_min

    if v < (t_min + 0.25 * dv):
        c[0] = 0
        c[1] = 4 * (v - t_min) / dv
    elif v < (t_min + 0.5 * dv):
        c[0] = 0
        c[2] = 1 + 4 * (t_min + 0.25 * dv - v) / dv
    elif v < (t_min + 0.75 * dv):
        c[0] = 4 * (v - t_min - 0.5 * dv) / dv
        c[2] = 0
    else:
        c[1] = 1 + 4 * (t_min + 0.75 * dv - v) / dv
        c[2] = 0

    return c


@ti.kernel
def temperature_to_color():
    for i, j in t_n:
        for k, l in ti.ndrange(scatter, scatter):
            pixels[i * scatter + k, j * scatter + l] = get_color(t_n[i, j])


# GUI
my_gui = ti.GUI("Diffuse", (res, res))
init()
i = 0
while my_gui.running:
    for e in my_gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.SPACE:
            paused = not paused
        elif e.key == 'i':
            save_images = not save_images
            print(f"Exporting images to images/output_{i:05}.png")
        elif e.key == 'r':
            init()
            i = 0

    if not paused:
        for sub in range(substep):
            diffuse(h / substep)
            update_source()
            commit()

    temperature_to_color()
    my_gui.set_image(pixels)
    if save_images and not paused:
        my_gui.show(f"images/output_{i:05}.png")
        i += 1
    else:
        my_gui.show()
