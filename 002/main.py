import taichi as ti
ti.init(ti.gpu)

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 1
PI = 3.141592653

# number of planets
N = 1000
# unit mass
m = 5
# mouse black hole mass
m0 = 10000
# galaxy size
galaxy_size = 0.4
# planet radius (for rendering)
planet_radius = 2
# init vel
init_vel = 120

# time-step size
h = 1e-5
# substepping
substepping = 10

# pos, vel and force of the planets
# Nx2 vectors
pos0 = ti.Vector.field(2, ti.f32, ())
pos = ti.Vector.field(2, ti.f32, N)
vel = ti.Vector.field(2, ti.f32, N)
force = ti.Vector.field(2, ti.f32, N)

@ti.kernel
def initialize():
    center=ti.Vector([0.5, 0.5])
    for i in range(N):
        theta = ti.random() * 2 * PI
        r = (ti.sqrt(ti.random()) * 0.7 + 0.3) * galaxy_size
        offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        pos[i] = center+offset
        vel[i] = [-offset.y, offset.x]
        vel[i] *= init_vel


@ti.kernel
def compute_force():
    # clear force
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])

    # compute gravitational force
    for i in range(N):
        p = pos[i]
        for j in range(N):  # double the computation for a better memory footprint and load balance
            if i != j:
                diff = p - pos[j]
                r = diff.norm(1e-5)

                # gravitational force -(GMm / r^2) * (diff/r) for i
                f = -G * m * m * (1.0 / r) ** 3 * diff

                # assign to each particle
                force[i] += f


@ti.kernel
def compute_force0():
    for i in range(N):
        diff = pos[i] - pos0[None]
        r = diff.norm(1e-5)

        # gravitational force -(GMm / r^2) * (diff/r) for i
        f = -G * m * m0 * (1.0 / r) ** 3 * diff

        # assign to each particle
        force[i] += f


@ti.kernel
def update():
    dt = h/substepping
    for i in range(N):
        # symplectic euler
        vel[i] += dt*force[i]/m
        pos[i] += dt*vel[i]


gui = ti.GUI('N-body problem', (512, 512))
is_pressed = False
initialize()
while gui.running:
    if not is_pressed:
        is_pressed = gui.get_event(ti.GUI.LMB)
    else:
        is_pressed = not gui.get_event(ti.GUI.RELEASE)
    pos0[None] = gui.get_cursor_pos()

    for i in range(substepping):
        compute_force()
        if is_pressed:
            compute_force0()
        update()

    gui.clear(0x112F41)
    gui.circles(pos.to_numpy(), color=0xffffff, radius=planet_radius)
    gui.show()
