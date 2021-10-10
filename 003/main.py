import taichi as ti
from celestial_objects import BlackHole, Star, Planet

if __name__ == "__main__":
    ti.init(arch=ti.cuda)

    # control
    is_pressed = False
    paused = False
    export_images = False

    # stars and planets
    black_hole = BlackHole(N=0, mass=10000)
    stars = Star(N=2, mass=1000)
    stars.initialize(0.5, 0.5, 0.2, 10)
    planets = Planet(N=1000, mass=1)
    planets.initialize(0.5, 0.5, 0.4, 10)

    # GUI
    my_gui = ti.GUI("Galaxy", (800, 800))
    h = 5e-5  # time-step size
    i = 0
    while my_gui.running:
        for e in my_gui.get_events(ti.GUI.PRESS, ti.GUI.LMB, ti.GUI.RELEASE):
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == ti.GUI.SPACE:
                paused = not paused
                print("paused =", paused)
            elif e.key == 'r':
                black_hole = BlackHole(N=0, mass=10000)
                stars.initialize(0.5, 0.5, 0.2, 10)
                planets.initialize(0.5, 0.5, 0.4, 10)
                i = 0
            elif e.key == 'i':
                export_images = not export_images
            elif not is_pressed and e.key == ti.GUI.LMB:
                is_pressed = True
                black_hole = BlackHole(N=1, mass=10000)
                pos = my_gui.get_cursor_pos()
                black_hole.initialize(pos[0], pos[1], 0, 0)
            elif is_pressed and e.key == ti.GUI.RELEASE:  # can't reach
                is_pressed = False
                black_hole = BlackHole(N=0, mass=10000)

        if not paused:
            black_hole.computeForce()
            stars.computeForce(black_hole)
            planets.computeForce(stars, black_hole)
            for celestial_obj in (black_hole, stars, planets):
                celestial_obj.update(h)
            i += 1

        black_hole.display(my_gui)
        stars.display(my_gui, radius=10, color=0xffd500)
        planets.display(my_gui)
        if export_images:
            my_gui.show(f"images\output_{i:05}.png")
        else:
            my_gui.show()
