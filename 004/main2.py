import taichi as ti

ti.init(ti.gpu)

res = 512

# flat layout
pixels = ti.field(ti.f32, (res, res))

# flat layout
# pixels = ti.field(ti.f32)
# block = ti.root.dense(ti.ij, (512, 512))
# block.place(pixels)

# hierarchical layout
# pixels = ti.field(ti.f32)
# block = ti.root.dense(ti.ij, (2, 2))
# block = block.dense(ti.ij, (256, 256))
# block.place(pixels)

# hierarchical layout
# pixels = ti.field(ti.f32)
# block = ti.root.dense(ti.ij, (256, 256))
# block = block.dense(ti.ij, (2, 2))
# block.place(pixels)


@ti.func
def fract(x):
    return x - ti.floor(x)


@ti.func
def lerp(x, y, w):
    return (y - x) * (3.0 - w * 2.0) * w * w + x


@ti.func
def dot(l, r):
    return l.dot(r)


# https://www.shadertoy.com/view/Xsl3Dl
@ti.func
def hash_3d(p):
    p = ti.Vector([
        p.dot(ti.Vector([127.1, 311.7, 74.7])),
        p.dot(ti.Vector([269.5, 183.3, 246.1])),
        p.dot(ti.Vector([113.5, 271.9, 124.6]))
    ])
    return -1 + 2 * fract(ti.sin(p) * 43758.5453123)


@ti.func
def gradient_noise_3d(p):
    i = ti.floor(p)
    f = fract(p)
    u = f * f * (3.0 - 2.0 * f)
    return lerp(lerp(lerp(dot(hash_3d(i + ti.Vector([0.0, 0.0, 0.0])), f - ti.Vector([0.0, 0.0, 0.0])),
                          dot(hash_3d(i + ti.Vector([1.0, 0.0, 0.0])), f - ti.Vector([1.0, 0.0, 0.0])), u.x),
                     lerp(dot(hash_3d(i + ti.Vector([0.0, 1.0, 0.0])), f - ti.Vector([0.0, 1.0, 0.0])),
                          dot(hash_3d(i + ti.Vector([1.0, 1.0, 0.0])), f - ti.Vector([1.0, 1.0, 0.0])), u.x), u.y),
                lerp(lerp(dot(hash_3d(i + ti.Vector([0.0, 0.0, 1.0])), f - ti.Vector([0.0, 0.0, 1.0])),
                          dot(hash_3d(i + ti.Vector([1.0, 0.0, 1.0])), f - ti.Vector([1.0, 0.0, 1.0])), u.x),
                     lerp(dot(hash_3d(i + ti.Vector([0.0, 1.0, 1.0])), f - ti.Vector([0.0, 1.0, 1.0])),
                          dot(hash_3d(i + ti.Vector([1.0, 1.0, 1.0])), f - ti.Vector([1.0, 1.0, 1.0])), u.x), u.y), u.z)


@ti.kernel
def paint(z: ti.f32):
    for P in ti.grouped(pixels):
        fp = P * 0.01
        p = ti.Vector([fp.x, fp.y, z])
        pixels[P] = gradient_noise_3d(p) * 0.5 + 0.5


def show():
    with ti.GUI("Perlin Noise", res=(res, res)) as gui:
        step = 0.0
        while gui.running:
            paint(step)
            step += 0.01
            gui.set_image(pixels)
            gui.show()


def save():
    result_dir = "./data"
    video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    step = 0.0
    for i in range(100):
        paint(step)
        step += 0.01
        video_manager.write_frame(pixels.to_numpy())
        print(f'\rFrame {i + 1}/1000 is recorded', end='')

    print()
    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=True)
    print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')


if __name__ == "__main__":
    show()
    # save()
