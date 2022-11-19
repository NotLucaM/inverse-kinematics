import math

import autograd.numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from autograd import grad

global target
target = np.array([-10, 3], dtype=float)


# joints (d, theta, r, alpha)
joints = np.array([
    np.array([0, 10, math.radians(0)], dtype=float),
    np.array([0, 7, math.radians(0)], dtype=float),
    np.array([0, 5, math.radians(0)], dtype=float)
], dtype=float)


def make_matrix(joint, t):
    transform = np.matrix([
        [np.cos(t), -np.sin(t), 0, 0],
        [np.sin(t), np.cos(t), 0, 0],
        [0, 0, 1, joint[0]],
        [0, 0, 0, 1]
    ])
    rotation = np.matrix([
        [1, 0, 0, joint[1]],
        [0, math.cos(joint[2]), -math.sin(joint[2]), 0],
        [0, math.sin(joint[2]), math.cos(joint[2]), 0],
        [0, 0, 0, 1]
    ])
    return transform, rotation


def matrix(all_joints, t):
    matrices = []

    for i, joint in enumerate(all_joints):
        m = make_matrix(joint, t[i])

        if i == 0:
            matrices.append(m[0] @ m[1])
            continue

        matrices.append(matrices[len(matrices) - 1] @ m[0] @ m[1])

    return matrices


def find_coords(matrices):
    x = [0]
    y = [0]

    for m in matrices:
        x.append(m[0, 3])
        y.append(m[1, 3])

    return x, y


def cost(t):
    m = matrix(joints, t).pop()
    l = np.array([m[0, 3], m[1, 3]])

    return np.linalg.norm(l - target) ** 2


c = grad(cost)

global theta
theta = np.array([0, 0, 0], dtype=float)

fig = plt.figure()
ax = plt.axes(xlim=(-23, 23), ylim=(-23, 23))
line, = ax.plot([], [], lw=3)
pt, = ax.plot([], [], 'ro')


def init():
    line.set_data([], [])
    pt.set_data([], [])
    return line,


def animate(i):
    global theta
    global target

    mat = matrix(joints, theta)

    theta -= 0.00075 * c(theta)

    x, y = find_coords(mat)
    line.set_data(x, y)
    pt.set_data([target[0]], [target[1]])
    return line, pt,


def onclick(event):
    global ix, iy
    global target
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(
        ix, iy))

    target = np.array([ix, iy])


anim = FuncAnimation(fig, animate, init_func=init,
                     frames=200, interval=20, blit=True)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
