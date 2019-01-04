from math import sin, cos, pi, sqrt

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from fem.fem import FiniteElement


def f(x):
    result = cos(sqrt(x[0] ** 2 + x[1] ** 2))
    return result


num_triangles = []
max_tri_edge = []
num_total_points = [2000, 3000, 4000, 5000, 8000, 16000, 32000]

radius = 15
for num_inner_points in [1000, 2000, 3000, 4000, 7000, 15000, 31000]:
    points = []
    boundary = []
    while len(points) < num_inner_points:
        x = (-1) ** (np.random.randint(10)) * np.random.rand() * radius
        y = (-1) ** (np.random.randint(10)) * np.random.rand() * radius
        if sqrt(x * x + y * y) < radius:
            points.append([x, y])

    gap = 2 * pi / 1000
    for i in range(1000):
        theta = gap * i
        points.append([sin(theta) * radius, cos(theta) * radius])
        boundary.append([len(points) - 1, 0])

    print('\n\n#Number of points: ', len(points))

    fem = FiniteElement(points, boundary, A=np.array([[1, 0], [0, 1]]), q=5, func=f)
    fem.solve()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(fem.points[:, 0], fem.points[:, 1], fem.triangles, fem.solution, cmap='rainbow')
    plt.title('%d points' % len(points), fontsize=14)
    plt.savefig('../results/fem/' + '%d-points-pde3' % len(points) + '.pdf')
    plt.show()

    max_edge = 0
    for tri in fem.triangles:
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            length = np.linalg.norm(np.array(points[v1]) - np.array(points[v2]), 2)
            max_edge = max(max_edge, length)
    max_tri_edge.append(max_edge)
