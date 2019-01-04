from math import sin, cos, pi

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from fem.fem import FiniteElement


def f(x):
    result = -6 + 2 * x[0] * x[0] + 2 * x[1] * x[1]
    return result


num_triangles = []
max_tri_edge = []
max_point_error = []
error_infinity_norm = []
num_total_points = [600, 1000, 2000, 3000, 4000, 8000, 16000]

for num_inner_points in [300, 700, 1700, 2700, 3700, 7700, 15700]:
    points = []
    boundary = []
    while len(points) < num_inner_points:
        x = (-1) ** (np.random.randint(10)) * np.random.rand()
        y = (-1) ** (np.random.randint(10)) * np.random.rand()
        if x * x + y * y < 1.0:
            points.append([x, y])

    gap = 2 * pi / 300
    for i in range(300):
        theta = gap * i
        points.append([sin(theta), cos(theta)])
        boundary.append([len(points) - 1, 0])

    print('\n\n#Number of points: ', len(points))

    fem = FiniteElement(points, boundary, A=np.array([[1, 0], [0, 1]]), q=2, func=f)
    fem.solve()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(fem.points[:, 0], fem.points[:, 1], fem.triangles, fem.solution, cmap='rainbow')
    plt.title('%d points' % len(points), fontsize=14)
    plt.savefig('../results/fem/' + '%d-points-pde1' % len(points) + '.pdf')
    plt.show()


    def sol(x):
        return x[0] * x[0] + x[1] * x[1] - 1


    error = np.array(fem.solution) - np.array([sol(x) for x in fem.points])

    max_point_error.append(max(abs(error)))
    num_triangles.append(len(fem.triangles))
    error_infinity_norm.append(np.linalg.norm(error, np.infty))

    max_edge = 0
    for tri in fem.triangles:
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            length = np.linalg.norm(np.array(points[v1]) - np.array(points[v2]), 2)
            max_edge = max(max_edge, length)
    max_tri_edge.append(max_edge)

print('\n# Total error results:')
print('Num_total_points', num_total_points)
print('Max_point_error:', max_point_error)
print('Num_triangles:', num_triangles)
print('Error_infinity_norm:', error_infinity_norm)
print('Max_tri_edge:', max_tri_edge)
