import json
from matplotlib import pyplot as plt
import os
from triangulation.delaunay import delaunay

shapes_dir = r'../dataset/shapes/'


def read_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
        return data


shape_files = ['deer-20.json', ]

for file in shape_files:

    shape = read_data(shapes_dir + file)

    shape_name = file[:-5]

    points = [[p['x'], p['y']] for p in shape['points']]
    points = points[:-1]
    points = points[::-1]
    xs = [xy[0] for xy in points]
    ys = [xy[1] for xy in points]
    num_points = len(points)

    plt.scatter(xs, ys, marker='o', s=15)
    plt.title(shape_name + ' scatter', fontsize=14)
    # plt.savefig('../results/triangulation/' + '%s-scatter-%d-points' % (shape_name, num_points) + '.pdf')
    plt.show()
    plt.close()

    plt.plot(xs, ys)
    plt.title(shape_name + ' line chart', fontsize=14)
    # plt.savefig('../results/triangulation/' + '%s-line-%d-points' % (shape_name, num_points) + '.pdf')
    plt.show()
    plt.close()

    borders = []
    for i in range(num_points):
        borders.append((i, (i + 1) % num_points))
    try:
        triangles = delaunay(points, constraint_borders=borders, shape_name=shape_name, dynamic_show=True)
    except:
        pass

