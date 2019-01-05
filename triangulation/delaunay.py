from math import sqrt

import numpy as np
from matplotlib import pyplot as plt


class Point:
    def __init__(self, x_y=(0.0, 0.0)):
        self.x, self.y = x_y[0], x_y[1]

    def __str__(self):
        return ('Point: ' + str([self.x, self.y]))


class Edge:
    def __init__(self, v1, v2, tri=None):

        # assert v1 != v2

        self.v1 = min(v1, v2)
        self.v2 = max(v1, v2)
        if tri == None:
            self.tri = []
        elif isinstance(tri, list):
            self.tri = tri
        else:
            self.tri = [tri]

    def __str__(self):
        return ('Edge: ' + str([self.v1, self.v2]))

    def __hash__(self):
        # tuple is hashable
        return hash((self.v1, self.v2))

    def __eq__(self, other):
        if other == None:
            return False

        if self.v1 != other.v1:
            return False
        if self.v2 != other.v2:
            return False
        if len(self.tri) != len(other.tri):
            return False
        elif len(self.tri):
            if (max(self.tri) != max(other.tri) or min(self.tri) != min(other.tri)):
                return False
        return True


class Triangle:
    def __init__(self, vlist, elist):
        # assert len(set(vlist)) == 3

        # assert vlist[0] != vlist[1]
        # assert vlist[0] != vlist[2]
        # assert vlist[1] != vlist[2]

        self.vlist = vlist
        self.elist = elist

    def __str__(self):
        return ('Triangle: ' + str([str(v) for v in self.vlist]) + str([str(e) for e in self.elist]))



# O(n) counting sort
def counting_sort(point_bin_num, point_sort):
    max_bin_num = max(point_bin_num)
    point_num = len(point_bin_num)
    aux = [0] * (max_bin_num + 1)
    for i in range(point_num):
        aux[point_bin_num[i]] += 1
    for i in range(1, max_bin_num + 1):
        aux[i] += aux[i - 1]
    for i in range(point_num - 1, -1, -1):
        point_sort[aux[point_bin_num[i]] - 1] = i
        aux[point_bin_num[i]] -= 1
    return point_sort


def is_point_in_triangle(p1, p2, p3, p):
    area = 0.5 * (-p2.y * p3.x + p1.y * (-p2.x + p3.x) + p1.x * (p2.y - p3.y) + p2.x * p3.y)
    u = 1.0 / (2 * area) * (p1.y * p3.x - p1.x * p3.y + (p3.y - p1.y) * p.x + (p1.x - p3.x) * p.y)
    v = 1.0 / (2 * area) * (p1.x * p2.y - p1.y * p2.x + (p1.y - p2.y) * p.x + (p2.x - p1.x) * p.y)
    return u, v


def is_point_in_triangle_circle(p1, p2, p3, p):
    x13 = p1.x - p3.x
    x23 = p2.x - p3.x
    x1p = p1.x - p.x
    x2p = p2.x - p.x
    y13 = p1.y - p3.y
    y23 = p2.y - p3.y
    y1p = p1.y - p.y
    y2p = p2.y - p.y
    cosA = x13 * x23 + y13 * y23
    cosB = x2p * x1p + y2p * y1p
    if (cosA >= 0 and cosB >= 0):
        return False
    if (cosA < 0 and cosB < 0):
        return True
    sinA = x13 * y23 - x23 * y13
    sinB = x2p * y1p - x1p * y2p
    sinAB = sinA * cosB + sinB * cosA
    if (sinAB < 0):
        return True
    else:
        return False


# cross product
def cross_mult(a, b, c):
    return (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y)


# judge whether edge cross or not
def is_intersect(a, b, c, d):
    if (max(a.x, b.x) < min(c.x, d.x)):
        return False
    if (max(a.y, b.y) < min(c.y, d.y)):
        return False
    if (max(c.x, d.x) < min(a.x, b.x)):
        return False
    if (max(c.y, d.y) < min(a.y, b.y)):
        return False
    if (cross_mult(c, b, a) * cross_mult(b, d, a) < 0):
        return False
    if (cross_mult(a, d, c) * cross_mult(d, b, c) < 0):
        return False
    return True


# judge whether the quadrilateral is convex or not
def is_convex_quadrilateral(a, b, c, d):
    u, v = is_point_in_triangle(a, c, d, b)
    if (u > 0 and v > 0 and u + v < 1):
        return False
    u, v = is_point_in_triangle(b, c, d, a)
    if (u > 0 and v > 0 and u + v < 1):
        return False
    return True


# find the relative position of c to vector a->b, return True for right position
def is_point_in_right_of_vector(a, b, c):
    if cross_mult(b, c, a) < 0:
        return True
    else:
        return False


# return the center position of a triangle
def triangle_center(a, b, c):
    return Point((np.mean([a.x, b.x, c.x]), np.mean([a.y, b.y, c.y])))


# change the diagonal of a quadrilateral
def change_quadrilateral_diagonal(p1, p2, p3, p, e, triangles):
    for k in range(3):
        if triangles[e.tri[0]].vlist[k] == p1:
            e01 = triangles[e.tri[0]].elist[k]
        elif triangles[e.tri[0]].vlist[k] == p2:
            e02 = triangles[e.tri[0]].elist[k]
        if triangles[e.tri[1]].vlist[k] == p1:
            e11 = triangles[e.tri[1]].elist[k]
        elif triangles[e.tri[1]].vlist[k] == p2:
            e12 = triangles[e.tri[1]].elist[k]
    # adjust edges
    # e02
    if e02.tri[0] == e.tri[0]:
        e02.tri[0] = e.tri[1]
    else:
        e02.tri[1] = e.tri[1]
    # e11
    if e11.tri[0] == e.tri[1]:
        e11.tri[0] = e.tri[0]
    else:
        e11.tri[1] = e11.tri[0]
        e11.tri[0] = e.tri[0]
    # e12
    if e12.tri[0] != e.tri[1]:
        e12.tri[1] = e12.tri[0]
        e12.tri[0] = e.tri[1]
    # e
    e.v1 = p
    e.v2 = p3

    # adjust triangles
    triangles[e.tri[0]] = Triangle([p, p2, p3], [e11, e, e01])
    triangles[e.tri[1]] = Triangle([p, p3, p1], [e12, e02, e])


def delaunay(points, constraint_borders=None, shape_name=None, dynamic_show=False):
    point_num = len(points)
    if (point_num < 3):
        return

    # A.1. (Normalize coordinates of points.)
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    x_max, x_min = max(xs), min(xs)
    y_max, y_min = max(ys), min(ys)
    d_max = max(x_max - x_min, y_max - y_min)
    xs = (xs - x_min) / d_max
    ys = (ys - y_min) / d_max
    points = np.array([[x, y] for x, y in zip(xs, ys)])
    x_max = (x_max - x_min) / d_max
    y_max = (y_max - y_min) / d_max

    # A.2. (Sort points into bins.)
    bin_num_row = int(sqrt(sqrt(point_num)))
    point_bin_num = [0] * point_num
    for i in range(point_num):
        r = int(0.99 * bin_num_row * points[i][1] / y_max)
        c = int(0.99 * bin_num_row * points[i][0] / x_max)
        if i % 2 == 0:
            point_bin_num[i] = r * bin_num_row + c + 1
        else:
            point_bin_num[i] = (r + 1) * bin_num_row - c
    point_sort = counting_sort(point_bin_num, list(range(point_num)))

    # A.3. (Establish the super-triangle.)
    bound = 3
    super_points = [[-bound + 2, -bound + 2], [bound, -bound + 2], [0.0, bound]]
    points = np.vstack([points, super_points])
    triangles = []
    triangle_exist = []
    e1, e2, e3 = Edge(point_num, point_num + 1, 0), Edge(point_num + 1, point_num + 2, 0), Edge(point_num + 2,
                                                                                                point_num, 0)
    triangles.append(Triangle([point_num, point_num + 1, point_num + 2], [e2, e3, e1]))
    triangle_exist.append(1)

    # A.4. (Loop over each point.) For each point in the list of sorted points, do steps 5-7.
    # A.5. (Insert new point in triangulation.)
    # A.6. (Initialize stack.)
    # A.7.（Restore Delaunay triangulation.)
    for i in range(point_num):
        # print('# Point', i, point_sort[i])
        # print('Triangles %s %s' % (str([str(t) for t in triangles]), str(triangle_exist)))
        pi = point_sort[i]

        # dynamic triangulation results show
        if dynamic_show:
            plt.scatter(x=[p[0] for k, p in enumerate(points[:point_num])],
                        y=[p[1] for k, p in enumerate(points[:point_num])],
                        c='slategray', marker='o', s=10)
            # plt.scatter(x=[p[0] for k, p in enumerate(points[:point_num]) if k == i], y=[p[1] for k, p in enumerate(points[:point_num]) if k == i],
            # c='aqua', marker='s', s=20)
            for n, tri in enumerate(triangles):
                if (triangle_exist[n] and max(tri.vlist) < point_num):
                    xs = [points[v][0] for v in tri.vlist]
                    ys = [points[v][1] for v in tri.vlist]
                    xs.append(points[tri.vlist[0]][0])
                    ys.append(points[tri.vlist[0]][1])
                    plt.plot(xs, ys)
            if shape_name:
                plt.title(shape_name, fontsize=14)
            # plt.savefig('../results/triangulation-gif/' + '%s-%d' % (shape_name, i) + '.pdf')
            plt.show()
            # time.sleep(0.05)
            # plt.close()

        p = Point(points[pi])
        for j in range(len(triangles) - 1, -1, -1):
            if triangle_exist[j] == 0:
                continue

            tri = triangles[j]
            u, v = is_point_in_triangle(Point(points[tri.vlist[0]]), Point(points[tri.vlist[1]]),
                                        Point(points[tri.vlist[2]]), p)

            # point in triangle
            if (u >= 0.0) and (v >= 0.0) and (u + v > 0.0) and (u + v <= 1.0):

                edge_contain_p = None
                if u == 0.0 and v < 1.0:
                    edge_contain_p = tri.elist[2]
                elif v == 0.0 and u < 1.0:
                    edge_contain_p = tri.elist[1]
                elif u + v == 1.0:
                    edge_contain_p = tri.elist[0]

                edge_stack = []
                cur_triangle_num = len(triangles)
                p1, p2, p3 = tri.vlist
                new_edges = [Edge(p1, pi), Edge(p2, pi), Edge(p3, pi)]

                triangle_exist[j] = 0

                for k in range(3):
                    edge = tri.elist[k]

                    if (edge.tri[0] == j):
                        edge.tri[0] = cur_triangle_num + k
                    else:
                        edge.tri[1] = edge.tri[0]
                        edge.tri[0] = cur_triangle_num + k
                    if len(edge.tri) > 1:
                        edge_stack.append(edge)

                    if (edge != edge_contain_p) or (edge == edge_contain_p and len(edge.tri) > 1):
                        new_edges[(k + 1) % 3].tri.append(cur_triangle_num + k)
                        new_edges[(k + 2) % 3].tri.append(cur_triangle_num + k)
                        triangles.append(
                            Triangle([pi, tri.vlist[(k + 1) % 3], tri.vlist[(k + 2) % 3]], [edge,
                                                                                            new_edges[(k + 2) % 3],
                                                                                            new_edges[(k + 1) % 3]]))
                        triangle_exist.append(1)
                    elif (edge == edge_contain_p) and (len(edge.tri) == 1):
                        triangles.append(
                            Triangle([pi, tri.vlist[(k + 1) % 3], tri.vlist[(k + 2) % 3]], [edge,
                                                                                            new_edges[(k + 2) % 3],
                                                                                            new_edges[(k + 1) % 3]]))
                        triangle_exist.append(0)

                while len(edge_stack) > 0:
                    e = edge_stack.pop()

                    assert e.tri[0] != e.tri[1]
                    assert pi in triangles[e.tri[0]].vlist
                    assert pi not in triangles[e.tri[1]].vlist

                    tri = triangles[e.tri[1]]
                    v1, v2 = e.v1, e.v2
                    for k in range(3):
                        v = tri.vlist[k]
                        if v != v1 and v != v2:
                            p3 = v
                            p1 = tri.vlist[(k + 1) % 3]
                            p2 = tri.vlist[(k + 2) % 3]
                            break

                    if is_point_in_triangle_circle(Point(points[p1]), Point(points[p2]), Point(points[p3]), p):

                        for k in range(3):
                            if triangles[e.tri[0]].vlist[k] == p1:
                                e01 = triangles[e.tri[0]].elist[k]
                            elif triangles[e.tri[0]].vlist[k] == p2:
                                e02 = triangles[e.tri[0]].elist[k]
                            if triangles[e.tri[1]].vlist[k] == p1:
                                e11 = triangles[e.tri[1]].elist[k]
                            elif triangles[e.tri[1]].vlist[k] == p2:
                                e12 = triangles[e.tri[1]].elist[k]

                        # e02
                        if e02.tri[0] == e.tri[0]:
                            e02.tri[0] = e.tri[1]
                        else:
                            e02.tri[1] = e.tri[1]
                        # e11
                        if e11.tri[0] == e.tri[1]:
                            e11.tri[0] = e.tri[0]
                        else:
                            e11.tri[1] = e11.tri[0]
                            e11.tri[0] = e.tri[0]
                        # e12
                        if e12.tri[0] != e.tri[1]:
                            e12.tri[1] = e12.tri[0]
                            e12.tri[0] = e.tri[1]
                        # e
                        e.v1 = pi
                        e.v2 = p3

                        # adjust triangles
                        triangles[e.tri[0]] = Triangle([pi, p2, p3], [e11, e, e01])
                        triangles[e.tri[1]] = Triangle([pi, p3, p1], [e12, e02, e])

                        if len(e11.tri) > 1:
                            edge_stack.append(e11)
                        if len(e12.tri) > 1:
                            edge_stack.append(e12)

                # move to next point
                break

    if dynamic_show:
        plt.scatter(x=[p[0] for k, p in enumerate(points[:point_num])],
                    y=[p[1] for k, p in enumerate(points[:point_num])],
                    c='slategray', marker='o', s=10)
        for n, tri in enumerate(triangles):
            if (triangle_exist[n] and max(tri.vlist) < point_num):
                xs = [points[v][0] for v in tri.vlist]
                ys = [points[v][1] for v in tri.vlist]
                xs.append(points[tri.vlist[0]][0])
                ys.append(points[tri.vlist[0]][1])
                plt.plot(xs, ys)
        if shape_name:
            plt.title(shape_name, fontsize=14)
        # plt.savefig('../results/triangulation-gif/' + '%s-%d' % (shape_name, point_num) + '.pdf')
        plt.show()

    for n, tri in enumerate(triangles):
        if (triangle_exist[n]) and max(tri.vlist) < point_num:
            xs = [points[v][0] for v in tri.vlist]
            ys = [points[v][1] for v in tri.vlist]
            xs.append(points[tri.vlist[0]][0])
            ys.append(points[tri.vlist[0]][1])
            plt.plot(xs, ys)
    if shape_name:
        plt.title(shape_name + ' triangulation', fontsize=14)
        # plt.savefig('../results/triangulation/' + '%s-tri-%d-points' % (shape_name, point_num) + '.pdf')
    plt.show()
    plt.close()

    # Section B. deal with constraint edges.
    if (constraint_borders != None):
        tot_edges = []
        tot_edges_set = set()
        constraint_edge_set = set()

        # record all current edges
        for i, tri in enumerate(triangles):
            if triangle_exist[i]:
                for j in range(3):
                    edge = Edge(tri.vlist[(j + 1) % 3], tri.vlist[(j + 2) % 3])
                    if edge not in tot_edges_set:
                        tot_edges_set.add(edge)
                        tot_edges.append(tri.elist[j])
        for border in constraint_borders:
            constraint_edge_set.add(Edge(border[0], border[1]))

        # B.1. (Loop over each constrained edge.)
        for border in constraint_borders:
            v1, v2 = border[0], border[1]
            con_edge = Edge(v1, v2)
            if con_edge in tot_edges_set:
                continue

            # B.2. (Find intersecting edges.)
            cross_edges = []
            for i, e in enumerate(tot_edges):
                if v1 == e.v1 or v1 == e.v2 or v2 == e.v1 or v2 == e.v2:
                    continue
                if is_intersect(Point(points[e.v1]), Point(points[e.v2]), Point(points[v1]), Point(points[v2])):
                    cross_edges.append(i)

            # B.3. (Remove intersecting edges.)
            new_edges = []
            while (len(cross_edges)):
                for i in range(len(cross_edges)):
                    if i >= len(cross_edges):
                        break
                    e = tot_edges[cross_edges[i]]
                    if is_convex_quadrilateral(Point(points[e.v1]), Point(points[e.v2]), Point(points[v1]),
                                               Point(points[v2])):
                        x, y = e.v1, e.v2

                        # find p1, p2, p3, p
                        for k in range(3):
                            v = triangles[e.tri[0]].vlist[k]
                            if (v != x and v != y):
                                p = v
                                p2 = triangles[e.tri[0]].vlist[(k + 1) % 3]
                                p1 = triangles[e.tri[0]].vlist[(k + 2) % 3]
                                break
                        for k in range(3):
                            v = triangles[e.tri[1]].vlist[k]
                            if (v != x and v != y):
                                p3 = v
                                break
                        tot_edges_set.remove(Edge(e.v1, e.v2))
                        change_quadrilateral_diagonal(p1, p2, p3, p, e, triangles)
                        tot_edges_set.add(Edge(e.v1, e.v2))

                        if v1 == e.v1 or v1 == e.v2 or v2 == e.v1 or v2 == e.v2:
                            cross = False
                        else:
                            cross = is_intersect(Point(points[e.v1]), Point(points[e.v2]), Point(points[v1]),
                                                 Point(points[v2]))
                        if not cross:
                            new_edges.append(cross_edges[i])

                            cross_num = len(cross_edges)
                            cross_edges[i] = cross_edges[cross_num - 1]
                            cross_edges = cross_edges[:-1]

            for i in range(len(new_edges)):
                e = tot_edges[new_edges[i]]
                x, y = e.v1, e.v2
                if ((x == v1 and y == v2) or (x == v2 and y == v1) or
                        (Edge(x, y) in constraint_edge_set)):
                    continue
                # find p1, p2, p3, p
                for k in range(3):
                    v = triangles[e.tri[0]].vlist[k]
                    if (v != x and v != y):
                        p = v
                        p2 = triangles[e.tri[0]].vlist[(k + 1) % 3]
                        p1 = triangles[e.tri[0]].vlist[(k + 2) % 3]
                        break
                for k in range(3):
                    v = triangles[e.tri[1]].vlist[k]
                    if (v != x and v != y):
                        p3 = v
                        break
                if is_point_in_triangle_circle(Point(points[v1]), Point(points[v2]), Point(points[p3]),
                                               Point(points[p])):
                    tot_edges_set.remove(Edge(e.v1, e.v2))
                    change_quadrilateral_diagonal(p1, p2, p3, p, e, triangles)
                    tot_edges_set.add(Edge(e.v1, e.v2))

    # B.5. (Remove superfluous triangles.)
    constraint_border_tuples = set()
    border_point_next = dict()
    if constraint_borders != None:
        for border in constraint_borders:
            border_point_next[border[0]] = border[1]
            constraint_border_tuples.add((border[0], border[1]))

    for i in range(len(triangles)):
        tri = triangles[i]
        if triangle_exist[i]:
            if max(tri.vlist) >= point_num:
                triangle_exist[i] = 0
                continue

            if constraint_borders != None:
                inner_flag = False
                for j in range(3):
                    cur_e = tri.elist[j]
                    cur_v = tri.vlist[j]
                    if (cur_e.v1, cur_e.v2) in constraint_border_tuples:
                        if is_point_in_right_of_vector(Point(points[cur_e.v1]), Point(points[cur_e.v2]),
                                                       Point(points[cur_v])):
                            triangle_exist[i] = 0
                            inner_flag = True
                            continue
                    elif (cur_e.v2, cur_e.v1) in constraint_border_tuples:
                        if is_point_in_right_of_vector(Point(points[cur_e.v2]), Point(points[cur_e.v1]),
                                                       Point(points[cur_v])):
                            triangle_exist[i] = 0
                            inner_flag = True
                            continue

                if not inner_flag:
                    strict_border = True
                    for v in tri.vlist:
                        if v not in border_point_next:
                            strict_border = False
                    if strict_border:
                        next = tri.vlist[0]
                        while next != tri.vlist[1] and next != tri.vlist[2] and next in border_point_next:
                            next = border_point_next[next]
                            if next == tri.vlist[2] and next != tri.vlist[1]:
                                triangle_exist[i] = 0

    if constraint_borders:
        for n, tri in enumerate(triangles):
            if (triangle_exist[n]):
                xs = [points[v][0] for v in tri.vlist]
                ys = [points[v][1] for v in tri.vlist]
                xs.append(points[tri.vlist[0]][0])
                ys.append(points[tri.vlist[0]][1])
                plt.plot(xs, ys)
        if shape_name:
            plt.title(shape_name +' constraint triangulation', fontsize=14)
            # plt.savefig('../results/triangulation/' + '%s-tri-constraint-%d-points' % (shape_name, point_num) + '.pdf')
        plt.show()
        # plt.close()

    exist_triangles = [tri for i, tri in enumerate(triangles) if triangle_exist[i]]

    return [tri.vlist for tri in exist_triangles]
