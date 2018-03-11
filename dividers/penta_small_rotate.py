from math import acos, sin, pi, cos, atan, asin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, product
import numpy as np
from scipy.optimize import fsolve

r = 1
a = 4 * r * (10 + 22 / 5 ** 0.5) ** (-0.5)  # length of the side
d = a * sin(3 * pi / 10) / sin(2 * pi / 5)  # distance between the center and the pentagon angle
h = d * sin(3 * pi / 10)  # hight to the middle of the side
angle = acos(- 5 ** (-0.5))  # between two surfaces
center_up = np.array([0, 0, 1])
center_down = np.array([0, 0, -1])
center = np.array([0, 0, 0])
min_distance = h / 2 # for decoding position
# cangle_bet_pangles = pi/5
# one_side_divide = 3
# step_rotangle = cangle_bet_pangles/one_side_divide
eps_length = 0.001
d_hor_angle = 2*pi/5
d_ver_angle = pi/3
n_y = 5
n_z = 5

########################### Getting dodecahedron points ######################################
def get_side_centers_hotizontal(point):
    '''
    :param point: center_point of dodecahedron
    :return: five points - centers of horizontal pentagon sides, Radious = 1 (global)
    '''
    initial_angle = 0
    centers = []
    for i in range(5):
        centers.append(np.array([h * cos(initial_angle) + point[0], h * sin(initial_angle) + point[1], point[2]]))
        initial_angle += 2 * pi / 5
    return centers

def get_up_surface_centers(point_from):
    '''
    :param point_from: the highest point of dodecahedron
    :return: points in the higher semispace of dodecahedron
    '''
    initial_angle = 0
    centers = [point_from]
    for i in range(5):
        centers.append(np.array([h * cos(initial_angle) * (1 - cos(angle)) + point_from[0], h * sin(initial_angle) * (1 - cos(angle)) + point_from[1], h * sin(angle) + point_from[2]]))
        initial_angle += 2 * pi / 5
    return centers

def get_down_surface_centers(point_from):
    '''
    :param point_from: the lowest point of dodecahedron
    :return: points in thr lower semispace of dodecahedron
    '''
    initial_angle = 0
    centers = [point_from]
    for i in range(5):
        centers.append(np.array([h * cos(initial_angle) * (1 - cos(angle)) + point_from[0], h * sin(initial_angle) * (1 - cos(angle)) + point_from[1], -h * sin(angle) + point_from[2]]))
        initial_angle += 2 * pi / 5
    return centers

def get_penta_points(point_center):
    '''
    :param point_center:
    :return: centers of dodecahedron (oriented with (0, 0, 0) degrees) sides
    '''
    center_up = np.array([i for i in point_center])
    center_up[2] += 1
    center_down = np.array([i for i in point_center])
    center_down[2] -= 1
    return np.concatenate((get_down_surface_centers(center_up), get_up_surface_centers(center_down)), axis=0)
########################### 3d plots #############################################
def show_points(points):
    '''
    :param points:
    :return: 3d-show array of points
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in points:
        ax.scatter(i[0], i[1], i[2])
    plt.show()

def show_named_points(points, labels = False):
    '''
    :param points: dictionary of points
    :param labels: False - show without annotate
    :return: 3d-show (optionally with annotate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if labels:
        for item, i in points.items():
            ax.scatter(i[0], i[1], i[2])
            ax.text(i[0], i[1], i[2], item)
    else:
        for item, i in points.items():
            ax.scatter(i[0], i[1], i[2])
        plt.show()
##################################Smth##########################################

def get_dictionary_coordinates(penta_points):
    '''
    :param penta_points: points to set nums of sections
    :return: dictionary of enumerate points
    '''
    dictionary = {}
    for i, j in enumerate(penta_points):
        dictionary.update({i+1: j})
    return dictionary

def section_num_by_coords(center_point, penta_point):
    '''
    :param center_point: point is center of dodecahedron
    :param penta_point: one point which has bound with center
    :return: num of section of center_point which penta_point belongs to
    '''
    dictionary = get_dictionary_coordinates(get_penta_points(center_point))
    for key, item in dictionary.items():
        diff = np.array(item) - np.array(penta_point)
        diff = np.linalg.norm(diff)
        if diff < d:
            return key

#########################Angles###########################
def point_to_angles(point):
    '''
    :param point: coordinates in Decart basis
    :return: two angles in spherical coordinates (without radius)
    '''
    if isinstance(point[0], (int, float)):
        if point[0] != 0:
            theta = atan(point[1] / point[0])
            if point[0] < 0:
                theta += pi
        else:
            if point[1] > 0:
                theta = pi / 2
            elif point[1] < 0:
                theta = -pi / 2
            else:
                theta = 0
        return np.array([theta, acos((point[2]))])
    else:
        angles = []
        for i in point:
            angles.append(point_to_angles(i))
        return angles

def get_penta_angles():
    '''
    :return: angles of dodecahedron points with O=(0,0,0) and R=1
    '''
    points = get_penta_points(np.array([0,0,0]))
    angles = []
    for i in points:
        angles.append(point_to_angles(i))
    return angles

def show_points_by_angles(center, angles, labels=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if labels:
        for item, i in angles.items():
            ax.scatter(center[0]+cos(i[0])*sin(i[1]), center[1] + sin(i[0])*sin(i[1]), center[2] + cos(i[1]))
            ax.text(center[0]+cos(i[0])*sin(i[1]), center[1] + sin(i[0])*sin(i[1]), center[2] + cos(i[1]), item)
    else:
        for item, i in angles.items():
            ax.scatter(center[0] + cos(i[0]) * sin(i[1]), center[1] + sin(i[0]) * sin(i[1]), center[2] + cos(i[1]))

    plt.show()

# def rotate(angles, alpha, betta):
#     for key, item in angles.items():
#         angles[key] = np.array([(angles[key][0]+alpha)%(2*pi), angles[key][1]+betta if abs(angles[key][1]+betta) < pi/2 else (pi - angles[key][1]-betta)])
#     return angles
#########################Rotations##############################################
def Rx(x_angle):
    '''
    :param x_angle:
    :return: matrix for rotation around x-axis with x-angle
    '''
    R_x = np.eye(3)
    R_x[1][1], R_x[1][2], R_x[2][1],R_x[2][2]=[cos(x_angle), -sin(x_angle),sin(x_angle), cos(x_angle)]
    return R_x

def Ry(y_angle):
    '''
    :param y_angle:
    :return: matrix for rotation around y-axis with y-angle
    '''
    R_y = np.eye(3)
    R_y[0][0], R_y[0][2], R_y[2][0],R_y[2][2]=[cos(y_angle), sin(y_angle),-sin(y_angle), cos(y_angle)]
    return R_y

def Rz(z_angle):
    '''
    :param z_angle:
    :return: matrix for rotation around z-axis with z-angle
    '''
    R_z = np.eye(3)
    R_z[0][0], R_z[0][1], R_z[1][0],R_z[1][1]=[cos(z_angle), -sin(z_angle),sin(z_angle), cos(z_angle)]
    return R_z

def Rxyz(x_angle, y_angle, z_angle):
    '''
    :param x_angle:
    :param y_angle:
    :param z_angle:
    :return: matrix for all rotations around (x,y,z)-axis
    '''
    return Rz(z_angle).dot(Ry(y_angle).dot(Rx(x_angle)))

def rotate_arr(points, a1, a2, a3):
    return [i.dot(Rxyz(a1,a2,a3)) for i in points]

def rotate(points, alpha, betta, gamma):
    '''
    :param points: dictionary of points to rotate
    :param alpha: angle for rotate around x-axis
    :param betta: angle for rotate around y-axis
    :param gamma: angle for rotate around z-axis
    :return: rotated dictionary of points
    '''
    for key, item in points.items():
        points[key] = points[key].dot(Rxyz(alpha, betta, gamma))
    return points

def functions_to_solve(p, data):
    '''
    :param p: variable to solve
    :param data: point0 - from old basis, point1 - from next basis
    :return: system with wanted basis angles
    '''
    x, y, z = p
    point0, point1 = (np.array(i) for i in data)
    system = point1-point0.dot(Rxyz(x, y, z))
    return (system[0], system[1], system[2])

def search_rotate(p0, p1):
    '''
    :param p0: coordinates from old basis
    :param p1: in new basis
    :return: new basis
    '''
    return fsolve(functions_to_solve, [0, 0, 0], args=[p0, p1])

def rotate_2d_on_sphere(points, y_angle, z_angle):
    rotate = Rz(z_angle).dot(Ry(y_angle))
    return [i.dot(rotate) for i in points]

pp = get_penta_points(np.array([0,0,0]))
pa = get_penta_angles()

def check_diff_rotate(n_y, n_z):
    my_dodecahedron = get_penta_points(np.array([0, 0, 0]))
    dhs = []
    for i in range(n_y):
        for j in range(n_z):
            a1, a2 = i * d_hor_angle / n_y, j * d_ver_angle / n_z
            dhs.append(np.array(point_to_angles(rotate_2d_on_sphere(my_dodecahedron, a1, a2))))
    diff = 1
    for i0, i in enumerate(dhs):
        for j0, j in enumerate(dhs):
            if i0 != j0:
                for k in range(12):
                    diff = min(diff, np.linalg.norm(i[k] - j[k]))
    return diff

###############Extra (useless now)########################
def vertical_rotate_angles(angles, betta):
    for key, item in angles.items():
        if item[0]>pi:
            v_angle = angles[key][1] + betta
        else:
            v_angle = angles[key][1] - betta
        if abs(v_angle) > pi/2:
            if v_angle>0:
                angles[key][1] = pi - v_angle
            else:
                angles[key][1] = - pi - v_angle
            angles[key][0] = (angles[key][0] + pi) % (2 * pi)
    return angles

def vector_mult3(a,b):
    '''
    P.S. find in numpy vector multiplication!!!!!!!!!
    :param a: first vector
    :param b: second vector
    :return: vector multiplication
    '''
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])

def rotate_around_vector(rotation_base, R1, angle):
    #may be later????
    return rotation_base.dot(R1)(1-cos(angle)) * rotation_base + vector_mult3(rotation_base, R1) * sin(angle) + R1 * cos(angle)
##########################################################
def rotation_3axis(points, parameters):
    '''
    :param points: to rotation
    :param parameters:angles to rotate around parameters
    :return:rotated points
    '''
    parameters = parameters[::]#*(np.array([n_y, n_z]))
    # print(parameters)
    if isinstance(points, dict):
        to_rotate = np.array(list(points.values()))
        return dict(enumerate(to_rotate.dot(Rxyz(parameters[0], parameters[1],parameters[2]))))
    return points.dot(Rxyz(parameters[0], parameters[1], parameters[2]))

def r2d_on_sphere(basis=np.array([0.,0.])):
    return rotate_2d_on_sphere(pp, d_hor_angle*basis[0]/n_z, d_ver_angle*basis[1]/n_z)
# print(r2d_on_sphere([0,0]))

# step_eps = 0.5*np.linalg.norm(get_penta_points(np.array([0,0,0]))[0]-rotation_3axis(get_penta_points(np.array([0,0,0]))[0], [0, step_rotangle,0]))
# in case n_y,n_z=(5,5), can be checked by check_diff_rotate
step_rot = 0.033
def get_section_of_rotation(point0, point1, basis0=[0,0,0]):
    rv = point1-point0
    pp1 = rotation_3axis(get_penta_points(np.array([0,0,0])), basis0)
    for i, j in enumerate(pp1):
        if np.linalg.norm(j-rv) < eps_length:
            return i

def get_section_and_basis(spoint, basis0=[0,0,0]):
    dodecahedron = rotation_3axis(get_penta_points(np.array([0, 0, 0])), basis0)
    for num, j in enumerate(dodecahedron):
        if np.linalg.norm(spoint - j) < step_rot * 25:
            for i in sorted(product((0, 1, 2), repeat=3), key=lambda x: sum(x)):
                checker = rotation_3axis(dodecahedron, i)
                if np.linalg.norm(checker[num] - spoint) < step_rot:
                    return [num, i]

def check_section(point0, basis0=[0,0], point1):
    # ang0 = point_to_angles(point0)
    # ang1 = point_to_angles(point1)
    pp1 = r2d_on_sphere(basis0)
    for i in pp1:
        pass

if __name__ == '__main__':
    position = [1,1,2]

        # print(i)
    # print(dhs[0][1]-dhs[1][1])
    # show_points(my_dodecahedron)
    # show_points(rotate_arr(my_dodecahedron, step_rotangle*position[0], step_rotangle*position[1], step_rotangle*position[2]))
    # spoint = rotation_3axis(my_dodecahedron[10], position)
    # print(get_section_and_basis(spoint))
    # print(rotation_3axis(my_dodecahedron, [1,2,1])[1])
    # print(get_section_of_rotation(np.array([0,0,0]), my_dodecahedron[1]))
    # rotated = rotate_arr(my_dodecahedron, 0, 0, 0)

    # print()

