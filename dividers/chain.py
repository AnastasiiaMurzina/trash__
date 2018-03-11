from dividers.penta import get_penta_points, get_dictionary_coordinates, section_num_by_coords, show_points
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
count_bonds = {'H': 1, 'C': 4, 'O': 2}
eps_length = 0.001

def chain1():
    return [[1,2,1], [1,3,2], [1,4,6]], {1: 'C', 2: 'H', 3: 'H', 4: 'H'}

def chain2():
    return [[1,2,1], [2,3,5], [3,4,7]], {1: 'H', 2: 'O', 3: 'O', 4: 'H'}

def chain3():
    return [[1,2,1], [2,3,5], [3,4,7], [3,5, 10], [5,6,12], [3,7,3]],\
           {1: 'H', 2: 'O', 3: 'C', 4: 'H', 5: 'O', 6: 'H', 7: 'H'}

def chain_v1():
    return {1: [[2, 4],[3,5],[4,10]],
            2: [[1,3]],
            3: [[1,3]],
            4: [[2,6]]}, {1: 'N', 2: 'H', 3: 'H', 4: 'H'}

def dimentional_chain_v(chain_v):
    visited = []
    dim_structure = {}#1: np.array([0,0,0])}
    chain_v_copy = {}#chain_v.copy()
    chain_v_copy.pop(1)
    for key, item in chain_v.items():
        for i in item:
            if not i[0] in visited:
                pass
            else:
                print('It must be rotated')
            if i[1] in dim_structure.keys():
                print('cycle!')

def show_with_bonds(bonds, dpoints, annotate=False, dictionary=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in bonds:
        ax.plot([dpoints[i[0]][0], dpoints[i[1]][0]],
                [dpoints[i[0]][1], dpoints[i[1]][1]],
                [dpoints[i[0]][2], dpoints[i[1]][2]])
        # print((dpoints[i[0]][0] - dpoints[i[1]][0]) ** 2 + (dpoints[i[0]][1] - dpoints[i[1]][1]) ** 2 +
        #       (dpoints[i[0]][2] - dpoints[i[1]][2]) ** 2)
    if annotate:
        for key, item in dpoints.items():
            ax.text(item[0]+0.05, item[1]+0.05, item[2]+0.05, dictionary[key])
    plt.show()

def show_points(dpoints, structure):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for key, i in dpoints.items():
        ax.scatter(i[0], i[1], i[2])
        ax.text(i[0]+0.15, i[1]+0.15, i[2]+0.15, structure[key])
    plt.show()

def dimentional_structure_from_list(bonds, element_names):

    first_node = [0, 0, 0]
    penta_points = get_penta_points(first_node)
    init_dict = get_dictionary_coordinates(penta_points)

    # list_structure, dict_structure = chain3()
    bonds_copy = bonds.copy()
    queue = [1,]
    viewed_elements = set([])
    structure_positions = {1: np.array([0, 0, 0])}
    while len(queue) != 0:
        current_node = queue.pop()
        pops = []
        for num, i in enumerate(bonds_copy):
            if i[0] == current_node and not i[0] in viewed_elements:
                structure_positions.update({i[1]: structure_positions[i[0]]+init_dict[i[2]]})
                queue.insert(0, i[1])
                pops.append(num)
        for i in pops.__reversed__():
            bonds_copy.pop(i)
    return structure_positions, element_names

def write_to_file_coord_point(file, elements_positions, elements_names):
    with open(file, 'w') as f:
        f.write(str(len(elements_positions))+'\n')
        for key, position in elements_positions.items():
            #position = dict_structure_position[key]
            string_arr = [elements_names[key], position[0], position[1], position[2]]
            string = ' '.join([str(i) for i in string_arr])
            f.write(string+'\n')

def read_xyz_file(file):
    with open(file, 'r') as file:
        n = int(file.readline())
        names = {}
        positions = {}
        for i in range(n):
            line = file.readline().split()
            names.update({i: line[0]})
            x, y, z = [float(j) for j in line[1::]]
            positions.update({i: [x,y,z]})
    return positions, names

def dictionary_for_checked_bond_count(element_names):
    valencies = element_names.copy()
    for key, item in valencies.items():
        valencies[key] = count_bonds[item]
    return valencies

def from_coordinates_to_list_bond(structure_positions, element_names):
    element_names_copy = dictionary_for_checked_bond_count(element_names)
    structure_positions_copy = structure_positions.copy()
    bonds_list = []
    for key, item in structure_positions.items():
        current_node = structure_positions_copy.pop(key)
        for i in range(element_names_copy[key]):
            for key2, item2 in structure_positions_copy.items():
                if abs(np.linalg.norm(current_node - item2) - 1) < eps_length and element_names_copy[key2] != 0:
                    bonds_list.append([key, key2, section_num_by_coords(current_node, item2)])
                    element_names_copy[key] -= 1
                    element_names_copy[key2] -= 1
                if element_names_copy[key] == 0: break
    return bonds_list

if __name__ == '__main__':
    struct = chain3()
    structure_positions, element_names = dimentional_structure_from_list(struct[0], struct[1])
    # print(structure_positions)
    # print(from_coordinates_to_list_bond(structure_positions, element_names))

    write_to_file_coord_point('output.txt', structure_positions, element_names)
    # read_xyz_file('output.txt')
    # show_with_bonds(struct[0], structure_positions, annotate=True, dictionary=element_names)


    # show_with_bonds(list_structure, dict_structure_position)
    # show_points(dict_structure_position, dict_structure)
