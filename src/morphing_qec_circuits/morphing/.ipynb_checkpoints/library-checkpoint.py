import numpy as np
from copy import deepcopy
from .morphing_specifications import MorphingSpecifications


def get_triangular_half_shifted_hex_morphing_specifications(end_cycle_d, final_cnots_directions = {"X": [0,1], "Z": [1,0]}, extra_layer = False):
    """
    Returns a morphing specification for a planar colour code. With the default settings, the morphing circuit corresponds to the one in Figure 18 from arXiv:2604.09797.
    Setting extra_layer = True instead returns a morphing circuit shown in Figure 28(a).
    
    Arguments:
        end_cycle_d is an Int representing the static, end-cycle distance of the returned morphing circuit.
        final_cnots_directions is a dictionary with keys "X" and "Z" and items that are length-2 lists of integers. final_cnots_directions 
            specifies the direction of the final CNOTs, and the subsequent placement of the measurements, for each contracting stabiliser
            in each contracting round. 0 indicates that measurement occurs on the lower y value qubit (stabiliser gets "pushed down"), 1 indicates that the
            measurement occurs on the higher y value qubit (stabiliser gets "pushed up"). The direction of the CNOT is determined by this.
            The default value is the one that maximises the average number of measurement outcomes that go into each detector.
            To instead minimise the average number of measurement outcomes that go into each detector, use {"X": [1,0], "Z": [0,1]}.
        extra_layer is a bool that, if True, adds an extra layer to the contraction circuit to remove one of the circuit-level-decreasing error mechanisms.
    
    Returns:
        The morphing specifications for the morphing circuit.
    """
    d = end_cycle_d
    def qubit_in_bulk(coords):
        x = coords[0]
        y = coords[1]
        if y < 0:
            return False
        elif x < 0:
            return False
        elif y >= d:
            if y == d and x >= 0 and x % 6 == 3 * (d%2) and x // 3 < d:
                return True
            else:
                return False
        elif 2*x - 6*y >= 3:
            return False
        else:
            return True

    coordinates_list = [[x, y] for x in range(3*d-2) for y in range(d+1) if qubit_in_bulk([x,y])]
    new_coordinates_list = []
    
    mid_cycle_stabilisers_XZ = []# the same for both X and Z
    stabiliser_contraction_rounds = {"X": [], "Z": []}
    stabiliser_annotations = {"X": [], "Z": []}
    measurement_lists = {"X": [[],[]], "Z": [[],[]]}
    for x in range(3*d - 2):
        for y in range(d):
            if (x+y) % 2 == 0:
                stabiliser_coordinates_to_add = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_bulk([x+i,y+j])]
                if len(stabiliser_coordinates_to_add) == 6:
                    mid_cycle_stabilisers_XZ.append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                    stabiliser_contraction_rounds["X"].append([y%2])
                    stabiliser_contraction_rounds["Z"].append([(y+1)%2])
                    stabiliser_annotations["X"].append((x,y,0,(3-x)%3))
                    stabiliser_annotations["Z"].append((x,y,0,3+(3-x)%3))
                    final_pair_coords = [[x-y%2,y], [x+(y+1)%2,y+1]]
                    final_pair = [coordinates_list.index(coords) for coords in final_pair_coords]
                    measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                    measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                elif len(stabiliser_coordinates_to_add) == 4:
                    stabiliser_to_append = [coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add]
                    stabiliser_contraction_rounds["X"].append([y%2])
                    stabiliser_contraction_rounds["Z"].append([(y+1)%2])
                    stabiliser_annotations["X"].append((x,y,0,(3-x)%3))
                    stabiliser_annotations["Z"].append((x,y,0,3+(3-x)%3))
                    final_pair_coords = [[x-y%2,y], [x+(y+1)%2,y+1]]
                    check = all([coords in stabiliser_coordinates_to_add for coords in final_pair_coords])
                    if not check:
                        new_qubit_coords = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if not qubit_in_bulk([x+i,y+j])]
                        new_stabiliser = [len(coordinates_list) + len(new_coordinates_list), len(coordinates_list) + len(new_coordinates_list) + 1]
                        stabiliser_to_append += new_stabiliser
                        new_coordinates_list += new_qubit_coords
                        mid_cycle_stabilisers_XZ.append(stabiliser_to_append)
                        final_pair = []
                        for coords in final_pair_coords:
                            if coords in stabiliser_coordinates_to_add:
                                final_pair.append(coordinates_list.index(coords))
                            else:
                                final_pair.append(len(coordinates_list) + new_coordinates_list.index(coords))
                        measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                        measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                        
                        new_stabiliser_annotation_coords = [[x+i, y - (-1)**j] for i in [-1,1] for j in [0,1] if not qubit_in_bulk([x+i,y+j])]
                        new_measurement_qubit_coords = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if (not qubit_in_bulk([x+i,y+j])) and (not [x+i,y+j] in final_pair_coords)]
                        if len(new_measurement_qubit_coords) != 1 or len(new_stabiliser_annotation_coords) != 1:
                            raise Exception("uh oh there's a bug... new_measurement_qubit_coords and new_stabiliser_annotation_coords should both have length 1...")
                        mid_cycle_stabilisers_XZ.append(new_stabiliser)
                        stabiliser_contraction_rounds["X"].append([(y+1)%2])
                        stabiliser_contraction_rounds["Z"].append([y%2])
                        measurement_lists["X"][(y+1)%2].append(len(coordinates_list) + new_coordinates_list.index(new_measurement_qubit_coords[0]))
                        measurement_lists["Z"][(y)%2].append(len(coordinates_list) + new_coordinates_list.index(new_measurement_qubit_coords[0]))
                        new_x, new_y = new_stabiliser_annotation_coords[0]
                        stabiliser_annotations["X"].append((new_x,new_y,0,(3-new_x)%3))
                        stabiliser_annotations["Z"].append((new_x,new_y,0,3+(3-new_x)%3))
                    else:
                        mid_cycle_stabilisers_XZ.append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                        final_pair = [coordinates_list.index(coords) for coords in final_pair_coords]
                        measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                        measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                        
    coordinates_list += new_coordinates_list
    mid_cycle_stabilisers = {pauli: deepcopy(mid_cycle_stabilisers_XZ) for pauli in ["X", "Z"]}

    n_layers = 4 if extra_layer else 3
    gate_lists = [[[] for layer in range(n_layers)] for c_round in range(2)]

    for x in range(3*d - 2):
        for y in range(d+1):
            if x%2 == 0:
                gate_coords = [[x,y],[x-1,y]]
                check = all([coords in coordinates_list for coords in gate_coords])
                c_round = 2 if (x%6 == 2 and y%2 == 0 and extra_layer) else 0
                if check:
                    gate_lists[0][c_round].append([coordinates_list.index(gate_coords[i]) for i in range(2)])
                    gate_lists[1][c_round].append([coordinates_list.index(gate_coords[1-i]) for i in range(2)])

                gate_coords = [[x,y],[x+1,y]]
                check = all([coords in coordinates_list for coords in gate_coords])
                c_round = 1
                if check:
                    gate_lists[0][c_round].append([coordinates_list.index(gate_coords[i if y%2 == 0 else 1-i]) for i in range(2)])
                    gate_lists[1][c_round].append([coordinates_list.index(gate_coords[(1-i) if y%2 == 0 else i]) for i in range(2)])

                gate_coords = [[x,y],[x+1,y+1]]
                check = all([coords in coordinates_list for coords in gate_coords])
                c_round = 3 if extra_layer else 2
                if check:
                    flip_0 = bool((final_cnots_directions[["X","Z"][y%2]][0] + y+1)%2)
                    flip_1 = bool((final_cnots_directions[["X","Z"][(y+1)%2]][1] + y)%2)
                    gate_lists[0][c_round].append([coordinates_list.index(gate_coords[i if flip_0 else 1-i]) for i in range(2)])
                    gate_lists[1][c_round].append([coordinates_list.index(gate_coords[i if flip_1 else 1-i]) for i in range(2)])
                    
    complex_coordinates_list = [
        coordinates[0] + 1j * coordinates[1]
        for coordinates in coordinates_list
    ]

    specifications = MorphingSpecifications(
        mid_cycle_stabilisers,
        stabiliser_contraction_rounds,
        gate_lists,
        measurement_lists,
        coordinates_list=complex_coordinates_list,
        stabiliser_annotations=stabiliser_annotations,
    )
    
    return specifications

def get_unoptimised_shifted_hex_morphing_specifications(end_cycle_d, final_cnots_directions = {"X": [0,0], "Z": [0,0]}):
    """
    Returns a morphing specification for a planar colour code. With the default settings, the morphing circuit corresponds to the one in Figure 4 from arXiv:2312.08813.
    
    Arguments:
        end_cycle_d is an Int representing the static, end-cycle distance of the returned morphing circuit.
        final_cnots_directions is a dictionary with keys "X" and "Z" and items that are length-2 lists of integers. final_cnots_directions
            specifies the direction of the final CNOTs, and the subsequent placement of the measurements, for each contracting stabiliser
            in each contracting round. 0 indicates that measurement occurs on the lower y value qubit (stabiliser gets "pushed down"), 1 indicates that the
            measurement occurs on the higher y value qubit (stabiliser gets "pushed up"). The direction of the CNOT is determined by this.
            The default value is the one that maximises the average number of measurement outcomes that go into each detector (although it doesn't make
            much difference here, unlike in the half-shifted case).
            To instead minimise the average number of measurement outcomes, set {"X": [0,1], "Z": [1,0]}.
    
    Returns:
        The morphing specifications for the morphing circuit.
    """
    d = 2*end_cycle_d - 1
    def qubit_in_bulk(coords):
        x = coords[0]
        y = coords[1]
        if y < 0:
            return False
        elif x < 0:
            return False
        elif 2*x + 6*y >= 6*d - 5:
            return False
        elif 2*x - 6*y >= 3:
            return False
        else:
            return True

    coordinates_list = [[x, y] for x in range(3*d-2) for y in range(d+1) if qubit_in_bulk([x,y])]
    new_coordinates_list = []
    
    mid_cycle_stabilisers_XZ = []# the same for both X and Z
    stabiliser_contraction_rounds = {"X": [], "Z": []}
    stabiliser_annotations = {"X": [], "Z": []}
    measurement_lists = {"X": [[],[]], "Z": [[],[]]}
    for x in range(3*d - 2):
        for y in range(d):
            if (x+y) % 2 == 0:
                stabiliser_coordinates_to_add = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_bulk([x+i,y+j])]
                if len(stabiliser_coordinates_to_add) == 6:
                    mid_cycle_stabilisers_XZ.append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                    stabiliser_contraction_rounds["X"].append([y%2])
                    stabiliser_contraction_rounds["Z"].append([(y+1)%2])
                    stabiliser_annotations["X"].append((x,y,0,(3-x)%3))
                    stabiliser_annotations["Z"].append((x,y,0,3+(3-x)%3))
                    final_pair_coords = [[x+(-1)**y,y], [x+(-1)**y,y+1]]
                    final_pair = [coordinates_list.index(coords) for coords in final_pair_coords]
                    measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                    measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                elif len(stabiliser_coordinates_to_add) == 4:
                    stabiliser_to_append = [coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add]
                    stabiliser_contraction_rounds["X"].append([y%2])
                    stabiliser_contraction_rounds["Z"].append([(y+1)%2])
                    stabiliser_annotations["X"].append((x,y,0,(3-x)%3))
                    stabiliser_annotations["Z"].append((x,y,0,3+(3-x)%3))
                    final_pair_coords = [[x+(-1)**y,y], [x+(-1)**y,y+1]]
                    check = all([coords in stabiliser_coordinates_to_add for coords in final_pair_coords])
                    if not check:
                        new_qubit_coords = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if not qubit_in_bulk([x+i,y+j])]
                        new_stabiliser = [len(coordinates_list) + len(new_coordinates_list), len(coordinates_list) + len(new_coordinates_list) + 1]
                        stabiliser_to_append += new_stabiliser
                        new_coordinates_list += new_qubit_coords
                        mid_cycle_stabilisers_XZ.append(stabiliser_to_append)
                        final_pair = []
                        for coords in final_pair_coords:
                            if coords in stabiliser_coordinates_to_add:
                                final_pair.append(coordinates_list.index(coords))
                            else:
                                final_pair.append(len(coordinates_list) + new_coordinates_list.index(coords))
                        measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                        measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                        
                        new_stabiliser_annotation_coords = [[x+i, y - (-1)**j] for i in [-1,1] for j in [0,1] if not qubit_in_bulk([x+i,y+j])]
                        new_measurement_qubit_coords = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if (not qubit_in_bulk([x+i,y+j])) and (not [x+i,y+j] in final_pair_coords)]
                        if len(new_measurement_qubit_coords) != 1 or len(new_stabiliser_annotation_coords) != 1:
                            raise Exception("uh oh there's a bug... new_measurement_qubit_coords and new_stabiliser_annotation_coords should both have length 1...")
                        mid_cycle_stabilisers_XZ.append(new_stabiliser)
                        stabiliser_contraction_rounds["X"].append([(y+1)%2])
                        stabiliser_contraction_rounds["Z"].append([y%2])
                        measurement_lists["X"][(y+1)%2].append(len(coordinates_list) + new_coordinates_list.index(new_measurement_qubit_coords[0]))
                        measurement_lists["Z"][(y)%2].append(len(coordinates_list) + new_coordinates_list.index(new_measurement_qubit_coords[0]))
                        new_x, new_y = new_stabiliser_annotation_coords[0]
                        stabiliser_annotations["X"].append((new_x,new_y,0,(3-new_x)%3))
                        stabiliser_annotations["Z"].append((new_x,new_y,0,3+(3-new_x)%3))
                    else:
                        mid_cycle_stabilisers_XZ.append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                        final_pair = [coordinates_list.index(coords) for coords in final_pair_coords]
                        measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                        measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                        
    coordinates_list += new_coordinates_list
    mid_cycle_stabilisers = {pauli: deepcopy(mid_cycle_stabilisers_XZ) for pauli in ["X", "Z"]}

    n_layers = 3
    gate_lists = [[[] for layer in range(n_layers)] for c_round in range(2)]

    for x in range(3*d - 2):
        for y in range(d+1):
            if x%2 == 0:
                gate_coords = [[x,y],[x-1,y]]
                check = all([coords in coordinates_list for coords in gate_coords])
                c_round = 0
                if check:
                    gate_lists[0][c_round].append([coordinates_list.index(gate_coords[i]) for i in range(2)])
                    gate_lists[1][c_round].append([coordinates_list.index(gate_coords[1-i]) for i in range(2)])

                gate_coords = [[x+1,y],[x,y]]
                check = all([coords in coordinates_list for coords in gate_coords])
                c_round = 1
                if check:
                    gate_lists[0][c_round].append([coordinates_list.index(gate_coords[i]) for i in range(2)])
                    gate_lists[1][c_round].append([coordinates_list.index(gate_coords[1-i]) for i in range(2)])

                gate_coords = [[x+(y+1)%2,y],[x+(y+1)%2,y+1]]
                check = all([coords in coordinates_list for coords in gate_coords])
                c_round = 2
                if check:
                    flip_0 = bool((final_cnots_directions[["X","Z"][y%2]][0] + y+1)%2)
                    flip_1 = bool((final_cnots_directions[["X","Z"][(y+1)%2]][1] + y)%2)
                    gate_lists[0][c_round].append([coordinates_list.index(gate_coords[i if flip_0 else 1-i]) for i in range(2)])
                    gate_lists[1][c_round].append([coordinates_list.index(gate_coords[i if flip_1 else 1-i]) for i in range(2)])
                    
    complex_coordinates_list = [
        coordinates[0] + 1j * coordinates[1]
        for coordinates in coordinates_list
    ]

    specifications = MorphingSpecifications(
        mid_cycle_stabilisers,
        stabiliser_contraction_rounds,
        gate_lists,
        measurement_lists,
        coordinates_list=complex_coordinates_list,
        stabiliser_annotations=stabiliser_annotations,
    )
    
    return specifications

def get_triangular_half_shifted_hex_morphing_specifications_extra_qubits(
    d,
    final_cnots_directions = {"X": [0,1], "Z": [1,0]},
    extra_layer = False,
):
    """
    Returns a morphing specification for a planar colour code. Setting extra_layer = True, the morphing circuit corresponds to the one in Figure 28(c) from arXiv:2604.09797.
    Setting extra_layer = False instead returns a morphing circuit shown in Figure 28(c) but without the extra CNOT layer defined by the edges highlighted in yellow.
    
    Arguments:
        d is an Int representing the static, end-cycle distance of the returned morphing circuit, which in this case also equals the circuit distance.
        final_cnots_directions is a dictionary with keys "X" and "Z" and items that are length-2 lists of integers. final_cnots_directions 
            specifies the direction of the final CNOTs, and the subsequent placement of the measurements, for each contracting stabiliser
            in each contracting round. 0 indicates that measurement occurs on the lower y value qubit (stabiliser gets "pushed down"), 1 indicates that the
            measurement occurs on the higher y value qubit (stabiliser gets "pushed up"). The direction of the CNOT is determined by this.
            The default value is the one that maximises the average number of measurement outcomes that go into each detector.
            To instead minimise the average number of measurement outcomes that go into each detector, use {"X": [1,0], "Z": [0,1]}.
        extra_layer is a bool that, if True, adds an extra layer to the contraction circuit to remove one of the circuit-level-decreasing error mechanisms.
    
    Returns:
        The morphing specifications for the morphing circuit.
    """

    def qubit_in_bulk(coords):
        x = coords[0]
        y = coords[1]
        if y < 0:
            return False
        elif x < 0:
            return False
        elif y >= d + 2*(d//4):
            if y == d + 2*(d//4) and x >= 0 and x % 6 == 3 * (d%2) and x // 3 < d:
                return True
            else:
                return False
        elif 2*x - 6*y >= 3 - 12*((x+5.5-(d%4)//2*6)//12):
            return False
        else:
            return True

    coordinates_list = [[x, y] for x in range(3*d-2) for y in range(d + 2*(d//4) + 1) if qubit_in_bulk([x,y])]
    new_coordinates_list = []
    
    mid_cycle_stabilisers_XZ = []# the same for both X and Z
    stabiliser_contraction_rounds = {"X": [], "Z": []}
    stabiliser_annotations = {"X": [], "Z": []}
    measurement_lists = {"X": [[],[]], "Z": [[],[]]}
    n_layers = 4
    gate_lists = [[[] for layer in range(n_layers)] for c_round in range(2)]
    for x in range(3*d - 2):
        for y in range(d + 2*(d//4)):
            if (x+y) % 2 == 0:
                stabiliser_coordinates_to_add = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_bulk([x+i,y+j])]
                if len(stabiliser_coordinates_to_add) == 6:
                    mid_cycle_stabilisers_XZ.append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                    stabiliser_contraction_rounds["X"].append([y%2])
                    stabiliser_contraction_rounds["Z"].append([(y+1)%2])
                    stabiliser_annotations["X"].append((x,y,0,(3-x)%3))
                    stabiliser_annotations["Z"].append((x,y,0,3+(3-x)%3))
                    final_pair_coords = [[x-y%2,y], [x+(y+1)%2,y+1]]
                    final_pair = [coordinates_list.index(coords) for coords in final_pair_coords]
                    measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                    measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                elif len(stabiliser_coordinates_to_add) == 4:
                    stabiliser_to_append = [coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add]
                    stabiliser_contraction_rounds["X"].append([y%2])
                    stabiliser_contraction_rounds["Z"].append([(y+1)%2])
                    stabiliser_annotations["X"].append((x,y,0,(3-x)%3))
                    stabiliser_annotations["Z"].append((x,y,0,3+(3-x)%3))
                    final_pair_coords = [[x-y%2,y], [x+(y+1)%2,y+1]]
                    check = all([coords in stabiliser_coordinates_to_add for coords in final_pair_coords])
                    if not check:
                        new_qubit_coords = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if not qubit_in_bulk([x+i,y+j])]
                        new_stabiliser = [len(coordinates_list) + len(new_coordinates_list), len(coordinates_list) + len(new_coordinates_list) + 1]
                        stabiliser_to_append += new_stabiliser
                        new_coordinates_list += new_qubit_coords
                        mid_cycle_stabilisers_XZ.append(stabiliser_to_append)
                        final_pair = []
                        for coords in final_pair_coords:
                            if coords in stabiliser_coordinates_to_add:
                                final_pair.append(coordinates_list.index(coords))
                            else:
                                final_pair.append(len(coordinates_list) + new_coordinates_list.index(coords))
                        measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                        measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                        
                        new_stabiliser_annotation_coords = [[x+i, y - (-1)**j] for i in [-1,1] for j in [0,1] if not qubit_in_bulk([x+i,y+j])]
                        new_measurement_qubit_coords = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if (not qubit_in_bulk([x+i,y+j])) and (not [x+i,y+j] in final_pair_coords)]
                        
                        mid_cycle_stabilisers_XZ.append(new_stabiliser)
                        if new_qubit_coords == [[x+1,y], [x+1,y+1]]:
                            gate_lists[y%2][1].append([len(coordinates_list) + new_coordinates_list.index(new_qubit_coords[i]) for i in range(2)])
                            gate_lists[(y+1)%2][1].append([len(coordinates_list) + new_coordinates_list.index(new_qubit_coords[i]) for i in reversed(range(2))])
                            measurement_lists["X"][y%2].append(len(coordinates_list) + new_coordinates_list.index(new_qubit_coords[0]))
                            measurement_lists["Z"][(y+1)%2].append(len(coordinates_list) + new_coordinates_list.index(new_qubit_coords[0]))
                            stabiliser_contraction_rounds["X"].append([y%2])
                            stabiliser_contraction_rounds["Z"].append([(y+1)%2])
                        elif len(new_measurement_qubit_coords) > 1 or len(new_stabiliser_annotation_coords) > 1:
                            print(new_measurement_qubit_coords, new_stabiliser_annotation_coords)
                            raise Exception("uh oh there's a bug... new_measurement_qubit_coords and new_stabiliser_annotation_coords should both have length <= 2...")
                        else:
                            stabiliser_contraction_rounds["X"].append([(y+1)%2])
                            stabiliser_contraction_rounds["Z"].append([y%2])
                            measurement_lists["X"][(y+1)%2].append(len(coordinates_list) + new_coordinates_list.index(new_measurement_qubit_coords[0]))
                            measurement_lists["Z"][(y)%2].append(len(coordinates_list) + new_coordinates_list.index(new_measurement_qubit_coords[0]))
                        new_x, new_y = new_stabiliser_annotation_coords[0]
                        stabiliser_annotations["X"].append((new_x,new_y,0,(3-new_x)%3))
                        stabiliser_annotations["Z"].append((new_x,new_y,0,3+(3-new_x)%3))
                    else:
                        mid_cycle_stabilisers_XZ.append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                        final_pair = [coordinates_list.index(coords) for coords in final_pair_coords]
                        measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                        measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                        
    coordinates_list += new_coordinates_list
    mid_cycle_stabilisers = {pauli: deepcopy(mid_cycle_stabilisers_XZ) for pauli in ["X", "Z"]}

    for x in range(3*d - 2):
        for y in range(d + 2*(d//4) + 1):
            if x%2 == 0:
                gate_coords = [[x,y],[x-1,y]]
                check = all([coords in coordinates_list for coords in gate_coords])
                if y%2 == 0:
                    c_round = 1
                else:
                    c_round = 0
                if check:
                    gate_lists[0][c_round].append([coordinates_list.index(gate_coords[i]) for i in range(2)])
                    gate_lists[1][c_round].append([coordinates_list.index(gate_coords[1-i]) for i in range(2)])

                gate_coords = [[x,y],[x+1,y]]
                check = all([coords in coordinates_list for coords in gate_coords])
                if y%2 == 0:
                    if x%6 == 4 and y == (d + 2*(d//4) - 1) and extra_layer:
                        c_round = 2
                    else:
                        c_round = 0
                else:
                    if x%12 == (6 - ((d%4)//2 * 6)) and y == (x//2 - (d%4)//2):
                        c_round = 2
                    else:
                        c_round = 1
                if check:
                    gate_lists[0][c_round].append([coordinates_list.index(gate_coords[i if y%2 == 0 else 1-i]) for i in range(2)])
                    gate_lists[1][c_round].append([coordinates_list.index(gate_coords[(1-i) if y%2 == 0 else i]) for i in range(2)])

                gate_coords = [[x,y],[x+1,y+1]]
                check = all([coords in coordinates_list for coords in gate_coords])
                c_round = 3
                if check:
                    flip_0 = bool((final_cnots_directions[["X","Z"][y%2]][0] + y+1)%2)
                    flip_1 = bool((final_cnots_directions[["X","Z"][(y+1)%2]][1] + y)%2)
                    gate_lists[0][c_round].append([coordinates_list.index(gate_coords[i if flip_0 else 1-i]) for i in range(2)])
                    gate_lists[1][c_round].append([coordinates_list.index(gate_coords[i if flip_1 else 1-i]) for i in range(2)])
    
    complex_coordinates_list = [
        coordinates[0] + 1j * coordinates[1]
        for coordinates in coordinates_list
    ]

    specifications = MorphingSpecifications(
        mid_cycle_stabilisers,
        stabiliser_contraction_rounds,
        gate_lists,
        measurement_lists,
        coordinates_list=complex_coordinates_list,
        stabiliser_annotations=stabiliser_annotations,
    )
    
    return specifications
    
def get_triangular_half_shifted_hex_morphing_specifications_even_more_layers(
    d,
):
    """
    Alias for get_fault_tolerant_triangular_half_shifted_hex_morphing_specifications, kept for backwards compatibility.
    """
    return get_fault_tolerant_triangular_half_shifted_hex_morphing_specifications(d)

def get_fault_tolerant_triangular_half_shifted_hex_morphing_specifications(
    d,
):
    """
    Returns a morphing specifications for a planar colour code. This circuit was foreshadowed in the discussion of arXiv:2604.09797 and will be published
        in the near future.
    
    Arguments:
        d is an Int representing the static, end-cycle distance of the returned morphing circuit, which in this case also equals the circuit distance.

    Raises:
        NotImplementedError: The fault-tolerant circuit is not implemented for even d
        
    Returns:
        The morphing specifications for the morphing circuit.
    """
    #Things get complicated if you allow for all those previous options (extra_layer, extra_flags, even_more_layers) to be True or False, so here we're getting rid of that and
    # just coding the circuit that works: even_more_layers = True, extra_flags = True, extra_layer = False (but ig it doesn't super matter any more).
    # even_more_layers adds even more layers to protect against the bulk mid-contraction-circuit errors. Requires the final cnots to go the right way,
    # does this atm by just ignoring final_cnots_directions and just making them go the right way (I think it's the default anyway).
    final_cnots_directions = {"X": [0,1], "Z": [1,0]}
    def qubit_in_bulk(coords):
        x = coords[0]
        y = coords[1]
        if y < 0:
            return False
        elif x < 0:
            return False
        elif y >= d:
            if y == d and x >= 0 and x % 6 == 3 * (d%2) and x // 3 < d:
                return True
            else:
                return False
        elif 2*x - 6*y >= 3:
            return False
        else:
            return True

    coordinates_list = [[x, y] for x in range(3*d-2) for y in range(d+1) if qubit_in_bulk([x,y])]
    new_coordinates_list = []
    
    mid_cycle_stabilisers_XZ = []# the same for both X and Z
    stabiliser_contraction_rounds = {"X": [], "Z": []}
    stabiliser_annotations = {"X": [], "Z": []}
    measurement_lists = {"X": [[],[]], "Z": [[],[]]}
    for x in range(3*d - 2):
        for y in range(d):
            if (x+y) % 2 == 0:
                stabiliser_coordinates_to_add = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_bulk([x+i,y+j])]
                if len(stabiliser_coordinates_to_add) == 6:
                    mid_cycle_stabilisers_XZ.append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                    stabiliser_contraction_rounds["X"].append([y%2])
                    stabiliser_contraction_rounds["Z"].append([(y+1)%2])
                    stabiliser_annotations["X"].append((x,y,0,(3-x)%3))
                    stabiliser_annotations["Z"].append((x,y,0,3+(3-x)%3))
                    final_pair_coords = [[x-y%2,y], [x+(y+1)%2,y+1]]
                    final_pair = [coordinates_list.index(coords) for coords in final_pair_coords]
                    measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                    measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                elif len(stabiliser_coordinates_to_add) == 4:
                    stabiliser_to_append = [coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add]
                    stabiliser_contraction_rounds["X"].append([y%2])
                    stabiliser_contraction_rounds["Z"].append([(y+1)%2])
                    stabiliser_annotations["X"].append((x,y,0,(3-x)%3))
                    stabiliser_annotations["Z"].append((x,y,0,3+(3-x)%3))
                    final_pair_coords = [[x-y%2,y], [x+(y+1)%2,y+1]]
                    check = all([coords in stabiliser_coordinates_to_add for coords in final_pair_coords])
                    if not check:
                        new_qubit_coords = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if not qubit_in_bulk([x+i,y+j])]
                        new_stabiliser = [len(coordinates_list) + len(new_coordinates_list), len(coordinates_list) + len(new_coordinates_list) + 1]
                        stabiliser_to_append += new_stabiliser
                        new_coordinates_list += new_qubit_coords
                        mid_cycle_stabilisers_XZ.append(stabiliser_to_append)
                        final_pair = []
                        for coords in final_pair_coords:
                            if coords in stabiliser_coordinates_to_add:
                                final_pair.append(coordinates_list.index(coords))
                            else:
                                final_pair.append(len(coordinates_list) + new_coordinates_list.index(coords))
                        measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                        measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                        
                        new_stabiliser_annotation_coords = [[x+i, y - (-1)**j] for i in [-1,1] for j in [0,1] if not qubit_in_bulk([x+i,y+j])]
                        new_measurement_qubit_coords = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if (not qubit_in_bulk([x+i,y+j])) and (not [x+i,y+j] in final_pair_coords)]
                        if len(new_measurement_qubit_coords) != 1 or len(new_stabiliser_annotation_coords) != 1:
                            raise Exception("uh oh there's a bug... new_measurement_qubit_coords and new_stabiliser_annotation_coords should both have length 1...")
                        mid_cycle_stabilisers_XZ.append(new_stabiliser)
                        stabiliser_contraction_rounds["X"].append([(y+1)%2])
                        stabiliser_contraction_rounds["Z"].append([y%2])
                        measurement_lists["X"][(y+1)%2].append(len(coordinates_list) + new_coordinates_list.index(new_measurement_qubit_coords[0]))
                        measurement_lists["Z"][(y)%2].append(len(coordinates_list) + new_coordinates_list.index(new_measurement_qubit_coords[0]))
                        new_x, new_y = new_stabiliser_annotation_coords[0]
                        stabiliser_annotations["X"].append((new_x,new_y,0,(3-new_x)%3))
                        stabiliser_annotations["Z"].append((new_x,new_y,0,3+(3-new_x)%3))
                    else:
                        mid_cycle_stabilisers_XZ.append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                        final_pair = [coordinates_list.index(coords) for coords in final_pair_coords]
                        measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                        measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                        
    coordinates_list += new_coordinates_list
    mid_cycle_stabilisers = {pauli: deepcopy(mid_cycle_stabilisers_XZ) for pauli in ["X", "Z"]}

    n_layers = 6
    gate_lists = [[[] for layer in range(n_layers)] for c_round in range(2)]

    for x in range(3*d - 2):
        for y in range(d+1):
            if x%2 == 0 and y%2 == 0:
                #y%2 ==0 is the "centred" rungs
                gate_coords_list = [[[x,y+1],[x-1,y+1]],[[x+1,y+1],[x,y+1]],[[x,y],[x+1,y+1]],[[x,y+1],[x+1,y+2]],[[x,y],[x-1,y]],[[x,y],[x+1,y]]]
                if x == 0:
                    c_round_list = [0,1,2,5,4,3]
                else:
                    c_round_list = [0,1,2,2,4,3]
                for i in range(len(gate_coords_list)):
                    gate_coords = gate_coords_list[i]
                    check = all([coords in coordinates_list for coords in gate_coords])
                    if check:
                        c_round = c_round_list[i]
                        gate_lists[0][c_round].append([coordinates_list.index(gate_coords[i]) for i in range(2)])
                        gate_lists[1][c_round].append([coordinates_list.index(gate_coords[1-i]) for i in range(2)])
    
    if d%2 == 0:
        raise NotImplementedError("The fault-tolerant circuit is not implemented for even d")
    coordinates_list += [[-1, y] for y in range(d-1)]
    for pauli in ["X", "Z"]:
        mid_cycle_stabilisers[pauli] += [[coordinates_list.index([-1, 2*i]), coordinates_list.index([-1, 2*i + 1])] for i in range((d-1)//2)]
        stabiliser_contraction_rounds[pauli] += [[0] if pauli == "X" else [1] for _ in range((d-1)//2)]
        stabiliser_annotations[pauli] += [(-1, 2*i, 0, 1 if pauli == "X" else 4) for i in range((d-1)//2)]
    #trying the Z's as flags in contraction round 0, if I'm wrong then swap it around lol
    gate_lists[0][0] += [[coordinates_list.index([-1, 2*i]), coordinates_list.index([-1, 2*i + 1])] for i in range((d-1)//2)]
    measurement_lists["X"][0] += [coordinates_list.index([-1, 2*i]) for i in range((d-1)//2)]
    gate_lists[1][0] += [[coordinates_list.index([-1, 2*i + 1]), coordinates_list.index([-1, 2*i])] for i in range((d-1)//2)]
    measurement_lists["Z"][1] += [coordinates_list.index([-1, 2*i]) for i in range((d-1)//2)]
    offset = [1,1] if final_cnots_directions["Z"][0] == 0 else [0,0]
    gate_lists[0][2] += [[coordinates_list.index([offset[0], 2*i + 1 + offset[0]]), coordinates_list.index([-1, 2*i + 1])] for i in range((d-1)//2)]
    offset = [1,1] if final_cnots_directions["X"][1] == 0 else [0,0]
    gate_lists[1][2] += [[coordinates_list.index([-1, 2*i + 1]), coordinates_list.index([offset[0], 2*i + 1 + offset[0]])] for i in range((d-1)//2)]
    
    complex_coordinates_list = [
        coordinates[0] + 1j * coordinates[1]
        for coordinates in coordinates_list
    ]

    specifications = MorphingSpecifications(
        mid_cycle_stabilisers,
        stabiliser_contraction_rounds,
        gate_lists,
        measurement_lists,
        coordinates_list=complex_coordinates_list,
        stabiliser_annotations=stabiliser_annotations,
    )
    
    return specifications

def get_rectangular_centred_hex_morphing_specifications(
    boundary_displacements,
    final_cnots_directions = {"X": [0,1], "Z": [1,0]},
):
    """
    Returns a morphing specifications for a planar colour code, as defined in Fig 30(a) of arXiv:2604.09797.
    
    Arguments:
        boundary_displaceents is either a length-4 List[Int] or an Int.
            If boundary_displacements is a length-4 List[Int] then it corresponds to the absolute value of the y-intercepts of Table 7 in arXiv:2604.09797.
            If boundary_displacements is an Int then this integer is interpreted as the circuit-level distance of the desired code. It is automatically
                converted to the boundary_displacements vector corresponding to the morphing circuit that achieves circuit-level distance d with the fewest
                number of physical qubits, as shown in Table 7 in arXiv:2604.09797. Currently only supported for d = 2, 3, 4, 5, 6, 7, or 8.
        final_cnots_directions is a dictionary with keys "X" and "Z" and items that are length-2 lists of integers. final_cnots_directions 
            specifies the direction of the final CNOTs, and the subsequent placement of the measurements, for each contracting stabiliser
            in each contracting round. 0 indicates that measurement occurs on the lower y value qubit (stabiliser gets "pushed down"), 1 indicates that the
            measurement occurs on the higher y value qubit (stabiliser gets "pushed up"). The direction of the CNOT is determined by this.
            The default value is the one that maximises the average number of measurement outcomes that go into each detector.
            To instead minimise the average number of measurement outcomes that go into each detector, use {"X": [1,0], "Z": [0,1]}.

    Raises:
        NotImplementedError: Converting this value of d to boundary_displacements is not yet supported
        ValueError: Every element of boundary_displacements should be positive
        ValueError: boundaries should be large enough that the (0,0) qubit is in the bulk
    
    Returns:
        The morphing specifications for the morphing circuit.
    """
    if type(boundary_displacements) == int:
        d = boundary_displacements
        d_to_displacements = {
            2: [3,5,8,6],
            3: [7,5,8,10],
            4: [11,9,12,14],
            5: [15,14,16,17],
            6: [17,18,20,19],
            7: [19,20,24,23],
            8: [23,22,26,27],
        }
        if not d in d_to_displacements.keys():
            raise NotImplementedError('Converting this value of d to boundary_displacements is not yet supported')
        boundary_displacements = d_to_displacements[d]
    
    for el in boundary_displacements:
        if el <= 0:
            raise ValueError("Every element of boundary_displacements should be positive")
    shifts = np.array(boundary_displacements) * 1/(2*np.sqrt(3)) + 1/(4*np.sqrt(3))
    shifts[2] = -shifts[2]
    shifts[3] = -shifts[3]

    def grid_to_hex_coords(coords):
        return (coords[0]/2, (coords[1]-0.5)*1.5/np.sqrt(3) + (-1)**(coords[0] + coords[1] + 1) * 0.25/np.sqrt(3))

    def qubit_in_bulk(coords):
        # here I use real (as opposed to integer) coordinates to define the boundaries so that the geometry of what's happening is more obvious.
        # Gradient of all boundaries should be plus or minus 2/np.sqrt(3)
        x, y = grid_to_hex_coords(coords)
        if y > -2*x/np.sqrt(3) + shifts[0]:
            return False
        elif y > 2*x/np.sqrt(3) + shifts[1]:
            return False
        elif y < -2*x/np.sqrt(3) + shifts[2]:
            return False
        elif y < 2*x/np.sqrt(3) + shifts[3]:
            return False
        else:
            return True
    if not qubit_in_bulk((0,0)):
        raise ValueError("boundaries should be large enough that the (0,0) qubit is in the bulk")
    
    top_x = (shifts[0] - shifts[1]) * np.sqrt(3)/4
    bottom_x = (shifts[2] - shifts[3]) * np.sqrt(3)/4
    right_y = (shifts[0] + shifts[3]) / 2
    left_y = (shifts[1] + shifts[2]) / 2
    
    def qubit_in_Z_phase(coords):
        if qubit_in_bulk(coords):
            return False
        x, y = grid_to_hex_coords(coords)
        if y >= right_y - 10**(-10) and x >= top_x - 10**(-10):
            return True
        elif y <= left_y - 10**(-10) and x <= bottom_x - 10**(-10):
            return True
        else:
            return False
    
    def qubit_in_X_phase(coords):
        return not (qubit_in_bulk(coords) or qubit_in_Z_phase(coords))

    bulk_coordinates_list = []
    box_size = 0
    keep_going = True
    while keep_going:
        keep_going = False
        for x in range(-box_size, box_size + 1):
            for y in range(-box_size, box_size + 1):
                if (abs(x) == box_size) or (abs(y) == box_size):
                    if qubit_in_bulk((x,y)):
                        keep_going = True
                        bulk_coordinates_list.append([x, y])
        box_size += 1
        if box_size > 100:
            raise Exception("bloody hell box_size has got pretty big, what on earth is going on?")
    
    for x in range(-box_size, box_size + 1):
        for y in range(-box_size, box_size + 1):
            if (x+y) % 2 == 0:
                stabiliser_coordinates_to_add = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_bulk([x+i,y+j])]
                if len(stabiliser_coordinates_to_add) == 1:
                    X_phase_coordinates = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_X_phase([x+i,y+j])]
                    Z_phase_coordinates = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_Z_phase([x+i,y+j])]
                    if len(X_phase_coordinates) > 0 and len(Z_phase_coordinates) > 0:
                        continue
                    if stabiliser_coordinates_to_add[0] in bulk_coordinates_list:
                        bulk_coordinates_list.remove(stabiliser_coordinates_to_add[0])

    def qubit_in_bulk(coords):
        return coords in bulk_coordinates_list

    coordinates_list = deepcopy(bulk_coordinates_list)
    new_coordinates_list = []
    
    mid_cycle_stabilisers = {pauli: [] for pauli in ["X", "Z"]}
    stabiliser_contraction_rounds = {"X": [], "Z": []}
    stabiliser_annotations = {"X": [], "Z": []}
    measurement_lists = {"X": [[],[]], "Z": [[],[]]}
    n_layers = 3
    gate_sets = [[set() for layer in range(n_layers)] for c_round in range(2)]
    possible_pointless_qubits = []
    necessary_qubits = []
    for x in range(-box_size, box_size + 1):
        for y in range(-box_size, box_size + 1):
            if (x+y) % 2 == 0:
                stabiliser_coordinates_to_add = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_bulk([x+i,y+j])]
                if len(stabiliser_coordinates_to_add) == 6:
                    for pauli in ["X", "Z"]:
                        mid_cycle_stabilisers[pauli].append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                    stabiliser_contraction_rounds["X"].append([y%2])
                    stabiliser_contraction_rounds["Z"].append([(y+1)%2])
                    stabiliser_annotations["X"].append((x,y,0,(3-x)%3))
                    stabiliser_annotations["Z"].append((x,y,0,3+(3-x)%3))
                    final_pair_coords = [[x,y], [x,y+1]]
                    final_pair = [coordinates_list.index(coords) for coords in final_pair_coords]
                    necessary_qubits += final_pair
                    
                    gate_sets[y%2][0].add((coordinates_list.index([x,y]),coordinates_list.index([x+(-1)**(y%2),y])))
                    gate_sets[y%2][0].add((coordinates_list.index([x,y+1]),coordinates_list.index([x+(-1)**(y%2),y+1])))
                    gate_sets[y%2][1].add((coordinates_list.index([x,y]),coordinates_list.index([x-(-1)**(y%2),y])))
                    gate_sets[y%2][1].add((coordinates_list.index([x,y+1]),coordinates_list.index([x-(-1)**(y%2),y+1])))
                    gate_sets[y%2][2].add((final_pair[final_cnots_directions["X"][y%2]],final_pair[(final_cnots_directions["X"][y%2]+1)%2]))
                    measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])

                    gate_sets[(y+1)%2][0].add((coordinates_list.index([x+(-1)**(y%2),y]),coordinates_list.index([x,y])))
                    gate_sets[(y+1)%2][0].add((coordinates_list.index([x+(-1)**(y%2),y+1]),coordinates_list.index([x,y+1])))
                    gate_sets[(y+1)%2][1].add((coordinates_list.index([x-(-1)**(y%2),y]),coordinates_list.index([x,y])))
                    gate_sets[(y+1)%2][1].add((coordinates_list.index([x-(-1)**(y%2),y+1]),coordinates_list.index([x,y+1])))
                    gate_sets[(y+1)%2][2].add((final_pair[(final_cnots_directions["Z"][(y+1)%2]+1)%2],final_pair[final_cnots_directions["Z"][(y+1)%2]]))
                    measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                elif len(stabiliser_coordinates_to_add) == 1:
                    X_phase_coordinates = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_X_phase([x+i,y+j])]
                    Z_phase_coordinates = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_Z_phase([x+i,y+j])]
                    if len(X_phase_coordinates) > 0 and len(Z_phase_coordinates) > 0:
                        continue
                    elif len(X_phase_coordinates) > 0:
                        pauli = "X"
                        c_round = y%2
                    elif len(Z_phase_coordinates) > 0:
                        pauli = "Z"
                        c_round = (y+1)%2
                    else:
                        raise Exception("err something is wrong lol, this stabiliser has weight less than 6 but isn't in either the X or the Z phase")
                    qubit = coordinates_list.index(stabiliser_coordinates_to_add[0])
                    mid_cycle_stabilisers[pauli].append([qubit])
                    stabiliser_contraction_rounds[pauli].append([c_round])
                    stabiliser_annotations[pauli].append((x, y, 0, (3-x)%3 + (0 if pauli == "X" else 3)))
                    measurement_lists[pauli][c_round].append(qubit)
                    possible_pointless_qubits.append(qubit)
                elif len(stabiliser_coordinates_to_add) > 1:
                    X_phase_coordinates = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_X_phase([x+i,y+j])]
                    Z_phase_coordinates = [[x+i,y+j] for i in [-1,0,1] for j in [0,1] if qubit_in_Z_phase([x+i,y+j])]
                    if len(X_phase_coordinates) > 0 and len(Z_phase_coordinates) > 0:
                        continue
                    elif len(X_phase_coordinates) > 0:
                        pauli = "X"
                    elif len(Z_phase_coordinates) > 0:
                        pauli = "Z"
                    else:
                        raise Exception("err something is wrong lol, this stabiliser has weight less than 6 but isn't in either the X or the Z phase")
                    stabiliser_to_append = [coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add]
                    c_round = y%2 if pauli == "X" else (y+1)%2
                    stabiliser_contraction_rounds[pauli].append([c_round])
                    stabiliser_annotations[pauli].append((x,y,0,(3-x)%3 + (3 if pauli == "Z" else 0)))
                    final_pair_coords = [[x,y], [x,y+1]]
                    single_row_stab = any([len([[x+i,y+j] for i in [-1,0,1] if qubit_in_bulk([x+i,y+j])]) == 0 for j in [0,1]])
                    check = all([coords in stabiliser_coordinates_to_add for coords in final_pair_coords]) or single_row_stab
                    if not check:
                        new_qubit_coords = [coords for coords in final_pair_coords if not qubit_in_bulk(coords)]
                        if len(new_qubit_coords) != 1:
                            print(stabiliser_coordinates_to_add)
                            print(new_qubit_coords)
                            raise Exception("again something has gone horribly wrong, new_qubit_coords should have exactly one qubit coords in it")
                        new_stabiliser = [len(coordinates_list) + len(new_coordinates_list)]
                        stabiliser_to_append += new_stabiliser
                        new_coordinates_list += new_qubit_coords
                        mid_cycle_stabilisers[pauli].append(stabiliser_to_append)
                        def relative_coords_to_index(i,j):
                            if qubit_in_bulk([x+i,y+j]):
                                return coordinates_list.index([x+i,y+j])
                            elif [x+i, y+j] == new_qubit_coords[0]:
                                return len(coordinates_list) + new_coordinates_list.index([x+i, y+j])
                            else:
                                return None
                        final_pair = [relative_coords_to_index(0,0), relative_coords_to_index(0,1)]
                        necessary_qubits += final_pair
                        gate_pairs = [[[[0,0],[(-1)**(y%2),0]],[[0,1],[(-1)**(y%2),1]]],
                                      [[[0,0],[-(-1)**(y%2),0]],[[0,1],[-(-1)**(y%2),1]]]]
                        if pauli == "X":
                            for layer in range(2):
                                for gate_pair in gate_pairs[layer]:
                                    gate_index_list = tuple([relative_coords_to_index(*gate_pair[k]) for k in range(2)])
                                    if not any([gate_index_list[k] == None for k in range(2)]):
                                        gate_sets[y%2][layer].add(gate_index_list)
                            gate_sets[y%2][2].add((final_pair[final_cnots_directions["X"][y%2]],final_pair[(final_cnots_directions["X"][y%2]+1)%2]))
                            measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                        else:
                            for layer in range(2):
                                for gate_pair in gate_pairs[layer]:
                                    gate_index_list = tuple([relative_coords_to_index(*gate_pair[k]) for k in reversed(range(2))])
                                    if not any([gate_index_list[k] == None for k in range(2)]):
                                        gate_sets[(y+1)%2][layer].add(gate_index_list)
                            gate_sets[(y+1)%2][2].add((final_pair[(final_cnots_directions["Z"][(y+1)%2]+1)%2],final_pair[final_cnots_directions["Z"][(y+1)%2]]))
                            measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                        
                        new_x, new_y = deepcopy(new_qubit_coords[0])
                        if new_x > 0:
                            new_x += 1
                        else:
                            new_x += -1
                        if new_y > 0:
                            new_y += 1
                        new_measurement_qubit_coords = deepcopy(new_qubit_coords)
                        c_round = y%2 if pauli == "Z" else (y+1)%2
                        mid_cycle_stabilisers[pauli].append(new_stabiliser)
                        stabiliser_contraction_rounds[pauli].append([c_round])
                        measurement_lists[pauli][c_round].append(len(coordinates_list) + new_coordinates_list.index(new_measurement_qubit_coords[0]))
                        stabiliser_annotations[pauli].append((new_x,new_y,0,(3-new_x)%3 + (3 if pauli == "Z" else 0)))
                    elif single_row_stab:
                        mid_cycle_stabilisers[pauli].append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                        def relative_coords_to_index(i,j):
                            if qubit_in_bulk([x+i,y+j]):
                                return coordinates_list.index([x+i,y+j])
                            else:
                                return None
                        final_qubit = [coordinates_list.index(coords) for coords in final_pair_coords if qubit_in_bulk(coords)][0]
                        necessary_qubits.append(final_qubit)
                        gate_pairs = [[[[0,0],[(-1)**(y%2),0]],[[0,1],[(-1)**(y%2),1]]],
                                      [[[0,0],[-(-1)**(y%2),0]],[[0,1],[-(-1)**(y%2),1]]],
                                     ]
                        if pauli == "X":
                            for layer in range(2):
                                for gate_pair in gate_pairs[layer]:
                                    gate_index_list = tuple([relative_coords_to_index(*gate_pair[k]) for k in range(2)])
                                    if not any([gate_index_list[k] == None for k in range(2)]):
                                        gate_sets[y%2][layer].add(gate_index_list)
                        else:
                            for layer in range(2):
                                for gate_pair in gate_pairs[layer]:
                                    gate_index_list = tuple([relative_coords_to_index(*gate_pair[k]) for k in reversed(range(2))])
                                    if not any([gate_index_list[k] == None for k in range(2)]):
                                        gate_sets[(y+1)%2][layer].add(gate_index_list)
                        c_round = y%2 if pauli == "X" else (y+1)%2
                        measurement_lists[pauli][c_round].append(final_qubit)
                    else:
                        mid_cycle_stabilisers[pauli].append([coordinates_list.index(qubit_coords) for qubit_coords in stabiliser_coordinates_to_add])
                        final_pair = [coordinates_list.index(coords) for coords in final_pair_coords]
                        necessary_qubits += final_pair
                        def relative_coords_to_index(i,j):
                            if qubit_in_bulk([x+i,y+j]):
                                return coordinates_list.index([x+i,y+j])
                            else:
                                return None
                        gate_pairs = [[[[0,0],[(-1)**(y%2),0]],[[0,1],[(-1)**(y%2),1]]],
                                      [[[0,0],[-(-1)**(y%2),0]],[[0,1],[-(-1)**(y%2),1]]]]
                        if pauli == "X":
                            for layer in range(2):
                                for gate_pair in gate_pairs[layer]:
                                    gate_index_list = tuple([relative_coords_to_index(*gate_pair[k]) for k in range(2)])
                                    if not any([gate_index_list[k] == None for k in range(2)]):
                                        gate_sets[y%2][layer].add(gate_index_list)
                            gate_sets[y%2][2].add((final_pair[final_cnots_directions["X"][y%2]],final_pair[(final_cnots_directions["X"][y%2]+1)%2]))
                            measurement_lists["X"][y%2].append(final_pair[final_cnots_directions["X"][y%2]])
                        else:
                            for layer in range(2):
                                for gate_pair in gate_pairs[layer]:
                                    gate_index_list = tuple([relative_coords_to_index(*gate_pair[k]) for k in reversed(range(2))])
                                    if not any([gate_index_list[k] == None for k in range(2)]):
                                        gate_sets[(y+1)%2][layer].add(gate_index_list)
                            gate_sets[(y+1)%2][2].add((final_pair[(final_cnots_directions["Z"][(y+1)%2]+1)%2],final_pair[final_cnots_directions["Z"][(y+1)%2]]))
                            measurement_lists["Z"][(y+1)%2].append(final_pair[final_cnots_directions["Z"][(y+1)%2]])
                        
    coordinates_list += new_coordinates_list
    gate_lists = [[[list(gate) for gate in gate_sets[i][j]] for j in range(len(gate_sets[i]))] for i in range(len(gate_sets))]
    
    complex_coordinates_list = [
        coordinates[0] + 1j * coordinates[1]
        for coordinates in coordinates_list
    ]



    specifications = MorphingSpecifications(
        mid_cycle_stabilisers,
        stabiliser_contraction_rounds,
        gate_lists,
        measurement_lists,
        coordinates_list=complex_coordinates_list,
        stabiliser_annotations=stabiliser_annotations,
    )
    
    return specifications

def get_3D_toric_code_morphing_specifications(d):
    '''
    Returns morphing specifications for the 3D toric code, as defined by Figure 21 of arXiv:2604.09797.
    The toric boundary conditions are chosen to improve the distance of the end-cycle code.
    The returned morphing specifications has a non-trivial meta-check attribute that identifies the product of each cube of Z-checks
        as the identity. This means that stability experiments run using this will have non-local redundancies as their observables
        and these local redundancies as "meta-check" detectors.

    Arguments:
        d is an even integer that is the distance of the end-cycle code.

    Raises:
        NotImplementedError: d must be divisible by two!

    Returns:
        A MorphingSpecifications object for the 3D toric code morphing circuit.
    '''
    if d%2 != 0:
        raise NotImplementedError("d must be divisible by two!")

    L = d//2

    def is_vertex(coords):
        return all([np.abs(coords[i] - round(coords[i])) < 10**(-10) for i in range(3)])

    def is_edge(coords):
        two_integers = sum([np.abs(coords[i] - round(coords[i])) < 10**(-10) for i in range(3)]) == 2
        one_half_integer = sum([np.abs(coords[i] - 0.5 - round(coords[i] - 0.5)) < 10**(-10) for i in range(3)]) == 1
        return two_integers and one_half_integer

    def is_face(coords):
        one_integer = sum([np.abs(coords[i] - round(coords[i])) < 10**(-10) for i in range(3)]) == 1
        two_half_integers = sum([np.abs(coords[i] - 0.5 - round(coords[i] - 0.5)) < 10**(-10) for i in range(3)]) == 2
        return one_integer and two_half_integers

    def face_orientation(coords):
        if is_face(coords):
            return [np.abs(coords[i] - round(coords[i])) < 10**(-10) for i in range(3)].index(True)

    def is_cube(coords):
        return all([np.abs(coords[i] - 0.5 - round(coords[i] - 0.5)) < 10**(-10) for i in range(3)])

    big_coordinates_list = [[X/2, Y/2, Z/2] for Z in range(2*L) for Y in range(2*L) for X in range(8*L)]
    all_coordinates_list = deepcopy(big_coordinates_list)
    periodicity_vectors = np.array([[4*L,0,0],[3*L,L,0],[3*L,0,L]])


    coordinates_list = [coords for coords in all_coordinates_list if is_edge(coords)]
    vertices_list = [coords for coords in all_coordinates_list if is_vertex(coords)]
    faces_list = [coords for coords in all_coordinates_list if is_face(coords)]
    cubes_list = [coords for coords in all_coordinates_list if is_cube(coords)]


    def add_coordinates(coords1, coords2):
        coords = [coords1[i] + coords2[i] for i in range(3)]
        coords[0] += L * (coords[2] // L)
        coords[0] += L * (coords[1] // L)
        return [coords[0] % (4*L), coords[1] % L, coords[2] % L]
    
    def coordinates_to_index(coords):
        return next(i for i, v in enumerate(coordinates_list) if np.all(np.isclose(v, coords)))

    def coordinates_to_Z_stabiliser_index(coords):
        return next(i for i, v in enumerate(faces_list) if np.all(np.isclose(v, coords)))
    
    new_coordinates_list = []
    
    mid_cycle_stabilisers = {pauli: [] for pauli in ["X", "Z"]}
    stabiliser_contraction_rounds = {"X": [], "Z": []}
    stabiliser_annotations = {"X": [], "Z": []}
    measurement_lists = {"X": [[] for c_round in range(6)], "Z": [[] for c_round in range(6)]}
    n_layers = 3
    gate_sets = [[set() for layer in range(n_layers)] for c_round in range(6)]
    meta_checks = {"X": [], "Z": []}
    
    X_relative_coordinates = []
    to_append = [0,0,0]
    for i in range(3):
        for j in range(2):
            to_append[i] += (-1)**j / 2
            X_relative_coordinates.append(to_append)
            to_append = [0,0,0]

    Z_relative_coordinates = [[] for _ in range(3)]
    to_append = [0,0,0]
    for i in range(3):
        for j in range(2):
            to_append[i] += (-1)**j / 2
            for k in range(3):
                if k != i:
                    Z_relative_coordinates[k].append(to_append)
            to_append = [0,0,0]

    for vertex in vertices_list:
        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        stabiliser_to_add = [coordinates_to_index(add_coordinates(vertex, coords)) for coords in X_relative_coordinates]
        mid_cycle_stabilisers["X"].append(stabiliser_to_add)
        c_rounds = [3*(round(x+y+z)%2), 3*(round(x+y+z+1)%2)+1, 3*(round(x+y+z)%2)+2]
        stabiliser_annotations["X"].append((x,y,z,0))
        gate_relative_indices = [[[[0,3],[1,4]],[[5,0],[2,1]],[[5,2]]],
                                 [[[3,0],[2,5]],[[4,3],[1,2]],[[1,4]]],
                                 [[[4,1],[5,2]],[[3,4],[0,5]],[[0,3]]],
                                ]
        for i in range(len(c_rounds)):
            for layer in range(3):
                for gate_pair in gate_relative_indices[i][layer]:
                    gate_index_list = tuple([stabiliser_to_add[gate_pair[k]] for k in range(2)])
                    gate_sets[c_rounds[i]][layer].add(gate_index_list)
            measurement_lists["X"][c_rounds[i]].append(stabiliser_to_add[gate_relative_indices[i][2][0][0]])
        c_rounds.sort()
        stabiliser_contraction_rounds["X"].append(c_rounds)
        
    for face in faces_list:
        orientation = face_orientation(face)
        x = face[0]
        y = face[1]
        z = face[2]
        stabiliser_to_add = [coordinates_to_index(add_coordinates(face, coords)) for coords in Z_relative_coordinates[orientation]]
        mid_cycle_stabilisers["Z"].append(stabiliser_to_add)
        c_rounds = [3*(round(x+y+z)%2), 3*(round(x+y+z+1)%2)+1, 3*(round(x+y+z)%2)+2]
        stabiliser_annotations["Z"].append((x,y,z,3))
        last_gate_relative_indices = [[[],[0,2],[1,3]],
                                      [[1,3],[],[2,0]],
                                      [[2,0],[3,1],[]],
                                     ]
        for i in range(len(c_rounds)):
            if i != orientation:
                gate_index_list = tuple([stabiliser_to_add[last_gate_relative_indices[i][orientation][k]] for k in range(2)])
                gate_sets[c_rounds[i]][2].add(gate_index_list)
                measurement_lists["Z"][c_rounds[i]].append(stabiliser_to_add[last_gate_relative_indices[i][orientation][1]])
        c_rounds.pop(orientation)
        c_rounds.sort()
        stabiliser_contraction_rounds["Z"].append(c_rounds)

    for cube in cubes_list:
        #X_relative_coordinates happens to work also for the cubes
        meta_check_to_add = [coordinates_to_Z_stabiliser_index(add_coordinates(cube, coords)) for coords in X_relative_coordinates]
        meta_checks["Z"].append(meta_check_to_add)
    
    coordinates_list += new_coordinates_list
    
    complex_coordinates_list = [
        coordinates[0] + 1j * coordinates[1] + (L)*2.5j * coordinates[2]
        for coordinates in coordinates_list
    ]

    gate_lists = [[[list(gate) for gate in gate_sets[i][j]] for j in range(len(gate_sets[i]))] for i in range(len(gate_sets))]

    specifications = MorphingSpecifications(
        mid_cycle_stabilisers,
        stabiliser_contraction_rounds,
        gate_lists,
        measurement_lists,
        coordinates_list=complex_coordinates_list,
        stabiliser_annotations=stabiliser_annotations,
        meta_checks = meta_checks,
    )
    
    return specifications
