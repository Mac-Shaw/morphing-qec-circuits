import numpy as np
from itertools import product, permutations, combinations
from copy import deepcopy
import galois  # for matrix calculations over a finite field
from ..misc import (
hnf_form,
hnf_iterator,
_standard_permutation_fast,
nkd_params,
qec_circuit_to_XZ_memory_experiment,
)
from ..morphing import MorphingSpecifications
from collections import defaultdict
from surface_sim.models import CircuitNoiseModel
import stim

F2 = galois.GF(2)

"""
Note to self: I have also recently written a get_standard_XZ_memory_circuit in the BBParams class that should be (trivially) adapted and brought over here once the rest of this is tidied up.

TO DO 14/5: Organise the morphing specifications search better. Write something that also automatically computes the distance and circuit-level distance, etc. DONE and tested. :)))
TO DO 7/8: up to going through and culling methods from BBParams. But I do want to ensure backwards compatibility. Also, I'm a bit puzzled as to why the standard morphing method here is so convoluted. But also, wtf who cares just leave it. That said, I would like to double check that my old code works. So perhaps have a look at the old Example Notebook and see if it runs as intended.

Also need to go through the other files to check they're ok. DONE!! :)
Need to fix my error bars in the plotting notebook with the bug that I found (about the scaling of the error bars whoops) DONE!! literally nothing changed. :)
And, go through the General Nonesense notebook for things to put in a script somewhere so that it is more organised. DONE!! :)
Then, format the whole thing as a package so that I don't have to keep fucking around with file placement etc.
And finally, write a new Example notebook that contains scripts to reproduce my plots from the paper.
And extra finally just check that I did action whatever response I get from Marc about uniform circuit level noise or whatever the heck is going on with that.
"""

class Abelian2BGAParams:
    """An Abelian Two-Block Group Algebra (2BGA) code.

    Represents an Abelian 2BGA code in both presentations: its group presentation (group, a_list, b_list) and its lattice presentation (lattice_signature, periodicity_vectors, n_copies) - see Appendix C2 of https://arxiv.org/pdf/2604.09797.
        group is a List[Int] representing an Abelian group in terms of a product of cyclic groups with orders specified by the list.
        a_list is a List[List[Int]] where each element of the list represents an element in the group.
        b_list is a List[List[Int]] where each element of the list represents an element in the group.
        lattice_signature is a length-two Tuple[Int] representing the dimensionality of the lattice presentation of the code.
        periodicity_vectors is an integer-valued numpy array storing a square matrix in Hermite Normal Form, representing the lattice generators of the lattice presentation of the code.
        n_copies is an Int representing the number of copies in the lattice presentation of the code.

    Given the group presentation of a code with a fixed ordering of elements a_list and b_list, the lattice presentation is unique; but the lattice presentation may change under permutations of elements in a_list and b_list. Conversely, two Abelian 2BGA codes are equivalent and have the same ordering of elements in a_list and b_list iff their lattice presentations are the same.
    
    Some simple relations between the two presentations:
        lattice_signature[0] = len(a_list) - 1,
        lattice_signature[1] = len(b_list) - 1,
        product(group) = np.linalg.det(periodicity_vectors) * n_copies.
    """
    
    def __init__(self, *args, input_format = "group", **kwargs):
        """Creates an Abelian Two-Block Group Algebra (2BGA) code, and computes both the group and lattice presentations of the code.

        Args:
            input_format: either a string ("group" or "vectors"/"lattice"), or an Int (0 or 1, respectively), specifying whether the Abelian 2BGA code is being given in its group or lattice presentation.
        """
        if input_format in ["group", 0]:
            self.from_group(*args, **kwargs)
        elif input_format in ["vectors", "lattice", 1]:
            self.from_vectors(*args, **kwargs)
        else:
            raise ValueError(f"input_format must be group, vectors or lattice but was instead {input_format}")

    def from_group(self, group, a_list, b_list, nkd = (None, None, None)):
        """Creates an Abelian 2BGA code from its group presentation, and computes its lattice presentation. The algorithm is not at all efficient, but is quick enough for reasonably sized codes.

        Args:
            group is a List[Int] representing an Abelian group in terms of a product of cyclic groups with orders specified by the list.
            a_list is a List[List[Int]] where each element of the list represents an element in the group.
            b_list is a List[List[Int]] where each element of the list represents an element in the group.
        
        Raises:
            ValueError: The nkd parameters provided do not match the number of physical qubits in the code!
            ValueError: An element of a_list does not have the same length as the group!
            ValueError: An element of b_list does not have the same length as the group!
        """
        for a_el in a_list:
            if len(a_el) != len(group):
                raise ValueError(f"An element of a_list {a_el} does not have the same length as the group!")
        for b_el in b_list:
            if len(b_el) != len(group):
                raise ValueError(f"An element of b_list {b_el} does not have the same length as the group!")
            
        self.group = group
        self.a_list = [[a_list[i][j] % self.group[j] for j in range(len(group))] for i in range(len(a_list))]
        self.b_list = [[b_list[i][j] % self.group[j] for j in range(len(group))] for i in range(len(b_list))]
        n_qubits = 2
        for i in self.group:
            n_qubits *= i
        if nkd[0] == None:
            self.nkd = (n_qubits, None, None)
        else:
            if nkd[0] != n_qubits:
                raise ValueError("The nkd parameters provided do not match the number of physical qubits in the code!")
            else:
                self.nkd = (nkd[0], nkd[1], nkd[2])
        
        if len(a_list) <= 1 and len(b_list) <= 1:
            self.periodicity_vectors = np.zeros((0,0)).astype(int)
            self.lattice_signature = [len(a_list) - 1, len(b_list) - 1]
            self.n_copies = 1
            for order in group:
                self.n_copies *= order
        else:
            a_els = [[(self.a_list[j][i] - self.a_list[0][i]) % group[i] for i in range(len(group))] for j in range(1, len(a_list))]
            b_els = [[(self.b_list[j][i] - self.b_list[0][i]) % group[i] for i in range(len(group))] for j in range(1, len(b_list))]
            lattice_signature = [len(a_list)-1, len(b_list)-1]
            els = a_els + b_els
            def order(el):
                to_return = 1
                for i in range(len(group)):
                    to_return = np.lcm(to_return, 1 if el[i] == 0 else np.lcm(el[i], group[i])//el[i])
                return to_return
            def group_mult(el1, el2):
                return [(el1[i] + el2[i]) % group[i] for i in range(len(group))]
            def group_power(el, n):
                return [(el[i] * n) % group[i] for i in range(len(group))]
                
            def thicken_index(index, thickened_shape):
                return [(index%np.prod(thickened_shape[k:]).astype(int))//np.prod(thickened_shape[(k+1):]).astype(int) for k in range(len(thickened_shape))]
    
            orders = [order(el) for el in els]
            periodicity_vectors = [[orders[0]] + [0 for _ in range(len(els) - 1)]]
            existing_subgroup = [group_power(els[0], i) for i in range(orders[0])]
            suborders = [orders[0]]
            for i in range(1, len(els)):
                new_periodicity_vector = [0 for _ in range(len(els))]
                new_periodicity_vector[i] = orders[i]
                new_suborder = orders[i]
                for j in range(1,orders[i]):
                    to_check = group_power(els[i], j)
                    if to_check in existing_subgroup:
                        flattened_index = existing_subgroup.index(to_check)
                        new_periodicity_vector = thicken_index(flattened_index, suborders)
                        new_periodicity_vector.append(-j)
                        new_periodicity_vector += [0 for _ in range(len(suborders)+1,len(els))]
                        new_suborder = j
                        break
                periodicity_vectors.append(new_periodicity_vector)
                suborders.append(new_suborder)
                existing_subgroup = [group_mult(existing_subgroup[k], group_power(els[i], j)) for k in range(len(existing_subgroup)) for j in range(new_suborder)]
            if np.abs(np.linalg.det(periodicity_vectors)) < 10**(-10):
                raise Exception("oh jeez something's gone horribly wrong here, the periodicity vectors array I generated has determinant zero...")
            hnf = hnf_form(np.array(periodicity_vectors, dtype = int))
            self.periodicity_vectors = hnf
            self.lattice_signature = lattice_signature
            self.n_copies = np.prod(group).astype(int) // round(np.linalg.det(self.periodicity_vectors))

    def from_vectors(self, lattice_signature, periodicity_vectors, n_copies = 1, nkd = (None, None, None)):
        """Creates an Abelian 2BGA code from its lattice presentation, and computes its group presentation. The algorithm is not at all efficient, but is quick enough for reasonably sized codes.

        Args:
            lattice_signature is a length-two Tuple[Int] representing the dimensionality of the lattice presentation of the code.
            periodicity_vectors is an integer-valued numpy array storing a square matrix in Hermite Normal Form, representing the lattice generators of the lattice presentation of the code.
            n_copies is an Int representing the number of copies in the lattice presentation of the code.
        
        Raises:
            ValueError: The nkd parameters provided do not match the number of physical qubits in the code!
            ValueError: periodicity_vectors is not full rank!
            ValueError: lattice_signature does not match the dimension of the periodicity_vectors!
            ValueError: Inputted periodicity vectors are invalid as they result in multiple elements of a_list being the same.
            ValueError: Inputted periodicity vectors are invalid as they result in multiple elements of b_list being the same.
        """
        if lattice_signature[0] == -1 and lattice_signature[1] == -1:
            self.periodicity_vectors = periodicity_vectors
            self.lattice_signature = lattice_signature
            self.n_copies = n_copies
            self.group = [n_copies]
            self.a_list = []
            self.b_list = []
            n_qubits = 2*n_copies
            if nkd[0] == None:
                self.nkd = (n_qubits, None, None)
            else:
                if nkd[0] != n_qubits:
                    raise ValueError("The nkd parameters provided do not match the number of physical qubits in the code!")
                else:
                    self.nkd = (nkd[0], nkd[1], nkd[2])
        elif lattice_signature[0] == -1 and lattice_signature[1] == 0:
            self.periodicity_vectors = periodicity_vectors
            self.lattice_signature = lattice_signature
            self.n_copies = n_copies
            self.group = [n_copies]
            self.a_list = []
            self.b_list = [[0]]
            n_qubits = 2*n_copies
            if nkd[0] == None:
                self.nkd = (n_qubits, None, None)
            else:
                if nkd[0] != n_qubits:
                    raise ValueError("The nkd parameters provided do not match the number of physical qubits in the code!")
                else:
                    self.nkd = (nkd[0], nkd[1], nkd[2])
        elif lattice_signature[0] == 0 and lattice_signature[1] == -1:
            self.periodicity_vectors = periodicity_vectors
            self.lattice_signature = lattice_signature
            self.n_copies = n_copies
            self.group = [n_copies]
            self.a_list = [[0]]
            self.b_list = []
            n_qubits = 2*n_copies
            if nkd[0] == None:
                self.nkd = (n_qubits, None, None)
            else:
                if nkd[0] != n_qubits:
                    raise ValueError("The nkd parameters provided do not match the number of physical qubits in the code!")
                else:
                    self.nkd = (nkd[0], nkd[1], nkd[2])
        elif lattice_signature[0] == 0 and lattice_signature[1] == 0:
            self.periodicity_vectors = periodicity_vectors
            self.lattice_signature = lattice_signature
            self.n_copies = n_copies
            self.group = [n_copies]
            self.a_list = [[0]]
            self.b_list = [[0]]
            n_qubits = 2*n_copies
            if nkd[0] == None:
                self.nkd = (n_qubits, None, None)
            else:
                if nkd[0] != n_qubits:
                    raise ValueError("The nkd parameters provided do not match the number of physical qubits in the code!")
                else:
                    self.nkd = (nkd[0], nkd[1], nkd[2])
        else:
            if np.linalg.matrix_rank(np.array(periodicity_vectors)) != np.array(periodicity_vectors).shape[0]:
                raise ValueError(f"periodicity_vectors {periodicity_vectors} is not full rank!")
            hnf = hnf_form(np.array(periodicity_vectors, dtype = int))
            n_gens = hnf.shape[0]
            if sum(lattice_signature) != n_gens:
                if lattice_signature[0] == -1 or lattice_signature[1] == -1:
                    if sum(lattice_signature) != n_gens - 1:
                        raise ValueError("lattice_signature does not match the dimension of the periodicity_vectors!")
                else:
                    raise ValueError("lattice_signature does not match the dimension of the periodicity_vectors!")
            suborders = [hnf[i,i] for i in range(n_gens)]
            self.periodicity_vectors = hnf
            self.lattice_signature = lattice_signature
            self.n_copies = n_copies
            
            """
            We have 1 = x0^hnf[0,0] = x0^hnf[1,0] * x1^hnf[1,1] = x0^hnf[2,0] * x1^hnf[2,1] * x2^hnf[2,2] = ...
            Group elements here are represented by a list [i0, i1, ...] = x0^i0 * x1^i1 * ...
            We can get the whole group by iterating each ij in range(hnf[j,j]).            
            """
            def group_power(el, n):
                """
                [i0, i1, ...]^n = x0^(i0*n) x1^(i1*n) x2^(i2*n) ... = 
                """
                to_return = np.zeros((n_gens)).astype(int)
                for i in reversed(range(n_gens)):
                    temp = to_return[i] + el[i]*n
                    to_return[i] = temp % hnf[i,i]
                    if i > 0:
                        to_return[:i] += - hnf[i,:i] * (temp // hnf[i,i])
                return list(to_return)
    
            def group_mult(el1, el2):
                temp = [el1[i] + el2[i] for i in range(n_gens)]
                return group_power(temp, 1)
    
            def is_identity(el):
                is_zero = True
                for i in range(n_gens):
                    if el[i] != 0:
                        is_zero = False
                        break
                return is_zero
                
            def order(el):
                if is_identity(el):
                    return 1
                else:
                    n = 1
                    while True:
                        if is_identity(group_power(el, n)):
                            return n
                        n += 1
                        
            def order_in_quotient(el, subgroup):
                if el in subgroup:
                    return 1
                else:
                    n = 1
                    while True:
                        if group_power(el, n) in subgroup:
                            return n
                        n += 1
            
            def thicken_index(index, thickened_shape):
                return [int((index%np.prod(thickened_shape[k:]).astype(int))//np.prod(thickened_shape[(k+1):]).astype(int)) for k in range(len(thickened_shape))]
    
            whole_group = [list(el) for el in product(*[range(suborder) for suborder in suborders])]
            orders = np.array([order(g) for g in whole_group], dtype=int)
            max_order = np.max(orders)
            generating_orders = [int(max_order)]
            max_element = thicken_index(np.argmax(orders), suborders)
            
            existing_subgroup = [group_power(max_element, n) for n in range(max_order)]
            orders_in_quotient = np.array([order_in_quotient(g, existing_subgroup) for g in whole_group], dtype=int)
            max_order_in_quotient = np.max(orders_in_quotient)
            num_loops = 0
            while max_order_in_quotient > 1:
                for_comparison = np.array([orders[i] if orders[i] == orders_in_quotient[i] else 0 for i in range(len(orders))], dtype = int)
                next_order = np.max(for_comparison)
                assert next_order == max_order_in_quotient
                next_element = thicken_index(np.argmax(for_comparison), suborders)
                existing_subgroup = [group_mult(existing_subgroup[k], group_power(next_element, j)) for k in range(len(existing_subgroup)) for j in range(next_order)]
                generating_orders.append(int(next_order))
                orders_in_quotient = np.array([order_in_quotient(g, existing_subgroup) for g in whole_group], dtype=int)
                max_order_in_quotient = np.max(orders_in_quotient)
                num_loops += 1
            if lattice_signature[0] == -1:
                self.a_list = []
            else:
                self.a_list = [[0 for _ in range(len(generating_orders))]]
                if n_copies > 1:
                    self.a_list[0].append(0)
            if lattice_signature[1] == -1:
                self.b_list = []
            else:
                self.b_list = [[0 for _ in range(len(generating_orders))]]
                if n_copies > 1:
                    self.b_list[0].append(0)
            for i in range(max(lattice_signature[0],0)):
                el = [0 for _ in range(n_gens)]
                el[i] = 1
                flattened_index = existing_subgroup.index(group_power(el,1))
                to_append = thicken_index(flattened_index, generating_orders)
                if n_copies > 1:
                    to_append.append(0)
                self.a_list.append(to_append)
            for i in range(max(lattice_signature[0],0), n_gens):
                el = [0 for _ in range(n_gens)]
                el[i] = 1
                flattened_index = existing_subgroup.index(group_power(el,1))
                to_append = thicken_index(flattened_index, generating_orders)
                if n_copies > 1:
                    to_append.append(0)
                self.b_list.append(to_append)
            
            if n_copies > 1:
                generating_orders.append(int(n_copies))
            self.group = generating_orders
            n_qubits = 2
            for i in self.group:
                n_qubits *= i
            if nkd[0] == None:
                self.nkd = (n_qubits, None, None)
            else:
                if nkd[0] != n_qubits:
                    raise ValueError("The nkd parameters provided do not match the number of physical qubits in the code!")
                else:
                    self.nkd = (nkd[0], nkd[1], nkd[2])
    
            a_set = set(map(tuple, self.a_list))
            b_set = set(map(tuple, self.b_list))
            if len(a_set) != len(self.a_list):
                raise ValueError("Inputted periodicity vectors are invalid as they result in multiple elements of a_list being the same")
            if len(b_set) != len(self.b_list):
                raise ValueError("Inputted periodicity vectors are invalid as they result in multiple elements of b_list being the same")
    
    def list_equivalent_params(self, swap_AB = True):
        """Lists all Abelian2BGA codes that are equivalent to the current Abelian 2BGA code but have different lattice presentations.

        Args:
            swap_AB is a bool that determines whether or not to also list codes that are equivalent up to swapping A and B.
        
        Returns:
            A list of equivalent Abelian 2BGA codes.
        """
        equivalent_codes = []
        for permuted_a_list in permutations(self.a_list):
            for permuted_b_list in permutations(self.b_list):
                to_append = Abelian2BGAParams(self.group, permuted_a_list, permuted_b_list, nkd = self.nkd, input_format = "group")
                if not to_append in equivalent_codes:
                    equivalent_codes.append(to_append)
                if swap_AB:
                    to_append = Abelian2BGAParams(self.group, permuted_b_list, permuted_a_list, nkd = self.nkd, input_format = "group")
                    to_append.nkd = deepcopy(self.nkd)
                    if not to_append in equivalent_codes:
                        equivalent_codes.append(to_append)
        return equivalent_codes

    def __eq__(self, other):
        """
        Checks if the lattice presentation of two Abelian2BGA codes are identical. Note that this does not check permutations of a_list or b_list.
        
        Returns:
            True if the lattice presentations are the same, False otherwise.
        """
        test2 = self.n_copies == other.n_copies
        test3 = self.lattice_signature == other.lattice_signature
        if test2 and test3:
            test1 = np.all(self.periodicity_vectors == other.periodicity_vectors)
            return bool(test1 and test2 and test3)
        else:
            return False

    def __ne__(self, other):
        """
        Checks if the lattice presentation of two Abelian2BGA codes are not identical. Note that this does not check permutations of a_list or b_list.
        
        Returns:
            False if the lattice presentations are the same, True otherwise.
        """
        return not (self == other)

    #def to_lattice_tuple(self):
    #    return tuple(self.lattice_signature + [self.n_copies] + list(self.periodicity_vectors.flatten()))

    def is_equivalent(self, other):
        """Checks if two Abelian2BGA codes are equivalent by iterating through permutations of elements in a_list and b_list of self and checking if any of these permuted codes have identical lattice presentation to other.
        
        Returns:
            True if the codes are the equivalent up to permutations, False otherwise.
        """
        test1a = self.lattice_signature == other.lattice_signature
        test1b = self.lattice_signature == list(reversed(other.lattice_signature))
        test2 = self.n_copies == other.n_copies
        test3 = np.prod(np.diag(self.periodicity_vectors)) == np.prod(np.diag(other.periodicity_vectors))
        if (test1a or test1b) and test2 and test3:
            other_to_check = other.periodicity_vectors
            for a_id_index in range(max(1,len(self.a_list))):
                for b_id_index in range(max(1,len(self.b_list))):
                    shifted_a_list = [self.a_list[(i - a_id_index) % len(self.a_list)] for i in range(len(self.a_list))]
                    shifted_b_list = [self.b_list[(i - b_id_index) % len(self.b_list)] for i in range(len(self.b_list))]
                    if test1a:
                        start = Abelian2BGAParams(self.group, shifted_a_list, shifted_b_list, input_format = "group").periodicity_vectors
                        for permutation_a in permutations(range(len(self.a_list)-1)):
                            for permutation_b in permutations(range(len(self.a_list)-1, len(self.a_list) + len(self.b_list)-2)):
                                permutation = [x for x in permutation_a] + [x for x in permutation_b]
                                new_periodicity_vectors = start[:,permutation]
                                to_check = hnf_form(new_periodicity_vectors)
                                if np.all(other_to_check == to_check):
                                    return True
                    if test1b:
                        start = Abelian2BGAParams(self.group, shifted_b_list, shifted_a_list, input_format = "group").periodicity_vectors
                        for permutation_b in permutations(range(len(self.b_list)-1)):
                            for permutation_a in permutations(range(len(self.b_list)-1, len(self.b_list) + len(self.a_list)-2)):
                                permutation = [x for x in permutation_b] + [x for x in permutation_a]
                                new_periodicity_vectors = start[:,permutation]
                                to_check = hnf_form(new_periodicity_vectors)
                                if np.all(other_to_check == to_check):
                                    return True
            return False
        else:
            return False

    def __lt__(self, other):
        """
        Determines if an Abelian 2BGA code is less than another based on a lexicographic ordering. The lexicographic ordering considers a flattened list containing
            1. lattice_signature,
            2. n_copies, and
            3. flattened(periodicity_vectors)
        of each Abelian 2BGA code in that order, and says that the code C1 < C2 if there exists an index i such that self_list[i] < other_list[i] and self_list[j] == other_list[j] for all j < i.

        Returns:
            True if self < other under the lexicographic ordering, otherwise False.
        """
        self_list = list(self.lattice_signature) + [self.n_copies] + list(self.periodicity_vectors.flatten())
        other_list = list(other.lattice_signature) + [other.n_copies] + list(other.periodicity_vectors.flatten())
        for i in range(len(self_list)):
            if self_list[i] < other_list[i]:
                return True
            elif self_list[i] > other_list[i]:
                return False
        return False

    def __le__(self, other):
        """
        Determines if an Abelian 2BGA code is less than or equal to another based on a lexicographic ordering of the parameters in each code's lattice presentation. The lexicographic ordering considers a flattened list containing
            1. lattice_signature,
            2. n_copies, and
            3. flattened(periodicity_vectors)
        of each Abelian 2BGA code in that order, and says that the code C1 < C2 if there exists an index i such that self_list[i] < other_list[i] and self_list[j] == other_list[j] for all j < i.

        Returns:
            True if self < other or self == other under the lexicographic ordering, otherwise False.
        """
        if self == other:
            return True
        else:
            return self < other

    def is_standard_permutation(self, swap_AB = True):
        """
        Determines if the permutation of the elements in A and B for the Abelian 2BGA code is the standard permutation. The standard permutation is the permutation of the elements in A and B that gives the minimal Abelian 2BGA code under the lexicographic ordering of the lattice presentation.

        Arguments:
            swap_AB is a bool that determines whether to also swap A and B when determining the standard permutation.
        
        Returns:
            True if the code is in its standard permutation, otherwise False.
        """
        return self == self.get_standard_permutation(swap_AB = swap_AB)

    def get_standard_permutation(self, swap_AB = True):
        """
        Returns the standard permutation of the Abelian 2BGA code. The standard permutation is the permutation of the elements in A and B that gives the minimal Abelian 2BGA code under the lexicographic ordering of the lattice presentation.

        Arguments:
            swap_AB is a bool that determines whether to also swap A and B when determining the standard permutation.
        
        Returns:
            The standard permutation of the Abelian 2BGA code.
        """
        if not swap_AB:
            minimal_periodicity_vectors = _standard_permutation_fast(
                np.array(self.lattice_signature, dtype = np.int64),
                np.array(self.periodicity_vectors, dtype = np.int64),
            )
            return Abelian2BGAParams(
                self.lattice_signature,
                minimal_periodicity_vectors,
                n_copies = self.n_copies,
                nkd = self.nkd,
                input_format = "vectors",
            )
        else:
            if self.lattice_signature[0] < self.lattice_signature[1]:
                minimal_periodicity_vectors = _standard_permutation_fast(
                    np.array(self.lattice_signature, dtype = np.int64),
                    np.array(self.periodicity_vectors, dtype = np.int64),
                )
                return Abelian2BGAParams(
                    self.lattice_signature,
                    minimal_periodicity_vectors,
                    n_copies = self.n_copies,
                    nkd = self.nkd,
                    input_format = "vectors",
                )
            elif self.lattice_signature[1] < self.lattice_signature[0]:
                swap_permutation = [i for i in range(self.lattice_signature[0], sum(self.lattice_signature))]
                swap_permutation += [i for i in range(self.lattice_signature[0])]
                swapped_lattice_signature = [self.lattice_signature[1], self.lattice_signature[0]]
                swapped_periodicity_vectors = self.periodicity_vectors[:, swap_permutation]
                minimal_periodicity_vectors = _standard_permutation_fast(
                    np.array(swapped_lattice_signature, dtype = np.int64),
                    np.array(swapped_periodicity_vectors, dtype = np.int64),
                )
                return Abelian2BGAParams(
                    self.lattice_signature,
                    minimal_periodicity_vectors,
                    n_copies = self.n_copies,
                    nkd = self.nkd,
                    input_format = "vectors",
                )
            else:
                minimal_periodicity_vectors = _standard_permutation_fast(
                    np.array(self.lattice_signature, dtype = np.int64),
                    np.array(self.periodicity_vectors, dtype = np.int64),
                )
                swap_permutation = [i for i in range(self.lattice_signature[0], sum(self.lattice_signature))]
                swap_permutation += [i for i in range(self.lattice_signature[0])]
                swapped_lattice_signature = [self.lattice_signature[1], self.lattice_signature[0]]
                swapped_periodicity_vectors = self.periodicity_vectors[:, swap_permutation]
                swapped_minimal_periodicity_vectors = _standard_permutation_fast(
                    np.array(swapped_lattice_signature, dtype = np.int64),
                    np.array(swapped_periodicity_vectors, dtype = np.int64),
                )
                for i in range(swapped_minimal_periodicity_vectors.shape[0]):
                    for j in range(swapped_minimal_periodicity_vectors.shape[1]):
                        if swapped_minimal_periodicity_vectors[i,j] < minimal_periodicity_vectors[i,j]:
                            return Abelian2BGAParams(
                                swapped_lattice_signature,
                                swapped_minimal_periodicity_vectors,
                                n_copies = self.n_copies,
                                nkd = self.nkd,
                                input_format = "vectors",
                            )
                        elif swapped_minimal_periodicity_vectors[i,j] > minimal_periodicity_vectors[i,j]:
                            return Abelian2BGAParams(
                                self.lattice_signature,
                                minimal_periodicity_vectors,
                                n_copies = self.n_copies,
                                nkd = self.nkd,
                                input_format = "vectors",
                            )
                return Abelian2BGAParams(
                    self.lattice_signature,
                    minimal_periodicity_vectors,
                    n_copies = self.n_copies,
                    nkd = self.nkd,
                    input_format = "vectors",
                )
    
    def __repr__(self):
        """
        Returns a string that specifies the group and lattice presentations of the Abelian 2BGA code.
        """
        cls = self.__class__.__name__
        return f'{cls}(group={self.group!r}, a_list={self.a_list!r}, b_list={self.b_list!r}; lattice_signature={self.lattice_signature!r}, periodicity_vectors={self.periodicity_vectors!r}, n_copies={self.n_copies!r}; [[n,k,d]]=[[{self.nkd[0]},{"?" if self.nkd[1] == None else self.nkd[1]},{"?" if self.nkd[2] == None else self.nkd[2]}]])'

    def check_consistent_parameters(self):
        """
        Checks if the group presentation of the Abelian 2BGA code is valid. Raises a ValueError if something is wrong.
        """
        for order in self.group:
            if order < 1:
                raise ValueError("Group must be a list of positive integers!")
        for a in self.a_list:
            if len(a) != len(self.group):
                raise ValueError("Each element of a_list should have the same length as group!")
        for b in self.b_list:
            if len(b) != len(self.group):
                raise ValueError("Each element of b_list should have the same length as group!")
        a_set = set([tuple(a_el) for a_el in self.a_list])
        if len(a_set) != len(self.a_list):
            raise ValueError("Each element of a_list must be unique!")
        b_set = set([tuple(b_el) for b_el in self.b_list])
        if len(b_set) != len(self.b_list):
            raise ValueError("Each element of b_list must be unique!")

    def nk_params(self):
        """
        Returns the number of physical qubits (n) and number of logical qubits (k) in the code.
        """
        hs = self.parity_check_matrices()
        n_qubits = hs["X"].shape[1]
        if 0 in hs["X"].shape:
            n_qubits = hs["Z"].shape[1]
        n_logicals = (
            n_qubits - np.linalg.matrix_rank(hs["X"]) - np.linalg.matrix_rank(hs["Z"])
        )
        self.nkd = (n_qubits, n_logicals, None)
        return (n_qubits, n_logicals)
    
    def nkd_params(self, recalculate = False, **kwargs):
        """
        Returns the number of physical qubits (n), number of logical qubits (k), and distance (d) of the code, and sets self.nkd to the returned value.

        Arguments:
            recalculate is a bool that if set to True will mean that the distance is recalculated, rather than using a stored value
                of the distance.
            verbose is a bool that determines whether to output text during the distance calculation.

        Returns:
            A length-3 tuple of Ints (n,k,d).
        """
        if recalculate or (None in self.nkd):
            if self.n_copies > 1:
                one_copy = Abelian2BGAParams(self.lattice_signature, self.periodicity_vectors, n_copies = 1, input_format = "vectors")
                hs = one_copy.parity_check_matrices()
                temp_nkd_params = nkd_params(hs, check_x_and_z = False, **kwargs)
                to_return = (temp_nkd_params[0] * self.n_copies, temp_nkd_params[1] * self.n_copies, temp_nkd_params[2])
                self.nkd = to_return
                return to_return
            else:
                hs = self.parity_check_matrices()
                to_return = nkd_params(hs, check_x_and_z = False, **kwargs)
                self.nkd = to_return
                return to_return
        else:
            return self.nkd

    def homomorphism_action_iterator(self, homomorphism_image):
        """
        Iterates over possible injective group homomorphisms from self.group to the Abelian group represeted by homomorphism_image.

        Arguments:
            homomorphism_image is a List[Int] representing the index of each cyclic group whose product gives the Abelian group that is the image of the homomorphism. homomorphism_image can only be [2], [3], [4], or [2,2]; otherwise a NotImplementedError will be raised.

        Raises:
            NotImplementedError: homomorphism_action_iterator currently only supports homomorphism_image of [2], [3], [4], or [2,2]

        Yields:
            homomorphism_image, a List[List[Int]] that represents the image of each generator of self.group under the homomorphism.
        """
        if homomorphism_image == [2]:
            homomorphism_action = [[0] for _ in range(len(self.group))]
            while True:
                for i in reversed(range(len(self.group))):
                    if homomorphism_action[i][0] == 1:
                        homomorphism_action[i][0] = 0
                    else:
                        homomorphism_action[i][0] = 1
                        break
                else:
                    return 
                if self.valid_homomorphism(homomorphism_image, homomorphism_action, verbose = False):
                    yield homomorphism_action
                    
        elif homomorphism_image == [3]:
            homomorphism_action = [[0] for _ in range(len(self.group))]
            while True:
                for i in reversed(range(len(self.group))):
                    if homomorphism_action[i][0] == 2:
                        homomorphism_action[i][0] = 0
                    else:
                        homomorphism_action[i][0] += 1
                        break
                else:
                    return 
                if not [1] in homomorphism_action:
                    continue
                if [2] in homomorphism_action:
                    if homomorphism_action.index([2]) < homomorphism_action.index([1]):
                        continue
                if self.valid_homomorphism(homomorphism_image, homomorphism_action):
                    yield homomorphism_action
        elif homomorphism_image == [4]:
            homomorphism_action = [[0] for _ in range(len(self.group))]
            while True:
                for i in reversed(range(len(self.group))):
                    if homomorphism_action[i][0] == 3:
                        homomorphism_action[i][0] = 0
                    else:
                        homomorphism_action[i][0] += 1
                        break
                else:
                    return 
                if not [1] in homomorphism_action:
                    continue
                if [3] in homomorphism_action:
                    if homomorphism_action.index([3]) < homomorphism_action.index([1]):
                        continue
                if self.valid_homomorphism(homomorphism_image, homomorphism_action):
                    yield homomorphism_action
        elif homomorphism_image == [2,2]:
            homomorphism_action = [[0,0] for _ in range(len(self.group))]
            while True:
                for i in reversed(range(len(self.group))):
                    for j in reversed(range(2)):
                        if homomorphism_action[i][j] == 1:
                            homomorphism_action[i][j] = 0
                        else:
                            homomorphism_action[i][j] += 1
                            break
                    else:
                        continue
                    break
                else:
                    return 
                elements_in_image = [x in homomorphism_action for x in [[0,1],[1,0],[1,1]]]
                if elements_in_image.count(True) < 2:
                    continue
                if not [0,1] in homomorphism_action:
                    continue
                if [1,0] in homomorphism_action:
                    if homomorphism_action.index([1,0]) < homomorphism_action.index([0,1]):
                        continue
                if [1,1] in homomorphism_action:
                    if not [1,0] in homomorphism_action:
                        continue
                    if homomorphism_action.index([1,1]) < homomorphism_action.index([1,0]):
                        continue
                if self.valid_homomorphism(homomorphism_image, homomorphism_action):
                    yield homomorphism_action
        else:
            raise NotImplementedError("homomorphism_action_iterator currently only supports homomorphism_image of [2], [3], [4], or [2,2]")
        
    def valid_homomorphism(self, homomorphism_image, homomorphism_action, verbose = False):
        """
        Checks if a homomorphism from self.group to homomorphism_image is valid.

        Arguments:
            homomorphism_image is a List[Int] representing the index of each cyclic group whose product gives the Abelian group that is the image of the homomorphism.
            homomorphism_action is a List[List[Int]] that represents the image of each generator of self.group under the homomorphism.
            verbose is a bool that specifies whether or not to print text explaining why an invalid homomorphism is not valid.

        Returns:
            True if the homomorphism is valid, otherwise False.
        """
        group = self.group
        if len(homomorphism_action) != len(group):
            if verbose:
                print("homomorphism is invalid because len(homomorphism_action) is not equal to len(group)")
            return False
        
        for output in homomorphism_action:
            if len(output) != len(homomorphism_image):
                if verbose:
                    print("homomorphism is invalid because one of its outputs has a different length to len(homomorphism_image)")
                return False

        for i in range(len(homomorphism_action)):
            for j in range(len(homomorphism_action[i])):
                if (homomorphism_action[i][j] * group[i]) % homomorphism_image[j] != 0:
                    if verbose:
                        print(f"homomorphism is invalid because the image of group generator {i} when raised to the power of its index does not return identity under the homomorphism")
                    return False

        return True
    
    def valid_morphing_parameters(self, contraction_tree, homomorphism_image, homomorphism_action, b_shift, method = "graph", verbose = False):
        """
        Checks if a given Abelian 2BGA code specification and homomorphism define a valid homomorphism-based morphing circuit (https://arxiv.org/pdf/2604.09797.
        
        This is sensitive to multiplications of a_list and b_list by a group generator, and to permutations of a_list and b_list (in other words, it does not attempt to find a shift/permutation that is valid for the given contraction tree).

        Arguments:
            contraction_tree is a List[List[List[Str]]] representing the contraction tree of the X stabiliser s(X,1) where 1 is the identity in self.group.
                Each contraction_tree[layer] corresponding to a different contraction gate layer in the contraction circuit.
                Each element contraction_tree[layer][i] is itself a length-2 list [el1, el2] corresponding to an edge in the contraction tree, where el1 gets contracted onto el2.
                The elements el1 and el2 themselves are strings that are one of "A0", "B0", "A1", "B1", etc.
                The contraction trees of other stabilisers are given by the definition of a homomorphism-based morphing circuit.
            homomorphism_image is a List[Int] representing the index of each cyclic group whose product gives the Abelian group that is the image of the homomorphism.
            homomorphism_action is a List[List[Int]] that represents the image of each generator of self.group under the homomorphism.
            b_shift is a list corresponding to an element in homomorphism_image that is multiplied to the Z stabiliser labels when determining their contraction round.
            method is a Str, either "contraction" or "graph" (default). The "contraction" method attempts to contract the s(X,1) stabiliser. If it encounters a problem during this (e.g. overlapping CNOT gates, or the stabiliser does not end up on its root qubit), it returns False; otherwise, it returns True. The "graph" method constructs the "simultaneously contractable graph" from the contraction tree, and checks that the given homomorphism condition is consistent with the simultaneously contractable graph. The "contraction" and "graph" methods coincide if the Abelian2BGA code has minimal overlap (i.e. all overlapping X checks intersect on only one qubit, all overlapping pairs of X and Z checks intersect on only two qubits); if the code does not have minimal overlap, the "graph" method might return False even if the morphing parameters are valid.
            verbose is a bool that specifies whether or not to print text explaining why an invalid set of morphing parameters is not valid.

        Raises:
            ValueError: method must be contraction or graph!
        
        Returns:
            True if the morphing parameters are valid, otherwise False. Setting method = "graph" sometimes gives false negatives if the Abelian 2BGA code does not have minimal overlap as defined above.
        """
        self.check_consistent_parameters()

        if method == "contraction":
            """
            First checking the parallelisability of the CNOT gates.
            Due to translational symmetry/ZX-duality, only need to check that one representative from each coset of left qubits participates in (no more than)
            one CNOT per layer.
            """
            identity_el = [0 for _ in range(len(homomorphism_image))]
            elements_dict = dict({"A" + str(i): self.a_list[i] for i in range(len(self.a_list))},
                                 **{"B" + str(i): self.b_list[i] for i in range(len(self.b_list))})

            def inverse(el):
                return [(-el[i]) % self.group[i] for i in range(len(self.group))]
        
            def homomorphism_function(el, shift = identity_el):
                out = [_ for _ in range(len(homomorphism_image))]
                for i in range(len(el)):
                    for j in range(len(out)):
                        out[j] += homomorphism_action[i][j] * el[i]
                return [(- shift[j] + out[j]) % homomorphism_image[j] for j in range(len(homomorphism_image))]
            
            support = {"L": [[self.a_list[i][j] % self.group[j] for j in range(len(self.group))] for i in range(len(self.a_list))],
                       "R": [[self.b_list[i][j] % self.group[j] for j in range(len(self.group))] for i in range(len(self.b_list))]}
            for layer in contraction_tree:
                #checking parallelisation first
                for coset_rep in product(*[range(homomorphism_image[i]) for i in range(len(homomorphism_image))]):
                    CNOT_participation = []
                    for gate in layer:
                        #CNOTs from X checks first
                        control_LR = {"A": "L", "B": "R"}[gate[1][0]]
                        control_element = elements_dict[gate[1]]
                        target_LR = {"A": "L", "B": "R"}[gate[0][0]]
                        target_element = elements_dict[gate[0]]
                        if control_LR == "L" and homomorphism_function(control_element) == list(coset_rep):
                            to_check = ["control",
                                        [(target_element[i] - control_element[i]) % self.group[i] for i in range(len(self.group))],
                                        target_LR,
                                       ]
                            if CNOT_participation == []:
                                CNOT_participation = to_check
                            elif CNOT_participation != to_check:
                                if verbose:
                                    print(f"two CNOT gates overlapped on the same qubit q(L,{control_element}) in layer {contraction_tree.index(layer)}")
                                return False
                        if target_LR == "L" and homomorphism_function(target_element) == list(coset_rep):
                            to_check = ["target",
                                        [(- target_element[i] + control_element[i]) % self.group[i] for i in range(len(self.group))],
                                        control_LR,
                                       ]
                            if CNOT_participation == []:
                                CNOT_participation = to_check
                            elif CNOT_participation != to_check:
                                if verbose:
                                    print(f"two CNOT gates overlapped on the same qubit q(L,{control_element}) in layer {contraction_tree.index(layer)}")
                                return False
                        #CNOTs from Z checks second
                        control_LR = {"A": "R", "B": "L"}[gate[0][0]]
                        control_element = inverse(elements_dict[gate[0]])
                        target_LR = {"A": "R", "B": "L"}[gate[1][0]]
                        target_element = inverse(elements_dict[gate[1]])
                        if control_LR == "L" and homomorphism_function(control_element, shift = b_shift) == list(coset_rep):
                            to_check = ["control",
                                        [(target_element[i] - control_element[i]) % self.group[i] for i in range(len(self.group))],
                                        target_LR,
                                       ]
                            if CNOT_participation == []:
                                CNOT_participation = to_check
                            elif CNOT_participation != to_check:
                                if verbose:
                                    print(f"two CNOT gates overlapped on the same qubit q(L,{control_element}) in layer {contraction_tree.index(layer)}")
                                return False
                        if target_LR == "L" and homomorphism_function(target_element, shift = b_shift) == list(coset_rep):
                            to_check = ["target",
                                        [(- target_element[i] + control_element[i]) % self.group[i] for i in range(len(self.group))],
                                        control_LR,
                                       ]
                            if CNOT_participation == []:
                                CNOT_participation = to_check
                            elif CNOT_participation != to_check:
                                if verbose:
                                    print(f"two CNOT gates overlapped on the same qubit q(L,{control_element}) in layer {contraction_tree.index(layer)}")
                                return False
                #now the contraction of the X stabiliser
                cnots_already_done = []
                for gate in layer:
                    #first the CNOTs from X stabiliser contraction
                    control_LR = {"A": "L", "B": "R"}[gate[1][0]]
                    control_element = elements_dict[gate[1]]
                    temp = deepcopy(support[control_LR])
                    for qubit in temp:
                        if homomorphism_function(qubit) == homomorphism_function(control_element):
                            target_qubit = [(elements_dict[gate[0]][i] - control_element[i] + qubit[i]) % self.group[i] for i in range(len(self.group))]
                            target_LR = {"A": "L", "B": "R"}[gate[0][0]]
                            to_check = [qubit, control_LR, target_qubit, target_LR]
                            if to_check in cnots_already_done:
                                continue
                            cnots_already_done.append(to_check)
                            if target_qubit in support[target_LR]:
                                support[target_LR].remove(target_qubit)
                            else:
                                support[target_LR].append(target_qubit)
                    #then the CNOTs from Z stabiliser contraction.
                    control_element = elements_dict[gate[0]]
                    control_LR = {"A": "R", "B": "L"}[gate[0][0]]
                    temp = deepcopy(support[control_LR])
                    for qubit in temp:
                        if homomorphism_function(qubit) == homomorphism_function(inverse(control_element), shift = b_shift):
                            target_qubit = [(- elements_dict[gate[1]][i] + control_element[i] + qubit[i]) % self.group[i] for i in range(len(self.group))]
                            target_LR = {"A": "R", "B": "L"}[gate[1][0]]
                            to_check = [qubit, control_LR, target_qubit, target_LR]
                            if to_check in cnots_already_done:
                                continue
                            cnots_already_done.append(to_check)
                            if target_qubit in support[target_LR]:
                                support[target_LR].remove(target_qubit)
                            else:
                                support[target_LR].append(target_qubit)
                        
            root = contraction_tree[-1][0][1]
            f_of_root = homomorphism_function(elements_dict[root])
            #check X measurements
            temp = deepcopy(support[{"A": "L", "B": "R"}[root[0]]])
            count = 0
            for qubit in temp:
                if homomorphism_function(qubit) == f_of_root:
                    count += 1
                    support[{"A": "L", "B": "R"}[root[0]]].remove(qubit)
            if count != 1:
                if verbose:
                    print("contracting tree is invalid because the contracting X stabiliser does not contract to exactly one X measured qubit!")
                return False
            
            if len(support["L"]) != 0 or len(support["R"]) != 0:
                if verbose:
                    print("contracting tree is invalid because the contracting stabiliser ends with support on a qubit that isn't measured in the X basis!")
                return False
            
            return True
        elif method == "graph":
            remaining_vertices = [["A", "B"][i] + str(j) for i in range(2) for j in range([len(self.a_list), len(self.b_list)][i])]
            vertex_rounds = {["A", "B"][i] + str(j): [] for i in range(2) for j in range([len(self.a_list), len(self.b_list)][i])}
            AB_gates = []
            for i in range(len(contraction_tree)):
                contraction_layer = contraction_tree[i]
                used_this_layer = []
                for edge in contraction_layer:
                    if (edge[1] in remaining_vertices) and (edge[0] in remaining_vertices) and not (edge[1] in used_this_layer):
                        remaining_vertices.remove(edge[0])
                        vertex_rounds[edge[0]].append(i)
                        vertex_rounds[edge[1]].append(i)
                        used_this_layer.append(edge[1])
                        if edge[0][0] != edge[1][0]:
                            AB_gates.append(set([edge[0], edge[1]]))
                    else:
                        if verbose:
                            print(f"contracting tree is invalid, possibly because it contains a cycle or because the layers are incorrect. In particular edge {edge} in layer {i} is invalid.")
                        return False
            if len(remaining_vertices) != 1:
                if verbose:
                    print("contracting tree is invalid because it contracts to more than one root")
                return False
    
            group = self.group
            if not self.valid_homomorphism(homomorphism_image, homomorphism_action, verbose = verbose):
                return False
            
            def homomorphism_function(el):
                out = [_ for _ in range(len(homomorphism_image))]
                for i in range(len(el)):
                    for j in range(len(out)):
                        out[j] += homomorphism_action[i][j] * el[i]
                return [out[j] % homomorphism_image[j] for j in range(len(homomorphism_image))]
    
            def simultaneously_contractible(label0, label1):
                if label0[0] == label1[0]:
                    if len(vertex_rounds[label0]) == 1:
                        if all(x > vertex_rounds[label0][0] for x in vertex_rounds[label1]):
                            return True
                    
                    if len(vertex_rounds[label1]) == 1:
                        if all(x > vertex_rounds[label1][0] for x in vertex_rounds[label0]):
                            return True
                    
                    return False
                    
                else:
                    if set([label0, label1]) in AB_gates:
                        if len(set(vertex_rounds[label0] + vertex_rounds[label1])) == len(vertex_rounds[label0]) + len(vertex_rounds[label1]) - 1:
                            return True
                    
                    return False
    
            element_subsets = defaultdict(list)
            for i in range(len(self.a_list)):
                element_subsets[tuple(homomorphism_function(self.a_list[i]))].append("A" + str(i))
            for j in range(len(self.b_list)):
                out = homomorphism_function(self.b_list[j])
                inv = [(-b_shift[i]-out[i]) % homomorphism_image[i] for i in range(len(homomorphism_image))]
                element_subsets[tuple(inv)].append("B" + str(j))
            
            for element_subset in element_subsets.values():
                for pair in combinations(element_subset, 2):
                    if not simultaneously_contractible(pair[0], pair[1]):
                        if verbose:
                            print(f"morphing parameters are invalid because {pair[0]} and {pair[1]} should be simultaneously contractible, but are not.")
                        return False
    
            return True
        else:
            raise ValueError("method must be contraction or graph!")
    
    def valid_standard_morphing_parameters(self, homomorphism_action):
        """
        Checks if the Abelian 2BGA code and homomorphism_action define a valid standard morphing circuit.
        
        A standard morphing circuit is a two-round homomorphism-based morphing circuit that most-straightforwardly generalises the contraction tree from the morphing circuits in arXiv:2407.16336 to higher-weight Abelian 2BGA codes. In the language of valid_morphing_parameters, we have:
            contraction_tree = [[[An, B0], [Bn, A0]], [[A(n-1), B0], [B(n-1), A0]], ..., [[A1, B0], [B1, A0]], [[A0, B0]]] (or something very similar), and
            homomorphism_image = [2].
        
        Moreover, the function checks over all permutations of A and B, and the two possible values of b_shift, to see if any of these give a valid standard morphing circuit, by checking if Criterion 1 from arXiv:2407.16336 is satisfied.

        Arguments:
            homomorphism_action is a List[List[Int]] that represents the image of each generator of self.group under the homomorphism. Since the morphing circuit is "standard" it has two rounds and homomorphism_image is [2]. For backwards compatibility, homomorphism_action may also be a string x, y, or xy.

        Returns:
            True if there exists a permutation of A and B, and a value of b_shift, such that the standard morphing circuit defined above is valid.
        """
        self.check_consistent_parameters()
        group = self.group
        
        if homomorphism_action == "x":
            homomorphism_action = [[1],[0]]
        elif homomorphism_action == "y":
            homomorphism_action = [[0],[1]]
        elif homomorphism_action == "xy":
            homomorphism_action = [[1],[1]]
            
        if len(homomorphism_action) != len(group):
            return False

        for i in range(len(homomorphism_action)):
            if homomorphism_action[i] == 1 and group[i] % 2 != 0:
                return False

        def homomorphism_function(el):
            out = 0
            for i in range(len(el)):
                out += homomorphism_action[i][0] * el[i]
            return F2((out%2))

        if all([len(self.a_list) > 1, len(self.b_list) > 1]):
            for c_list in [self.a_list, self.b_list]:
                homomorphism_list = [homomorphism_function(c_el) for c_el in c_list]
                if (homomorphism_list.count(0) == 1) and (
                    homomorphism_list.count(1) == len(c_list) - 1
                ):
                    continue
                elif (homomorphism_list.count(1) == 1) and (
                    homomorphism_list.count(0) == len(c_list) - 1
                ):
                    continue
                else:
                    return False
            return True
        else:
            return False
    
    def prepare_for_standard_morphing(self, homomorphism_action):
        """
        Reorders the elements in A and B so that the Abelian 2BGA code is valid for standard morphing. Specifically, it finds the unique element in A that gives a different value under the homomorphism than the other elements in A, and swaps it with the first element in A, and similar for B.
        
        Arguments:
            homomorphism_action is a List[List[Int]] that represents the image of each generator of self.group under the homomorphism. Since the morphing circuit is "standard" it has two rounds and homomorphism_image is [2]. For backwards compatibility, homomorphism_action may also be a string x, y or xy.

        Raises:
            ValueError: The length of the homomorphism_action must equal the length of the group!
            ValueError: The homomorphism_action is incompatible with the group
            ValueError: The morphing specifications are not suitable for standard morphing!
            ValueError: The morphing specifications are not suitable for standard morphing!
        """
        self.check_consistent_parameters()
        group = self.group
        if homomorphism_action == "x":
            homomorphism_action = [[1],[0]]
        elif homomorphism_action == "y":
            homomorphism_action = [[0],[1]]
        elif homomorphism_action == "xy":
            homomorphism_action = [[1],[1]]
            
        if len(homomorphism_action) != len(group):
            raise ValueError("The length of the homomorphism_action must equal the length of the group!")

        for i in range(len(homomorphism_action)):
            if homomorphism_action[i] == 1 and group[i] % 2 != 0:
                raise ValueError("The homomorphism_action is incompatible with the group")

        def homomorphism_function(el):
            out = 0
            for i in range(len(el)):
                out += homomorphism_action[i][0] * el[i]
            return F2(out%2)

        if all([len(self.a_list) > 1, len(self.b_list) > 1]):
            for c_list in [self.a_list, self.b_list]:
                homomorphism_list = [homomorphism_function(c_el) for c_el in c_list]
                if (homomorphism_list.count(0) == 1) and (
                    homomorphism_list.count(1) == len(c_list) - 1
                ):
                    special_value = 0
                elif (homomorphism_list.count(1) == 1) and (
                    homomorphism_list.count(0) == len(c_list) - 1
                ):
                    special_value = 1
                else:
                    raise ValueError(
                        "The morphing specifications are not suitable for standard morphing!"
                    )
                to_swap = homomorphism_list.index(special_value)
                c_list[0], c_list[to_swap] = c_list[to_swap], c_list[0]
        else:
            raise ValueError(
                "The morphing specifications are not suitable for standard morphing!"
            )

    def parity_check_matrices(self):
        """
        Returns the X- and Z-basis parity check matrices hx and hz of the Abelian 2BGA code.
        
        Returns:
            A dictionary {"X": hx, "Z": hz} containing the X-basis and Z-basis parity check matrices of the code.
        """
        self.check_consistent_parameters()
        group = self.group
        size_group = 1
        for order in group:
            size_group *= order
        a_list = self.a_list
        b_list = self.b_list

        n_a = len(a_list)
        n_b = len(b_list)
        permutation_matrices = [F2(np.diag(np.ones((group[i] - 1)), 1).astype(int)) for i in range(len(group))]
        for i in range(len(group)):
            permutation_matrices[i][group[i]-1, 0] = 1
        a_matrices = [np.linalg.matrix_power(permutation_matrices[0], a[0]) for a in a_list]
        b_matrices = [np.linalg.matrix_power(permutation_matrices[0], b[0]) for b in b_list]
        for i in range(1, len(group)):
            a_matrices = [np.kron(a_matrices[j], np.linalg.matrix_power(permutation_matrices[i], a_list[j][i])) for j in range(n_a)]
            b_matrices = [np.kron(b_matrices[j], np.linalg.matrix_power(permutation_matrices[i], b_list[j][i])) for j in range(n_b)]
        
        a_matrix = F2(np.zeros((size_group, size_group)).astype(int))
        b_matrix = F2(np.zeros((size_group, size_group)).astype(int))
        for i in range(n_a):
            a_matrix += a_matrices[i]
        for i in range(n_b):
            b_matrix += b_matrices[i]

        h = {
            "X": np.concatenate((a_matrix, b_matrix), axis=1),
            "Z": np.concatenate((b_matrix.T, a_matrix.T), axis=1),
        }

        return h
    
    def _get_stabilisers_and_contraction_rounds(self, contraction_round_functions=None):
        group = self.group
        a_list = self.a_list
        b_list = self.b_list

        h = self.parity_check_matrices()

        mid_cycle_stabilisers = {pauli: [] for pauli in ["X", "Z"]}
        stabiliser_contraction_rounds = (
            {pauli: [] for pauli in ["X", "Z"]}
            if contraction_round_functions != None
            else None
        )
        index = 0
        for el in product(*[range(i) for i in group]):
            mid_cycle_stabilisers["X"].append(
                np.nonzero(h["X"][index])[0].tolist()
            )
            mid_cycle_stabilisers["Z"].append(
                np.nonzero(h["Z"][index])[0].tolist()
            )
            if contraction_round_functions != None:
                x_round = int(contraction_round_functions["X"](el))
                stabiliser_contraction_rounds["X"].append([x_round])
                z_round = int(contraction_round_functions["Z"](el))
                stabiliser_contraction_rounds["Z"].append([z_round])
            index += 1
        return mid_cycle_stabilisers, stabiliser_contraction_rounds

    def _get_real_coordinates_list(self):
        coordinates_list = []
        """
        First coordinate represents left (0) or right (1), the remaining coordinates the group element x^i y^j z^k ... .
        """
        for lr in range(2):
            for el in product(*[range(i) for i in self.group]):
                coordinates_list.append([lr] + [i for i in el])
        return coordinates_list

    def two_round_end_cycle_code(self, contraction_tree, homomorphism_action, b_shift):
        """
        Returns the end-cycle code of a two-round homomorphism-based morphing circuit.

        Arguments:
            contraction_tree is a List[List[List[Str]]] representing the contraction tree of the X stabiliser s(X,1) where 1 is the identity in self.group.
                Each contraction_tree[layer] corresponding to a different contraction gate layer in the contraction circuit.
                Each element contraction_tree[layer][i] is itself a length-2 list [el1, el2] corresponding to an edge in the contraction tree, where el1 gets contracted onto el2.
                The elements el1 and el2 themselves are strings that are one of "A0", "B0", "A1", "B1", etc.
                The contraction trees of other stabilisers are given by the definition of a homomorphism-based morphing circuit.
            homomorphism_action is a List[List[Int]] that represents the image of each generator of self.group under the homomorphism. For backwards compatibility, homomorphism_action may also be a string x, y, or xy.
            b_shift is a list corresponding to an element in homomorphism_image that is multiplied to the Z stabiliser labels when determining their contraction round.
            
        Returns:
            An Abelian 2BGA code corresponding to the end-cycle code of the morphing circuit.
        """
        homomorphism_image = [2]
        if homomorphism_action == "x":
            homomorphism_action = [[1],[0]]
        elif homomorphism_action == "y":
            homomorphism_action = [[0],[1]]
        elif homomorphism_action == "xy":
            homomorphism_action = [[1],[1]]
        
        identity_el = [0 for _ in range(len(homomorphism_image))]
        elements_dict = dict({"A" + str(i): self.a_list[i] for i in range(len(self.a_list))},
                             **{"B" + str(i): self.b_list[i] for i in range(len(self.b_list))})
        
        def homomorphism_function(el, shift = identity_el):
            out = [_ for _ in range(len(homomorphism_image))]
            for i in range(len(el)):
                for j in range(len(out)):
                    out[j] += homomorphism_action[i][j] * el[i]
            return [(shift[j] + out[j]) % homomorphism_image[j] for j in range(len(homomorphism_image))]
        
        """
        Propagates the X-check s(X,1) through the second contraction circuit. Then projects it onto the end-cycle code data qubits, and applies the
            appropriate map s.t. the end-cycle group is the kernel of the homomorphism.
        """
        
        support = {"L": [[self.a_list[i][j] % self.group[j] for j in range(len(self.group))] for i in range(len(self.a_list))],
                   "R": [[self.b_list[i][j] % self.group[j] for j in range(len(self.group))] for i in range(len(self.b_list))]}
        for layer in contraction_tree:
            cnots_already_done = []
            for gate in layer:
                #first the CNOTs from X stabiliser contraction
                control_element = elements_dict[gate[1]]
                control_LR = {"A": "L", "B": "R"}[gate[1][0]]
                temp = deepcopy(support[control_LR])
                for qubit in temp:
                    if homomorphism_function(qubit) != homomorphism_function(control_element):
                        target_qubit = [(elements_dict[gate[0]][i] - control_element[i] + qubit[i]) % self.group[i] for i in range(len(self.group))]
                        target_LR = {"A": "L", "B": "R"}[gate[0][0]]
                        to_check = [qubit, control_LR, target_qubit, target_LR]
                        if to_check in cnots_already_done:
                            continue
                        cnots_already_done.append(to_check)
                        if target_qubit in support[target_LR]:
                            support[target_LR].remove(target_qubit)
                        else:
                            support[target_LR].append(target_qubit)
                #then the CNOTs from Z stabiliser contraction, only if the CNOT gates don't coincide ofc.
                control_element = elements_dict[gate[0]]
                control_LR = {"A": "R", "B": "L"}[gate[0][0]]
                temp = deepcopy(support[control_LR])
                for qubit in temp:
                    if homomorphism_function(qubit) != homomorphism_function(control_element, shift = b_shift):
                        target_qubit = [(- elements_dict[gate[1]][i] + control_element[i] + qubit[i]) % self.group[i] for i in range(len(self.group))]
                        target_LR = {"A": "R", "B": "L"}[gate[1][0]]
                        to_check = [qubit, control_LR, target_qubit, target_LR]
                        if to_check in cnots_already_done and gate[0][0] != gate[1][0]:
                            continue
                        cnots_already_done.append(to_check)
                        if target_qubit in support[target_LR]:
                            support[target_LR].remove(target_qubit)
                        else:
                            support[target_LR].append(target_qubit)
                    
        root = contraction_tree[-1][0][1]
        f_of_root = homomorphism_function(elements_dict[root])
        need_to_shift = {"L": False, "R": False}
        #remove X measurements
        temp = deepcopy(support[{"A": "L", "B": "R"}[root[0]]])
        for qubit in temp:
            if homomorphism_function(qubit) != f_of_root:
                support[{"A": "L", "B": "R"}[root[0]]].remove(qubit)
                if homomorphism_function(qubit) == identity_el:
                    need_to_shift[{"A": "L", "B": "R"}[root[0]]] = True
        #remove Z measurements
        temp = deepcopy(support[{"A": "R", "B": "L"}[root[0]]])
        for qubit in temp:
            if homomorphism_function(qubit, shift = b_shift) != f_of_root:
                support[{"A": "R", "B": "L"}[root[0]]].remove(qubit)
                print("contraction_tree", contraction_tree)
                print("homomorphism_action", homomorphism_action)
                print("b_shift", b_shift)
                thingy = {"A": "R", "B": "L"}[root[0]]
                print(f"s(X,1) end-cycle {thingy} support", temp)
                print("Z measurement", qubit)
                raise Exception()
            if homomorphism_function(qubit) != identity_el:
                need_to_shift[{"A": "R", "B": "L"}[root[0]]] = True
        
        for group_element in product(*[range(self.group[i]) for i in range(len(self.group))]):
            if homomorphism_function(group_element) != identity_el:
                shift_element = group_element
                break
        
        a_shift = shift_element if need_to_shift["L"] else [0 for _ in range(len(self.group))]
        b_shift = shift_element if need_to_shift["R"] else [0 for _ in range(len(self.group))]
        temp_a_list = []
        temp_b_list = []
        for qubit in support["L"]:
            temp_a_list.append([(qubit[i] + a_shift[i]) % self.group[i] for i in range(len(self.group))])
        for qubit in support["R"]:
            temp_b_list.append([(qubit[i] + b_shift[i]) % self.group[i] for i in range(len(self.group))])
        temp_code = Abelian2BGAParams(self.group, temp_a_list, temp_b_list, input_format = "group")
        if temp_code.n_copies % 2 != 0 and temp_code.lattice_signature != [0,0]:
            print("mid-cycle code", self)
            print("contraction_tree", contraction_tree)
            print("homomorphism_action", homomorphism_action)
            print("b_shift", b_shift)
            print(f"s(X,1) end-cycle support", support)
            print("shifts", a_shift, b_shift)
            print("temp_code", temp_code)
            raise Exception()
        try:
            new_code = Abelian2BGAParams(temp_code.lattice_signature, temp_code.periodicity_vectors, n_copies = temp_code.n_copies//2, input_format = "vectors")
        except ValueError:
            print(temp_code)
            raise Exception()
        return new_code

    def has_minimal_overlap(self):
        raise NotImplementedError()

    def morphing_performance_parameters(
        self,
        contraction_tree,
        homomorphism_image,
        homomorphism_action,
        b_shift,
        separate_X_and_Z = False,
    ):
        n_c_rounds = 1
        for i in homomorphism_image:
            n_c_rounds *= i
        if homomorphism_action == "x":
            homomorphism_action = [[1],[0]]
        elif homomorphism_action == "y":
            homomorphism_action = [[0],[1]]
        elif homomorphism_action == "xy":
            homomorphism_action = [[1],[1]]
        
        identity_el = [0 for _ in range(len(homomorphism_image))]
        def homomorphism_function(el, shift = identity_el):
            out = [_ for _ in range(len(homomorphism_image))]
            for i in range(len(el)):
                for j in range(len(out)):
                    out[j] += homomorphism_action[i][j] * el[i]
            return [(shift[j] + out[j]) % homomorphism_image[j] for j in range(len(homomorphism_image))]
    
        def homomorphism_output_to_contraction_round(out):
            round = 0
            multiplier = 1
            for i in reversed(range(len(out))):
                round += out[i] * multiplier
                multiplier *= homomorphism_image[i]
            return round
    
        if separate_X_and_Z:
            n_c_rounds *= 2
            contraction_round_functions = {
                "X": lambda el: homomorphism_output_to_contraction_round(homomorphism_function(el)),
                "Z": lambda el: homomorphism_output_to_contraction_round(homomorphism_function(el, shift = b_shift)) + n_c_rounds//2,
            }
        else:
            contraction_round_functions = {
                "X": lambda el: homomorphism_output_to_contraction_round(homomorphism_function(el)),
                "Z": lambda el: homomorphism_output_to_contraction_round(homomorphism_function(el, shift = b_shift)),
            }
        
        depth = len(contraction_tree)
        
        #enumerates the required connectivity of the qubit q(L,1).
        #also counts the gates q(L,1) participates in by checking if q(L,1) participates in a gate in each layer of each contraction round.
        #By translational/ZX-duality symmetry all qubits have the same connectivity degree and gate count.
        left_connections = set()
        right_connections = set()
        participates_in_gate = [[False for _ in range(depth)] for _ in range(n_c_rounds)]
        elements_dict = dict({"A" + str(i): self.a_list[i] for i in range(len(self.a_list))},
                             **{"B" + str(i): self.b_list[i] for i in range(len(self.b_list))})
        gate_count = 0
        for layer in range(len(contraction_tree)):
            contraction_layer = contraction_tree[layer]
            for gate in contraction_layer:
                if gate[0][0] == gate[1][0]:
                    #if the edge goes from Ai to Aj or from Bi to Bj.
                    left_connections.add(((elements_dict[gate[0]][i] - elements_dict[gate[1]][i]) % self.group[i] for i in range(len(self.group))))
                    left_connections.add(((elements_dict[gate[1]][i] - elements_dict[gate[0]][i]) % self.group[i] for i in range(len(self.group))))
                else:
                    #if the edge goes between Ai and Bj
                    a_gate = gate[0] if gate[0][0] == "A" else gate[1]
                    b_gate = gate[1] if gate[0][0] == "A" else gate[0]
                    right_connections.add(((elements_dict[b_gate][i] - elements_dict[a_gate][i]) % self.group[i] for i in range(len(self.group))))
                for el in gate:
                    #q(L,1) is the el element for the stabiliser s(P, el^-1) where P is X if el is Ai and P is Z if el is Bi.
                    #This calculates the contraction round in which s(P, el^-1) is contracting.
                    if el[0] == "A":
                        this_stabiliser_type = "X"
                        this_stabiliser_el = [-elements_dict[el][i] % self.group[i] for i in range(len(self.group))]
                    else:
                        this_stabiliser_type = "Z"
                        this_stabiliser_el = [elements_dict[el][i] % self.group[i] for i in range(len(self.group))]
                    this_c_round = contraction_round_functions[this_stabiliser_type](this_stabiliser_el)
                    participates_in_gate[this_c_round][layer] = True
        
        connectivity = len(left_connections) + len(right_connections)
        gate_count = 0
        for temp_list in participates_in_gate:
            for el in temp_list:
                gate_count += int(el)
        gate_count *= 2
        
        swap_ancillas = True
        if contraction_tree[-1][0][0][0] != contraction_tree[-1][0][1][0] and not separate_X_and_Z and layer == (depth - 1) and i == 1:
            #Consider the first contraction round, i.e. the contraction round in which s(X,1) is contracting.
            #is the qubit it does a CNOT with the ancilla qubit for s(X,1) also an ancilla in the same round?
            #The answer is (obviously) no if separate_X_and_Z, and (less obviously) no if the last gate is Ai -> Aj or Bi -> Bj.
            #But if the last gate is between an Ai and a Bj, it depends on if the Z stabiliser s(Z, Ai^-1 Bj^-1) it shares the gate with contracts in the
            # same contraction round.
            #So if s(Z, Ai^-1 Bj^-1) is contracting in the first contraction round, swap_ancillas is False. Otherwise it is True.
            #By translational symmetry this applies for all other stabilisers/contraction rounds.
            first_el = elements_dict[contraction_tree[-1][0]]
            second_el = elements_dict[contraction_tree[-1][0]]
            other_Z_stabiliser = [(-first_el[i] - second_el[i]) % self.group[i] for i in range(len(self.group))]
            swap_ancillas = contraction_round_functions["Z"](other_Z_stabiliser) != 0
        
        return (n_c_rounds, depth, connectivity, gate_count, swap_ancillas)
    
    def get_morphing_specifications(self, contraction_tree, homomorphism_image, homomorphism_action, b_shift, safe = True, separate_X_and_Z = False, swap_AB_rounds = []):
        """
        Returns a morphing specification for a purely-contracting homomorphism-based morphing circuit.
        
        Arguments:
        contraction_tree is a List[List[List[Str]]] representing the contraction tree of the X stabiliser s(X,1) where 1 is the identity in self.group.
                Each contraction_tree[layer] corresponding to a different contraction gate layer in the contraction circuit.
                Each element contraction_tree[layer][i] is itself a length-2 list [el1, el2] corresponding to an edge in the contraction tree, where el1 gets contracted onto el2.
                The elements el1 and el2 themselves are strings that are one of "A0", "B0", "A1", "B1", etc.
                The contraction trees of other stabilisers are given by the definition of a homomorphism-based morphing circuit.
            homomorphism_image is a List[Int] representing the index of each cyclic group whose product gives the Abelian group that is the image of the homomorphism.
            homomorphism_action is a List[List[Int]] that represents the image of each generator of self.group under the homomorphism. For backwards compatibility, homomorphism_action may also be a string "x", "y", or "xy".
            b_shift is a list corresponding to an element in homomorphism_image that is multiplied to the Z stabiliser labels when determining their contraction round.
            safe is a bool, if it is True then the parameters are first checked to see if they are valid.
            separate_X_and_Z is a bool, if it is True then the X and Z checks are contracted in separate contraction rounds.
            swap_AB_rounds is a List[Int]. Each element corresponds to a contraction round, each contraction round in the list will have its contraction tree modified by swapping A and B.
                This can be convenient for colour code morphing circuits where the relationship between the two contraction rounds involves a ZX-duality instead of just a translation.

        Returns:
            The morphing specifications for the morphing circuit.
        """

        swapped_contraction_tree = [[[{"A": "B", "B": "A"}[contraction_tree[i][j][k][0]] + contraction_tree[i][j][k][1:] for k in range(len(contraction_tree[i][j]))]
                                     for j in range(len(contraction_tree[i]))
                                    ] for i in range(len(contraction_tree))
                                   ]
        if safe:
            if not self.valid_morphing_parameters(contraction_tree, homomorphism_image, homomorphism_action, b_shift):
                raise ValueError(
                    "The inputted midcycle parameters do not form a valid morphing circuit."
                )
            if len(swap_AB_rounds) > 0 and not self.valid_morphing_parameters(swapped_contraction_tree, homomorphism_image, homomorphism_action, b_shift):
                raise ValueError(
                    "The inputted midcycle parameters with swapped contraction tree do not form a valid morphing circuit."
                )
        if homomorphism_action == "x":
            homomorphism_action = [[1],[0]]
        elif homomorphism_action == "y":
            homomorphism_action = [[0],[1]]
        elif homomorphism_action == "xy":
            homomorphism_action = [[1],[1]]

        group = self.group
        a_list = self.a_list
        b_list = self.b_list
        if len(b_shift) != len(homomorphism_image):
            raise ValueError("The length of b_shift does not match the length of homomorphism_image")
        identity_el = [0 for _ in range(len(homomorphism_image))]

        n_a = len(a_list)
        n_b = len(b_list)
        n_c_rounds = 1
        for i in range(len(homomorphism_image)):
            n_c_rounds *= homomorphism_image[i]
        
        coordinates_list = self._get_real_coordinates_list()

        def homomorphism_function(el, shift = identity_el):
            out = [_ for _ in range(len(homomorphism_image))]
            for i in range(len(el)):
                for j in range(len(out)):
                    out[j] += homomorphism_action[i][j] * el[i]
            return [(shift[j] + out[j]) % homomorphism_image[j] for j in range(len(homomorphism_image))]

        def homomorphism_output_to_contraction_round(out):
            round = 0
            multiplier = 1
            for i in reversed(range(len(out))):
                round += out[i] * multiplier
                multiplier *= homomorphism_image[i]
            return round

        if separate_X_and_Z:
            n_c_rounds *= 2
            contraction_round_functions = {
                "X": lambda el: homomorphism_output_to_contraction_round(homomorphism_function(el)),
                "Z": lambda el: homomorphism_output_to_contraction_round(homomorphism_function(el, shift = b_shift)) + n_c_rounds//2,
            }
        else:
            contraction_round_functions = {
                "X": lambda el: homomorphism_output_to_contraction_round(homomorphism_function(el)),
                "Z": lambda el: homomorphism_output_to_contraction_round(homomorphism_function(el, shift = b_shift)),
            }

        mid_cycle_stabilisers, stabiliser_contraction_rounds = (
            self._get_stabilisers_and_contraction_rounds(
                contraction_round_functions=contraction_round_functions
            )
        )

        gate_lists = [[[] for layer in range(len(contraction_tree))] for c_round in range(n_c_rounds)]
        measurement_lists = {
            pauli: [[] for c_round in range(n_c_rounds)] for pauli in ["X", "Z"]
        }

        for el in product(*[range(i) for i in group]):
            round_X = contraction_round_functions["X"](el)
            qubits_index_dict = {}
            for i in range(len(a_list)):
                qubits_index_dict["A" + str(i)] = coordinates_list.index([0] + [(a_list[i][j] + el[j]) % group[j] for j in range(len(el))])
            for i in range(len(b_list)):
                qubits_index_dict["B" + str(i)] = coordinates_list.index([1] + [(b_list[i][j] + el[j]) % group[j] for j in range(len(el))])

            if round_X in swap_AB_rounds:
                this_contraction_tree = deepcopy(swapped_contraction_tree)
            else:
                this_contraction_tree = deepcopy(contraction_tree)

            for i in range(len(this_contraction_tree)):
                for gate in this_contraction_tree[i]:
                    to_append = [qubits_index_dict[gate[1]], qubits_index_dict[gate[0]]]
                    if to_append in gate_lists[round_X][i]:
                        pass
                    else:
                        gate_lists[round_X][i].append(to_append)

            measurement_lists["X"][round_X].append(qubits_index_dict[this_contraction_tree[-1][0][-1]])

            round_Z = contraction_round_functions["Z"](el)
            qubits_index_dict = {}
            for i in range(len(a_list)):
                qubits_index_dict["A" + str(i)] = coordinates_list.index([1] + [(-a_list[i][j] + el[j]) % group[j] for j in range(len(el))])
            for i in range(len(b_list)):
                qubits_index_dict["B" + str(i)] = coordinates_list.index([0] + [(-b_list[i][j] + el[j]) % group[j] for j in range(len(el))])

            if round_Z in swap_AB_rounds:
                this_contraction_tree = deepcopy(swapped_contraction_tree)
            else:
                this_contraction_tree = deepcopy(contraction_tree)

            for i in range(len(this_contraction_tree)):
                for gate in this_contraction_tree[i]:
                    to_append = [qubits_index_dict[gate[0]], qubits_index_dict[gate[1]]]
                    if to_append in gate_lists[round_Z][i]:
                        pass
                    else:
                        gate_lists[round_Z][i].append(to_append)

            measurement_lists["Z"][round_Z].append(qubits_index_dict[this_contraction_tree[-1][0][-1]])

        complex_coordinates_list = []
        for coordinates in coordinates_list:
            complex_coordinate = (1 + 1j) * coordinates[0]
            factor = 1
            for i in range(0,len(group),2):
                complex_coordinate += 2 * factor * coordinates[i+1]
                factor *= group[i] + 1
            factor = 1
            for i in range(1, len(group), 2):
                complex_coordinate += 2j * factor * coordinates[i+1]
                factor *= group[i] + 1
            complex_coordinates_list.append(complex_coordinate)

        specifications = MorphingSpecifications(
            mid_cycle_stabilisers,
            stabiliser_contraction_rounds,
            gate_lists,
            measurement_lists,
            coordinates_list=complex_coordinates_list,
        )

        return specifications
        

    def get_standard_morphing_specifications(self, homomorphism_action):
        """
        Returns a morphing specification for a standard morphing circuit. A standard morphing circuit is a two-round homomorphism-based morphing circuit that most-straightforwardly generalises the contraction tree from the morphing circuits in Table IV of https://arxiv.org/abs/2407.16336 to higher-weight Abelian 2BGA codes. In the language of valid_morphing_parameters, we have:
            contraction_tree = [[[An, B0], [Bn, A0]], [[A(n-1), B0], [B(n-1), A0]], ..., [[A1, B0], [B1, A0]], [[A0, B0]]] (or something very similar), and
            homomorphism_image = [2].
        
        Arguments:
            homomorphism_action is a List[List[Int]] that represents the image of each generator of self.group under the homomorphism. For backwards compatibility homomorpism_action may also be a string "x", "y", or "xy".

        Raises:
            ValueError: The inputted midcycle parameters do not form a valid middle-out circuit.
        
        Returns:
            The morphing specifications for the morphing circuit.
        """

        if not self.valid_standard_morphing_parameters(homomorphism_action):
            raise ValueError(
                "The inputted midcycle parameters do not form a valid middle-out circuit."
            )

        old_a_list = deepcopy(self.a_list)
        old_b_list = deepcopy(self.b_list)

        self.prepare_for_standard_morphing(homomorphism_action)

        group = self.group
        a_list = self.a_list
        b_list = self.b_list
        if homomorphism_action == "x":
            homomorphism_action = [[1],[0]]
        elif homomorphism_action == "y":
            homomorphism_action = [[0],[1]]
        elif homomorphism_action == "xy":
            homomorphism_action = [[1],[1]]

        n_a = len(a_list)
        n_b = len(b_list)

        coordinates_list = self._get_real_coordinates_list()

        def homomorphism_function(el):
            out = 0
            for i in range(len(el)):
                out += homomorphism_action[i][0] * el[i]
            return F2(out%2)

        contraction_round_functions = {
            "X": lambda el: homomorphism_function(el) + homomorphism_function(a_list[0]),
            "Z": lambda el: homomorphism_function(el) + homomorphism_function(b_list[0]) + F2(1),
        }

        mid_cycle_stabilisers, stabiliser_contraction_rounds = (
            self._get_stabilisers_and_contraction_rounds(
                contraction_round_functions=contraction_round_functions
            )
        )

        gate_lists = [[[] for layer in range(max([n_a, n_b]))] for c_round in range(2)]
        measurement_lists = {
            pauli: [[] for c_round in range(2)] for pauli in ["X", "Z"]
        }

        for el in product(*[range(i) for i in group]):
            round_g = int(homomorphism_function(el))  # this is the round in which the left qubit labelled by el corresponds to the element g in Table IV
            round_h = (round_g + 1) % 2  # same but for h in Table IV
            this_left_qubit = coordinates_list.index([0] + [i for i in el])
            right_g_qubits = [coordinates_list.index([1] + [(-a_list[0][i] + bel[i] + el[i]) % group[i] for i in range(len(el))]) for bel in b_list]
            right_h_qubits = [coordinates_list.index([1] + [(-ael[i] + b_list[0][i] + el[i]) % group[i] for i in range(len(el))]) for ael in a_list]
            for k in range(1, n_b):
                gate_lists[round_g][k - 1].append([this_left_qubit, right_g_qubits[k]])
            for k in range(1, n_a):
                gate_lists[round_h][k - 1].append([right_h_qubits[k], this_left_qubit])
            gate_lists[round_g][-1].append([right_g_qubits[0], this_left_qubit])
            gate_lists[round_h][-1].append([this_left_qubit, right_h_qubits[0]])
            measurement_lists["X"][round_g].append(right_g_qubits[0])
            measurement_lists["Z"][round_h].append(right_h_qubits[0])

        complex_coordinates_list = []
        for coordinates in coordinates_list:
            complex_coordinate = (1 + 1j) * coordinates[0]
            factor = 1
            for i in range(0,len(group),2):
                complex_coordinate += 2 * factor * coordinates[i+1]
                factor *= group[i] + 1
            factor = 1
            for i in range(1, len(group), 2):
                complex_coordinate += 2j * factor * coordinates[i+1]
                factor *= group[i] + 1
            complex_coordinates_list.append(complex_coordinate)

        specifications = MorphingSpecifications(
            mid_cycle_stabilisers,
            stabiliser_contraction_rounds,
            gate_lists,
            measurement_lists,
            coordinates_list=complex_coordinates_list,
        )

        self.a_list = old_a_list
        self.b_list = old_b_list

        return specifications
    
    def get_logical_basis(self):
        """
        Returns a logical basis for the Abelian 2BGA code. Note that the logical basis is NOT symplectic.
        
        Returns:
            A dictionary with keys "X" and "Z", and items that are a numpy array corresponding to a generating set of X- and Z-logical operators.
                Note that the logical basis is NOT symplectic (so the X1 operator does not necessarily anticommute with Z1 and does not necessarily
                commute with all other Z operators).
        """
        h = self.parity_check_matrices()

        stabilisers = {pauli: [] for pauli in ["X", "Z"]}
        for i in range(self.nkd[0]//2):
            stabilisers["X"].append(np.nonzero(h["X"][i])[0].tolist())
            stabilisers["Z"].append(np.nonzero(h["Z"][i])[0].tolist())

        hker = {pauli: h[pauli].null_space() for pauli in ["X", "Z"]}
        stabs_to_logicals = {pauli: h[pauli].row_reduce() for pauli in ["X", "Z"]}
        observables = {pauli: [] for pauli in ["X", "Z"]}

        for pauli in ["X", "Z"]:
            other_pauli = "Z" if pauli == "X" else "X"
            for i in range(hker[other_pauli].shape[0]):
                new_vecs = np.concatenate(
                    (stabs_to_logicals[pauli], [hker[other_pauli][i]]), axis=0
                )
                if np.linalg.matrix_rank(new_vecs) != np.linalg.matrix_rank(
                    stabs_to_logicals[pauli]
                ):
                    stabs_to_logicals[pauli] = new_vecs
                    observables[pauli].append(
                        np.nonzero(hker[other_pauli][i])[0].tolist()
                    )
        return observables

    def get_standard_XZ_memory_circuit(self, *args, **kwargs):
        """
        An alias of get_bare_ancilla_XZ_memory_circuit, kept for backwards compatibility.
        """
        return self.get_bare_ancilla_XZ_memory_circuit(*args, **kwargs)
    
    def get_bare_ancilla_XZ_memory_circuit(
        self,
        X_order,
        Z_order,
        noise_setup,
        T,
        noise_model_generator=lambda qubit_dictionary, noise_setup: CircuitNoiseModel(
            qubit_dictionary, noise_setup
        ),
    ):
        """
        Returns a stim circuit for a memory experiment using a bare-ancilla syndrome extraction circuit.
        The memory experiment contains observables for all the logical operators of the code (both X and Z). It begins and ends with one round of
            ideal QEC, with T rounds of noisy QEC in between.
        For example, the syndrome extraction circuit from Table 5 of arXiv:2308.07915 is given by setting
            X_order = ["", "A1", "B1", "B0", "B2", "A0", "A2"], and
            Z_order = ["A0", "A2", "B0", "B1", "B2", "A1", ""].
        Note that the returned circuit does NOT stagger the measurements so that the extra idling steps are
            amortized throughout the memory experiment. This is unlike the get_IBM_XZ_memory_circuit method of the BBParams subclass.
        Note that this method does NOT check whether the order specified is "proper" in the sense that the order of the CNOT gates correctly
            measures the stabilisers.
        
        Arguments:
            X_order is a List[Str] that indicates the order in which each X-ancilla performs a CNOT with the corresponding data qubits.
                The data qubits are labelled by a string "A0", "B0", "A1", etc, with "A0" for example indicating that each X-ancilla q(X,g) performs a
                CNOT gate with q(L, A0 * g), and "B0" indicates a CNOT gate with q(R, B0 * g).
            Z_order is a List[Str] that indicates the order in which each Z-ancilla performs a CNOT with the corresponding data qubits.
                The data qubits are labelled by a string "A0", "B0", "A1", etc, with "A0" for example indicating that each Z-ancilla q(Z,g) performs a
                CNOT gate with q(R, A0^-1 * g), and "B0" indicates a CNOT gate with q(L, B0^-1 * g).
            noise_setup is a surface_sim.Setup object that sets up the noise model to be used in the circuit.
            T is the number of noisy QEC rounds in the memory experiment.
            noise_model_generator is a function that takes two arguments: a qubit dictionary, and a noise setup, and returns a surface_sim.Model object
                that defines the noise model of the memory experiment.

        Raises:
            ValueError: X_order and Z_order must have the same length!
            ValueError: X_order[i] and Z_order[i] are both non-trivial but have different matrix labels.
            ValueError: X_order[i] refers to an element that is out of the range of the list.
            ValueError: Z_order[i] refers to an element that is out of the range of the list.
        
        Returns:
            A stim circuit for the memory experiment.
        """

        ab_lists = {"A": self.a_list, "B": self.b_list}

        if len(X_order) != len(Z_order):
            raise ValueError("X_order and Z_order must have the same length!")

        for i in range(len(X_order)):
            if (not X_order[i] in ["", None]) and (not Z_order[i] in ["", None]):
                if not X_order[i][0] == Z_order[i][0]:
                    raise ValueError(f"X_order[{i}] and Z_order[{i}] are both non-trivial but have different matrix labels.")
                if int(X_order[i][1]) >= len(ab_lists[X_order[i][0]]):
                    raise ValueError(f"X_order[{i}] refers to an element {X_order[i]} that is out of the range of the {X_order[i][0]}-list.")
                if int(Z_order[i][1]) >= len(ab_lists[Z_order[i][0]]):
                    raise ValueError(f"Z_order[{i}] refers to an element {Z_order[i]} that is out of the range of the {Z_order[i][0]}-list.")

        group_coords_list = [
            (label, el)
            for label in ["L", "R", "X", "Z"]
            for el in product(*[range(self.group[i]) for i in range(len(self.group))])
        ]

        gate_lists = [[] for _ in range(len(X_order))]
        measure_lists = {pauli: [] for pauli in ["X", "Z"]}
        reset_lists = {pauli: [] for pauli in ["X", "Z"]}

        for el in product(*[range(self.group[i]) for i in range(len(self.group))]):
            x_ancilla = group_coords_list.index(("X", el))
            z_ancilla = group_coords_list.index(("Z", el))

            left_x_qubits = [
                group_coords_list.index(("L", tuple([(ael[i] + el[i]) % self.group[i] for i in range(len(self.group))])))
                for ael in self.a_list
            ]
            right_x_qubits = [
                group_coords_list.index(("R", tuple([(bel[i] + el[i]) % self.group[i] for i in range(len(self.group))])))
                for bel in self.b_list
            ]
            x_qubits = {"A": left_x_qubits, "B": right_x_qubits}
            left_z_qubits = [
                group_coords_list.index(("L", tuple([(-bel[i] + el[i]) % self.group[i] for i in range(len(self.group))])))
                for bel in self.b_list
            ]
            right_z_qubits = [
                group_coords_list.index(("R", tuple([(-ael[i] + el[i]) % self.group[i] for i in range(len(self.group))])))
                for ael in self.a_list
            ]
            z_qubits = {"A": right_z_qubits, "B": left_z_qubits}

            measure_lists["X"].append(x_ancilla)
            reset_lists["X"].append(x_ancilla)
            measure_lists["Z"].append(z_ancilla)
            reset_lists["Z"].append(z_ancilla)

            for k in range(len(X_order)):
                if not X_order[k] in ["", None]:
                    gate_lists[k].append(x_ancilla)
                    gate_lists[k].append(x_qubits[X_order[k][0]][int(X_order[k][1])])
                if not Z_order[k] in ["", None]:
                    gate_lists[k].append(z_qubits[Z_order[k][0]][int(Z_order[k][1])])
                    gate_lists[k].append(z_ancilla)

        qec_circuit = stim.Circuit()
        qec_circuit.append("RX", reset_lists["X"])
        qec_circuit.append("RZ", reset_lists["Z"])
        qec_circuit.append("TICK")
        for i in range(len(gate_lists)):
            qec_circuit.append("CNOT", gate_lists[i])
            qec_circuit.append("TICK")
        qec_circuit.append("MX", measure_lists["X"])
        qec_circuit.append("MZ", measure_lists["Z"])
        qec_circuit.append("TICK")

        detector_circuit = stim.Circuit()
        for i in range(self.nkd[0]):
            detector_circuit.append("DETECTOR", [stim.target_rec(-self.nkd[0]+i), stim.target_rec(-2*self.nkd[0]+i)])

        offsets = {"L": 0, "R": 1 + 1j, "X": 1, "Z": 1j}
        complex_coordinates_list = []
        for coordinates in group_coords_list:
            complex_coordinate = offsets[coordinates[0]]
            factor = 1
            for i in range(0,len(self.group),2):
                complex_coordinate += 2 * factor * coordinates[1][i]
                factor *= self.group[i] + 1
            factor = 1
            for i in range(1, len(self.group), 2):
                complex_coordinate += 2j * factor * coordinates[1][i]
                factor *= self.group[i] + 1
            complex_coordinates_list.append(complex_coordinate)

        #copy/paste of the coordinates part of the get_IBM_XZ_memory_circuit method below.

        coordinate_circuit_str = ""

        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0
        for i in range(len(complex_coordinates_list)):
            coord = complex_coordinates_list[i]
            min_x = min(round(coord.real), min_x)
            max_x = max(round(coord.real), max_x)
            min_y = min(round(coord.imag), min_y)
            max_y = max(round(coord.imag), max_y)
        for i in range(len(complex_coordinates_list)):
            coord = complex_coordinates_list[i]
            coordinate_circuit_str += f"QUBIT_COORDS({-min_x + round(coord.real)}, {(max_y - round(coord.imag))}) {i}\n"
            # the minus makes sure y always is pointing up.
            # Also should be such that all coords are always positive (to copy/paste into crumble if necessary)

        coordinate_circuit = stim.Circuit(coordinate_circuit_str)

        observables = self.get_logical_basis()
        logical_operator_basis = {pauli: [] for pauli in ["X", "Z"]}
        zs = np.zeros(2*self.nkd[0], dtype = np.bool_)
        for i in range(len(observables["X"])):
            xs = np.zeros(2*self.nkd[0], dtype = np.bool_)
            for j in observables["X"][i]:
                xs[j] = 1
            logical_operator_basis["X"].append(stim.PauliString.from_numpy(xs = xs, zs = zs))
        xs = np.zeros(2*self.nkd[0], dtype = np.bool_)
        for i in range(len(observables["Z"])):
            zs = np.zeros(2*self.nkd[0], dtype = np.bool_)
            for j in observables["Z"][i]:
                zs[j] = 1
            logical_operator_basis["Z"].append(stim.PauliString.from_numpy(xs = xs, zs = zs))

        return qec_circuit_to_XZ_memory_experiment(
            qec_circuit,
            coordinate_circuit,
            detector_circuit,
            logical_operator_basis,
            noise_setup,
            T,
            noise_model_generator=noise_model_generator,
        )

def Abelian_2BGA_iterator(size_G, signature):
    """
    Iterates over Abelian Two-Block Group Algebra (2BGA) codes with group G with size |G| = size_G, and with defining subsets A and B with size |A| = signature[0] + 1, |B| = signature[1] + 1. Iterator only returns codes that are inequivalent up to group automorphisms, translations of elements in A and B, permutations of elements within A and B, and (if signature[0] = signature[1]) interchanging A and B.

    Args:
        size_G: An Int setting the size of the group G in the Abelian 2BGA code. The number of physical qubits in the code is 2 * size_G.
        signature: A length-two List[Int] setting the size of the subsets A and B, with |A| = signature[0] + 1 and |B| = signature[1] + 1. Equivalently, this is the lattice signature of the lattice presentation of the Abelian 2BGA code. The weight of each stabiliser is |A| + |B| = sum(signature) + 2.

    Yields:
        Abelian2BGA codes with |G| = size_G, |A| = signature[0] + 1, and |B| = signature[1] + 1.

    Raises:
        ValueError: signature must have length 2.
    """
    if len(signature) != 2:
        raise ValueError("signature must have length 2")
    
    n_gens = sum(signature)
    already_iterated = set()
    swap_AB = signature[0] == signature[1]

    for hnf in hnf_iterator(n_gens, size_G):
        for i in range(signature[0]):
            if hnf[i,i] == 1:
                if np.all(hnf[i,:i] == 0):
                    #this means that the hnf is going to generate a non-trivial group element corresponding to the identity group element.
                    break
                for j in range(i):
                    test1 = np.all(hnf[i,:j] == hnf[j,:j])
                    test2 = hnf[i,j] == hnf[j,j]-1
                    test3 = np.all(hnf[i,(j+1):i] == 0)
                    if test1 and test2 and test3:
                        #this means that two elements in A are going to be the same
                        break
                else:
                    continue
                break
        else:
            for i in range(signature[0], signature[0]+signature[1]):
                if hnf[i,i] == 1:
                    if np.all(hnf[i,:i] == 0):
                        #this means that the hnf is going to generate a non-trivial group element corresponding to the identity group element.
                        break
                    for j in range(signature[0], i):
                        test1 = np.all(hnf[i,:j] == hnf[j,:j])
                        test2 = hnf[i,j] == hnf[j,j]-1
                        test3 = np.all(hnf[i,(j+1):i] == 0)
                        if test1 and test2 and test3:
                            #this means that two elements in B are going to be the same
                            break
                    else:
                        continue
                    break
            else:
                hnf_tuple = tuple(map(tuple, hnf))
                if not hnf_tuple in already_iterated:
                    to_yield = Abelian2BGAParams(signature, hnf, input_format = "vectors")
                    yield to_yield
                    for equiv in to_yield.list_equivalent_params(swap_AB = (signature[0] == signature[1])):
                        equiv_hnf_tuple = tuple(map(tuple, equiv.periodicity_vectors))
                        already_iterated.add(equiv_hnf_tuple)

def self_dual_Abelian_2BGA_iterator(size_G, signature):
    """
    Iterates over Abelian Two-Block Group Algebra (2BGA) codes with group G with size |G| = size_G, and with defining subsets A and B with size |A| = signature[0] + 1, |B| = signature[1] + 1. Iterator only returns codes that are inequivalent up to group automorphisms, translations of elements in A and B, and permutations of elements in A and B.

    Args:
        size_G: An Int setting the size of the group G in the Abelian 2BGA code. The number of physical qubits in the code is 2 * size_G.
        signature: A length-two List[Int] setting the size of the subsets A and B, with |A| = signature[0] + 1 and |B| = signature[1] + 1. Equivalently, this is the lattice signature of the lattice presentation of the Abelian 2BGA code. The weight of each stabiliser is |A| + |B| = sum(signature) + 2.

    Yields:
        Abelian2BGA codes with |G| = size_G, |A| = signature[0] + 1, and |B| = signature[1] + 1.

    Raises:
        ValueError: signature must have length 2.
        ValueError: signature of a self-dual code must have signature[0] = signature[1].
    """
    if len(signature) != 2:
        raise ValueError("signature must have length 2")
    elif signature[0] != signature[1]:
        raise ValueError("signature of a self-dual code must have signature[0] = signature[1]")
    
    n_gens = signature[0]
    already_iterated = set()

    for hnf in hnf_iterator(n_gens, size_G):
        for i in range(signature[0]):
            if hnf[i,i] == 1:
                if np.all(hnf[i,:i] == 0):
                    #this means that the hnf is going to generate a non-trivial group element corresponding to the identity group element.
                    break
                for j in range(i):
                    test1 = np.all(hnf[i,:j] == hnf[j,:j])
                    test2 = hnf[i,j] == hnf[j,j]-1
                    test3 = np.all(hnf[i,(j+1):i] == 0)
                    if test1 and test2 and test3:
                        #this means that two elements in A are going to be the same
                        break
                else:
                    continue
                break
        else:
            hnf_tuple = tuple(map(tuple, hnf))
            if not hnf_tuple in already_iterated:
                hnf_to_yield = np.block([[hnf,np.zeros((n_gens, n_gens), dtype = int)],[np.eye(n_gens, dtype = int), np.eye(n_gens, dtype = int)]])
                to_yield = Abelian2BGAParams(signature, hnf_to_yield, input_format = "vectors")
                yield to_yield
                for equiv in to_yield.list_equivalent_params(swap_AB = False):
                    equiv_hnf_tuple = tuple(map(tuple, equiv.periodicity_vectors[:signature[0],:signature[0]]))
                    already_iterated.add(equiv_hnf_tuple)