from itertools import permutations
from copy import deepcopy
from collections import defaultdict

def parentheses_iterator(n_brackets):
    """Iterates over forests (i.e. sets of rooted acyclic graphs) with n_brackets nodes, represented by a sequence of parentheses. Uses Algorithm P from Chapter 7.2.1.6 of The Art of Computer Programming by Donald Knuth.

    Args:
        n_brackets: An Int setting the number of (opening) parentheses in the sequence of parentheses; equivalently, this is the number of nodes in the forest.

    Yields:
        Parentheses representing a forest with n_brackets nodes.
    """
    parentheses = [')']
    for i in range(n_brackets):
        parentheses += ['(', ')']
    yield parentheses[1:]
    
    m = 2*n_brackets - 1
    j = 1
    while True:
        parentheses[m] = ')'
        if parentheses[m-1] == ')':
            parentheses[m-1] = '('
            m += -1
            yield parentheses[1:]
        else:
            j = m-1
            k = 2*n_brackets - 1
            while parentheses[j] == '(':
                parentheses[j] = ')'
                parentheses[k] = '('
                j += -1
                k += -2
            if j == 0:
                break
            else:
                parentheses[j] = '('
                m = 2*n_brackets - 1
                yield parentheses[1:]

def contraction_tree_iterator(n_nodes):
    """Iterates over contraction trees (i.e. rooted acyclic graphs, labelled with integers on each edge that make up a valid contraction tree) with n_nodes nodes, up to permutations of vertices.

    Args:
        n_nodes: An Int setting the number of nodes in the contraction tree.

    Yields:
        Contraction trees containing n_nodes nodes. This is returned as a List[List[Tuple[Int]]], with the first list representing the ordering of edges that would implement the contraction tree, and each layer has a list of edges that are in each layer.
    """
    already_returned = []
    for parentheses in parentheses_iterator(n_nodes-1):
        """
        This works by starting at the root node and picking one of the edges to add to the last layer of the contraction circuit.
        Then, it picks any (simultaneously executable) subset of the new list of children to add to the second-last layer, and so on, until
        all the edges have been assigned a layer.
        """
        children_dict = defaultdict(list)
        current_branch = [0]
        newest_node = 0
        for parenthesis in parentheses:
            if parenthesis == '(':
                children_dict[current_branch[-1]].append(newest_node + 1)
                current_branch.append(newest_node + 1)
                newest_node += 1
            else:
                current_branch.pop(-1)
        """
        available_edges[-1] is a list of lists of edges, such that at most one edge from each list can be chosen to add to the next layer.
        The other elements available_edges[i] are simply records of the previous choices that could've been made.
        index_choices lists the choices that were made at each step. If it's ever -1 this indicates that a choice has not yet been made.
        """
        available_edges = [[[(i, 0) for i in children_dict[0]]]]
        index_choices = [[-1]]
        current_inverse_layer = 0
        expansion_tree = []
        still_iterating = True
        while still_iterating:
            """
            First, check if there are any available_edges. If not, proceed to yield the current contraction tree.
            """
            for i in range(len(available_edges[-1])):
                if len(available_edges[-1][i]) > 0:
                    break
            else:
                proposed_tree = list(reversed(expansion_tree))
                """
                Before we go ahead and yield the tree, check that a permutation of the current tree hasn't already been yielded.
                """
                for permutation in permutations(range(n_nodes)):
                    to_check = [[tuple([permutation[proposed_tree[i][j][k]] for k in range(2)]) for j in range(len(proposed_tree[i]))] for i in range(len(proposed_tree))]
                    if to_check in already_returned:
                        break
                else:
                    already_returned.append(proposed_tree)
                    yield proposed_tree
                """
                Now we move on to the next tree to iterate over.
                """
                if current_inverse_layer == 0:
                    still_iterating = False
                    break
                else:
                    """
                    Because we just generated a valid tree, this means that available_edges[-1] and index_choices[-1] are trivial, so we pop them.
                    We also get rid of expansion_tree[-1] because we're gonna make a different choice there in a sec.
                    Finally we move current_inverse_layer back one so that we know we're iterating over the last layer that still has some choices in it.
                    """
                    current_inverse_layer += -1
                    available_edges.pop(-1)
                    index_choices.pop(-1)
                    expansion_tree.pop(-1)

            """
            If we made it here, that means there are some available_edges in the current_inverse_layer.
            This iterates through the different branches from which a new choice can be made at the current inverse_layer.
            """
            for i in reversed(range(len(index_choices[current_inverse_layer]))):
                """
                First we check to see if index_choices can be iterated without changing layers.
                """
                if index_choices[current_inverse_layer][i] == len(available_edges[current_inverse_layer][i])-1:
                    """
                    If we're here, then it means that the index_choice in this branch has reached its maximum value. So we check if there's another
                    branch we can still iterate the index_choice over.
                    """
                    if i == 0:
                        """
                        If we're here, it means that we've iterated over all of the branches in this current_layer. So we need to go back a layer
                        to iterate index_choice there.
                        """
                        if current_inverse_layer == 0:
                            """
                            If we're here, it means we're actually already back at the start, so we're done.
                            """
                            still_iterating = False
                            break
                        else:
                            """
                            If we're here, that means we go back a layer and see if there's anything to iterate over there!
                            We break the for loop because it's kinda meaningless now. It was looping over the branches in this layer, but we're moving
                            back a layer now.
                            """
                            current_inverse_layer += -1
                            available_edges.pop(-1)
                            index_choices.pop(-1)
                            expansion_tree.pop(-1)
                            break
                    else:
                        """
                        If we're here, it means there's another branch to iterate over. We set the index_choice of this branch back to its initial value
                        of -1 (which means that no edges have been chosen from this branch in this layer).
                        The for loop will continue because we need to check the next branch to iterate up by one there.
                        """
                        index_choices[current_inverse_layer][i] = -1
                else:
                    """
                    If we're here, then it means we've found a branch whose index choice can be iterated over!
                    We increase the index choice for this current layer and branch by one.
                    With this choice, we can update the expansion_tree with the index_choices we made.
                    Then we calculate the available_edges in the next layer, and initialise the index_choices for that layer to [[-1], [-1], ...]
                    Finally, the current_inverse_layer is moved forward one.
                    We break the for loop because we're moving layers (we'll go back and check if we have any more layers to go through or if we
                    have a complete contraction tree that we can (possibly) return.
                    """
                    index_choices[current_inverse_layer][i] = index_choices[current_inverse_layer][i] + 1
                    new_layer = []
                    new_available_edges = deepcopy(available_edges[current_inverse_layer])
                    for j in range(len(index_choices[current_inverse_layer])):
                        if index_choices[current_inverse_layer][j] >= 0:
                            new_edge = available_edges[current_inverse_layer][j][index_choices[current_inverse_layer][j]]
                            new_layer.append(new_edge)
                            new_available_edges[j].pop(index_choices[current_inverse_layer][j])
                            if new_edge[0] in children_dict.keys():
                                new_available_edges.append([(i, new_edge[0]) for i in children_dict[new_edge[0]]])
                    expansion_tree.append(new_layer)
                    available_edges.append(new_available_edges)
                    index_choices.append([-1 for _ in range(len(new_available_edges))])
                    current_inverse_layer += 1
                    break