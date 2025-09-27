#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finite-State Machines for automatic MPO generation.

This module defines the classes and logic for building a Matrix Product Operator
(MPO) representation of a Hamiltonian using a Finite-State Machine (FSM).
The FSM is constructed by adding Hamiltonian terms one by one, with the FSM
graph being automatically compressed to minimize the resulting MPO bond dimension.
"""

import numpy as np


def bound_sort(list_a, list_b):
    """Sorts two lists simultaneously based on the values in the first list.

    Args:
        list_a (list): The list whose values will determine the sorting order.
        list_b (list): The list to be sorted in correspondence with list_a.

    Returns:
        tuple[list, list]: A tuple containing the two lists after being sorted
            according to the ascending order of `list_a`.
    """
    # zip() packs elements into tuples, sorted() sorts them based on the
    # first element of each tuple, and zip(*) unpacks them back into lists.
    list_a, list_b = (list(bound_element) for bound_element in zip(*sorted(zip(list_a, list_b))))
    return list_a, list_b


def is_ordered(int_list):
    """Checks if a list of integers is in non-decreasing order.

    Args:
        int_list (list[int]): The list of integers to check.

    Returns:
        bool: True if the list is sorted from smallest to largest, False otherwise.
    """
    for i in range(len(int_list) - 1):
        if int_list[i] >= int_list[i + 1]:
            return False
    return True


def is_close(a, b, abs_tol=1e-14):
    """Checks if two floating-point numbers are close to each other.

    Args:
        a (float): The first number.
        b (float): The second number.
        abs_tol (float): The absolute tolerance for the comparison.

    Returns:
        bool: True if the absolute difference between `a` and `b` is less than
            `abs_tol`, False otherwise.
    """
    return abs(a - b) < abs_tol


class NamedData:
    """A wrapper class to associate a string name with a data object.

    This allows for quick equality checks based on the name string, which is
    more efficient than comparing the data arrays directly, especially for
    identifying operators like 'Sz' or 'Id'.
    """

    def __init__(self, name_str, data):
        """Initializes the NamedData instance.

        Args:
            name_str (str): The name to associate with the data.
            data: The data object, typically a NumPy array for an operator.
        """
        self.name = name_str
        self.data = data

    def __eq__(self, other):
        """Two NamedData objects are equal if they have the same name."""
        return self.name == other.name


class FsmEdge:
    """Represents a directed edge in the Finite-State Machine graph.

    An edge connects two `FsmNode` objects and is associated with a local
    operator and a numerical weight, which together form a component of the MPO.

    Attributes:
        named_op (NamedData): The local operator on this edge.
        weight (float): The numerical weight or coefficient on this edge.
        node_from (FsmNode): The node from which this edge originates.
        node_to (FsmNode): The node to which this edge points.
        out_idx (int): The index of this edge in the outgoing edges list of
            `node_from`.
        in_idx (int): The index of this edge in the incoming edges list of
            `node_to`.
    """

    def __init__(self, named_op, weight, node_from, node_to):
        """Initializes an FsmEdge instance.

        Args:
            named_op (NamedData): The local operator to associate with this edge.
            weight (float): The numerical weight for this edge.
            node_from (FsmNode): The starting node of the edge.
            node_to (FsmNode): The ending node of the edge.
        """
        # --- Edge Attributes: Operator and Weight ---
        self.named_op = named_op
        self.weight = weight

        # --- Edge Connectivity: Start and End Nodes ---
        # Defines the direction: (node_from) --edge--> (node_to)
        self.node_from = node_from
        self.node_to = node_to

        # --- Edge Indexing for fast access during graph manipulation ---
        # This edge is the `out_idx`-th outgoing edge of `node_from`.
        self.out_idx = node_from.out_degree()
        # This edge is the `in_idx`-th incoming edge of `node_to`.
        self.in_idx = node_to.in_degree()

    def is_equal(self, other):
        """Checks if another edge has the same operator and weight.

        This method is used to identify edges that can be merged during the
        FSM compression.

        Args:
            other (FsmEdge): The other edge to compare against.

        Returns:
            bool: True if both edges have the same operator and a close enough
                weight, False otherwise.
        """
        return (self.named_op == other.named_op) and is_close(self.weight, other.weight)

    def matches(self, wanted_named_op, wanted_weight):
        """Checks if this edge matches a specific operator and weight.

        Args:
            wanted_named_op (NamedData): The desired operator.
            wanted_weight (float): The desired weight.

        Returns:
            bool: True if the edge's attributes match the desired ones.
        """
        return (self.named_op == wanted_named_op) and is_close(self.weight, wanted_weight)


class FsmNode:
    """Represents a node (or state) in the Finite-State Machine.

    Each node exists in a specific "layer" of the FSM, which corresponds to a
    physical site in the quantum lattice.

    Attributes:
        layer_idx (int): The index of the layer this node belongs to, corresponding
            to the physical site index.
        bond_idx (int): The index of this node within its layer. This corresponds
            to a row/column index in the final MPO tensor.
        list_edge_out (list[FsmEdge]): A list of edges originating from this node.
        list_edge_in (list[FsmEdge]): A list of edges pointing to this node.
    """

    def __init__(self, layer_idx, bond_idx):
        """Initializes an FsmNode instance.

        Args:
            layer_idx (int): The layer index for the node (physical site).
            bond_idx (int): The bond index for the node within the layer.
        """
        # The (layer_idx, bond_idx) tuple uniquely identifies the node's position.
        self.layer_idx = layer_idx
        self.bond_idx = bond_idx

        # Lists to store connections to other nodes.
        self.list_edge_out = []
        self.list_edge_in = []

    def __eq__(self, other):
        """Checks if two nodes are identical based on their location."""
        return (self.layer_idx == other.layer_idx) and (self.bond_idx == other.bond_idx)

    def out_degree(self):
        """Returns the number of outgoing edges."""
        return len(self.list_edge_out)

    def in_degree(self):
        """Returns the number of incoming edges."""
        return len(self.list_edge_in)

    def new_edge_out(self, next_node, named_op, weight):
        """Creates a new outgoing edge from this node to another.

        This is the primary method for building connections in the FSM graph.

        Args:
            next_node (FsmNode): The destination node for the new edge.
            named_op (NamedData): The operator for the new edge.
            weight (float): The weight for the new edge.

        Returns:
            FsmEdge: The newly created edge object.
        """
        # Create the edge instance, linking self to the next node.
        new_edge = FsmEdge(named_op, weight, self, next_node)

        # Register this new edge on both the source and destination nodes.
        self.list_edge_out.append(new_edge)
        next_node.list_edge_in.append(new_edge)
        return new_edge

    def search_edge_out(self, wanted_named_op, wanted_weight):
        """Finds the first outgoing edge matching an operator and weight.

        Args:
            wanted_named_op (NamedData): The operator to search for.
            wanted_weight (float): The weight to search for.

        Returns:
            FsmEdge | None: The first matching edge, or None if not found.
        """
        for edge in self.list_edge_out:
            if edge.matches(wanted_named_op, wanted_weight):
                return edge
        return None

    def search_edge_in(self, wanted_named_op, wanted_weight):
        """Finds the first incoming edge matching an operator and weight.

        Args:
            wanted_named_op (NamedData): The operator to search for.
            wanted_weight (float): The weight to search for.

        Returns:
            FsmEdge | None: The first matching edge, or None if not found.
        """
        for edge in self.list_edge_in:
            if edge.matches(wanted_named_op, wanted_weight):
                return edge
        return None

    def search_host_edge_out(self, merged_edge):
        """Finds a suitable existing outgoing edge to merge into.

        A "host" edge is an existing edge that has the same operator and weight
        as `merged_edge` but points to a different destination node. This is a
        key step in the forward merging (compression) process.

        Args:
            merged_edge (FsmEdge): The new edge that is a candidate for merging.

        Returns:
            FsmEdge | None: A suitable host edge, or None if none exists.
        """
        for edge in self.list_edge_out:
            if merged_edge.is_equal(edge) and (edge.node_to != merged_edge.node_to):
                return edge
        return None

    def search_host_edge_in(self, merged_edge):
        """Finds a suitable existing incoming edge to merge into.

        A host edge is an existing edge with the same properties as `merged_edge`
        but originating from a different source node. Used in backward merging.

        Args:
            merged_edge (FsmEdge): The new edge that is a candidate for merging.

        Returns:
            FsmEdge | None: A suitable host edge, or None if none exists.
        """
        for edge in self.list_edge_in:
            if merged_edge.is_equal(edge) and (edge.node_from != merged_edge.node_from):
                return edge
        return None

    def search_op_out(self, wanted_named_op):
        """Finds the first outgoing edge with a specific operator, ignoring weight.

        This is used to find existing paths for combining like terms.

        Args:
            wanted_named_op (NamedData): The operator to search for.

        Returns:
            FsmEdge | None: The first matching edge, or None if not found.
        """
        for edge in self.list_edge_out:
            if edge.named_op == wanted_named_op:
                return edge
        return None


class FSM:
    """A Finite-State Machine for building Matrix Product Operators.

    This class manages the entire FSM graph, which represents a Hamiltonian.
    Users can add terms of the Hamiltonian, and the class automatically builds
    and compresses the FSM. Finally, it can generate the MPO representation.

    The FSM consists of `site_num + 1` layers of nodes, from layer 0 to
    `site_num`. Layer `i` is connected to layer `i+1` by edges that represent
    the local operators at physical site `i`.

    Attributes:
        site_num (int): The number of physical sites in the system.
        head (FsmNode): The single starting node at layer 0.
        tail (FsmNode): The single ending node at the last layer.
        locator (list[list[FsmNode]]): A 2D list where `locator[i]` contains
            all nodes in layer `i`. This provides fast access to any node.
        name_Id (str): The string identifier for the identity operator.
    """

    def __init__(self, site_num, identity_name='Id'):
        """Initializes the FSM instance for a given system size.

        Args:
            site_num (int): The number of sites in the physical system.
            identity_name (str, optional): The name for the identity operator.
                Defaults to 'Id'.
        """
        # The number of sites determines the number of MPO tensors.
        # The number of node layers = site_num + 1 (from layer 0 to site_num).
        self.site_num = site_num

        # The FSM always has a single entry (head) and a single exit (tail) point.
        self.head = FsmNode(layer_idx=0, bond_idx=0)
        self.tail = FsmNode(layer_idx=site_num, bond_idx=0)

        # `locator` is a data structure for efficient node management. It's a
        # list of lists, where each inner list holds the nodes of a layer.
        self.locator = [[] for _ in range(site_num + 1)]
        self.locator[0].append(self.head)
        self.locator[-1].append(self.tail)

        # Store the name of the identity operator for consistent use.
        self.name_Id = identity_name

    def add_term(self, coef, phys_ops, sites, insert_ops=None, print_form=None):
        """Adds a new term of the Hamiltonian to the FSM.

        This is the main user-facing method. Each call adds one term, such as
        `J * Sz_i * Sz_j`, to the machine, which updates the graph structure.

        Args:
            coef (float): The coefficient of the Hamiltonian term.
            phys_ops (list[NamedData]): A list of the physical operators in
                the term (e.g., [Sz, Sz]).
            sites (list[int]): A list of the site indices where the physical
                operators act (e.g., [i, j]).
            insert_ops (list[NamedData], optional): A list of operators to
                insert between the physical operators. If None, identity
                operators are used by default.
            print_form (str, optional): Controls printing of the added term.
                Can be 'phys' (only physical operators) or 'all' (all operators
                on the chain). Defaults to None.
        """
        # --- Pre-computation and Validation ---
        if is_close(coef, 0):
            return  # Ignore terms with a coefficient of zero.

        # If no inserted operators are provided, fill with Identity operators.
        # The structure is: insert_op - phys_op - insert_op - ... - phys_op - insert_op
        if insert_ops is None:
            opr_shape = phys_ops[0].data.shape
            identity_op = NamedData(self.name_Id, np.identity(opr_shape[0]))
            insert_ops = [identity_op] * (len(phys_ops) + 1)

        assert len(phys_ops) == len(sites), \
            "Error: Unmatched length of physical operators and site lists."
        assert len(phys_ops) + 1 == len(insert_ops), \
            "Error: Unmatched length of inserted operators list."
        for site in sites:
            assert site < self.site_num, \
                f"Error: Site index {site} exceeds system size {self.site_num}."

        # Ensure operators are sorted by site index for consistent path creation.
        if not is_ordered(sites):
            sites, phys_ops = bound_sort(sites, phys_ops)
            print("----Warning: Physical operators were not in order and have been "
                  "sorted automatically. Inserted operators remain unchanged.----")

        # --- Prepare Operator and Weight Lists for Path Creation ---
        # Create a list of weights, one for each site. Default is 1.0.
        list_weight = [1.0] * self.site_num
        # By convention, the term's coefficient is placed on the weight of the
        # edge corresponding to the first physical operator.
        if not sites:  # Handle terms with no physical operators (e.g., global identity).
            if self.site_num > 0: list_weight[0] = coef
        else:
            list_weight[sites[0]] = coef

        # Assemble the full operator string for all sites from 0 to N-1.
        list_named_op = []
        phys_op_idx = 0
        for site_idx in range(self.site_num):
            is_phys_site = (phys_op_idx < len(sites) and site_idx == sites[phys_op_idx])
            if is_phys_site:
                # Place a physical operator at this site.
                list_named_op.append(phys_ops[phys_op_idx])
                phys_op_idx += 1
            else:
                # Fill with an inserted (usually identity) operator.
                list_named_op.append(insert_ops[phys_op_idx])

        # Optional: Print the term that was just processed for debugging.
        if print_form == 'all':
            print(f"{coef} {' '.join(op.name for op in list_named_op)}")
        elif print_form == 'phys':
            phys_terms = [f"{op.name}_{site}" for op, site in zip(phys_ops, sites)]
            print(f"{coef} {' '.join(phys_terms)}")

        # Create a new path in the FSM for this complete operator string.
        self._new_path(list_named_op, list_weight)

    def _new_path(self, list_named_op, list_weight):
        """Creates a new path in the FSM for a Hamiltonian term.

        This method first attempts to combine the term with an existing identical
        path ("like terms"). If no such path exists, it creates a new path of
        nodes and edges, and then attempts to compress the graph by merging
        redundant parts of this new path with existing paths.

        Args:
            list_named_op (list[NamedData]): The full list of operators for
                each site.
            list_weight (list[float]): The list of weights for each site's edge.
        """
        assert len(list_named_op) == self.site_num, "Operator list length mismatch."

        # A list to keep track of the new edges created for this path.
        new_edge_register = []

        # First, check if this exact operator sequence already exists.
        if self._combine_like_term(list_named_op, list_weight):
            return  # If so, weights are combined, and we are done.

        # --- If it's a new term, build a new path from head to tail ---
        last_node = self.head
        # Iterate through sites 0 to site_num-2 to create intermediate nodes/edges.
        for layer_idx in range(self.site_num - 1):
            the_node = self._new_successor(
                last_node, list_named_op[layer_idx], list_weight[layer_idx])
            new_edge_register.append(the_node.list_edge_in[0])
            last_node = the_node

        # Connect the last intermediate node to the tail node.
        final_edge = self._new_edge(
            last_node, self.tail, list_named_op[self.site_num - 1],
            list_weight[self.site_num - 1])
        new_edge_register.append(final_edge)

        # --- Compress the FSM by merging the new path where possible ---
        # 1. Forward merge: from head towards tail.
        for new_edge in new_edge_register:
            if new_edge.node_from.in_degree() > 1: break
            host_edge = new_edge.node_from.search_host_edge_out(new_edge)
            if host_edge is None or host_edge.node_to.in_degree() > 1: break

            is_merged = self._merge_successor(new_edge, host_edge)
            if not is_merged: break

        # 2. Backward merge: from tail towards head.
        new_edge_register.reverse()
        for new_edge in new_edge_register:
            if new_edge.node_from.out_degree() > 1: break
            host_edge = new_edge.node_to.search_host_edge_in(new_edge)
            if host_edge is None or host_edge.node_from.out_degree() > 1: break

            is_merged = self._merge_precursor(new_edge, host_edge)
            if not is_merged: break

    def _combine_like_term(self, list_named_op, list_weight):
        """Checks for and combines "like terms".

        If a path with the exact same sequence of operators already exists, this
        method finds it and adds the new coefficient to the weight of the
        first non-identity operator's edge.

        Args:
            list_named_op (list[NamedData]): The sequence of operators.
            list_weight (list[float]): The sequence of weights.

        Returns:
            bool: True if a like term was found and combined, False otherwise.
        """
        current_node = self.head
        first_phys_edge = None

        # Traverse the FSM following the operator sequence.
        for layer_idx in range(self.site_num):
            # Find an outgoing edge with the matching operator.
            next_edge = current_node.search_op_out(list_named_op[layer_idx])
            if next_edge is None:
                return False  # Path deviates, so it's not a like term.

            # Path continues, move to the next node.
            current_node = next_edge.node_to
            # Keep track of the first edge with a physical (non-Id) operator.
            if (list_named_op[layer_idx].name != self.name_Id) and (first_phys_edge is None):
                first_phys_edge = next_edge

        # If the entire path was traversed, it's a like term.
        # Add the new coefficient to the weight of the first physical edge.
        if first_phys_edge:
            first_phys_edge.weight += list_weight[first_phys_edge.node_from.layer_idx]
            print("----Warning: A like term was found and has been combined automatically.----")
            return True
        return False

    @staticmethod
    def _new_edge(node_from, node_to, named_op, weight):
        """Creates a single new edge between two existing nodes."""
        return node_from.new_edge_out(node_to, named_op, weight)

    def _new_successor(self, last_node, named_op, weight):
        """Creates a new node in the next layer and an edge connecting to it."""
        # Determine the position (layer and bond index) of the new node.
        new_node_layer_idx = last_node.layer_idx + 1
        new_node_bond_idx = len(self.locator[new_node_layer_idx])

        # Create the new node and register it in the locator.
        new_node = FsmNode(new_node_layer_idx, new_node_bond_idx)
        self.locator[new_node_layer_idx].append(new_node)

        # Create the edge connecting the previous node to the new one.
        self._new_edge(last_node, new_node, named_op, weight)
        return new_node

    @staticmethod
    def _precursor_coincide(node0, node1):
        """Checks if two nodes share any immediate common precursor (parent) node."""
        for edge0 in node0.list_edge_in:
            for edge1 in node1.list_edge_in:
                if edge0.node_from == edge1.node_from:
                    return True
        return False

    @staticmethod
    def _successor_coincide(node0, node1):
        """Checks if two nodes share any immediate common successor (child) node."""
        for edge0 in node0.list_edge_out:
            for edge1 in node1.list_edge_out:
                if edge0.node_to == edge1.node_to:
                    return True
        return False

    def _merge_precursor(self, merged_edge, host_edge):
        """Merges the precursor node of `merged_edge` into the precursor of `host_edge`.

        This is part of the backward merge process. It redirects all incoming
        edges from the `merged_node` to the `host_node` and removes the now
        redundant `merged_node`.

        Args:
            merged_edge (FsmEdge): The edge whose starting node will be merged.
            host_edge (FsmEdge): The edge whose starting node will receive the
                connections.

        Returns:
            bool: True if the merge was successful, False otherwise.
        """
        host_node = host_edge.node_from
        merged_node = merged_edge.node_from

        # --- Pre-merge Sanity Checks ---
        assert host_node.layer_idx == merged_node.layer_idx
        assert host_node.out_degree() <= 1, "Unexpected cross-edge at host node."
        assert merged_node.out_degree() <= 1, "Unexpected cross-edge at merged node."
        if self._precursor_coincide(host_node, merged_node):
            return False  # Cannot merge if nodes share a parent.

        # --- Perform the Merge ---
        # 1. Delete the edge leading from the node-to-be-merged.
        self._del_backward_merged_edge(merged_edge)
        # 2. Reroute all incoming edges from the merged_node to the host_node.
        self._deliver_edge_in(merged_node, host_node)
        # 3. Update bond indices of subsequent nodes in the same layer.
        for node in self.locator[merged_node.layer_idx][(merged_node.bond_idx + 1):]:
            node.bond_idx -= 1
        # 4. Remove the redundant node from the locator.
        del self.locator[merged_node.layer_idx][merged_node.bond_idx]
        del merged_node
        return True

    def _merge_successor(self, merged_edge, host_edge):
        """Merges the successor node of `merged_edge` into the successor of `host_edge`.

        This is part of the forward merge process. It redirects all outgoing
        edges from the `merged_node` to the `host_node` and removes the
        redundant `merged_node`.

        Args:
            merged_edge (FsmEdge): The edge whose destination node will be merged.
            host_edge (FsmEdge): The edge whose destination node will receive the
                connections.

        Returns:
            bool: True if the merge was successful, False otherwise.
        """
        host_node = host_edge.node_to
        merged_node = merged_edge.node_to

        # --- Pre-merge Sanity Checks ---
        assert host_node.layer_idx == merged_node.layer_idx
        assert host_node.in_degree() <= 1, "Unexpected cross-edge at host node."
        assert merged_node.in_degree() <= 1, "Unexpected cross-edge at merged node."
        if self._successor_coincide(host_node, merged_node):
            return False  # Cannot merge if nodes share a child.

        # --- Perform the Merge ---
        # 1. Delete the edge leading to the node-to-be-merged.
        self._del_forward_merged_edge(merged_edge)
        # 2. Reroute all outgoing edges from the merged_node to the host_node.
        self._deliver_edge_out(merged_node, host_node)
        # 3. Update bond indices of subsequent nodes in the same layer.
        for node in self.locator[merged_node.layer_idx][(merged_node.bond_idx + 1):]:
            node.bond_idx -= 1
        # 4. Remove the redundant node from the locator.
        del self.locator[merged_node.layer_idx][merged_node.bond_idx]
        del merged_node
        return True

    @staticmethod
    def _del_forward_merged_edge(merged_edge):
        """Helper to clean up connections for a forward merge."""
        for edge in merged_edge.node_from.list_edge_out[(merged_edge.out_idx + 1):]:
            edge.out_idx -= 1
        del merged_edge.node_from.list_edge_out[merged_edge.out_idx]
        del merged_edge.node_to.list_edge_in[:]

    @staticmethod
    def _del_backward_merged_edge(merged_edge):
        """Helper to clean up connections for a backward merge."""
        for edge in merged_edge.node_to.list_edge_in[(merged_edge.in_idx + 1):]:
            edge.in_idx -= 1
        del merged_edge.node_to.list_edge_in[merged_edge.in_idx]
        del merged_edge.node_from.list_edge_out[:]

    @staticmethod
    def _deliver_edge_out(merged_node, host_node):
        """Reroutes all outgoing edges from `merged_node` to `host_node`."""
        for edge in merged_node.list_edge_out:
            edge.node_from = host_node
            edge.out_idx = host_node.out_degree()
            host_node.list_edge_out.append(edge)
        del merged_node.list_edge_out[:]

    @staticmethod
    def _deliver_edge_in(merged_node, host_node):
        """Reroutes all incoming edges from `merged_node` to `host_node`."""
        for edge in merged_node.list_edge_in:
            edge.node_to = host_node
            edge.in_idx = host_node.in_degree()
            host_node.list_edge_in.append(edge)
        del merged_node.list_edge_in[:]

    def to_mpo(self, sort_order=None):
        """Generates the list of MPO tensors from the FSM graph.

        This method traverses the final, compressed FSM graph layer by layer and
        constructs the MPO tensors.

        Args:
            sort_order (str, optional): Specifies how to sort nodes within each
                layer before generation. 'top' or 'bottom' can produce more
                structured MPOs (e.g., upper or lower triangular).
                Defaults to None.

        Returns:
            list[np.ndarray]: A list of the MPO tensors. Each tensor is a
                4-index numpy array with shape (dim_in, dim_out, d, d),
                where d is the local physical dimension.
        """
        if sort_order is not None:
            self._sort_layers(sort_order)

        opr_shape = self.head.list_edge_out[0].named_op.data.shape
        list_mpo = []

        # Iterate through each layer, corresponding to a physical site.
        for layer_idx in range(self.site_num):
            dim_in = len(self.locator[layer_idx])
            dim_out = len(self.locator[layer_idx + 1])
            # Initialize MPO tensor: shape (in, out, phys, phys)
            mpo_tensor_np = np.zeros((dim_in, dim_out) + opr_shape, dtype=complex)

            # Populate the MPO tensor based on the edges between layers.
            for node in self.locator[layer_idx]:
                for edge in node.list_edge_out:
                    i = edge.node_from.bond_idx
                    j = edge.node_to.bond_idx
                    mpo_tensor_np[i, j, :, :] = edge.weight * edge.named_op.data

            list_mpo.append(mpo_tensor_np)
        return list_mpo

    def to_symbolic_mpo(self, sort_order='bottom'):
        """Generates a symbolic representation of the MPO.

        This is a useful debugging tool to visualize the MPO's structure.

        Args:
            sort_order (str, optional): Node sorting order ('top' or 'bottom').
                Defaults to 'bottom'.

        Returns:
            list[list[list[str]]]: A nested list where each element is a
                string like "1.0*Sz", representing the MPO tensor elements.
        """
        if sort_order is not None:
            self._sort_layers(sort_order)

        list_symbol_mpo = []
        for layer_idx in range(self.site_num):
            dim_in = len(self.locator[layer_idx])
            dim_out = len(self.locator[layer_idx + 1])
            symbol_tensor = [['0' for _ in range(dim_out)] for _ in range(dim_in)]

            for node in self.locator[layer_idx]:
                for edge in node.list_edge_out:
                    i = edge.node_from.bond_idx
                    j = edge.node_to.bond_idx
                    symbol_tensor[i][j] = f"{edge.weight:.2f}*{edge.named_op.name}"
            list_symbol_mpo.append(symbol_tensor)
        return list_symbol_mpo

    def print_symbolic_mpo(self, sort_order='bottom'):
        """Prints the symbolic MPO to the console in a readable format."""
        symbol_mpo = self.to_symbolic_mpo(sort_order)
        print('')
        for layer_idx, tensor in enumerate(symbol_mpo):
            print(f"--- MPO at Site {layer_idx} ---")
            for row in tensor:
                print(row)
            print('')

    def print_bond_dimensions(self):
        """Prints the bond dimension at each cut of the chain."""
        print('')
        bond_dims = [len(layer) for layer in self.locator]
        print("Bond Dimensions: ", ' '.join(map(str, bond_dims)))
        print('')

    def _sort_layers(self, sort_order='bottom'):
        """Sorts nodes within each layer to achieve a canonical MPO form.

        This method identifies the path of identity operators through the FSM
        and moves the nodes on this path to either the top or bottom of their
        respective layers. This often results in an MPO that is upper or lower
        triangular.

        Args:
            sort_order (str): Either 'top' or 'bottom'. Specifies where to move
                the identity path nodes.

        Raises:
            ValueError: If an invalid `sort_order` is provided.
        """
        opr_shape = self.head.list_edge_out[0].named_op.data.shape
        identity_op = NamedData(self.name_Id, np.identity(opr_shape[0]))

        if sort_order == 'top':
            current_node = self.head
            while True:
                id_edge = current_node.search_edge_out(identity_op, 1.0)
                if id_edge is None: break
                if id_edge.node_to.bond_idx != 0:
                    self._move_node_to_top(id_edge.node_to)
                current_node = id_edge.node_to
            current_node = self.tail
            while True:
                id_edge = current_node.search_edge_in(identity_op, 1.0)
                if id_edge is None: break
                if id_edge.node_from.bond_idx != len(self.locator[id_edge.node_from.layer_idx]) - 1:
                    self._move_node_to_bottom(id_edge.node_from)
                current_node = id_edge.node_from

        elif sort_order == 'bottom':
            current_node = self.head
            while True:
                id_edge = current_node.search_edge_out(identity_op, 1.0)
                if id_edge is None: break
                num_nodes = len(self.locator[id_edge.node_to.layer_idx])
                if id_edge.node_to.bond_idx != num_nodes - 1:
                    self._move_node_to_bottom(id_edge.node_to)
                current_node = id_edge.node_to
            current_node = self.tail
            while True:
                id_edge = current_node.search_edge_in(identity_op, 1.0)
                if id_edge is None: break
                if id_edge.node_from.bond_idx != 0:
                    self._move_node_to_top(id_edge.node_from)
                current_node = id_edge.node_from
        else:
            raise ValueError(f"Error: Illegal sort_order '{sort_order}'!")

    def _move_node_to_top(self, node_to_move):
        """Moves a given node to the top (index 0) of its layer."""
        layer = self.locator[node_to_move.layer_idx]
        old_idx = node_to_move.bond_idx

        # Increment bond_idx of nodes currently above the target node.
        for node in layer[0:old_idx]:
            node.bond_idx += 1
        # Remove node from its old position and re-insert at the top.
        del layer[old_idx]
        node_to_move.bond_idx = 0
        layer.insert(0, node_to_move)

    def _move_node_to_bottom(self, node_to_move):
        """Moves a given node to the bottom (last index) of its layer."""
        layer = self.locator[node_to_move.layer_idx]
        old_idx = node_to_move.bond_idx

        # Decrement bond_idx of nodes currently below the target node.
        for node in layer[(old_idx + 1):]:
            node.bond_idx -= 1
        # Remove node from its old position and append to the end.
        del layer[old_idx]
        node_to_move.bond_idx = len(layer)
        layer.append(node_to_move)
