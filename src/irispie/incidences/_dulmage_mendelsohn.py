r"""
"""


#[

from __future__ import annotations

import numpy as np
import networkx as nx
from itertools import product

#]


def dulmage_mendelsohn_reordering(A: np.ndarray) -> tuple[list[int], list[int], np.ndarray]:
    """
    Reorders the incidence matrix using the Dulmage-Mendelsohn decomposition
    to produce the maximal number of smallest sequentially independent blocks.

    Parameters:
        A: np.ndarray
            Binary incidence matrix with shape (m equations, n variables)

    Returns:
        A tuple (row_order, col_order, A_reordered) where:
        - row_order: list of row indices
        - col_order: list of column indices
        - A_reordered: incidence matrix reordered according to row/column permutations
    """
    m, n = A.shape
    B = nx.Graph()

    # Construct bipartite graph: equations as ('e', i), variables as ('v', j)
    eq_idxs, var_idxs = np.where(A)
    for i, j in zip(eq_idxs, var_idxs):
        B.add_edge(("e", int(i)), ("v", int(j)))

    # Find maximum matching on the bipartite graph
    matching = nx.bipartite.maximum_matching(B, top_nodes={("e", i) for i in range(m)})

    # Build directed graph for Dulmage-Mendelsohn decomposition
    G = nx.DiGraph()
    for u, v in B.edges():
        # Reverse direction if edge is in the matching
        if matching.get(u) == v or matching.get(v) == u:
            if u[0] == "e":
                G.add_edge(v, u)
            else:
                G.add_edge(u, v)
        else:
            # Keep direction from equation to variable otherwise
            if u[0] == "e":
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)

    # Unmatched equations and variables
    unmatched_eqs = [("e", i) for i in range(m) if ("e", i) not in matching]
    unmatched_vars = [("v", j) for j in range(n) if ("v", j) not in matching]

    # Compute forward reachable nodes from unmatched equations
    F_reachable = set()
    for e in unmatched_eqs:
        F_reachable.update(nx.descendants(G, e) | {e})

    # Compute backward reachable nodes from unmatched variables (in reversed graph)
    GT = G.reverse()
    T_reachable = set()
    for v in unmatched_vars:
        T_reachable.update(nx.descendants(GT, v) | {v})

    # Remaining nodes define well-determined block(s)
    remaining = set(G.nodes()) - F_reachable - T_reachable

    # Find strongly connected components in the well-determined region
    subgraph = G.subgraph(remaining).copy()
    sccs = list(nx.strongly_connected_components(subgraph))

    # Collect nodes by type from SCCs
    block_eqs = []
    block_vars = []
    block_list = []
    for component in sccs:
        eqs = sorted([n[1] for n in component if n[0] == "e"])
        vars_ = sorted([n[1] for n in component if n[0] == "v"])
        block_eqs.extend(eqs)
        block_vars.extend(vars_)
        block_list.append((eqs, vars_))

    # Extract nodes from F and T partitions
    F_eqs = [n[1] for n in F_reachable if n[0] == "e"]
    F_vars = [n[1] for n in F_reachable if n[0] == "v"]
    T_eqs = [n[1] for n in T_reachable if n[0] == "e"]
    T_vars = [n[1] for n in T_reachable if n[0] == "v"]

    # Final row/column permutation orders
    row_order = F_eqs + block_eqs + T_eqs
    col_order = F_vars + block_vars + T_vars

    # Apply permutation to matrix
    A_reordered = A[np.ix_(row_order, col_order)]

    return row_order, col_order, A_reordered

def extract_dm_blocks(A: np.ndarray, row_order: list[int], col_order: list[int]) -> list[dict[str, object]]:
    """
    Splits a reordered matrix into sequential blocks assuming DM reordering.

    Parameters:
        A: np.ndarray
            Original incidence matrix
        row_order: list[int]
            Row permutation from DM reordering
        col_order: list[int]
            Column permutation from DM reordering

    Returns:
        A list of dictionaries with keys:
        - 'block': submatrix for the block
        - 'rows': list of original row indices in the block
        - 'cols': list of original column indices in the block
    """
    A_reordered = A[np.ix_(row_order, col_order)]

    blocks = []
    start_row = 0
    for i in range(1, len(row_order) + 1):
        if i == len(row_order) or not np.any(A_reordered[start_row:i, :]):
            if i > start_row:
                block = A_reordered[start_row:i, :]
                if block.any():
                    blocks.append({
                        'block': block,
                        'rows': row_order[start_row:i],
                        'cols': col_order
                    })
            start_row = i

    return blocks

# Example usage:
if __name__ == '__main__':
    A = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1]
    ], dtype=bool)
    row_order, col_order, A_reordered = dulmage_mendelsohn_reordering(A)
    print("Row order:", row_order)
    print("Column order:", col_order)
    print("Reordered matrix:\n", A_reordered)

    blocks = extract_dm_blocks(A, row_order, col_order)
    for idx, blk in enumerate(blocks):
        print(f"\nBlock {idx} shape: {blk['block'].shape}")
        print("Rows:", blk['rows'])
        print("Cols:", blk['cols'])
        print(blk['block'])

