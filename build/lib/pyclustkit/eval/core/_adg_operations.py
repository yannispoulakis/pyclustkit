from collections import defaultdict, deque
import networkx as nx
import io
from pyclustkit.eval.core._shared_processes import process_adg
# from PIL import Image


def get_subgraph(processes, target):
    """

    :param processes:
    :type processes: dict
    :param target:
    :type target: str
    :return:
    :rtype:
    """
    assert target in processes.keys(), f'{target} is not a valid process'
    subgraph = defaultdict(list)
    visited = set()
    queue = deque([target])

    while queue:
        process = queue.popleft()
        if process not in visited:
            visited.add(process)
            if process not in subgraph:
                subgraph[process] = []
            for dep in processes[process]:
                subgraph[process].append(dep)
                queue.append(dep)
    return subgraph


def topological_sort(subgraph):
    indegree = {u: 0 for u in subgraph}  # Initialize indegrees of all vertices to 0
    for u in subgraph:
        for v in subgraph[u]:
            indegree[v] += 1  # Compute indegree of each vertex

    queue = [u for u in subgraph if indegree[u] == 0]  # Collect all vertices with 0 indegree
    top_order = []

    while queue:
        u = queue.pop(0)
        top_order.append(u)

        # Decrease the indegree of all the neighbors
        for v in subgraph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if len(top_order) == len(subgraph):
        return top_order
    else:
        raise Exception("Graph has a cycle, topological sorting not possible.")


def execute_graph(sorted_subgraph, operations_dict):
    """

    Args:
        sorted_subgraph ():
        operations_dict ():

    Returns:

    """
    sorted_subgraph.reverse()
    process_results = []
    for process in sorted_subgraph:
        if operations_dict[process]['value'] is None:
            parameters = [operations_dict[x]['value'] for x in operations_dict[process]['requires']]
            operations_dict[process]['value'] = operations_dict[process]['method'](*parameters)
            process_results.append(operations_dict[process]['value'])
        else:
            process_results.append(operations_dict[process]['value'])
    return dict(zip(sorted_subgraph,process_results))


# Visualization method removed to minimize requirements
"""
def visualize_subgraph_as_tree(target, layout='-Grankdir=LR', save_img_path=None, return_as_bytes=True):

    subgraph = get_subgraph(process_adg, target)

    # Create a directed graph
    G = nx.DiGraph()
    # - Add edges to the graph
    for process, dependencies in subgraph.items():
        for dep in dependencies:
            G.add_edge(dep, process)

    # Use a tree layout (dot, hierarchy) for better tree visualization
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args=layout)  # 'dot' is good for tree structures

    # Draw the tree
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=3000,
            font_size=10, font_weight='bold', arrows=True, arrowsize=15)
    plt.title(f"Dependency Tree for Process '{target}'")

    if save_img_path is not None:
        plt.savefig(save_img_path
                    )
    if return_as_bytes:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()  # Close the plot to avoid display in notebooks or other environments
        img = Image.open(buf)
        return img
    else:
        return None
"""