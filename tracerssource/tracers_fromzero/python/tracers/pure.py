# contains the pure python functions for the tracers package

from typing import List, Tuple
import networkx as nx
from numpy import delete, unique, abs, diff, array
from tracers import _tracers  


def thaversine_distance(point1: Tuple[float, float, int], point2: Tuple[float, float, int]) -> float:
    """
    haversine distance between two points, but if the time difference between the two points is greater than 20 minutes
    the distance is increased by a very large number
    """
    (x1, y1, t1) = point1
    (x2, y2, t2) = point2

    distance = _tracers.rs_haversine_distance(
        (x1, y1), (x2, y2)) + 0.001 * abs(t2 - t1)

    # if the time difference between the two points is greater than 20 minutes
    # add a very large number to the distance
    if abs(t2 - t1) > 60*20:
        return distance + 100000000  # a very large number

    return distance

# complexity upperbound is O(n^3) where n is the number of points in the trace
# heuristic is guaranteed to find a path of length within 3/2 of the optimal path length
# this method will force the points with the largest time difference to be the first and last points in the path
# if you have a short input and a lot of error on the timestamps, this method will not work well


def tsp_reorder_points(intrace: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    """
    Reorders the points in the given trace to form a traveling salesman problem (TSP) solution.

    Args:
        intrace (List[Tuple[float, float, int]]): The input trace containing points with coordinates and timestamps.

    Returns:
        List[Tuple[float, float, int]]: The reordered trace forming a TSP solution.

    Note:
        complexity upperbound is O(n^3) where n is the number of points in the trace
        heuristic is guaranteed to find a path of length within 3/2 of the optimal path length
        this method will force the points with the largest time difference to be the first and last points in the path
        if you have a short input and a lot of error on the timestamps, this method will not work well

    """
    G = nx.Graph()
    # unique points only
    trace = sorted([(a, b, int(c)) for a, b, c in unique(intrace, axis=0)],key=lambda x: x[2])
    for point in trace:
        G.add_node(node_for_adding=point)

    # add edges between all nodes
    for node1 in trace:
        for node2 in trace:
            if node1 != node2:
                G.add_edge(
                    node1, node2, weight=thaversine_distance(node1, node2))

    # find the tsp solution
    tsp_solution = nx.algorithms.approximation.traveling_salesman_problem(
        G, cycle=True)
    # tsp_solution is a cycle, find the item with the smallest timestamp
    # and rotate the cycle so that the smallest timestamp is first
    tsp_solution = array(tsp_solution)
    # make the cycle a trace again by removing the item that is doubled
    # find the transition with the largest time difference
    max_i, max_item = max(
        enumerate(abs(diff([point[2] for point in tsp_solution]))), key=lambda x: x[1])
    # cut the cycle here by rotating the list
    tsp_solution = tsp_solution[max_i+1:] + tsp_solution[:max_i+1]
    # remove the first doubled item, knowing that they are next to each other
    for i in range(len(tsp_solution)-1):
        if tsp_solution[i] == tsp_solution[i+1]:
            delete(tsp_solution, i)
            break
    # if timestamp of first is greater than timestamp of last, reverse the list
    if tsp_solution[0][2] > tsp_solution[-1][2]:
        tsp_solution = tsp_solution[::-1]

    return [(a[0], a[1], b[2]) for a, b in zip(tsp_solution, trace)]


def tsp_chunked_reorder_points(intrace: List[Tuple[float, float, int]], overlap: int = 200) -> List[Tuple[float, float, int]]:
    """
    Chunked version of tsp_reorder_points, this method will reorder the points in the given trace to form a traveling salesman problem (TSP) solution.
    It cuts on the complexity by assuming points that are far enough apart in time have correct ordering. 
    For this assumption to work, the overlap parameter needs to be large enough.
    If not, this method will fail on asserting that the length of the output trace is equal to the length of the input trace.
    """
    intrace = sorted(unique(intrace, axis=0), key=lambda x: x[2])
    outtrace = intrace.copy()

    # case left border is 0 for first subtrace
    outtrace[0:4 *
             overlap] = tsp_reorder_points(intrace[0:4*overlap])[:4*overlap]

    for i in range(1, len(outtrace)//(2*overlap)):
        outtrace[2*overlap*i:2*overlap*i + 4*overlap] = tsp_reorder_points(
            intrace[2*overlap*i:2*overlap*i + 4*overlap])[overlap:4*overlap]

    assert (len(outtrace) == len(
        intrace)), "length of outtrace and intrace are not equal, this probably means that the overlap region was not large enough"
    return outtrace


def tsp_chopped_reorder_points(intrace: List[Tuple[float, float, int]], T: int = 20*60) -> List[Tuple[float, float, int]]:
    """
    Chopped version of tsp_reorder_points, this method will reorder the points in the given trace to form a traveling salesman problem (TSP) solution.
    See also tsp_chunked_reorder_points.
    This method is the same but instead of taking equal sized chunks, it splits the input trace such that there is at least a x minute time difference between the chunks.
    """

    intrace = sorted(unique(intrace, axis=0), key=lambda x: x[2])
    outtrace = intrace.copy()

    # starting at ts0
    tsiter = intrace[0][2]
    i_splits = [0]

    for i, item in enumerate(intrace):
        if item[2] - tsiter > T:
            tsiter = item[2]
            i_splits.append(i)

    i_splits.append(len(intrace))

    # check if chopping is needed
    if len(i_splits) < 4:
        return tsp_reorder_points(intrace)

    # print(i_splits)

    for L, R, LW in zip(i_splits[:-3],  i_splits[3:], [0]+i_splits[2:]):
        # print(L, LW, R)
        outtrace[LW:R] = tsp_reorder_points(
            intrace[L:R])[LW-L:]

    return outtrace
