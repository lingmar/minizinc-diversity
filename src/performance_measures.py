from typing import List, Tuple
from scipy import integrate


def is_dominated(x, frontier):
    """
    Returns True iff x is dominated by the frontier
    """
    assert len(x) == len(frontier[0])
    dim = len(x)
    for point in frontier:
        if all([pi <= xi for pi, xi in zip(point, x)]):
            return True
    return False


def hypervolume(points):
    dim = len(points[0])
    mxs = [max([p[i] for p in points]) for i in range(dim)]
    ranges = [[0, m] for m in mxs]

    r = integrate.nquad(lambda *args: 1 if is_dominated(args, points) else 0,
                        ranges)
    return r


if __name__ == '__main__':
    h = hypervolume([[1, 1], [1.5, 0], [0, 2], [2, 0]])
    print(h)
    print(
        hypervolume([[1, 1, 1], [1.5, 0, 0.5], [0, 2, 0], [2, 0, 0], [0, 0,
                                                                      2]]))
