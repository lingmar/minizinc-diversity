import datetime
from abc import abstractmethod

import common.common as common


class IterSolver():
    @abstractmethod
    def __init__(self, solver, instance, nsols, timeout):
        self.instance = instance
        self.solver = solver
        self.nsols = nsols
        self.timeout = None if timeout is None else datetime.timedelta(
            seconds=timeout)
        self.store = []

    @abstractmethod
    def solve_first(self):
        pass

    @abstractmethod
    def solve_next(self):
        pass

    def solve(self):
        # TODO: add additional conditions for stopping
        failed = False

        # Given start solutions?
        if common.args.start_sols:
            self.store = common.parse_start_solutions(common.args.start_sols)
        # Otherwise, find first solution
        else:
            first, failed = self.solve_first()
            if not failed:
                self.store = [first]

        # Main loop
        while not failed and len(self.store) < self.nsols:
            r, failed = self.solve_next()
            if not failed:
                self.store += [r]

        return self.store
