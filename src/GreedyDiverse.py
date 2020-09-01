# TODO: add typing
from IterSolver import IterSolver
import minizinc
import common.common as common

from typing import List, Callable

Store = List[str]


class GreedyDiverse(IterSolver):
    normalised_dist_ub = common.DIVERSITY_UB
    unnormalised_dist_ub = common.DIVERSITY_UB

    def __init__(self,
                 solver,
                 files: List[str],
                 first_objective: str,
                 next_objectives: Callable[[Store], str],
                 inter_solution_constraints: List[Callable[[Store], str]],
                 intra_solution_constraints: List[Callable[[Store], str]],
                 timeout,
                 nsols: int,
                 distance_vars,
                 init_scaling,
                 update_scaling,
                 update_ub,
                 search_ann=''):
        model = minizinc.Model(files=files)
        instance = minizinc.Instance(
            model=model, solver=minizinc.Solver.lookup(solver))
        super().__init__(
            minizinc.Solver.lookup(solver), instance, nsols, timeout)

        self.first_objective = first_objective
        self.next_objectives = next_objectives
        self.inter_solution_constraints = inter_solution_constraints
        self.intra_solution_constraints = intra_solution_constraints

        self.scaling = init_scaling
        self.update_scaling = update_scaling
        self.update_ub = update_ub
        self.distance_vars = distance_vars

        self.search_ann = search_ann
        self.distance_type = 'float' if common.args.float else 'int'
        if self.distance_type == 'float':
            self.scaling = list(map(float, self.scaling))

    def solve_first(self):
        r = common.solve_one_instance(
            instance=self.instance,
            solver=self.solver,
            timeout=self.timeout,
            objective=self.first_objective)
        if r.status == minizinc.result.Status.OPTIMAL_SOLUTION or (
                common.args.use_best_found and r.status.has_solution()):
            failed = False
        else:
            failed = True

        return r, failed

    def solve_next(self):
        results = common.lex_max(
            objectives=[
                o(self.store, self.scaling) for o in self.next_objectives
            ],
            instance=self.instance,
            solver=self.solver,
            timeout=self.timeout,
            add_to_model=[
                c(self.store) for c in self.inter_solution_constraints
            ] + [c(self.store) for c in self.intra_solution_constraints] +
            [self.define_scaling_array()] + [common.include_distance_functions()],
            relax=common.args.lex_float_relax
            if self.distance_type == 'float' else None)

        if common.args.lex == 'seq':
            if None in results:
                return None, True
            optimal_result = results[-1]
        else:
            if results is None:
                return None, True
            optimal_result = results

        if common.args.report_diversity:
            GreedyDiverse.unnormalised_dist_ub, GreedyDiverse.normalised_dist_ub = self.update_ub(
                optimal_result, self.store, self.scaling)

        self.update_scaling(optimal_result, self.scaling)
        return optimal_result, False

    def define_scaling_array(self):
        return 'array[int] of {distance_type}: scaling {ann} = {scaling};'.format(
            scaling=self.scaling,
            distance_type=self.distance_type,
            ann=':: add_to_output' if common.args.dist_add_to_output else '')
