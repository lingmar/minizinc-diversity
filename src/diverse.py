#!/usr/bin/env python3

import random
from functools import reduce

import minizinc

import common.common as common
import performance_measures
from GreedyDiverse import GreedyDiverse
from Parser import Parser

HELPER_ARRAY = 'dist'
INT_MAX = 50000000


def compute_dists(dist_funs, store):
    dists = [compute_dist(dist_fun, store) for dist_fun in dist_funs]
    ret = 'array[1..{distances}, 1..{nsolutions}] of var {lb}{suffix}..{ub}{suffix}: {helper_array} {annotation};'.format(
        distances=len(dist_funs),
        nsolutions=len(store),
        helper_array=HELPER_ARRAY,
        lb=common.DIVERSITY_LB,
        ub=common.DIVERSITY_UB,
        suffix='.0' if common.args.float else '',
        annotation=':: add_to_output'
        if common.args.dist_add_to_output else '')
    for i in range(len(dist_funs)):
        ret += 'constraint {dist}[{measureidx},1..{nsolutions}] = {expr};'.format(dist=HELPER_ARRAY,
            measureidx=i + 1, nsolutions=len(store), expr=dists[i])

    return ret


def compute_dist(dist_fun, store):
    var_name, fun_name = dist_fun
    distances = [
        '{f}({var},{svar})'.format(f=fun_name, var=var_name, svar=s[var_name])
        for s in store
    ]
    ret = '[{}]'.format(','.join(distances))
    common.log.debug('compute_dist({}, {}, {}) = {}'.format(
        var_name, fun_name, store, ret))
    return ret


def objective_diversity_measure(objectives, store):
    def str_diff(e1, e2):
        return '({}) - ({})'.format(e1, e2)

    def inner_min(s):
        return 'max ([{}])'.format(','.join(
            [str_diff(s[o], o) for o in objectives]))

    return '[{}]'.format(','.join([inner_min(s) for s in store]))


def mult_comprehension(array):
    product = reduce(lambda x, y: x * y, array)
    common.log.debug('product is {}'.format(product))
    return product


def gen_weighted_sum(variables, weights):
    return ' + '.join([
        '{weight}*({var})'.format(weight=w, var=v)
        for w, v in zip(weights, variables)
    ])


def combine_and_aggregate(combinator,
                          aggregator,
                          scaling,
                          scaling_var_name='scaling'):
    assert combinator, 'Combinator cannot be empty'
    assert aggregator, 'Aggregator cannot be empty'

    # Improvement for floating points: don't use min function
    if aggregator == 'min' and combinator == 'min' and common.args.float:
        return 'FLOAT_OBJECTIVE'

    if not common.args.auto_scaling:
        return '{aggregator}([{combinator}([{dist}[m, s] | m in index_set_1of2({dist})]) | s in index_set_2of2({dist})])'.format(
            aggregator=aggregator, combinator=combinator, dist=HELPER_ARRAY)

    elif common.args.float:
        return '{aggregator}([{combinator}([{dist}[m, s] / {scaling_var}[m] | m in index_set_1of2({dist})]) | s in index_set_2of2({dist})])'.format(
            aggregator=aggregator,
            combinator=combinator,
            dist=HELPER_ARRAY,
            scaling_var=scaling_var_name)

    else:
        product = mult_comprehension(scaling)

        compensation = 1
        while product >= INT_MAX:
            product = int(product / 1000)
            compensation *= 1000

        return '{aggregator}([{combinator}([{dist}[m, s] * ({scaling_factors} div ({scaling_var}[m] div {compensation})) | m in index_set_1of2({dist})]) | s in index_set_2of2({dist})])'.format(
            aggregator=aggregator,
            combinator=combinator,
            dist=HELPER_ARRAY,
            scaling_factors=product,
            scaling_var=scaling_var_name,
            compensation=compensation)


def update_scaling(solution, scaling):
    if args.fixed_scaling:
        scaling = args.fixed_scaling
    else:
        dist_mtx = solution[HELPER_ARRAY]
        for i in range(len(scaling)):
            max_dist = max(map(abs, dist_mtx[i][:]))
            scaling[i] = max(scaling[i], max_dist)


def str_to_function(function_name):
    if function_name == 'min':
        agg = min
    elif function_name == 'max':
        agg = max
    elif function_name == 'sum':
        agg = sum
    else:
        assert False, 'function name {} not supported'.format(function_name)
    return agg


def objective_weights(nitems):
    if common.args.randomise_weights:
        #return [random.randint(1, 100) for _ in range(nitems)]
        return [
            random.randint(1, 100000),
            random.randint(1, 1000),
            random.randint(1, 10)
        ]
    else:
        return [1 for _ in range(nitems)]


def compute_dists_from_mtx(dist_mtx, aggregator, combinator, scaling):
    agg = str_to_function(aggregator)
    com = str_to_function(combinator)

    nbr_measures = len(dist_mtx)
    nbr_sols = len(dist_mtx[0])

    unnormalised = agg([com(dist_mtx[i][s] for i in range(nbr_measures))]
                       for s in range(nbr_sols))
    normalised = agg(
        [com(dist_mtx[i][s] / scaling[i] for i in range(nbr_measures))]
        for s in range(nbr_sols))

    return unnormalised, normalised


def mzn_compute_dists(store, solution, distance_functions, agg_function,
                      comb_function, scaling):

    model = []
    nbr_distances = len(distance_functions)
    ## Include distance functions
    model.append(common.include_distance_functions())

    ## Post 'variable' arrays of the solution
    dist_vars = list(set([v for v, _ in distance_functions]))
    for v in dist_vars:
        model.append(common.mzn_declaration(v, solution[v]))

    ## Post distance array
    model.append(compute_dists(distance_functions, store))

    ## Post scaling arrays
    model.append(
        'array[1..{nbrdistances}] of 1..1: dummy_scaling = {scaling};'.format(
            nbrdistances=nbr_distances,
            scaling=[1 for _ in range(nbr_distances)]))
    model.append(
        'array[1..{nbrdistances}] of {t}: scaling = {scaling};'.format(
            nbrdistances=nbr_distances,
            scaling=scaling,
            t='float' if common.args.float else 'int'))

    ## Post variable for the normalised and unnormalised distances
    model.append('var {}: d_unnormalised {};'.format(
        'float' if common.args.float else 'int', ':: add_to_output'
        if args.dist_add_to_output else ''))
    model.append('constraint d_unnormalised = {};'.format(
        combine_and_aggregate(
            comb_function,
            agg_function, [1 for _ in range(nbr_distances)],
            scaling_var_name='dummy_scaling')))
    model.append('var {}: d_normalised {};'.format(
        'float' if common.args.float else 'int', ':: add_to_output'
        if args.dist_add_to_output else ''))
    model.append('constraint d_normalised = {};'.format(
        combine_and_aggregate(comb_function, agg_function, scaling)))

    ## Solve model
    r = common.solve_one_instance(
        instance=minizinc.Instance(solver=minizinc.Solver.lookup('gecode')),
        solver=minizinc.Solver.lookup('gecode'),
        add_to_model=model,
        timeout=None)

    return r['d_unnormalised'], r['d_normalised']


if __name__ == '__main__':
    common.setup_argparser('Generate diverse solutions to MiniZinc models.')
    args = common.args

    parser = Parser(args.models + args.data_files + [args.solve_item])
    multi_objective = len(parser.objectives) > 1

    aggregator = parser.aggregator
    combinator = parser.combinator
    # Check consistency for multi-objective
    if multi_objective:
        assert not aggregator or aggregator == 'min', 'Must use aggregator min for multi-objective problems'
        assert not combinator or combinator == 'max', 'Must use combinator max for multi-objective problems'
        aggregator = 'min'
        combinator = 'max'
    # Use default combinator if empty
    if not combinator:
        combinator = parser.aggregator

    inter_sol_constr = []

    # Improvement for floating points: don't use min function
    if aggregator == 'min' and combinator == 'min' and common.args.float:
        inter_sol_constr.append(lambda _: 'var float: FLOAT_OBJECTIVE;')
        inter_sol_constr.append(
            lambda _: 'constraint forall(m in index_set_1of2({dist}), s in index_set_2of2({dist}))(FLOAT_OBJECTIVE <= {dist}[m, s]);'.format(dist=HELPER_ARRAY)
        )

    distance_functions = list(zip(parser.vararrays, parser.distance_functions))
    if multi_objective:
        # TODO: support arbitrary distance function on objectives
        for o in parser.objectives:
            distance_functions.append((str(o), 'pairwise_diff'))

    # Post helper matrix dist[i][j] = distance to solution j in measure i
    inter_sol_constr.append(
        lambda store: compute_dists(dist_funs=distance_functions, store=store))

    # Add inter-diversity constraint
    if parser.inter_diversity_constraint:
        # Slice the diversity measures that are not related to objectives
        inter_sol_constr.append(lambda _: 'constraint {name}({dist}[1..{ndistances},..]);'.format(name=parser.inter_diversity_constraint, dist=HELPER_ARRAY,ndistances=len(parser.distance_functions)))
    if multi_objective:
        start_idx_objectives = len(parser.distance_functions) + 1
        inter_sol_constr.append(lambda _store: 'constraint non_dominated({dist}[{start}..{end},..]);'.format(dist=HELPER_ARRAY, start=start_idx_objectives, end=len(parser.objectives) + start_idx_objectives - 1))

    # intra-solution constraint
    intra_sol_constr = []
    if parser.intra_diversity_constraint:
        intra_sol_constr.append(lambda store: 'constraint {}({});'.format(parser.intra_diversity_constraint,
                                                                          store[0].objective))

    # First and Next objectives
    weighted_sum = gen_weighted_sum(parser.objectives,
                                    [1 for _ in parser.objectives])

    combinator = aggregator if not combinator else combinator
    next_objectives = [lambda _, scaling: combine_and_aggregate(combinator=combinator, aggregator=aggregator, scaling=scaling),
                       lambda _store, _scaling: '-({})'.format(weighted_sum)]

    if not args.float:
        update_ub = lambda sol, store, scaling: mzn_compute_dists(store=store, solution=sol, distance_functions=distance_functions, agg_function=aggregator, comb_function=combinator, scaling=scaling)
    else:
        update_ub = lambda sol, store, scaling: compute_dists_from_mtx(dist_mtx=sol[HELPER_ARRAY], aggregator=aggregator, combinator=combinator, scaling=scaling)

    gd = GreedyDiverse(
        solver=args.solver,
        files=args.models + args.data_files,
        first_objective=gen_weighted_sum(
            parser.objectives, objective_weights(len(parser.objectives))),
        next_objectives=next_objectives,
        inter_solution_constraints=inter_sol_constr,
        intra_solution_constraints=intra_sol_constr,
        timeout=args.timeout,
        distance_vars=parser.objectives + parser.vararrays,
        nsols=args.nsols,
        init_scaling=[1 for _ in range(len(distance_functions))]
        if not args.fixed_scaling else args.fixed_scaling,
        update_scaling=update_scaling,
        search_ann=common.PPL_search_ann() if args.ppl else '',
        update_ub=update_ub)

    sols = gd.solve()
    common.log.info('Number of found solutions: {}'.format(len(sols)))

    if len(parser.objectives) > 1:
        common.visualise_pareto(
            store=sols,
            objectives=parser.objectives,
            name='-'.join(args.data_files).split('.mzn')[0] + '-diverse')

    print('Solutions are:')
    for s in sols:
        print('\n'.join(
            '{}: {}'.format(key, value)
            for key, value in common.get_assignments(s)))
        print('objective: {}'.format(s.objective))
        print()

    if multi_objective:
        if len(sols) > 0:
            print(
                'Hypervolume:',
                performance_measures.hypervolume(
                    [[s[o] for o in parser.objectives] for s in sols]))

    if common.args.report_diversity:
        print('Normalised diversity:', GreedyDiverse.normalised_dist_ub,
              gd.scaling)
        print('Unnormalised diversity:', GreedyDiverse.unnormalised_dist_ub)
