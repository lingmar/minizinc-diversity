import argparse
import json
import logging
import math
import re
import os
import sys
from enum import Enum

import matplotlib.pyplot as plt
import minizinc
from minizinc.result import Status
from mpl_toolkits.mplot3d import Axes3D
from numpy import array

log = logging.getLogger()
DIVERSITY_UB = 1000000
DIVERSITY_LB = -DIVERSITY_UB


def setup_arguments(argument_parser):
    argument_parser.add_argument(
        '-m',
        '--models',
        required=True,
        nargs='+',
        help=("One or several MiniZinc model files"))
    argument_parser.add_argument(
        '-d',
        '--data-files',
        required=False,
        default=[],
        nargs='*',
        help=("One or several MiniZinc data files (dzn or json)."))
    argument_parser.add_argument(
        '-n',
        '--nsols',
        type=int,
        required=False,
        default=1,
        help=("Number of solutions or 0 for all"))
    argument_parser.add_argument(
        '-i',
        '--solve-item',
        required=True,
        help=("File containing solve statement (and nothing else)"))
    argument_parser.add_argument(
        '-s',
        '--solver',
        type=str,
        required=False,
        help=("Name of solver to use (default: %(default)s)"),
        default='gecode')
    argument_parser.add_argument(
        '--timeout', '-t', required=False, type=int, help='timeout in seconds')
    argument_parser.add_argument(
        '--verbose',
        '-v',
        action='count',
        help=("Enable more verbose logging."
              " Supply several times for even"
              " more verbosity"),
        default=0)
    argument_parser.add_argument(
        '--processes',
        '-p',
        required=False,
        type=int,
        default=4,
        help='Number of threads used for solving')
    argument_parser.add_argument(
        '--use-best-found',
        '-b',
        required=False,
        action='store_true',
        help=
        'Use the best found solution (if any) after timeout. If not supplied, program aborts after timeout.'
    )
    argument_parser.add_argument(
        '--lex',
        '-l',
        required=False,
        choices=['seq', 'epsilon', 'none'],
        default='seq',
        help=
        """Method to use for lexicographic optimisation: sequentially optimise
objectives (seq), multiply objectives (epsilon) with constants, or don't use
lexicographic optimisation (none) (default: %(default)s)""")
    argument_parser.add_argument(
        '--lex-weights',
        '-lw',
        required=False,
        nargs='*',
        help='Weights for --lex epsilon')
    argument_parser.add_argument(
        '--lex-float-relax',
        required=False,
        default=0.1,
        type=float,
        help=
        'Relaxation to use for floating point diversity when using --lex seq (default: %(default)s)'
    )
    argument_parser.add_argument(
        '--start-sols',
        required=False,
        help=
        'Use set of solutions in this json file as start solutions for the algorithm'
    )
    argument_parser.add_argument(
        '--report-diversity',
        required=False,
        action='store_true',
        help='Report the value of the diversity measure after solving')
    argument_parser.add_argument(
        '--max-diversity',
        required=False,
        default=1000000,
        type=int,
        help='Maximum value of diversity (default: %(default)s)')
    argument_parser.add_argument(
        '--randomise-weights',
        required=False,
        action='store_true',
        help='Randomise weights on objectives multi-objective optimisation')
    # argument_parser.add_argument(
    #     '--output-intermediate',
    #     required=False,
    #     action='store_true',
    #     default=False,
    #     help='Output intermediate solutions')
    argument_parser.add_argument(
        '--auto-scaling',
        required=False,
        action='store_true',
        default=False,
        help='Use auto-scaling in combinator function')
    argument_parser.add_argument(
        '--fixed-scaling',
        required=False,
        type=int,
        nargs='*',
        help='Define scaling factors for combinator function manually')
    argument_parser.add_argument(
        '--dist-add-to-output',
        required=False,
        action='store_true',
        help=
        'Use ::add_to_output annotation on distance variables (necessary in some cases)'
    )
    argument_parser.add_argument(
        '--ppl',
        required=False,
        action='store_true',
        help='Enables use of the search strategy of the plant layout problem')
    argument_parser.add_argument(
        '--float',
        '-f',
        required=False,
        action='store_true',
        help='Set this flag if diversity measures are floats')
    argument_parser.add_argument(
        '--chuffed-disable-free-search',
        required=False,
        action='store_true',
        help=("Disable free search in Chuffed"))
    argument_parser.add_argument(
        '--warm-start',
        required=False,
        type=str,
        help=("Warm start search from solution in this file"))
    argument_parser.add_argument(
        '--output-mzn',
        required=False,
        type=str,
        help=("Output model to this file name"))
    argument_parser.add_argument(
        '--photo',
        required=False,
        action='store_true',
        help=("Enable search strategy for photo"))


class ArtificialSolution:
    def __init__(self, assignments):
        self.assignments = assignments

    def __str__(self):
        return str(self.__dict__)


class ArtificialResult:
    def __init__(self, sol_dict):
        self._solutions = [ArtificialSolution(sol_dict)]
        self.objective = sol_dict['objective']

    def __getitem__(self, key):
        return self._solutions[0].assignments[key]

    def __str__(self):
        return str(self.__dict__)


def parse_start_solutions(filename):
    results = []
    with open(filename) as json_file:
        data = json.load(json_file)
        for sol_dict in data['solutions']:
            r = ArtificialResult(sol_dict)
            results.append(r)
            log.info('Artificial result object: {}'.format(r))
    return results


def setup_logging(log, verbose_count):
    """
    Configure the system logging facility depending on how many times -v
    was provided on the command-line.
    """
    log_level = (max(3 - verbose_count, 0) * 10)
    log.setLevel(log_level)
    ch = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)


def setup_argparser(msg):
    global args

    argument_parser = argparse.ArgumentParser(description=msg)
    setup_arguments(argument_parser)
    args = argument_parser.parse_args()
    setup_logging(log, args.verbose)
    args.nsols = float('inf') if args.nsols == 0 else args.nsols

    global DIVERSITY_UB
    global DIVERSITY_LB
    DIVERSITY_UB = args.max_diversity
    DIVERSITY_LB = -args.max_diversity


def lex(instance,
        objectives,
        solver,
        timeout,
        add_to_model=[],
        method='minimize',
        relax=None):
    """
    Lexicographically optimise objectives
    """
    if args.lex == 'seq':
        return lex_seq(instance, objectives, solver, timeout, add_to_model,
                       method, relax)
    elif args.lex == 'epsilon':
        return lex_epsilon(instance, objectives, solver, timeout, add_to_model,
                           method, relax)
    else:
        log.warn(
            'Lexicographic optimisation is disabled, only optimising first objective!'
        )
        return lex_seq(instance, [objectives[0]], solver, timeout,
                       add_to_model, method, relax)[0]


def lex_epsilon(instance,
                objectives,
                solver,
                timeout,
                add_to_model=[],
                method='minimize',
                relax=None):
    """
    Lexicographically optimise objectives by multiplying the later ones with small constants
    """
    assert method == 'maximize', 'Lex epsilon with minimization not supported'  # TODO
    assert len(objectives
               ) == 2, 'Only two objectives supported for lex epsilon'  # TODO

    log.warn('Argument relax is not used for lex epsilon')

    if args.float:
        constants = [100, 0.0001]
    else:
        constants = [100000, 1]

    # Construct search annotations
    search_anns = []
    if args.ppl:
        search_anns.extend(PPL_search_anns())
    elif args.photo:
        search_anns.append(photo_search_ann())
    if args.warm_start:
        search_anns.extend(mzn_warm_start_from_file(args.warm_start))

    # Construct solve item
    solve_item = 'solve {search_ann} {method} {objective};'.format(
        search_ann=aggregate_search_anns(search_anns),
        method=method,
        objective=' + '.join([
            '({c} * ({o}))'.format(c=c, o=o)
            for c, o in zip(constants, objectives)
        ]))

    # Solve
    r = solve_one_instance(
        instance=instance,
        solver=solver,
        timeout=timeout,
        add_to_model=add_to_model + [solve_item])

    # Timeout?
    if r.status == Status.OPTIMAL_SOLUTION or (r.status.has_solution()
                                               and args.use_best_found):
        obj_val = r.objective
    else:
        return None

    log.info('Objective {} was {}'.format(solve_item, obj_val))
    if args.solver == 'gurobi':
        log.info('Objective gap was {}%'.format(obj_gap(r)))

    return r


def lex_seq(instance,
            objectives,
            solver,
            timeout,
            add_to_model=[],
            method='minimize',
            relax=None):
    """
    Lexicographically optimise objectives by sequentially optimising them
    """
    # TODO: handle timeout
    log.debug('lex_seq {} {}'.format(method, objectives))
    constraints = []
    last_result = None
    all_results = []
    for o in objectives:
        # Construct search annotations
        search_anns = []
        if last_result:
            search_anns.extend(mzn_warm_start(last_result))
        elif args.warm_start:
            search_anns.extend(mzn_warm_start_from_file(args.warm_start))
        if args.ppl:
            search_anns.extend(PPL_search_anns())
        elif args.photo:
            search_anns.append(photo_search_ann())

        # Solve
        last_result = solve_one_instance(
            instance=instance,
            solver=solver,
            timeout=timeout,
            add_to_model=add_to_model + constraints + [
                'solve {} {} {};'.format(
                    aggregate_search_anns(search_anns), method, o)
            ])

        # Timeout?
        if last_result.status == Status.OPTIMAL_SOLUTION or (
                last_result.status.has_solution() and args.use_best_found):
            obj_val = last_result.objective
        else:
            return [None for _ in objectives]

        log.info('Objective {} was {}'.format(o, obj_val))
        if args.solver == 'gurobi':
            log.info('Objective gap was {}%'.format(obj_gap(last_result)))

        if relax is None:
            if args.float:
                log.warn(
                    'Using floats with no relax can cause model infeasibility because of roundoff errors'
                )

            constraints.append('constraint {obj} {rel} {obj_val};'.format(
                obj=o,
                obj_val=obj_val,
                rel='<=' if method == 'minimize' else '>='))
        else:
            constraints.append(
                'constraint {obj} >= {obj_val} - {relax};'.format(
                    obj=o, obj_val=obj_val, relax=relax))

        all_results.append(last_result)

    return all_results


def obj_gap(result):
    """
    Find objective gap for result object
    """
    for line in result.stderr.decode().split('\n'):
        gap = re.findall(", gap \d+\.\d+\%", line)
        if gap:
            return float(gap[0].split('gap')[1].split('%')[0])
    return None


def lex_min(instance, objectives, solver, timeout, add_to_model=[],
            relax=None):
    """
    Lexicographically minimise objectives
    """
    return lex(
        instance,
        objectives,
        solver,
        timeout,
        add_to_model,
        method='minimize',
        relax=relax)


def lex_max(instance, objectives, solver, timeout, add_to_model=[],
            relax=None):
    """
    Lexicographically maximise objectives
    """
    return lex(
        instance,
        objectives,
        solver,
        timeout,
        add_to_model,
        method='maximize',
        relax=relax)

def get_assignments(result):
    """
    Get the variable assignments from a Result object

    FIXME: This is a hacky solution that heavily depends on how Results objects
    are implemented.
    """
    return [(k,v) for k, v in result.solution.__dict__.items()
            if (not k.startswith('_') and k != 'objective') ]


def include_distance_functions():
    """
    Returns the MiniZinc statement for including distance functions.
    """
    return 'include "{}";'.format(os.environ['MINIZINC_DIVERSITY_HOME'] + '/src/distance_functions.mzn')


def depth(mtx):
    """
    Return the depth of a Python matrix
    """
    return len(array(mtx).shape)


def flatten(l):
    """
    Return a flat version of a Python matrix
    """
    return [item for sublist in l for item in sublist]


def to_mzn1darray(key, value):
    """
    Return a MiniZinc 1D array conversion of the given key and value
    """
    d = depth(value)
    if d == 0:
        ret = '[{}]'.format(key), [value]
    elif d == 1:
        ret = key, value
    elif d == 2:
        ret = 'array1d({})'.format(key), flatten(value)
    else:
        log.warn('Cannot flatten matrix of depth {}: {} = {}'.format(
            depth, key, value))
        ret = None, None
    log.debug('to_mzn1darray({},{}) = {}'.format(key, value, ret))
    return ret


def contains_float(s):
    """
    Checks whether s contains a floating point number

    Shamelessly stolen from:
    https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
    """
    return re.findall("\d+\.\d+", s)


def mzn_warm_start_from_file(filename):
    """
    Create warm start annotations from solution in filename
    """
    warm_starts = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            if '=' in l:
                key, value = l.split('=')
                value = value.split(';')[0]
                if 'array1d' not in value:
                    value = '[{}]'.format(value)
                    key = '[{}]'.format(key)
                # Do not warm start from float values
                if not contains_float(value):
                    warm_starts.append('warm_start({vars},{vals})'.format(
                        vars=key, vals=value))
    return warm_starts


def mzn_warm_start(solution):
    warm_starts = []
    for key, value in get_assignments(solution):
        array1d_key, array1d_val = to_mzn1darray(key, value)
        # Do not warmstart with float values
        if not isinstance(array1d_val[0], float):
            warm_starts.append('warm_start({vars},{vals})'.format(
                vars=array1d_key, vals=array1d_val))
    return warm_starts


def aggregate_search_anns(anns):
    if anns:
        return ':: seq_search([{}])'.format(','.join(anns))
    else:
        return ''


def mzn_declaration(var, val):
    d = depth(val)
    t = 'float' if args.float else 'int'
    if d == 0:
        return '{t}: {varname} = {val};'.format(t=t, varname=var, val=val)
    elif d == 1:
        return 'array[1..{length}] of {t}: {varname} = {val};'.format(
            length=len(val), t=t, varname=var, val=val)
    else:
        assert False, 'depth {} not supported for {} = {}'.format(d, var, val)


def to_mzn_array_output(value):
    d = depth(value)
    if d == 0:
        return value
    elif d == 1:
        return 'array1d(1..{length}, {value})'.format(
            length=len(value), value=value)
    elif d == 2:
        return 'array2d(1..{nrows}, 1..{ncols}, {flattened})'.format(
            nrows=len(value), ncols=len(value[0]), flattened=flatten(value))
    else:
        log.warn('Cannot transform object of depth {}: {}'.format(d, value))
        return value


def mzn_output(result):
    return '\n'.join([
        '{} = {};'.format(key, to_mzn_array_output(value))
        for key, value in get_assignments(result)
    ] + ['----------'])


def mzn_output_sol(sol):
    return '\n'.join([
        '{} = {};'.format(key, to_mzn_array_output(value))
        for key, value in sol.assignments.items()
    ] + ['----------'])


def write_instance_to_file(instance, filename, strings=[]):
    """
    Output the model being solved (for debugging)
    """
    with open(filename, 'w') as f:
        # Model files
        for m in args.models:
            with open(m, 'r') as mf:
                f.write(mf.read())
        # Data files
        for m in args.data_files:
            with open(m, 'r') as mf:
                f.write(mf.read())
        # Additional strings
        for s in strings:
            f.write(s)


def solve_one_instance(instance,
                       solver,
                       timeout,
                       add_to_model=[],
                       objective=None):
    """
    Solve a MiniZinc instance

    Provide objective explicitly to use search annotations and warm starts given by command line arguments
    """
    with instance.branch() as i:
        i._method = minizinc.instance.Method.MINIMIZE

        # Add constraints and objective
        if objective:
            # Construct search annotations
            search_anns = []
            if args.ppl:
                search_anns.extend(PPL_search_anns())
            if args.photo:
                search_anns.append(photo_search_ann())
            if args.warm_start:
                search_anns.extend(mzn_warm_start_from_file(args.warm_start))

            solve_stm = 'solve {search_ann} minimize {obj};'.format(
                search_ann=aggregate_search_anns(search_anns), obj=objective)
            add_to_model.append(solve_stm)

        for c in add_to_model:
            i.add_string(c)

        log.debug('Solve with additional constraints: {}'.format(
            '\n'.join(add_to_model)))
        log.debug('timeout is set to {}'.format(timeout))
        log.debug('Using solver {}'.format(instance._solver.name))

        # Output model to a file?
        if args.output_mzn:
            write_instance_to_file(
                instance=i, filename=args.output_mzn, strings=add_to_model)

        # Solve the instance
        # Chuffed does not support -p flag
        if solver.name == 'Chuffed':
            if not args.chuffed_disable_free_search:
                # Use free search by default
                r = i.solve(timeout=timeout, free_search=True, verbose=True)
            else:
                r = i.solve(timeout=timeout, verbose=True)
        else:
            r = i.solve(
                timeout=timeout, processes=args.processes, verbose=True)

        # Report status of solution
        # TODO: support satisfaction problems
        if r.status == Status.OPTIMAL_SOLUTION:
            log.info('Found optimal solution:\n{}'.format(mzn_output(r)))
            log.info('Stats {}'.format(r.statistics))
        elif r.status == Status.UNSATISFIABLE:
            log.info('Model was unsatisfiable!')
            log.info('Stats {}'.format(r.statistics))
        elif r.status.has_solution():
            log.info(
                'Found a solution, but it is not proven to be optimal: {}'.
                format(mzn_output(r)))
            log.info('Stats {}'.format(r.statistics))
        else:
            log.info('No solution found, probably because of timeout!')
            log.info('Status was: {}'.format(r.status))
            #exit(0)
        return r


def visualise_pareto(store, objectives, name):
    if len(objectives) == 2:
        ## Plot the first two objective functions
        plt.scatter(
            x=[r[objectives[0]] for r in store],
            y=[r[objectives[1]] for r in store])

        plt.xlabel(objectives[0])
        plt.ylabel(objectives[1])
        plt.ylim(-10, 150)
        plt.xlim(-10, 190)

        plt.savefig('{}.png'.format(name))
        log.info('xs: {}'.format([r[objectives[0]] for r in store]))
        log.info('ys: {}'.format([r[objectives[1]] for r in store]))
        log.info('plot saved as {}.png'.format(name))
    elif len(objectives) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = [r[objectives[0]] for r in store]
        ys = [r[objectives[1]] for r in store]
        zs = [r[objectives[2]] for r in store]

        ax.scatter(xs, ys, zs)
        ax.scatter(xs, ys, [0 for _ in range(len(xs))])
        ax.scatter(xs, [0 for _ in range(len(xs))], zs)
        ax.scatter([0 for _ in range(len(xs))], ys, zs)

        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[2])
        #plt.show()
        plt.savefig('{}.png'.format(name))
        log.info('{}: {}'.format(objectives[0], xs))
        log.info('{}: {}'.format(objectives[1], ys))
        log.info('{}: {}'.format(objectives[2], zs))
        log.info('plot saved as {}.png'.format(name))

    else:
        print('Cannot visualise {} parameters'.format(len(obj_vars)))


def PPL_search_anns():
    return [
        'warm_start( nPosXYZ_1D_VAR_WS, nPosXYZ_1D_WS)',
        'warm_start( nSzWLH_1D_VAR_WS, nSzWLH_1D_WS)',
        'warm_start( r_VAR_WS, r_WS)', 'warm_start( rel_VAR_WS, rel_WS)',
        'warm_start( [ iMSupport[ aSupportedMdl[ i ] ] | i in index_set(aSupportedMdl) ],[ 1 + aSupportingMdl[ i ] | i in index_set(aSupportedMdl) ] )',
        'int_search( [ [nPosXYZ_1D[ (aMdlSort[ m ]-1)*3 +1 ],nPosXYZ_1D[ (aMdlSort[ m ]-1)*3 +2 ],nPosXYZ_1D[ (aMdlSort[ m ]-1)*3 +3 ] ][k] | m in MODULES, k in 1..3 ],input_order, indomain_split, complete)',
        'int_search([ r[ aMdlSort[ m ] ] | m in MODULES ],input_order, indomain_min, complete)'
    ]


def PPL_search_ann():
    return "::seq_search( [warm_start( nPosXYZ_1D_VAR_WS, nPosXYZ_1D_WS),warm_start( nSzWLH_1D_VAR_WS, nSzWLH_1D_WS),warm_start( r_VAR_WS, r_WS),warm_start( rel_VAR_WS, rel_WS),warm_start( [ iMSupport[ aSupportedMdl[ i ] ] | i in index_set(aSupportedMdl) ],[ 1 + aSupportingMdl[ i ] | i in index_set(aSupportedMdl) ] ),int_search( [ [nPosXYZ_1D[ (aMdlSort[ m ]-1)*3 +1 ],nPosXYZ_1D[ (aMdlSort[ m ]-1)*3 +2 ],nPosXYZ_1D[ (aMdlSort[ m ]-1)*3 +3 ] ][k] | m in MODULES, k in 1..3 ],input_order, indomain_split, complete),int_search([ r[ aMdlSort[ m ] ] | m in MODULES ],input_order, indomain_min, complete),] )"


def photo_search_ann():
    return "int_search(pos, first_fail, indomain, complete)"
