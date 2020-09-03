# MiniZinc Diversity

Finding diverse solutions to [MiniZinc](https://www.minizinc.org/) problems.

**NOTE** The current implementation is a prototype. Most likely the
implementation has bugs.

## Getting Started

### Installation

The first step is to clone this repository and set the environment variable
`MINIZINC_DIVERSITY_HOME` to point to the root of the repository:

```
> git clone https://github.com/lingmar/minizinc-diversity.git
> cd minizinc-diversity
> export MINIZINC_DIVERSITY_HOME=`pwd`
```

The next step is making sure all dependencies are satisfied. The following
Python packages, available via `pip`, are required:

* `minizinc` (See https://minizinc-python.readthedocs.io/en/latest/ for dependencies.)
* `numpy`
* `matplotlib`
* `scipy`

When all requirements are satisfied, the following command:

```
> ./src/diverse.py --help
```

should run without errors and show all available command line options of the
tool.

### Basic Usage

The tool requires at least two inputs: a *model file* and a *solve item file*.
The solve item file includes the diversity annotations and the solve statement
(`solve minimize ...`). The tool currently only supports minimisation problems.
The model file must not include any diversity annotations or solve statements.

The following shows an example of a small model file.

```
include "alldifferent.mzn";

array[1..5] of var 1..5: x;

var 1..5: y = x[1];

constraint alldifferent(x);

% Enforce solutions to be optimal
predicate optimal(int: optimum) =
  y = optimum;

% Enforce solutions to have some distance
predicate dist_apart(array[int,int] of var int: dist) =
  % dist[m,i] : distance between the i'th solution pair in the m'th
  % distance measure.
  forall (i in index_set_2of2(dist)) (dist[1,i] >= 4) /\
  forall (i in index_set_2of2(dist)) (dist[2,i] >= 6);
```

The following shows an example of an accompanying solve item file. We have two
measures of diversity, Hamming distance and Manhattan distance, each measured on
the variable array `x`. A full explanation of the diversity annotations can be
found [in this paper](https://aaai.org/ojs/index.php/AAAI/article/view/5512).

```
:: diverse_pairwise(x, "hamming_pairwise")
:: diverse_pairwise(x, "manhattan_pairwise")
:: diversity_aggregator("min")
:: diversity_combinator("min")
:: intra_diversity_constraint("optimal")
:: inter_diversity_constraint("dist_apart")

solve minimize y;
```

Assuming the contents of the above model and solve items are in files
`model.mzn` and `solve.mzn`, respectively, the following command finds `n=3`
diverse solutions to the problem:

```
> ./src/diverse.py -m model.mzn -i solve.mzn -n 3
Solutions are:
x: [1, 5, 4, 3, 2]
objective: 1

x: [1, 4, 2, 5, 3]
dist: [[4], [6]]
objective: -1

x: [1, 3, 5, 2, 4]
dist: [[4, 4], [6, 8]]
objective: -1
```

In this rather raw solution output, apart from the values of the variables in
`x`, two additional variables are shown: `objective` and `dist`. For the first
solution, the value of `objective` is the value of the problem objective (that
is, variable `y`). For the subsequent solutions, this value comes from the
internal diversity maximisation and is meaningless (in most cases, it is not
equal to the overall diversity measure). The `dist` variable array is an
internal helper variable for keeping track of diversity. In the current
implementation, the array `dist[m,..]` are the distances in measure `m` from the
previous solutions. The order of the measures is decided by the order in which
they are stated in the solve item file.

For more command line options of the tool, we for now refer to invoking the help
command:

```
> ./src/diverse.py --help
```

## Known Issues and Limitations

* The helper variable used during solving is hard-coded to `dist` and can
  therefore not be used in the model.
* The parser used for the solve item file (the file containing the diversity
  annotations and `solve minimize` statement) is currently very limited. The
  variable array provided as argument to the ´diverse_pairwise´ annotation must
  be a name of a variable defined in the model file (and not a general MiniZinc
  expression). Same goes for the objective function: an explicit variable must
  be defined for it in the model file.
