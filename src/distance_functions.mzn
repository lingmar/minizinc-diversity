% Hamming distance
% int
function var int: hamming_pairwise(array[int] of var int: x, array[int] of var int: y) =
  sum (i in index_set(x)) (x[i] != y[i]);
function var int: hamming_pairwise(array[int] of var int: x, array[int] of int: y) =
  sum (i in index_set(x)) (x[i] != y[i]);
function int: hamming_pairwise(array[int] of int: x, array[int] of int: y) =
  sum (i in index_set(x)) (x[i] != y[i]);
% float
function var float: hamming_pairwise(array[int] of var float: x, array[int] of var float: y) =
sum (i in index_set(x)) (x[i] != y[i]);
function var float: hamming_pairwise(array[int] of var float: x, array[int] of float: y) =
sum (i in index_set(x)) (x[i] != y[i]);

% Manhattan distance (L1 norm)
function var int: manhattan_pairwise(array[int] of var int: x, array[int] of var int: y) =
sum (i in index_set(x)) (abs(x[i] - y[i]));
function var int: manhattan_pairwise(array[int] of var int: x, array[int] of int: y) =
sum (i in index_set(x)) (abs(x[i] - y[i]));

function var float: manhattan_pairwise(array[int] of var float: x, array[int] of var float: y) =
sum (i in index_set(x)) (abs(x[i] - y[i]));
function var float: manhattan_pairwise(array[int] of var float: x, array[int] of float: y) =
sum (i in index_set(x)) (abs(x[i] - y[i]));
function float: manhattan_pairwise(array[int] of float: x, array[int] of float: y) =
sum (i in index_set(x)) (abs(x[i] - y[i]));


% % Jaccard's matching coefficient
% function var int: jaccard_matching_pw_bool(array[int] of var bool: x, array[int] of var bool: y) =
% -(sum(i in index_set(x)) (x[i] == y[i]) div sum (i in index_set(x)) (x[i] \/ y[i]));

% % Used for distance from objectives (assuming objectives are minimised)
% function var int: alpha(array[int] of var int: o, array[int] of var int: r) =
% min (i in index_set(o)) (r[i] - o[i]);

% function var int: pairwise_diff(var int: a1, var int: a2) = a2 - a1;
% function var int: pairwise_diff(int: a1, var int: a2) = a2 - a1;
% function var int: pairwise_diff(int: a1, int: a2) = a2 - a1;
% function var float: pairwise_diff(var float: a1, var float: a2) = a2 - a1;
% function var float: pairwise_diff(var float: a1, float: a2) = a2 - a1;
% function var float: pairwise_diff(float: a1, float: a2) = a2 - a1;

% predicate non_dominated(array[int, int] of var int: dist) =
% min([max([dist[m, s] | m in index_set_1of2(dist)]) | s in index_set_2of2(dist)]) > 0;

% predicate non_dominated(array[int, int] of var float: dist) =
% min([max([dist[m, s] | m in index_set_1of2(dist)]) | s in index_set_2of2(dist)]) > 1e-05;

% predicate one_apart(array[int, int] of var int: dist) =
% forall (i in index_set_1of2(dist), j in index_set_2of2(dist)) (dist[i,j] >= 1);
