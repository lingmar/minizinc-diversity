from typing import List

import common.common as common

INTER_DIVERSITY_CONSTRAINT = 'inter_diversity_constraint'
INTRA_DIVERSITY_CONSTRAINT = 'intra_diversity_constraint'
MAX_DIVERSE_PAIRWISE = 'diverse_pairwise'
COMBINATOR = 'diversity_combinator'
AGGREGATOR = 'diversity_aggregator'


class Parser():
    """
    Class for parsing MiniZinc files for diversity problems. Identifies
    diversity annotations and objectives.
    """

    def __init__(self, files: List[str]):
        self.objectives: List[str] = []
        self.distance_functions: List[str] = []
        self.vararrays: List[str] = []
        self.inter_diversity_constraint: str = ''
        self.intra_diversity_constraint: str = ''
        self.aggregator: str = ''
        self.combinator: str = ''

        self.parse(files)

        common.log.info('Parser results: {}'.format(self))

    def parse(self, files: List[str]):
        """
        Parse model files
        """
        for f in files:
            with open(f, 'r') as instream:
                self.parse_one(instream)

    def parse_one(self, instream):
        """
        Scan a file for diversity annotation and solve statements.
        """
        for line in instream:
            if '::' in line:
                ann_body = line.split('::')[1]
                ann_id = ann_body.split('(')[0].strip()
                if ann_id.startswith('add_to_output'):
                    continue
                ann_args = ann_body.split('(')[1].split(')')[0].split(',')
                self.parse_annotation(ann_id, Parser.process_args(ann_args))
            elif 'solve minimize' in line:
                self.parse_objective(line)
            elif 'solve maximize' in line:
                assert False, 'maximisation not implemented yet'

    @staticmethod
    def process_args(args):
        return [a.replace('"', '') for a in args]

    def parse_annotation(self, ann_id: str, ann_args: List[str]):
        if ann_id == MAX_DIVERSE_PAIRWISE:
            assert len(ann_args) == 2
            self.vararrays.append(ann_args[0])
            self.distance_functions.append(ann_args[1])

        elif ann_id == INTER_DIVERSITY_CONSTRAINT:
            assert len(ann_args) == 1
            self.inter_diversity_constraint = ann_args[0]

        elif ann_id == INTRA_DIVERSITY_CONSTRAINT:
            assert len(ann_args) == 1
            self.intra_diversity_constraint = ann_args[0]

        elif ann_id == AGGREGATOR:
            assert len(ann_args) == 1
            self.aggregator = ann_args[0]

        elif ann_id == COMBINATOR:
            assert len(ann_args) == 1
            self.combinator = ann_args[0]

    def parse_objective(self, line):
        """
        Parse minimize / maximize statements in model file
        """
        splitted = line.split()
        # Assume var name is after minimize
        index_of_minimize = splitted.index('minimize')
        var_name = splitted[index_of_minimize + 1].split(';')[0]
        self.objectives.append(var_name)

    def __str__(self):
        return str(self.__dict__)
