from platypus import NSGAII, Problem, Real
import numpy as np

class Belegundu(Problem):

    def __init__(self):
        super().__init__(2, 2, 2)
        self.types[0] = Real(0, 5)
        self.types[1] = Real(0, 3)
        self.constraints[:] = "<=0"

    def evaluate(self, solution):
        x = np.array(solution.variables[:]).reshape((2,1))
        y = solution.variables[1]

        solution.objectives[:] = [-2*x[0,0] + x[1,0], 2*x[0,0] + x[1,0]]
        solution.constraints[:] = [-x[0,0] + x[1,0] - 1, x[0,0] + x[1,0] - 7]

algorithm = NSGAII(Belegundu())
algorithm.run(10000)