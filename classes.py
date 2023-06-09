import numpy as np
from prettytable import PrettyTable
from sale import gauss, gauss_rectangle, lu_solve, gauss_seidel, iteration, dd, is_pos_def


class Equations:
    def __init__(self, a: np.array, b: np.array, method='all', tol=1e-10, max_iter=100000):
        """
        :param a: numpy array
        :param b: numpy array
        :param method: any of ['all', 'Gauss', 'Gauss rectangle', 'LU', 'Gauss Seidel', 'Iteration']
        :param tol: float number
        :param max_iter: int number
        """

        self.a = np.array(a, dtype=np.double)
        self.b = np.array(b, dtype=np.double)
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.x = None

    def solve(self):
        """
        Prints out solutions for a system of equations.
        If method not equal to 'all' Equations.x is the solution.

        :return: nothing
        """

        tb = PrettyTable()
        tb.field_names = ["Method", "Answer", "Time"]

        match self.method:
            case 'all':
                gauss_res, gauss_time = gauss(self.a, self.b)
                tb.add_row(['Gauss', gauss_res, gauss_time], divider=True)

                gauss_rec_res, gauss_rec_time = gauss_rectangle(self.a, self.b)
                tb.add_row(['Gauss rectangle', gauss_rec_res, gauss_rec_time], divider=True)

                lu_res, lu_time = lu_solve(self.a, self.b)
                tb.add_row(['LU', lu_res, lu_time], divider=True)

                gs_res, gs_time = gauss_seidel(self.a, self.b, self.tol, self.max_iter)
                it_res, it_time = iteration(self.a, self.b, self.tol, self.max_iter)

                if dd(self.a) or is_pos_def(self.a):
                    tb.add_row(['Gauss Seidel', gs_res, gs_time], divider=True)

                    tb.add_row(['Iteration', it_res, it_time], divider=True)
                else:
                    tb.add_row(['Gauss Seidel', str(gs_res) + '\n' + 'may be no convergence', gs_time], divider=True)

                    tb.add_row(['Iteration', str(it_res) + '\n' + 'may be no convergence', it_time], divider=True)

            case 'Gauss':
                gauss_res, gauss_time = gauss(self.a, self.b)
                tb.add_row(['Gauss', gauss_res, gauss_time], divider=True)

                self.x = gauss_res

            case 'Gauss rectangle':
                gauss_rec_res, gauss_rec_time = gauss_rectangle(self.a, self.b)
                tb.add_row(['Gauss rectangle', gauss_rec_res, gauss_rec_time], divider=True)

                self.x = gauss_rec_res

            case 'LU':
                lu_res, lu_time = lu_solve(self.a, self.b)
                tb.add_row(['LU', lu_res, lu_time], divider=True)

                self.x = lu_res

            case 'Gauss Seidel':
                gs_res, gs_time = gauss_seidel(self.a, self.b, self.tol, self.max_iter)
                tb.add_row(['Gauss Seidel', gs_res, gs_time], divider=True)

                self.x = gs_res

            case 'Iteration':
                it_res, it_time = iteration(self.a, self.b, self.tol, self.max_iter)
                tb.add_row(['Iteration', it_res, it_time], divider=True)

                self.x = it_res

        print(tb)
