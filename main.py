from classes import Equations
import numpy as np

if __name__ == '__main__':
    a = [[16, 23],
         [7, -11]]

    b = [11,
         13]

    # a = [[10, -1, 2, 0],
    #      [-1, 11, -1, 3],
    #      [2, -1, 10, -1],
    #      [0, 3, -1, 8]]
    #
    # b = [6,
    #      25,
    #      -11,
    #      15]

    my_eq = Equations(a, b, max_iter=1000)
    my_eq.solve()
    print(my_eq.x)
