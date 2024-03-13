#!/usr/bin/env python3
"""Computes dominant eigenvalue of matrix in HDF5 file via Power Method

   This program reads data for a square NxN matrix from an HDF5
   data file and uses the Power Method to compute an estimate of
   the dominant eigenvalue.  It assumes the matrix is stored in
   column-major order (FORTRAN, and not default C/C++/Python order).

   Written: January 2022 <jonathan.senning@gordon.edu>
   Updated: January 2024 <jonathan.senning@gordon.edu>
"""

import sys, time, argparse
import h5py as h5
import numpy as np

def usage(name):
    """Display program usage

    Parameters
    ----------
    name : str
        The path of the executable given on the command line
    """
    
    print(f'Usage: {name} [-q] [-v] [-e tol] [-m maxiter] filename')

def main():
    # Process command line
    parser = argparse.ArgumentParser(description='Power Method Demo')
    parser.add_argument('-e', '--tolerance', type=float, default=1e-6)
    parser.add_argument('-m', '--maxiter', type=int, default=1000)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('filename')
    args = parser.parse_args()

    # Copy arg values to new variables
    tolerance = args.tolerance
    maxiter = args.maxiter
    quiet = args.quiet
    verbose = args.verbose
    filename = args.filename

    # Read matrix from HDF5 file
    t1 = time.time()
    f = h5.File(filename, 'r')
    A = np.array(f['A/value']).transpose() # convert to column-major
    n, m = np.shape(A)
    t2 = time.time()
    read_time = t2 - t1

    # Initialize estimate of normalized eigenvector
    x = np.ones(m) / np.sqrt(m)

    # Main power method loop
    lambda_new = 0.0
    lambda_old = lambda_new + 2 * tolerance
    delta = np.abs(lambda_new - lambda_old)
    iter = 0
    t1 = time.time()
    while delta >= tolerance and iter <= maxiter:
        iter += 1

        # compute new eigenvector estimate y = A*x
        y = np.matmul(A,x)

        # compute new estimate of eigenvalue lambda = x'Ax
        lambda_old = lambda_new
        lambda_new = x.dot(y)

        # update estimated normalized eigenvector
        x = y / np.linalg.norm(y)

        delta = np.abs(lambda_new - lambda_old)
        if verbose:
            print(f'{iter:3d}: lambda = {lambda_new:12.9f},',
                  f'delta = {delta:.4e}')
    t2 = time.time()
    compute_time = t2 - t1

    # Report
    if iter > maxiter:
        print('*** WARNING ****: maximum number of iterations exceeded',
                file=sys.stderr)

    if quiet:
        print(f'{n:5d} x {m:5d} {iter:5d} {lambda_new:10.6f}',
              f'{read_time:10.6f} {compute_time:10.6f}')
    else:
        print(f'matrix dimensions: {n} x {n};',
              f'tolerance: {tolerance:8.4e}; max iterations: {maxiter}')
        print(f'elapsed HDF5 read time = {read_time:10.6f} seconds')
        print(f'elapsed compute time   = {compute_time:10.6f} seconds')
        print(f'eigenvalue = {lambda_new:.6f} found in {iter} iterations')

if __name__ == '__main__':
    main()
