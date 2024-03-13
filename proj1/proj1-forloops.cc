/*
 * $Smake: g++ -Wall -O3 -o %F %f -lcblas -latlas -lhdf5
 *
 * Computes a matrix-matrix product
 */

#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
#include "../wtime.h"
#include "../wtime.c"
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

/* Macro to index matrices in column-major (Fortran) order */
#define IDX(i,j,stride) ((i)+(j)*(stride))  /* column major */

/* Check return values from HDF5 routines */
#define CHKERR(status,name) if (status) \
     fprintf(stderr, "Warning: nonzero status (%d) in %s\n", status, name)

double tolerance = 1e-6;
int maxiter = 1000;

/*----------------------------------------------------------------------------
 * Display string showing how to run program from command line
 *
 * Input:
 *   char* program_name (in)  name of executable
 * Output:
 *   writes to stderr
 * Returns:
 *   nothing
 */
void usage(char* program_name)
{
    fprintf(stderr, "Usage: %s [-v] input-file\n", program_name);
}

/*----------------------------------------------------------------------------
 * Read Matrix
 */
void readMatrix(char* fname, const char* name, double** a, int* rows, int* cols)
{
    hid_t   file_id, dataset_id, file_dataspace_id, dataspace_id;
    herr_t status;
    hsize_t* dims;
    int rank;
    int ndims;
    hsize_t num_elem;

    /* Open existing HDF5 file */
    file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open existing first dataset */
    dataset_id = H5Dopen(file_id, name, H5P_DEFAULT);

    /* Determine dataset parameters */
    file_dataspace_id = H5Dget_space(dataset_id);
    rank = H5Sget_simple_extent_ndims(file_dataspace_id);
    dims = new hsize_t[rank];
    ndims = H5Sget_simple_extent_dims(file_dataspace_id, dims, NULL);
    if (ndims != rank)
    {
        fprintf(stderr, "Warning: expected dataspace to be dimension ");
        fprintf(stderr, "%d but appears to be %d\n", rank, ndims);
    }

    /* Allocate matrix */
    num_elem = H5Sget_simple_extent_npoints(file_dataspace_id);
    // *a = (double*) malloc(num_elem * sizeof(double));
    *a = new double[num_elem];
    *cols = dims[0]; /* reversed since we're using Fortran-style ordering */
    *rows = dims[1];

    /* Create dataspace */
    dataspace_id = H5Screate_simple(rank, dims, NULL);

    /* Read matrix data from file */
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, dataspace_id,
                     file_dataspace_id, H5P_DEFAULT, *a);
    CHKERR(status, "H5Dread()");

    /* Close resources */
    status = H5Sclose(dataspace_id); CHKERR(status, "H5Sclose()");
    status = H5Sclose(file_dataspace_id); CHKERR(status, "H5Sclose()");
    status = H5Dclose(dataset_id); CHKERR(status, "H5Dclose()");
    status = H5Fclose(file_id); CHKERR(status, "H5Fclose()");
    delete[] dims;
}

/*----------------------------------------------------------------------------
 * Main program
 */
int main(int argc, char* argv[])
{
    char* in_name;
    double* a;             /* initial matrix */
    double* y;             /* eigenvector */
    int nrow_a, ncol_a;    /* dimensions of initial matrix */
    int nrow_y;    /* dimensions of eigenvector */
    int verbose = 0;       /* nonzero for extra output */

    /* Process command line */
    int ch;
    while ((ch = getopt(argc, argv, "v")) != -1)
    {
        switch (ch)
        {
            case 'v':
                verbose++;
                break;
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
        }
    }
    argv[optind - 1] = argv[0];
    argv += (optind - 1);
    argc -= (optind - 1);

    /* Make sure there are no additional arguments */
    if (argc != 2)
    {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    in_name  = argv[1];

    // read matrix data and optionally display it
    double t1 = wtime();
    readMatrix(in_name, "/A/value", &a, &nrow_a, &ncol_a);
    double t2 = wtime();
    double read_time = t2 - t1;

    // Initialize estimate of normalized eigenvector
    vector<double> x;
    for (int i = 0; i < nrow_a; i++) {
        x.push_back(1);
    }

    double norm = 0.0;
    for (int i = 0; i < nrow_a; i++) {
        norm += x[i] * x[i];
    }
    norm = sqrt(norm);

    for(int i = 0; i < nrow_a; ++i) {
        x[i] = x[i] / norm;
    }

    nrow_y = nrow_a;
    y = new double[nrow_y];

    // Main power method loop
    double lambda_new = 0.0;
    double lambda_old = lambda_new + 2 * tolerance;
    double delta = abs(lambda_new - lambda_old);
    int iter = 0;
    t1 = wtime();
    while (delta >= tolerance and iter <= maxiter) {
        iter++;

        // compute new eigenvector estimate c = A*x
        // initialize vector to hold product
        for (int i = 0; i < nrow_a; i++) {
            y[i] = 0.0;
        }
        // compute product
        for (int j = 0; j < nrow_a; j++) {
            for (int i = 0; i < nrow_a; i++) {
                // IDX(i,j,stride) ((i)+(j)*(stride))
                y[i] += a[IDX(i,j,nrow_a)] * x[j];
            }
        }

        // compute new estimate of eigenvalue lambda = x'Ax
        lambda_old = lambda_new;
        lambda_new = 0.0;
        for ( int i = 0; i < nrow_a; i++) {
            lambda_new += x[i] * y[i];
        }

        // update estimated normalized eigenvector
        norm = 0.0;
        for (int i = 0; i < nrow_a; i++) {
            norm += y[i] * y[i];
        }
        norm = sqrt(norm);
        for(int i = 0; i < nrow_a; ++i) {
            x[i] = y[i] / norm;
        }

        delta = abs(lambda_new - lambda_old);

        if (verbose > 0)
        {
            printf("%3d: lambda = %12.9f delta = %.4e\n", iter, lambda_new, delta);
        }
    }
    t2 = wtime();
    double compute_time = t2 - t1;

    printf("matrix dimensions: %d x %d; tolerance: %8.4e; max iterations: %d\n", nrow_a, nrow_a, tolerance, maxiter);
    printf("elapsed HDF5 read time = %10.6f seconds\n", read_time);
    printf("elapsed compute time   = %10.6f seconds\n", compute_time);
    printf("eigenvalue = %.6f found in %d iterations\n", lambda_new, iter);

    /* Clean up and quit */
    delete[] a;
    delete[] y;
    return 0;
}
