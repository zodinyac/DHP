/*
 * Task 2, Variant 2
 * Nikita Belov <zodiac.nv@gmail.com>
 * 2016
 *
 * Compile.
 * * Bluegene: mpixlc_r -o dhp dhp.c
 * * Bluegene (OMP): mpixlc_r -qsmp=omp -DUSE_OMP -o dhp dhp.c
 * * Lomonosov: mpicc -std=c99 -o dhp dhp.c
 * Run. 
 * * Bluegene: mpisubmit.bg -n <proc_count> -w 00:05:00 --stdout stdout_<proc_count>_<nodes1>_<nodes2>.txt ./dhp -- <nodes1> <nodes2>
 * * Bluegene: mpisubmit.bg -n <proc_count> -w 00:05:00 --stdout stdout_omp_<proc_count>_<nodes1>_<nodes2>.txt ./dhp -- <nodes1> <nodes2>
 * * Lomonosov: sbatch -n<proc_count> -p test -o stdout_<proc_count>_<nodes1>_<nodes2>.txt ompi ./dhp <nodes1> <nodes2>
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#include <mpi.h>
#include "array.h"

#define DEBUG_PRINT

// the number of a process topology dimensions
#define D 2

// enums for arrays
#define DEFINE_ENUM(dim1, dim2) \
    enum {                      \
        dim1 = 0,               \
        dim2 = 1                \
    }
DEFINE_ENUM(X, Y);
DEFINE_ENUM(DOWN, UP);
DEFINE_ENUM(LEFT, RIGHT);
DEFINE_ENUM(START, END);

// structure to call neighbors_exchange
typedef struct {
    int *neighbors;
    MPI_Comm comm;
    int *n;
    int (*grid_coords)[D];
    int *coords;
    int *dims;
    array(double) (*send)[D];
    array(double) (*receive)[D];
    array(double) vect;
} neighbors_exchange_data;

// structure to call make_solution
typedef struct {
    array(double) sol_vect;
    array(double) sol_buf;
    MPI_Comm comm;
    int proc_count;
    int rank;
    int (*grid_coords)[D];
    int *k;
    int *n;
    int *old_n;
} make_solution_data;

// domain size
const double A = 3.0;
const double B = 3.0;

// mesh nodes
int mesh_n[D];

// get power of 2
int log2i(uint32_t n)
{
    // if n is power of 2
    if (n > 0 && !(n & (n - 1))) {
        // from https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
        static const int tbl32[32] = 
        {
            0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
            31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
        };
        return tbl32[(uint32_t)(n * 0x077CB531U) >> 27];
    }
    
    return -1;
}

// string to integer
int stoi(const char *str, int *n)
{
    char *endPtr;
    errno = 0;
    long int longval = strtol(str, &endPtr, 10);
    
    if ((longval == LONG_MIN || longval == LONG_MAX) && errno == ERANGE) {
        return -1;
    }
    if (sizeof(long) > sizeof(int)) {
        if (longval > INT_MAX || longval < INT_MIN) {
            return -1;
        }
    }
    
    *n = longval;
    return (*endPtr == '\0');
}

// splitting procedure of proc_count power
// integer px is calculated such that
// abs(mesh_n[X] / px - mesh_n[Y] / (power - px)) --> min
int split_function(int power)
{
    double n[] = {
        [X] = mesh_n[X],
        [Y] = mesh_n[Y]
    };
    int px = 0;
    
    for (int i = 0; i < power; i++) {
        if (n[X] > n[Y]) {
            n[X] /= 2;
            px++;
        } else {
            n[Y] /= 2;
        }
    }
    
    return px;
}

double min(double a, double b)
{
    return (a < b ? a : b);
}

double max(double a, double b)
{
    return (a > b ? a : b);
}

double sqr(double val)
{
    return val * val;
}

// function F(x, y)
double F(double x, double y)
{
    return ((sqr(x) + sqr(y)) / sqr(x * y + 1));
}

// grid starting index for this process
#define grid_coords_first(d) \
    min(coords[d], k[d]) * (n[d] + 1) + max(coords[d] - k[d], 0) * n[d]

// left part of equation -Laplace u
#define left_part(P, i, j)                                                                                  \
    ((-(P[mesh_n[X]*(j)+i+1]-P[mesh_n[X]*(j)+i])/h[X]+(P[mesh_n[X]*(j)+i]-P[mesh_n[X]*(j)+i-1])/h[X])/h[X]+ \
    (-(P[mesh_n[X]*(j+1)+i]-P[mesh_n[X]*(j)+i])/h[Y]+(P[mesh_n[X]*(j)+i]-P[mesh_n[X]*(j-1)+i])/h[Y])/h[Y])

// calculate right part
void right_part(array(double) rhs_vect, double *h)
{
    for (int j = 0; j < mesh_n[Y]; j++) {
        for (int i = 0; i < mesh_n[X]; i++) {
            rhs_vect[j * mesh_n[X] + i] = F(i * h[X], j * h[Y]);
        }
    }
}

// function phi(x, y). it's also solution function
double boundary_value(double x, double y)
{
	return log(1.0 + x * y);
}

// exchange values from/to neighbour processes
void neighbors_exchange(neighbors_exchange_data ned)
{
    // for all 4 neighbours
    MPI_Request send_reqs[4];
    MPI_Request receive_reqs[4];
    
    // used for loops for all 4 neighbours
    int if_conds[]  = { 0, 0, ned.dims[X] - 1, ned.dims[Y] - 1 };
    int add_vals[]  = { 0, 0, ned.n[X] - 1, mesh_n[X] * (ned.n[Y] - 1) };
    int add_vals2[] = { 0, 0, ned.n[X], mesh_n[X] * ned.n[Y] };
    int mul_vals[]  = { mesh_n[X], 1 };
    
    // send our values
    for (int i = 0; i < 4; i++) {
        if (ned.coords[i % 2] != if_conds[i]) {
            for (int j = 0; j < ned.n[(i + 1) % 2]; j++) {
                ned.send[i % 2][i / 2][j] = ned.vect[mesh_n[X] * ned.grid_coords[Y][DOWN] + ned.grid_coords[X][LEFT] + j * mul_vals[i % 2] + add_vals[i]];
            }
            MPI_Isend(ned.send[i % 2][i / 2], ned.n[(i + 1) % 2], MPI_DOUBLE, ned.neighbors[i], 0, ned.comm, send_reqs + i);
        }
    }
	
    // receive values
    int receive_count = 0;
    for (int i = 0; i < 4; i++) {
        if (ned.coords[i % 2] != if_conds[i]) {
            MPI_Irecv(ned.receive[i % 2][i / 2], ned.n[(i + 1) % 2], MPI_DOUBLE, ned.neighbors[i], 0, ned.comm, receive_reqs + receive_count);
            receive_count++;
        }
    }
	MPI_Waitall(receive_count, receive_reqs, MPI_STATUSES_IGNORE);
    
    // save received values
    for (int i = 0; i < 4; i++) {
        if (ned.coords[i % 2] != if_conds[i]) {
            for (int j = 0; j < ned.n[(i + 1) % 2]; j++) {
                ned.vect[mesh_n[X] * (ned.grid_coords[Y][DOWN] - (i == 1)) + ned.grid_coords[X][LEFT] - (i == 0) + j * mul_vals[i % 2] + add_vals2[i]] = ned.receive[i % 2][i / 2][j];
            }
        }
    }
}

// make solution: getting all blocks from all processes to root (rank 0)
void make_solution(make_solution_data msd)
{
    // create temporary arrays
    array(int) indexes[][D] = {
        [X] = {
            [LEFT] = array_new(msd.proc_count, int),
            [RIGHT] = array_new(msd.proc_count, int)
        },
        [Y] = {
            [DOWN] = array_new(msd.proc_count, int),
            [UP] = array_new(msd.proc_count, int)
        }
    };

    // containing the number of elements that are received from each process
    array(int) receive_counts = array_new(msd.proc_count, int);
    // specifies the displacement relative to recvbuf at which to place the incoming data from process i
    array(int) displs = array_new(msd.proc_count, int);

    array(double) send_buf = array_new((msd.old_n[X] + 1) * (msd.old_n[Y] + 1), double);
    
    array(double) receive_buf;
    // receive only at rank 0
    if (msd.rank == 0) {
        receive_buf = array_new(mesh_n[X] * mesh_n[Y], double);
    }

    int current_size = 0;
    // calculate indexes, receive_counts and displs for each proc
    for (int i = 0; i < msd.proc_count; i++) {
        int coords[2];
        MPI_Cart_coords(msd.comm, i, D, coords);
        
        receive_counts[i] = 1;
        for (int j = 0; j < D; j++) {
            int proc_n = msd.old_n[j % 2] + (coords[j % 2] < msd.k[j % 2]);
            receive_counts[i] *= proc_n;
        
            indexes[j % 2][START][i] = min(coords[j % 2], msd.k[j % 2]) * (msd.old_n[j % 2] + 1) + max(coords[j % 2] - msd.k[j % 2], 0) * msd.old_n[j % 2];
            indexes[j % 2][END][i] = indexes[j % 2][START][i] + proc_n;
        }
		displs[i] = current_size;
		current_size += receive_counts[i];
    }
    
    // create send_buf for each process
    for (int i = msd.grid_coords[Y][DOWN], j = 0; i < msd.grid_coords[Y][UP]; i++) {
        for (int k = msd.grid_coords[X][LEFT]; k < msd.grid_coords[X][RIGHT]; k++) {
            send_buf[j++] = msd.sol_vect[mesh_n[X] * i + k];
        }
    }
    // gather blocks
    MPI_Gatherv(send_buf, msd.n[X] * msd.n[Y], MPI_DOUBLE, receive_buf, receive_counts, displs, MPI_DOUBLE, 0, msd.comm);
    
    if (msd.rank == 0) {
        // create solution vector
        for (int i = 0, j = 0; i < msd.proc_count; i++) {
            for (int k = indexes[Y][DOWN][i]; k < indexes[Y][UP][i]; k++) {
                for (int l = indexes[X][LEFT][i]; l < indexes[X][RIGHT][i]; l++) {
                    msd.sol_buf[mesh_n[X] * k + l] = receive_buf[j++];
                }
            }
        }
    
        array_delete(&receive_buf);
    }
    
    // free memory
    array_delete(&receive_counts);
    array_delete(&displs);
    array_delete(&send_buf);
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            array_delete(&indexes[i][j]);
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
usage_label:
        fprintf(stderr, "Usage: dhp <nodes number 1> <nodes number 2> [<save|loadsave> <iteration count> | <load>].\n");
        return 1;
    }
    if (stoi(argv[1], &mesh_n[X]) == -1 || mesh_n[X] < 1 ||
            stoi(argv[2], &mesh_n[Y]) == -1 || mesh_n[Y] < 1) {
        fprintf(stderr, "Mesh numbers should be positive.\n");
        return 2;
    }
    
    int load = 0, save = 0, loadsave = 0;
    int iteration_count = 0;
    if (argc >= 4) {
        if (strcmp(argv[3], "save") == 0) {
            save = 1;
        } else if (strcmp(argv[3], "loadsave") == 0) {
            loadsave = 1;
        } else if (strcmp(argv[3], "load") == 0) {
            load = 1;
        } else {
            goto usage_label;
        }
        
        if (save || loadsave) {
            if (argc < 5) {
                goto usage_label;
            } else {
                if (stoi(argv[4], &iteration_count) == -1 || iteration_count < 1) {
                    fprintf(stderr, "Iteration count should be positive.\n");
                    return 2;
                }
            }
        }
    }
    
    FILE *file;
    char fname[256];
    double previous_time = 0, work_time = 0;
    
#ifdef USE_OMP
    omp_set_num_threads(3);
#endif
    
    MPI_Init(&argc, &argv);
    
    // the number of processes and rank in communicator
    int rank, proc_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
    
    if (rank == 0) {
        work_time = MPI_Wtime();
    }
    
    // proc_count = 2^(power)
    int power;
    if ((power = log2i(proc_count)) < 0) {
        if (rank == 0) {
            fprintf(stderr, "Processors should be power of 2.\n");
        }
        MPI_Finalize();
        return 3;
    }
    
    // power splits into sum p[X] + p[Y]
    int sf_ret = split_function(power);
    int p[] = {
        [X] = sf_ret,
        [Y] = power - sf_ret
    };

    // dims[i] = 2^p[i]
    // for all i = {X, Y}
    int dims[] = {
        [X] = 1u << p[X],
        [Y] = 1u << p[Y]
    };
    
    // mesh_n[i] = n[i] * dims[i] + k[i]
    // for all i = {X, Y}
    // number of i points in mesh for the process
    int n[] = {
        [X] = mesh_n[X] >> p[X],
        [Y] = mesh_n[Y] >> p[Y]
    };
    int k[] = {
        [X] = mesh_n[X] - dims[X] * n[X],
        [Y] = mesh_n[Y] - dims[Y] * n[Y]
    };
    
    // mesh steps on X and Y axes
    double h[] = {
        [X] = A / (mesh_n[X] - 1),
        [Y] = B / (mesh_n[Y] - 1)
    };
    
#ifdef DEBUG_PRINT
    if (rank == 0) {
        printf("The number of processes proc_count = 2^%d. It is split into %d x %d processes.\n"
               "The number of nodes mesh_n[X] = %d, mesh_n[Y] = %d.\n"
               "Blocks B(i,j) have size:\n",
            power, dims[X], dims[Y],
            mesh_n[X], mesh_n[Y]);
        if (k[X] > 0 && k[Y] > 0) {
            printf("-->\t %d x %d iff i = 0 .. %d, j = 0 .. %d;\n", n[X] + 1, n[Y] + 1, k[X] - 1, k[Y] - 1);
        }
        if (k[Y] > 0) {
            printf("-->\t %d x %d iff i = %d .. %d, j = 0 .. %d;\n", n[X], n[Y] + 1, k[X], dims[X] - 1, k[Y] - 1);
        }
        if (k[X] > 0) {
            printf("-->\t %d x %d iff i = 0 .. %d, j = %d .. %d;\n", n[X] + 1, n[Y], k[X] - 1, k[Y], dims[Y] - 1);
        }
        printf("-->\t %d x %d iff i = %d .. %d, j = %d .. %d.\n", n[X], n[Y], k[X], dims[X] - 1, k[Y], dims[Y] - 1);
    }
#endif

    // handler of a new communicator with cartesian topology
    MPI_Comm COMM_GRID;
    
    // the process coordinates in the cartesian topology created for mesh
    int coords[2];
    
    // create the cartesian topology of processes
    MPI_Cart_create(
        MPI_COMM_WORLD, // input communicator
        D,              // number of dimensions of cartesian grid
        dims,           // integer array of size ndims specifying the number of processes in each dimension 
        (int[2]){0, 0}, // logical array of size ndims specifying whether the grid is periodic (true) or not (false) in each dimension
        1,              // ranking may be reordered (true) or not (false)
        &COMM_GRID      // communicator with new cartesian topology
    );
    // get rank of current process
    MPI_Comm_rank(COMM_GRID, &rank);
    // determine process coords in cartesian topology given rank in group 
    MPI_Cart_coords(COMM_GRID, rank, D, coords);
    
    if (coords[X] < k[X]) {
        n[X]++;
    }
    if (coords[Y] < k[Y]) {
        n[Y]++;
    }
    
    // global grid left/top starting index for this process
    int gc_xl = grid_coords_first(X),
        gc_yd = grid_coords_first(Y);
    // grid coords for this process [start; end) for each dimension
    int grid_coords[][D] = {
        [X] = {
            [LEFT] = gc_xl,
            [RIGHT] = gc_xl + n[X]
        },
        [Y] = {
            [DOWN] = gc_yd,
            [UP] = gc_yd + n[Y]
        }
    };
    // grid cropped indexes for this process
    int indexes[][D] = {
        [X] = {
            [START] = max(1, grid_coords[X][LEFT]),
            [END] = min(mesh_n[X] - 1, grid_coords[X][RIGHT])
        },
        [Y] = {
            [START] = max(1, grid_coords[Y][DOWN]),
            [END] = min(mesh_n[Y] - 1, grid_coords[Y][UP])
        }
    };
    
    // the neighbours of the process
    int left, right, up, down;
    
    // get the shifted source and destination ranks, given a shift direction and amount 
    MPI_Cart_shift(
        COMM_GRID,  // communicator with cartesian structure
        X,          // coordinate dimension of shift (X or Y)
        1,          // displacement (> 0: upwards shift, < 0: downwards shift)
        &left,      // rank of source process
        &right      // rank of destination process
    );
    MPI_Cart_shift(
        COMM_GRID,  // communicator with cartesian structure
        Y,          // coordinate dimension of shift (X or Y)
        1,          // displacement (> 0: upwards shift, < 0: downwards shift)
        &down,      // rank of source process
        &up         // rank of destination process
    );

#ifdef DEBUG_PRINT
    printf("My Rank in COMM_GRID is %d. My topological coords is (%d, %d). Domain size is %d x %d nodes.\n"
           "My neighbours: left = %d, right = %d, down = %d, up = %d.\n",
            rank, coords[X], coords[Y], n[X], n[Y],
            left, right, down, up);
#endif

    // the solution array
    array(double) sol_vect = array_new(mesh_n[X] * mesh_n[Y], double);
    // the residual array
    array(double) res_vect = array_new(mesh_n[X] * mesh_n[Y], double);
    // the right hand side of Puasson equation
    array(double) rhs_vect = array_new(mesh_n[X] * mesh_n[Y], double);

    // initialize arrays
    array_fill(sol_vect, 0);
    array_fill(res_vect, 0);
    right_part(rhs_vect, h);
    
    // generate mesh
	for (int i = 0; i < mesh_n[X]; i++) {
		sol_vect[i] = boundary_value(i * h[X], 0.0);
		sol_vect[mesh_n[X] * (mesh_n[Y] - 1) + i] = boundary_value(i * h[X], B);
	}
	for (int j = 0; j < mesh_n[Y]; j++) {
		sol_vect[mesh_n[X] * j] = boundary_value(0.0, j * h[Y]);
		sol_vect[mesh_n[X] * j + (mesh_n[X] - 1)] = boundary_value(A, j * h[Y]);
	}

#ifdef DEBUG_PRINT
    if (rank == 0) {
		printf("\nSteep descent iterations begin ...\n");
	}
#endif

    // auxiliary values
    double sp, tau, alpha, norm, err;
    double tsp, ttau, talpha, terr;
    
    // epsilon to stop iterative process
    double eps = 0.0001;

    // the residual vector r(k) = Ax(k) - f is calculating...
#ifdef USE_OMP
    // distribute for iterations between parallel threads
    #pragma omp parallel for
#endif
	for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
		for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
			res_vect[j * mesh_n[X] + i] = -rhs_vect[j * mesh_n[X] + i];
        }
    }

    // the value of product (r(k),r(k)) is calculating...
	sp = 0.0;
#ifdef USE_OMP
    // distribute for iterations between parallel threads
    // reduction is for fast sum sp
    #pragma omp parallel for reduction(+:sp)
#endif
	for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
		for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
			sp += sqr(res_vect[mesh_n[X] * j + i]);
        }
    }
	ttau = sp * h[X] * h[Y];
	MPI_Allreduce(&ttau, &tau, 1, MPI_DOUBLE, MPI_SUM, COMM_GRID);
    
    // the value of product sp = (Ar(k),r(k)) is calculating...
	sp = 0.0;
#ifdef USE_OMP
    #pragma omp parallel for reduction(+:sp)
#endif
	for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
		for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
			sp += left_part(res_vect, i, j) * res_vect[mesh_n[X] * j + i];
        }
    }
	tsp = sp * h[X] * h[Y];
	MPI_Allreduce(&tsp, &sp, 1, MPI_DOUBLE, MPI_SUM, COMM_GRID);
	tau = tau / sp;

    // the x(k+1) is calculating...
#ifdef USE_OMP
    #pragma omp parallel for
#endif
	for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
		for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
			sol_vect[mesh_n[X] * j + i] = -tau * res_vect[mesh_n[X] * j + i];
        }
    }
    
#ifdef DEBUG_PRINT
	if (rank == 0) {
		printf("The Steep Descent iteration has been performed.\n");
	}
#endif

    // the vector of A-orthogonal system in CGM
    // g(0) = r(k-1).
	array(double) basis_vect = res_vect;
    
    res_vect = array_new(mesh_n[X] * mesh_n[Y], double);
    array_fill(res_vect, 0);
	
    // CGM iterations begin...
    // sp == (Ar(k-1),r(k-1)) == (Ag(0),g(0)), k=1
#ifdef DEBUG_PRINT
	if (rank == 0) {
		printf("\nCGM iterations begin ...\n");
	}
#endif

    if (rank == 0) {
#ifdef USE_OMP
        sprintf(fname, "PuassonParallel_ECGM_OMP_%dx%d_n%d.log", mesh_n[X], mesh_n[X], proc_count);
#else
        sprintf(fname, "PuassonParallel_ECGM_%dx%d_n%d.log", mesh_n[X], mesh_n[X], proc_count);
#endif
        // if load saved data append file instead of clear
        if (load || loadsave) {
            file = fopen(fname, "a");
        } else {
            file = fopen(fname, "w");
        }
    }
    
    // fill data for neighbors_exchange
    array(double) send[D][D];
    array(double) receive[D][D];
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            send[i][j] = array_new(n[(i + 1) % 2], double);
            receive[i][j] = array_new(n[(i + 1) % 2], double);
        }
    }

    neighbors_exchange_data ned = {
        (int[4]){ left, down, right, up },
        COMM_GRID,
        n,
        grid_coords,
        coords,
        dims,
        send,
        receive,
        NULL // vect
    };

    // the current iteration number
    int counter = 0;
    // the current run iteration number
    int counter_run = 0;
    
    // load saved data
    if (load || loadsave) {
        if (rank == 0) {
            FILE *save_data = fopen("save_data.txt", "r");
            if (save_data) {
                fscanf(save_data, "%d %lf", &counter, &previous_time);
                printf("Loaded previous iterations %d with working time %f.\n", counter, previous_time);
                fclose(save_data);
                remove("save_data.txt");
            } else {
                fprintf(stderr, "Could not open 'save_data.txt' file. Aborting.\n");
                goto end_label;
            }
        }
        
        sprintf(fname, "save_data_%d_%d_%d_%d.dat", left, down, right, up);
        FILE *data = fopen(fname, "rb");
        if (data) {
            fread(&sp, sizeof(sp), 1, data);
            fread(sol_vect, array_item_size(sol_vect), array_size(sol_vect), data);
            fread(res_vect, array_item_size(res_vect), array_size(res_vect), data);
            fread(basis_vect, array_item_size(basis_vect), array_size(basis_vect), data);
            fclose(data);
            remove(fname);
        } else {
            fprintf(stderr, "Could not open '%s' file. Aborting.\n", fname);
            goto end_label;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    do {
        // get p(i, j) values from the neighbours of the process
        ned.vect = sol_vect;
        neighbors_exchange(ned);

        // the residual vector r(k) is calculating...
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
            for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
                res_vect[mesh_n[X] * j + i] = left_part(sol_vect, i, j) - rhs_vect[mesh_n[X] * j + i];
            }
        }

        // send r(k) values from neighbours
        ned.vect = res_vect;
        neighbors_exchange(ned);

        // the value of product (Ar(k),g(k-1)) is calculating...
        alpha = 0.0;
#ifdef USE_OMP
        #pragma omp parallel for reduction(+:alpha)
#endif
        for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
            for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
                alpha += left_part(res_vect, i, j) * basis_vect[mesh_n[X] * j + i];
            }
        }
        talpha = alpha * h[X] * h[Y] / sp;
        MPI_Allreduce(&talpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, COMM_GRID);
        
        // the new basis vector g(k) is being calculated...
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
            for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
                basis_vect[mesh_n[X] * j + i] = res_vect[mesh_n[X] * j + i] - alpha * basis_vect[mesh_n[X] * j + i];
            }
        }
        
        // the value of product (r(k),g(k)) is being calculated...
        tau = 0.0;
#ifdef USE_OMP
        #pragma omp parallel for reduction(+:tau)
#endif
        for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
            for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
                tau += res_vect[mesh_n[X] * j + i] * basis_vect[mesh_n[X] * j + i];
            }
        }
        ttau = tau * h[X] * h[Y];
        MPI_Allreduce(&ttau, &tau, 1, MPI_DOUBLE, MPI_SUM, COMM_GRID);
        
        ned.vect = basis_vect;
        neighbors_exchange(ned);
        
        // the value of product sp = (Ag(k),g(k)) is being calculated...
        sp = 0.0;
#ifdef USE_OMP
        #pragma omp parallel for reduction(+:sp)
#endif
        for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
            for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
                sp += left_part(basis_vect, i, j) * basis_vect[mesh_n[X] * j + i];
            }
        }
        tsp = sp * h[X] * h[Y];
        MPI_Allreduce(&tsp, &sp, 1, MPI_DOUBLE, MPI_SUM, COMM_GRID);
        tau = tau / sp;
        
        // the x(k+1) is being calculated...
        err = 0.0;
#ifdef USE_OMP
        #pragma omp parallel for reduction(+:err)
#endif
        for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
            for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
                double new_value = sol_vect[mesh_n[X] * j + i] - tau * basis_vect[mesh_n[X] * j + i];
                err = max(err, fabs(new_value - sol_vect[mesh_n[X] * j + i]));
                sol_vect[mesh_n[X] * j + i] = new_value;
            }
        }

        // send err value
        MPI_Allreduce(&err, &norm, 1, MPI_DOUBLE, MPI_MAX, COMM_GRID);
    
        if (rank == 0) {
            fprintf(file, "\nThe iteration %d of conjugate gradient method has been finished.\n"
                            "The value of \\alpha(k) = %f, \\tau(k) = %f. The difference value is %f.\n",
                    counter, alpha, tau, norm);
        }

        counter++;
        counter_run++;
        
        if ((save || loadsave) && counter_run == iteration_count) {
            MPI_Barrier(MPI_COMM_WORLD);
            
            if (rank == 0) {
                fclose(file);
        
                FILE *save_data = fopen("save_data.txt", "w");
                fprintf(save_data, "%d %f\n", counter, MPI_Wtime() - work_time + previous_time);
                fclose(save_data);
            }
            
            sprintf(fname, "save_data_%d_%d_%d_%d.dat", left, down, right, up);
            FILE *data = fopen(fname, "wb");
            fwrite(&sp, sizeof(sp), 1, data);
            fwrite(sol_vect, array_item_size(sol_vect), array_size(sol_vect), data);
            fwrite(res_vect, array_item_size(res_vect), array_size(res_vect), data);
            fwrite(basis_vect, array_item_size(basis_vect), array_size(basis_vect), data);
            fclose(data);
            
            goto end_label;
        }
    } while (norm > eps); // check exit condition

    // make solution from all processes to root process (rank 0)
    array(double) sol_buf;
	if (rank == 0) {
		work_time = MPI_Wtime() - work_time + previous_time;
		printf("The %d iteration of CGM method has been carried out.\n", counter);
		sol_buf = array_new(mesh_n[X] * mesh_n[Y], double);
	}
    make_solution_data msd = {
        sol_vect,
        sol_buf,
        COMM_GRID,
        proc_count,
        rank,
        grid_coords,
        k,
        n,
        (int[2]){ n[X] - (coords[X] < k[X]), n[Y] - (coords[Y] < k[Y]) }
    };
    make_solution(msd);
    
    // compute error
	terr = 0.0;
    for (int j = indexes[Y][START]; j < indexes[Y][END]; j++) {
        for (int i = indexes[X][START]; i < indexes[X][END]; i++) {
            terr = max(terr, fabs(boundary_value(i * h[X], j * h[Y]) - sol_vect[mesh_n[X] * j + i]));
        }
    }
	MPI_Reduce(&terr, &err, 1, MPI_DOUBLE, MPI_MAX, 0, COMM_GRID);

    // the end of CGM iterations

    // printing some results...
	if (rank == 0) {
		fprintf(file, "\nThe error of iterations is estimated by %.12f.\n", err);
		fprintf(file, "Time spent: %f seconds\n", work_time);
        fclose(file);
        
#ifdef USE_OMP
        sprintf(fname, "PuassonParallel_ECGM_OMP_%dx%d_n%d.dat", mesh_n[X], mesh_n[Y], proc_count);
#else
        sprintf(fname, "PuassonParallel_ECGM_%dx%d_n%d.dat", mesh_n[X], mesh_n[Y], proc_count);
#endif
        // if load saved data append file instead of clear
        if (load || loadsave) {
            file = fopen(fname, "a");
        } else {
            file = fopen(fname, "w");
        }
		fprintf(file, "# This is the conjugate gradient method for descrete Puasson equation.\n"
                        "# A = %f, B = %f, N[0,A] = %d, N[0,B] = %d.\n"
                        "# One can draw it by gnuplot by the command: splot 'MyPath\\FileName.dat' with lines\n",
                A, B, mesh_n[X], mesh_n[Y]);
		for (int j = 0; j < mesh_n[Y]; j++) {
			for (int i = 0; i < mesh_n[X]; i++) {
				fprintf(file, "\n%f %f %f", i * h[X], j * h[Y], sol_buf[mesh_n[X] * j + i]);
            }
			fprintf(file, "\n");
		}
        fclose(file);
	}

    // free memory
    if (rank == 0) {
        array_delete(&sol_buf);
    }
    
end_label:
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            array_delete(&send[i][j]);
            array_delete(&receive[i][j]);
        }
    }
    array_delete(&basis_vect);
    array_delete(&rhs_vect);
    array_delete(&res_vect);
    array_delete(&sol_vect);
    
    MPI_Finalize();
    
    return 0;
}
