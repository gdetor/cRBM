#ifndef RBM_H
#define RBM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>
#include <sys/mman.h>

#include "pcg_basic.h"

#define BATCH 0

#define alloc(type_t, size) (type_t *) malloc((size) * sizeof(type_t))
#define alloc_zeros(type_t, size) (type_t *) calloc(size, sizeof(type_t))
#define mem_test(mem) {if(!mem) { \
                         perror("Cannot allocate memory!\n");   \
                         exit(-2);} }
#define dealloc(ptr) {free(ptr); \
                      ptr = NULL; }

#define XORSWAP(a, b)   ((&(a) == &(b)) ? (a) : ((a)^=(b),(b)^=(a),(a)^=(b)))


/* ***********************************************************************
 * Restricted Boltzmann Machine (RBM) data structure. This structure
 * contains all the necessary arrays and parameters of the RBM. 
 *
 * num_visibles : Number of visible units
 * num_hiddens  : Number of hiddne units
 * num_batches  : Number of batches
 * batch_size   : Size of a batch (number of data per batch)
 * v            : Visible neurons probability (0)
 * pv           : Visible neurons state
 * h            : Hidden neurons probability
 * ph           : Hidden neurons state
 * V            : Actual input to the RBM
 * W            : RBM weights matrix (visible units to hidden ones) 
 *                (num_visibles x num_hiddens)
 * dW_data      : Weights increment in data (positive) phase 
 * dW_recon     : Weights increment in reconstruction (negative) phase
 * a            : Visible units biases
 * da_data      : Visible units biases increment in data (positive) phase 
 * da_recon     : Visible units biases increment in reconstruction (negative)
 *                phase
 * b            : Hidden units biases
 * db_data      : Hidden units biases increment in data (positive) phase 
 * db_recon     : hidden units biases increment in reconstruction (negative)
 *                phase
 * eta          : Learning rate
 * alpha        : Regularization parameter (only for Non-negative matrix
 *                factorization) -- default value is 0
 * ***********************************************************************/
typedef struct rmb_s {
    double *v;
    double *pv;
    double *h;
    double *ph;
    double *V;
    double *W;
    double *dW_data;
    double *dW_recon;
    double *a;
    double *da_data;
    double *da_recon;
    double *b;
    double *db_data;
    double *db_recon;
    double eta;
    double alpha;
    size_t num_visibles;
    size_t num_hiddens;
    size_t num_batches;
    size_t batch_size;
} rbm_t;

/* Memory functions */
void allocate_resources(rbm_t *);
void cleanup_resources(rbm_t *);

/* Writting to files functions */
void write_weights2file(char *, double *, size_t, size_t);
void write_error2file(char *, double *, size_t);

/* Load/read data functions */
double **read_mnist_images(size_t);
int *read_mnist_labels(size_t);
void read_bst_images(char *, double ***, double **, size_t, size_t);

/* Random Number Generator funcions */
double uniform(double, double, pcg32_random_t *);
void uniform_array(double **, double, double, size_t, pcg32_random_t *);
double normal(double, double);
void normal_array(double **, double, double, size_t);
int *shuffle_indices(size_t, pcg32_random_t *);
int *generate_batches(rbm_t *, size_t, pcg32_random_t *);

/* RBM functions */
void test_rbm(rbm_t *restrict, pcg32_random_t *);
void rbm(rbm_t *, pcg32_random_t *);
void rbm_batch(rbm_t *, double **, int *, pcg32_random_t *);
void contrastive_divergence(rbm_t *);
void visible2hidden(rbm_t *, pcg32_random_t *);
void hidden2visible(rbm_t *, pcg32_random_t *);
void hidden2visible_test(rbm_t *, pcg32_random_t *);
void visible2hidden_test(rbm_t *, pcg32_random_t *);

void run_mnist_training(int);
void run_bars_stripes_training(int);

#endif  /* RBM_H  */
