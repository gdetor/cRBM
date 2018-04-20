#include "rbm.h"


/* ***********************************************************************
 * WRITE_WEIGHTS2FILE Writes the weights matrix W into a text file. 
 *
 * Args:
 *  fname (char *)          : Filename (to write to)
 *  W  (double *)           : RBM weight matrix
 *  num_visuals (size_t)    : Number of visible units  
 *  num_hiddens (size_t)    : Number of hidden units
 *
 * Return:
 *
 * ***********************************************************************/
void write_weights2file(char * fname,
                double *W,
                size_t num_visuals,
                size_t num_hiddens) {
    FILE *fp = NULL;

    if(!(fp = fopen(fname, "wb"))) {
        printf("Cannot open file %s!\n", fname);
        printf("Nothing written!\n");
    } else {
        fwrite(W, sizeof(double), num_visuals * num_hiddens, fp);
        fclose(fp);
    }
}


/* ***********************************************************************
 * WRITE_ERROR2FILE Writes the training error into a text file. 
 *
 * Args:
 *  fname (char *)          : Filename (to write to)
 *  error  (double *)       : Training error vector
 *  size (size_t)           : Size of the error vectror
 *
 * Return:
 *
 * ***********************************************************************/
void write_error2file(char *fname, double *error, size_t size) {
    FILE *fp=NULL;

    if(!(fp = fopen(fname, "wb"))) {
        printf("Cannot open file\n");
        exit(-1);
    } else {
        fwrite(error, sizeof(double), size, fp);
        fclose(fp);
    }
}


/* ***********************************************************************
 * ALLOCATE_RESOURCES Allocates memory for all the necessary vectors of
 * the Restricted Boltzmann Machine neural network.
 *
 * Args:
 *  nn (rbm_t *)          : Pointer to an RBM data structure
 *
 * Return:
 *
 * ***********************************************************************/
void allocate_resources(rbm_t *nn) {
    size_t len = nn->num_visibles * nn->num_hiddens;

    nn->v = alloc_zeros(double, nn->num_visibles);
    nn->pv = alloc_zeros(double, nn->num_visibles);
    nn->h = alloc_zeros(double, nn->num_hiddens);
    nn->ph = alloc_zeros(double, nn->num_hiddens);
    nn->a = alloc_zeros(double, nn->num_visibles);
    nn->da_data = alloc_zeros(double, nn->num_visibles);
    nn->da_recon = alloc_zeros(double, nn->num_visibles);
    nn->b = alloc_zeros(double, nn->num_hiddens);
    nn->db_data = alloc_zeros(double, nn->num_hiddens);
    nn->db_recon = alloc_zeros(double, nn->num_hiddens);
    nn->V = alloc_zeros(double, nn->num_visibles);
    nn->W = alloc_zeros(double, len); 
    nn->dW_data = alloc_zeros(double, len);
    nn->dW_recon = alloc_zeros(double, len);
}


/* ***********************************************************************
 * CLEANUP_RESOURCES Deallocates memory for all RBM vectors.
 *
 * Args:
 *  nn (rbm_t *)          : Pointer to an RBM data structure
 *
 * Return:
 *
 * ***********************************************************************/
void cleanup_resources(rbm_t *nn) {
    dealloc(nn->W);
    dealloc(nn->v);
    dealloc(nn->pv);
    dealloc(nn->h);
    dealloc(nn->ph);
    dealloc(nn->V);
    dealloc(nn->a);
    dealloc(nn->da_data);
    dealloc(nn->da_recon);
    dealloc(nn->b);
    dealloc(nn->db_data);
    dealloc(nn->db_recon);
    dealloc(nn->dW_data);
    dealloc(nn->dW_recon);
}


/* ***********************************************************************
 * UNIFORM  Returns a uniform number in the interval [low, upper). It
 * uses the PCG random number generator.
 *
 * Args:
 *  low (double)             : Lower boundary
 *  upper (double)           : Upper boundary
 *  rng (*pcg32_random_t)    : Pointer to the Random number generator 
 *
 * Return:
 *  Double -- A random number uniformly drawn from [low, upper).
 * ***********************************************************************/
double uniform(double low, double upper, pcg32_random_t *rng) {
    double rand_n = pcg32_random_r(rng) / (1.0 + UINT32_MAX);
    double range = (upper - low);
    return (rand_n * range) + low;
}


/* ***********************************************************************
 * UNIFORM_ARRAY  Fills in an array with random numbers (uniformly 
 * distributed).
 *
 * Args:
 *  x (double **)           : Double array to fill in
 *  a (double a)            : Lower boundary
 *  b (double b)            : Upper boundary
 *  len (size_t)            : The size of the array
 *  rng (pcg32_random_t)    : Pointer to the RNG
 *
 * Return:
 *
 * ***********************************************************************/
void uniform_array(double **x, double a, double b, size_t len,
                   pcg32_random_t *rng) {
    size_t i;

    for(i = 0; i < len; ++i) {
        (*x)[i] = uniform(a, b, rng);
    }
}


/* ***********************************************************************
 * NORMAL Returns a double drawn from a normal distribution based on 
 * Box-Muller. 
 *
 * Args:
 *  mu (double)     : The mean of the distribution
 *  sigma (double)  : The variance of the distribution
 *
 * Return:
 *  Double -- Drawn from a normal distribution.
 * ***********************************************************************/
double normal(double mu, double sigma)
{
	const double epsilon = DBL_MIN;
	const double two_pi = 2.0*3.14159265358979323846;

	static double z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   /* u1 = rand() * (1.0 / RAND_MAX); */
       u1 = (double) pcg32_boundedrand(100) / (double) 100;
	   /* u2 = rand() * (1.0 / RAND_MAX); */
       u2 = (double) pcg32_boundedrand(100) / (double) 100;
	 }
	while ( u1 <= epsilon );

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}


/* ***********************************************************************
 * NORMAL_ARRAY  Fills in an array with random numbers drawn from a 
 * normal distribution.
 *
 * Args:
 *  x (double **)           : Array to populate
 *  mean (double)           : Mean of the distribution
 *  sigma (double)          : Variance of the distribution
 *  len  (size_t)           : Size of the array
 *  rng (pcg32_random_t *)  : Pointer to RNG
 * Return:
 * ***********************************************************************/
void normal_array(double **x, double mean, double sigma, size_t len) {
    size_t i;

    for(i = 0; i < len; ++i) {
        (*x)[i] = normal(mean, sigma);
    }
}


/* ***********************************************************************
 * SHUFFLE_INDICES  Generates and returns a vector of integers shuffled. 
 *
 * Args:
 *  num_samples  (size_t)   : Number of integers to generate
 *  rng (pcg32_random_t *)  : Pointer to RNG
 *
 * Return:
 *  Integer's array
 * ***********************************************************************/
int *shuffle_indices(size_t num_samples, pcg32_random_t *rng) {
    size_t i;
    int p;
    int *im_indices=NULL;

    im_indices = alloc(int, num_samples);

    for (i = 0; i < num_samples; ++i) {
        im_indices[i] = i;
    }

    for (i = num_samples; i > 1; --i) {
        p = (int) pcg32_boundedrand_r(rng, i-1);
        XORSWAP(im_indices[i-1], im_indices[p]);
    }
    
    return im_indices;
}


/* ***********************************************************************
 * GENERATE_BATCHES  Builds all the batches for training based on the 
 * number of samples (input data set) and the size of batches. 
 *
 * The shuffling in this function is performed according to the
 * Fisher-Yates algorithm.
 *
 * Args:
 *  nn (*rbm_t)             : Pointer to the RBM data structure (by reference)
 *  num_samples (int)       : Number of samples
 *  rng (*pcg32_random_t)   : Poiinter to the RNG
 *
 * Return:
 *  Pointer to integer -- An array containing shuffled indices of the
 *  input data set.
 * ***********************************************************************/
int *generate_batches(rbm_t *nn, size_t num_samples, pcg32_random_t *rng) {
    size_t i, b;
    int p;
    int *indices = NULL, *im_indices=NULL;

    indices = alloc(int, num_samples);
    im_indices = alloc(int, nn->num_batches * nn->batch_size);

    for (i = 0; i < num_samples; ++i) {
        indices[i] = (int) i;
    }

    /* The Fisher-Yates algorithm -- Simplicity is beauty */
    for (i = num_samples; i > 1; --i) {
        p = (int) pcg32_boundedrand_r(rng, i-1);
        XORSWAP(indices[i-1], indices[p]);    /* Magic XOR */
    }
    
    for (b = 0; b < nn->num_batches; ++b) {
        for (i = 0; i < nn->batch_size; ++i) {
            im_indices[b*nn->batch_size+i] = indices[b*nn->batch_size+i];
        }
    }

    dealloc(indices);
    
    return im_indices;
}
