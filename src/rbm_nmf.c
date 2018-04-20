#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>

#include "pcg_basic.h"


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



/* ***********************************************************************
 * WRITE2FILE Writes the weights matrix W into a text file. 
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

    if(!(fp = fopen(fname, "w"))) {
        printf("Cannot open file\n");
        exit(-1);
    }

    for(int i = 0; i < num_visuals; ++i) {
        for (int j = 0; j < num_hiddens; ++j) {
            fprintf(fp, "%f ", W[i*num_hiddens+j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}


/* ***********************************************************************
 * REVERSE_INT Inverses the bits of the input integer number.
 *
 * Args:
 *  x (int)     : Input integer
 * Return:
 *  Integer -- with reversed bits
 * ***********************************************************************/
static inline int reverse_int (int x) {
    unsigned char ch1, ch2, ch3, ch4;

    ch1 = x & 255;
    ch2 = (x >> 8) & 255;
    ch3 = (x >> 16) & 255;
    ch4 = (x >> 24) & 255;

    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}


/* ***********************************************************************
 * READ_MNIST_IMAGES  Read the MNIST images data set.
 *
 * Args:
 *  num_images  (size_t)       : NUmber of input images to read
 *
 * Return:
 *  A double pointer to double (to the images)
 * ***********************************************************************/
double **read_mnist_images(size_t num_images) {
    int n_rows=0, n_cols=0;
    int magic_number = 0;
    int number_of_images = 0;

    unsigned char temp = 0;
    double **tmp = NULL;

    FILE *fp = NULL;
    // char *fname  = "/home/gdetorak/Datasets/mnist/t10k-images-idx3-ubyte";
    char *fname  = "/home/gdetorak/mnist/train-images-idx3-ubyte";

    if (!(fp = fopen(fname, "rb"))) {
        printf("File not found!\n");
        exit(-1);
    }

    fread((char*)&magic_number, sizeof(magic_number), 1, fp);
    magic_number = reverse_int(magic_number);

    fread((char*)&number_of_images, sizeof(number_of_images), 1, fp);
    number_of_images = reverse_int(number_of_images);

    fread((char*)&n_rows, sizeof(n_rows), 1, fp);
    n_rows = reverse_int(n_rows);

    fread((char*)&n_cols, sizeof(n_cols), 1, fp);
    n_cols = reverse_int(n_cols);

    tmp = (double **)malloc(num_images * sizeof(double *));
    for(int i = 0; i < num_images; ++i) {
        tmp[i] = (double *)malloc(n_rows * n_cols * sizeof(double));
    }


    for(int i = 0; i < num_images; ++i) {
        for(int r = 0; r < n_rows; ++r) {
            for(int c = 0; c < n_cols; ++c) {
                temp = 0;
                fread((char*)&temp, 1, sizeof(temp), fp);
                tmp[i][(n_rows*r)+c] = (double)temp / 255.0;
            }
        }
    }

    fclose(fp);
    fp = NULL;

    return tmp;
}


/* ***********************************************************************
 * READ_MNIST_LABELS Reads the MNIST data set labels. In consistency 
 * with the MNIST data set images.
 *
 * Args:
 *  num_images  (size_t)    : Number of images
 *
 * Return:
 *  Pointer to integer -- Array containing the labels of MNIST images. 
 * ***********************************************************************/
int *read_mnist_labels(size_t num_images) {
    int n_labels = 0;
    int magic_number = 0;

    unsigned char temp = 0;
    int *tmp = NULL;

    FILE *fp = NULL;
    // char *fname  = "/home/gdetorak/Datasets/mnist/t10k-labels-idx1-ubyte";
    char *fname  = "/home/gdetorak/mnist/train-labels-idx1-ubyte";

    if (!(fp = fopen(fname, "rb"))) {
        printf("File not found!\n");
        exit(-1);
    }

    tmp = (int *)alloc(int, num_images);

    fread((char*)&magic_number, sizeof(magic_number), 1, fp);
    magic_number = reverse_int(magic_number);

    fread((char*)&n_labels, sizeof(n_labels), 1, fp);
    n_labels = reverse_int(n_labels);

    for(int i = 0; i < num_images; ++i) {
        temp = 0;
        fread((char*)&temp, 1, sizeof(temp), fp);
        tmp[i] = (int)temp;
    }

    fclose(fp);
    fp = NULL;

    return tmp;
}


/* ***********************************************************************
 * SIGMOID  Just the sigmoid function. 
 *
 * Args:
 *  x (double)  : Input double to the sigmoid function
 * 
 * Return:
 *  Double -- the value of the sigmoid on x
 *
 * ***********************************************************************/
static inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


/* ***********************************************************************
 * NEGATIVE Negative rectification. 
 *
 * Args:
 *  x (double)  : Input value
 *
 * Return:
 *  Double -- The value of x if x is negative, zero otherwise. 
 * ***********************************************************************/
static inline double negative(double x) {
    return (x < 0) ? x : 0;
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
 * ***********************************************************************/
void uniform_array(double **x, double a, double b, size_t len,
                   pcg32_random_t *rng) {
    int i;

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
void normal_array(double **x, double mean, double sigma, size_t len,
                  pcg32_random_t *rng) {
    int i;

    for(i = 0; i < len; ++i) {
        (*x)[i] = normal(mean, sigma);
    }
}


/* ***********************************************************************
 * VISIBLE2HIDDEN Implements the positive phase of the CD0, where the
 * input is clamped to the visible units. States and probabilities of 
 * hiddens units are computed. 
 *
 * Args:
 *  nn  (rbm_t *)          : Pointer to RBM data structure (by reference)
 *  rng (pcg32_random_t)   : Pointer to RNG 
 *
 * Return:
 *
 * ***********************************************************************/
void visible2hidden(rbm_t *nn, pcg32_random_t *rng) {
    int i, j;
    double mysum = 0, prob;

    for (j = 0; j < nn->num_hiddens; ++j) {
        mysum = 0;
        for (i = 0; i < nn->num_visibles; ++i) {
            mysum += nn->W[i*nn->num_hiddens+j] * nn->V[i];
        }
        /* Probabilities */
        nn->ph[j] = sigmoid(mysum + nn->b[j]);

        /* States */
        prob = uniform(0, 1, rng);
        if (nn->ph[j] > prob) {
            nn->h[j] = 1;
        } else {
            nn->h[j] = 0;
        }
    }

    /* Associations CD positive phase -- Weights increment */
    for (i = 0; i < nn->num_visibles; ++i) {
        for (j = 0; j < nn->num_hiddens; ++j) {
           nn->dW_data[i*nn->num_hiddens+j] += nn->V[i] * nn->ph[j];
        }
    }

    /* Hidden units biases increment */
    for(j = 0; j < nn->num_hiddens; ++j)
        nn->db_data[j] += nn->ph[j];

    /* Visible units biases increment */
    for(i = 0; i < nn->num_visibles; ++i)
        nn->da_data[i] += nn->V[i];

}


/* ***********************************************************************
 * HIDDEN2VISIBLE Implements the negative phase of the CD0, where the
 * input data are reconstructed by clamping the hidden units and set 
 * the visibles free. 
 *
 * Args:
 *  nn  (rbm_t *)          : Pointer to RBM data structure (by reference)
 *  rng (pcg32_random_t)   : Pointer to RNG 
 *
 * Return:
 *
 * ***********************************************************************/
void hidden2visible(rbm_t *nn, pcg32_random_t *rng) {
    int i, j;
    double mysum;
    double *tmp_ph;

    tmp_ph = alloc_zeros(double, nn->num_hiddens);

    for (i = 0; i < nn->num_visibles; ++i) {
        mysum = 0;
        for (j = 0; j < nn->num_hiddens; ++j) {
            mysum += nn->W[i*nn->num_hiddens+j] * nn->h[j];
        }
        /* Probabilities */
        nn->pv[i] = sigmoid(mysum + nn->a[i]);
    }

    for (j = 0; j < nn->num_hiddens; ++j) {
        mysum = 0;
        for (i = 0; i < nn->num_visibles; ++i) {
            mysum += nn->W[i*nn->num_hiddens+j] * nn->pv[i];
        }
        /* Probabilities */
        tmp_ph[j] = sigmoid(mysum + nn->b[j]);
    }

    /* Reconstruction weights increment */
    for (i = 0; i < nn->num_visibles; ++i) {
        for (j = 0; j < nn->num_hiddens; ++j) {
           nn->dW_recon[i*nn->num_hiddens+j] += nn->pv[i] * tmp_ph[j];
        }
    }

    /* Hidden units biases increment */
    for(j = 0; j < nn->num_hiddens; ++j)
        nn->db_recon[j] += tmp_ph[j];

    /* Visible units biases increment */
    for(i = 0; i < nn->num_visibles; ++i)
        nn->da_recon[i] += nn->pv[i];

    free(tmp_ph);
}


/* ***********************************************************************
 * CONTRASTIVE_DIVERGENCE  The main implementation of CD0 algorithm. 
 * This function subtracts the positive from negative phases increments.
 * If the value of alpha is other than zero then the RBM computes the
 * non-negative matrix factorization (NMF).
 *
 * Args:
 *  nn  (* rbm_t)   : A pointer to the RBM data structure (by reference)
 *
 * Return:
 *
 * ***********************************************************************/
void contrastive_divergence(rbm_t *nn) {
    int i, j;

    /* RBM weights update based on CD0 */
    for(i = 0; i < nn->num_visibles; ++i) {
        for(j = 0; j < nn->num_hiddens; ++j) {
            nn->W[i*nn->num_hiddens+j] += nn->eta * 
                ((nn->dW_data[i*nn->num_hiddens+j] - nn->dW_recon[i*nn->num_hiddens+j]) / nn->num_batches
                 - nn->alpha * negative(nn->W[i*nn->num_hiddens+j]));
        }
    }

    /* Hidden units biases update */
    for(j = 0; j < nn->num_hiddens; ++j)
        nn->b[j] += nn->eta * (nn->db_data[j] - nn->db_recon[j]);

    /* Visible units biases update */
    for(i = 0; i < nn->num_visibles; ++i)
        nn->a[i] += nn->eta * (nn->da_data[i] - nn->da_recon[i]);

}


/* ***********************************************************************
 * RBM  The main RBM learning process. This function iterates over a 
 * predefined number of batches. For each sample in the batch calls the
 * visible2hidden and hidden2visible functions. For each batch calls
 * the contrastive_divergence function.
 *
 * Args:
 *  nn (*rbm_t)             : Pointer to the RBM data structure (by reference)
 *  images (**double )      : Data set
 *  idx (int *)             : Shuffled indexing for the input data
 *  rng (*pcg32_random_t)   : Pointer to the random number generator
 *
 * Return:
 * 
 * ***********************************************************************/
void rbm(rbm_t *nn, double **images, int *idx, pcg32_random_t *rng) {
    size_t i, j, ii = 0;
    size_t size = nn->num_hiddens * nn->num_visibles;

    for (i = 0; i < nn->num_batches; ++i) {
        for (j = 0; j < nn->batch_size; ++j) {
            memcpy(nn->V, images[idx[ii]], sizeof(double) * nn->num_visibles);
            visible2hidden(nn, rng);
            hidden2visible(nn, rng);
            ii++;
        }
        contrastive_divergence(nn);
        memset(nn->dW_data, 0, sizeof(double) * size);
        memset(nn->db_data, 0, sizeof(double) * nn->num_hiddens);
        memset(nn->da_data, 0, sizeof(double) * nn->num_visibles);
        memset(nn->dW_recon, 0, sizeof(double) * size);
        memset(nn->db_recon, 0, sizeof(double) * nn->num_hiddens);
        memset(nn->da_recon, 0, sizeof(double) * nn->num_visibles);
    }
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
int *generate_batches(rbm_t *nn, int num_samples, pcg32_random_t *rng) {
    int i, p;
    int *im_indices = NULL;

    im_indices = alloc(int, num_samples);
    nn->num_batches = (size_t) num_samples / nn->batch_size;

    for (i = 0; i < num_samples; ++i) {
        im_indices[i] = i;
    }

    /* The Fisher-Yates algorithm -- Simplicity is beauty */
    for (i = num_samples; i > 1; --i) {
        p = (int) pcg32_boundedrand_r(rng, i-1);
        XORSWAP(im_indices[i-1], im_indices[p]);    /* Magic XOR */
    }

    return im_indices;
}


/* ***********************************************************************
 * READ_BST_IMAGES  Reads bars and stripes data set images. 
 *
 * Args:
 *  fname (char *)      : Input filename
 *  x (double **)       : Array to be populated
 *  n (size_t)          : Number of samples
 *  m (size__t)         : Size of image (flattened) 
 *
 * Return:
 *
 * ***********************************************************************/
void read_bst_images(char *fname, double ***x, size_t n, size_t m) {
    size_t i, j;	
    double *tmp = NULL;
    FILE *fp = NULL;

    if(!(fp = fopen(fname, "rb"))) {
        printf("File %s not found!\n", fname);
        exit(-1);
    }

    tmp = alloc_zeros(double, n * m);
        
    fread(tmp, sizeof(double), m * n, fp);
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            (*x)[i][j] = tmp[i*m + j];
        }
    }

    fclose(fp);
    dealloc(tmp);
}


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
 * run_mnist_training Implememts the MNIST RBM learning. Batch training
 * is used. The number of epochs is passed as a command line argument. 
 *
 * Args:
 *  epochs  (int)   : Number of epochs
 *
 * Return:
 *  
 * FIXME: This has to be split into multiple functions and make 
 *        parametrization easier!!
 * ***********************************************************************/
void run_mnist_training(int epochs) {
    int e;
    size_t num_images = 32, len;
    double **images = NULL;
    int *idx = NULL;

    /* Set the Random Number Generator */
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, time(NULL), (intptr_t)&rng);

    rbm_t nn;

    /* Define the parameters of the problem*/
    nn.num_visibles = 18;
    nn.num_hiddens = 36;
    nn.batch_size = 1;
    nn. eta = .1;
    nn.alpha = .0;

    len = nn.num_visibles * nn.num_hiddens;

    /* Allocate memory for all the arrays */
    allocate_resources(&nn);
    
    /* Weights initialization */
    normal_array(&nn.W, 0, 0.1, len, &rng);
    normal_array(&nn.b, 0, 0.1, nn.num_hiddens, &rng);
    normal_array(&nn.a, 0, 0.1, nn.num_visibles, &rng);
    
    /* Load MNIST images */
    printf("Loading data...\n");
    // images = read_mnist_images(num_images);
    // idx = generate_batches(&nn, num_images, &rng);
    printf("...Done!\n");
    images = (double **)malloc(sizeof(double *) * 32);
    for(int i = 0; i < 32; ++i) {
        images[i] = (double *) malloc(sizeof(double) * 16);
    }	
    read_bst_images("../bst_images.dat", &images, 32, 16);
    idx = generate_batches(&nn, num_images, &rng);

    printf("Start training RBM!\n");
    for(e = 0; e < epochs; ++e) {
        /* printf("Epoch #: %d\n", e); */
        rbm(&nn, images, idx, &rng);
    }
    printf("Training done!\n");

    write_weights2file("weights_c.dat", nn.W, nn.num_visibles, nn.num_hiddens);

    /* Deallocate previously allocated memory */
    cleanup_resources(&nn);
    dealloc(idx);
    for(int i = 0; i < num_images; ++i) {
        free(images[i]);
    }
    free(images);
}

/* ***********************************************************************
 * The MAIN function :)
 *
 * Args:
 *  The second command line argument is the number of Epochs
 *  FIXME This will change in the next version
 * ***********************************************************************/
int main(int argc, char **argv) {
    if (argc == 2) {
       run_mnist_training(atoi(argv[1]));
    } else {
       perror("Please provide the number of epochs\n");
       exit(-1);
    	// run_simple_test(5000);
    }
    return 0;
}
