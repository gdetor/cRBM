#include "rbm.h"


/* ***********************************************************************
 * HEAVISIDE A variable heaviside function.
 *
 * Args:
 *  x (double)  : Input value
 *  y (double)  : Input value
 *
 * Return:
 *  If x > y returns 0, 1 otherwise.
 * ***********************************************************************/
static inline double heaviside(double x, double y)
{
    if (x > y) {
        return 0;
    } else {
        return 1;
    }
}


/* ***********************************************************************
 * run_mnist_training Implememts the MNIST RBM learning. On-line training
 * is used. The number of epochs is passed as a command line argument. 
 *
 * Args:
 *  epochs  (int)   : Number of epochs
 *
 * Return:
 *  
 * ***********************************************************************/
void run_bars_stripes_training(int epochs)
{
    int e;
    int *index=NULL;
    size_t i;
    int *idx = NULL;

    size_t num_images = 32, len, im_size = 16;
    double **train_images = NULL, **test_images=NULL;
    double *err = NULL;
    double *label = NULL;

    pcg32_random_t rng;
    pcg32_srandom_r(&rng, time(NULL), (intptr_t)&rng);

    rbm_t nn;

    /* Set the basic simulation parameters */
    nn.num_visibles = 16;
    nn.num_hiddens = 36;
    nn.eta = .1;
    nn.alpha = .0;
    nn.num_batches = 32;
    nn.batch_size = 1;

    /* Allocate memory for the RBM network */
    allocate_resources(&nn);

    /* Allocate auxiliary vectors */
    label = alloc_zeros(double, num_images);
    
    /* Weights initialization */
    len = nn.num_visibles * nn.num_hiddens;
    normal_array(&nn.W, 0, 0.1, len);
    normal_array(&nn.b, 0, 0.1, nn.num_hiddens);
    normal_array(&nn.a, 0, 0.1, nn.num_visibles);
    
    /* Load input samples */
    train_images = (double **)malloc(sizeof(double *) * num_images);
    test_images = (double **)malloc(sizeof(double *) * num_images);
    for(int i = 0; i < 32; ++i) {
        train_images[i] = (double *) malloc(sizeof(double) * im_size);
        test_images[i] = (double *) malloc(sizeof(double) * im_size);
    }	
    read_bst_images("/shares/data/bs/bs_train_data.dat",
                    &train_images,
                    &label,
                    num_images,
                    im_size);
    idx = generate_batches(&nn, num_images, &rng);

    for(e = 0; e < epochs; ++e) {
        /* Training in on-line fashion */
        int ii = (int) pcg32_boundedrand_r(&rng, num_images);
        memcpy(nn.V, train_images[ii], sizeof(double) * nn.num_visibles);
        rbm(&nn, &rng);
    }

    /* Write synaptic weights and error to files */
    write_weights2file("./data/weights_rbm.dat", nn.W, nn.num_visibles, nn.num_hiddens);

    /* Clean up RBM network and deallocate resources */
    cleanup_resources(&nn);

    /* Deallocate auxiliary and input arrays */
    dealloc(idx);
    dealloc(err);
    dealloc(index);
    dealloc(label);
    for(i = 0; i < num_images; ++i) {
        free(train_images[i]);
        free(test_images[i]);
    }
    free(train_images);
    free(test_images);
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
 * ***********************************************************************/
void run_mnist_training(int epochs)
{
    int e;
    size_t num_images = 5000, len, i;
    double **images = NULL;
    int *idx = NULL;

    /* Set the Random Number Generator */
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, time(NULL), (intptr_t)&rng);

    rbm_t nn;

    /* Define the parameters of the problem*/
    nn.num_visibles = 784;
    nn.num_hiddens = 36;
    nn.batch_size = 10;
    nn.num_batches = (int) num_images / nn.batch_size;
    nn. eta = .1;
    nn.alpha = .0;

    len = nn.num_visibles * nn.num_hiddens;

    /* Allocate memory for all the arrays */
    allocate_resources(&nn);
    
    /* Weights initialization */
    normal_array(&nn.W, 0, 0.01, len);
    normal_array(&nn.b, 0, 0.01, nn.num_hiddens);
    normal_array(&nn.a, 0, 0.01, nn.num_visibles);
    
    /* Load MNIST images */
    printf("Loading data...\n");
    images = read_mnist_images(num_images);
#if BATCH == 1
    idx = generate_batches(&nn, num_images, &rng);
#endif
    printf("...Done!\n");

    printf("Start training RBM!\n");
    for(e = 0; e < epochs; ++e) {
        /* printf("Epoch #: %d\n", e); */
#if BATCH == 0
        int ii = (int) pcg32_boundedrand_r(&rng, num_images);
        memcpy(nn.V, images[ii], sizeof(double) * nn.num_visibles);
        rbm(&nn, &rng);
#else
        rbm_batch(&nn, images, idx, &rng);
#endif
    }
    printf("Training done!\n");

    write_weights2file("./data/weights_mnist_16.dat", nn.W, nn.num_visibles, 
                       nn.num_hiddens);

    /* Deallocate previously allocated memory */
    cleanup_resources(&nn);
    dealloc(idx);
    for(i = 0; i < num_images; ++i) {
        free(images[i]);
    }
    free(images);
}
