#include "rbm.h"


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
static inline double sigmoid(double x)
{
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
static inline double negative(double x)
{
    return (x < 0) ? x : 0;
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
void visible2hidden(rbm_t *restrict nn, pcg32_random_t *restrict rng)
{
    size_t i, j;
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

#if BATCH == 0
    /* Associations CD positive phase */
    for (i = 0; i < nn->num_visibles; ++i) {
        for (j = 0; j < nn->num_hiddens; ++j) {
           nn->dW_data[i*nn->num_hiddens+j] = nn->V[i] * nn->ph[j];
        }
    }

    /* Hidden units biases increment */
    for(j = 0; j < nn->num_hiddens; ++j)
        nn->db_data[j] = nn->ph[j];

    /* Visible units biases increment */
    for(i = 0; i < nn->num_visibles; ++i)
        nn->da_data[i] = nn->V[i];
#else
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
#endif

}

/* just a test function 
 * Use this function only when learning is not required. 
 * */
void visible2hidden_test(rbm_t *restrict nn, pcg32_random_t *restrict rng)
{
    size_t i, j;
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
void hidden2visible(rbm_t *restrict nn, pcg32_random_t *restrict rng)
{
    size_t i, j;
    double mysum, prob;
    double *tmp_ph;

    tmp_ph = alloc_zeros(double, nn->num_hiddens);

    for (i = 0; i < nn->num_visibles; ++i) {
        mysum = 0;
        for (j = 0; j < nn->num_hiddens; ++j) {
            mysum += nn->W[i*nn->num_hiddens+j] * nn->h[j];
        }
        /* Probabilities */
        nn->pv[i] = sigmoid(mysum + nn->a[i]);

        /* States */
        prob = uniform(0, 1, rng);
        if (nn->pv[i] > prob) {
            nn->v[i] = 1;
        } else {
            nn->v[i] = 0;
        }
    }

    for (j = 0; j < nn->num_hiddens; ++j) {
        mysum = 0;
        for (i = 0; i < nn->num_visibles; ++i) {
            mysum += nn->W[i*nn->num_hiddens+j] * nn->pv[i];
        }
        /* Probabilities */
        tmp_ph[j] = sigmoid(mysum + nn->b[j]);
    }

#if BATCH == 0
    for (i = 0; i < nn->num_visibles; ++i) {
        for (j = 0; j < nn->num_hiddens; ++j) {
           nn->dW_recon[i*nn->num_hiddens+j] = nn->pv[i] * tmp_ph[j];
        }
    }

    for(j = 0; j < nn->num_hiddens; ++j)
        nn->db_recon[j] = tmp_ph[j];

    for(i = 0; i < nn->num_visibles; ++i)
        nn->da_recon[i] = nn->pv[i];

#else
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
#endif

    free(tmp_ph);
}


/* 
 * test function. Use this function when learning is not required! 
 * Validate results. 
 *
 * */
void hidden2visible_test(rbm_t *restrict nn, pcg32_random_t *restrict rng)
{
    size_t i, j;
    double mysum, prob;
    double *tmp_ph;

    tmp_ph = alloc_zeros(double, nn->num_hiddens);

    for (i = 0; i < nn->num_visibles; ++i) {
        mysum = 0;
        for (j = 0; j < nn->num_hiddens; ++j) {
            mysum += nn->W[i*nn->num_hiddens+j] * nn->h[j];
        }
        /* Probabilities */
        nn->pv[i] = sigmoid(mysum + nn->a[i]);

        /* States */
        prob = uniform(0, 1, rng);
        if (nn->pv[i] > prob) {
            nn->v[i] = 1;
        } else {
            nn->v[i] = 0;
        }
    }

    for (j = 0; j < nn->num_hiddens; ++j) {
        mysum = 0;
        for (i = 0; i < nn->num_visibles; ++i) {
            mysum += nn->W[i*nn->num_hiddens+j] * nn->pv[i];
        }
        /* Probabilities */
        tmp_ph[j] = sigmoid(mysum + nn->b[j]);
    }

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
void contrastive_divergence(rbm_t *restrict nn)
{
    size_t i, j;

    for(i = 0; i < nn->num_visibles; ++i) {
        for(j = 0; j < nn->num_hiddens; ++j) {
#if BATCH == 0
            nn->W[i*nn->num_hiddens+j] += nn->eta * 
                (nn->dW_data[i*nn->num_hiddens+j] - nn->dW_recon[i*nn->num_hiddens+j]
                 - nn->alpha * negative(nn->W[i*nn->num_hiddens+j]));
#else
            nn->W[i*nn->num_hiddens+j] += nn->eta * 
                ((nn->dW_data[i*nn->num_hiddens+j] - nn->dW_recon[i*nn->num_hiddens+j]) / nn->batch_size
                 - nn->alpha * negative(nn->W[i*nn->num_hiddens+j]));
#endif
        }
    }

#if BATCH == 0
    for(j = 0; j < nn->num_hiddens; ++j)
        nn->b[j] += nn->eta * (nn->db_data[j] - nn->db_recon[j]);

    for(i = 0; i < nn->num_visibles; ++i)
        nn->a[i] += nn->eta * (nn->da_data[i] - nn->da_recon[i]);
#else
    for(j = 0; j < nn->num_hiddens; ++j)
        nn->b[j] += nn->eta * (nn->db_data[j] - nn->db_recon[j]) / nn->batch_size;

    for(i = 0; i < nn->num_visibles; ++i)
        nn->a[i] += nn->eta * (nn->da_data[i] - nn->da_recon[i]) / nn->batch_size;
#endif
}


/* ***********************************************************************
 * RBM_BATCH  The main RBM learning process. This function iterates over a 
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
void rbm_batch(rbm_t *restrict nn, double **restrict images,
               int *restrict idx, pcg32_random_t *restrict rng)
{
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
 * RBM  The main RBM learning process. This function iterates over a 
 * predefined number of epochs. For each sample it calls the
 * visible2hidden, hidden2visible and the contrastive_divergence functions.
 *
 * Args:
 *  nn (*rbm_t)             : Pointer to the RBM data structure (by reference)
 *  rng (*pcg32_random_t)   : Pointer to the random number generator
 *
 * Return:
 * 
 * ***********************************************************************/
void rbm(rbm_t *restrict nn, pcg32_random_t *rng)
{
    visible2hidden(nn, rng);
    hidden2visible(nn, rng);
    contrastive_divergence(nn);
}


/* ***********************************************************************
 * TEST_RBM  Runs a test by calling visible2hidden and hidden2visible 
 * functions. It assumes learning has taken place. 
 *
 * Args:
 *  nn (*rbm_t)             : Pointer to the RBM data structure (by reference)
 *  rng (*pcg32_random_t)   : Pointer to the random number generator
 *
 * Return:
 * 
 * ***********************************************************************/
void test_rbm(rbm_t *restrict nn, pcg32_random_t *rng)
{ 
    visible2hidden_test(nn, rng);
    hidden2visible_test(nn, rng);
}
