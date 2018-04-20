#include "rbm.h"


/* ***********************************************************************
 * REVERSE_INT  Reverses the bits of an integer.
 *
 * Args:
 *  x  (int)       : Input integer
 *
 * Return:
 *  Reversed bits of the input integer.
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
    char *fname  = "/home/gdetorak/Datasets/mnist/train-images-idx3-ubyte";

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
    for(size_t i = 0; i < num_images; ++i) {
        tmp[i] = (double *)malloc(n_rows * n_cols * sizeof(double));
    }


    for(size_t i = 0; i < num_images; ++i) {
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
    size_t i;
    int n_labels = 0;
    int magic_number = 0;

    unsigned char temp = 0;
    int *tmp = NULL;

    FILE *fp = NULL;
    char *fname  = "/home/gdetorak/Datasets/mnist/t10k-labels-idx1-ubyte";
    /* char *fname  = "/home/gdetorak/mnist/train-labels-idx1-ubyte";   */

    if (!(fp = fopen(fname, "rb"))) {
        printf("File not found!\n");
        exit(-1);
    }

    tmp = (int *)alloc(int, num_images);

    fread((char*)&magic_number, sizeof(magic_number), 1, fp);
    magic_number = reverse_int(magic_number);

    fread((char*)&n_labels, sizeof(n_labels), 1, fp);
    n_labels = reverse_int(n_labels);

    for(i = 0; i < num_images; ++i) {
        temp = 0;
        fread((char*)&temp, 1, sizeof(temp), fp);
        tmp[i] = (int)temp;
    }

    fclose(fp);
    fp = NULL;

    return tmp;
}


/* ***********************************************************************
 * READ_BST_IMAGES Reads the Bars & Stripes data set.
 *
 * Args:
 *  fname (*char)       : Data set file
 *  x (***double)       : Pointer to pointers of pointers array (store the images)
 *  label (*double)     : Pointer to pointers array (store the labels)
 *  n (size_t)          : Number of samples in the dataset
 *  m (size_t)          : Image dimensions
 *
 * Return:
 *  
 * ***********************************************************************/
void read_bst_images(char *fname, double ***x, double **label, size_t n,
                     size_t m) {
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
        (*label)[i] = tmp[i*m+j-1];
    }

    fclose(fp);
    dealloc(tmp);
}
