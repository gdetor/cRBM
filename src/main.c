#define _GNU_SOURCE
#include "rbm.h"
#include <sched.h>

int main(int argc, char **argv) {
    mlockall(MCL_CURRENT|MCL_FUTURE);

    cpu_set_t  mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    int result = sched_setaffinity(0, sizeof(mask), &mask);
    if (!result)
        printf("CPU  %d  has been assigned succesfully!\n", 0);

    if (argc == 2) {
       /* run_mnist_training(atoi(argv[1])); */
       run_bars_stripes_training(atoi(argv[1]));
    } else {
       perror("Please provide the number of epochs\n");
       exit(-1);
    }
    return 0;
}
