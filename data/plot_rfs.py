import sys
import numpy as np
import matplotlib.pylab as plt
from struct import unpack


def read_bin_file(fname):
    with open(fname, 'rb') as f:
        c = f.read()
    size = len(c)
    tmp = np.array(unpack('d'*int(size // 8), c), 'd')
    return tmp


if __name__ == '__main__':
    im_size_x, im_size_y = int(sys.argv[2]), int(sys.argv[2])
    nn_size_x, nn_size_y = int(sys.argv[3]), int(sys.argv[3])

    W = read_bin_file(sys.argv[1]).reshape(im_size_x*im_size_y,
                                           nn_size_x*nn_size_y)

    R = np.zeros((im_size_x*nn_size_x, im_size_y*nn_size_x))
    for i in range(nn_size_x):
        for j in range(nn_size_y):
            ww = W[:, i*nn_size_x+j].reshape(im_size_x, im_size_y)
            R[i*im_size_x:(i+1)*im_size_x, j*im_size_y:(j+1)*im_size_y] = ww
    print(R.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(E[:-1], 'b', lw=2)
    # ax.plot(err[:-1], lw=2, c='orange')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if sys.argv[4] != 'b':
        im = ax.imshow(R, interpolation='nearest', cmap=plt.cm.gray)
    else:
        im = ax.imshow(R, interpolation='nearest', cmap=plt.cm.gray_r)
    ax.set_xticks(np.arange(-0.5, im_size_x*nn_size_x-.5, im_size_x))
    ax.set_yticks(np.arange(-0.5, im_size_y*nn_size_y-.5, im_size_y))
    ax.grid(color='r', linestyle='-', linewidth=1.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.colorbar(im)
    plt.show()
