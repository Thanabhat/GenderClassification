import os
import theano
import theano.tensor as T
import numpy
import random
from PIL import Image
# from logistic_sgd import load_data
from mlp import test_mlp
from DBN import test_DBN

IMG_SIZE = 32, 32


def openImage(inputFilePath):
    thumbnailFilePath = os.path.splitext(inputFilePath)[0] + ".thumbnail"
    if os.path.isfile(thumbnailFilePath):
        try:
            img = Image.open(thumbnailFilePath)
            img = img.convert("L")
            return img
        except IOError:
            print("cannot open thumbnail for '%s'" % inputFilePath)
    else:
        try:
            img = Image.open(inputFilePath)
            img = img.resize(IMG_SIZE, Image.ANTIALIAS)
            img = img.convert("L")
            img.save(thumbnailFilePath, "JPEG")
            return img
        except IOError:
            print("cannot create thumbnail for '%s'" % inputFilePath)


def printPixel(pix):
    for i in range(0, IMG_SIZE[0]):
        for y in range(0, IMG_SIZE[1]):
            print(pix[i, y]),
        print


def load_data():
    M_IMAGE_COUNT = 20
    F_IMAGE_COUNT = 20

    all_data_x = []
    all_data_y = []

    # load data from image

    for image_ind in range(1, 1 + M_IMAGE_COUNT):
        img = openImage("data/m%03d.jpg" % image_ind)
        pix = img.load()

        data_x = []
        for i in range(0, IMG_SIZE[0]):
            for j in range(0, IMG_SIZE[1]):
                data_x.append(float(pix[i, j]) / 256)

        data_y = 0

        all_data_x.append(data_x)
        all_data_y.append(data_y)

    for image_ind in range(1, 1 + F_IMAGE_COUNT):
        img = openImage("data/f%03d.jpg" % image_ind)
        pix = img.load()

        data_x = []
        for i in range(0, IMG_SIZE[0]):
            for j in range(0, IMG_SIZE[1]):
                data_x.append(float(pix[i, j]) / 256)

        data_y = 1

        all_data_x.append(data_x)
        all_data_y.append(data_y)

    all_data_len = len(all_data_x)

    # shuffle data

    all_data_x_shuffle = []
    all_data_y_shuffle = []
    index_shuffle = range(all_data_len)
    random.shuffle(index_shuffle)
    for i in index_shuffle:
        all_data_x_shuffle.append(all_data_x[i])
        all_data_y_shuffle.append(all_data_y[i])
    all_data_x = all_data_x_shuffle
    all_data_y = all_data_y_shuffle

    train_set_x = []
    train_set_y = []
    for i in range(0, all_data_len / 2):
        train_set_x.append(all_data_x[i])
        train_set_y.append(all_data_y[i])
    train_set = train_set_x, train_set_y

    valid_set_x = []
    valid_set_y = []
    for i in range(all_data_len / 2, all_data_len / 2 + all_data_len / 4):
        valid_set_x.append(all_data_x[i])
        valid_set_y.append(all_data_y[i])
    valid_set = valid_set_x, valid_set_y

    test_set_x = []
    test_set_y = []
    for i in range(all_data_len / 2 + all_data_len / 4, all_data_len):
        test_set_x.append(all_data_x[i])
        test_set_y.append(all_data_y[i])
    test_set = test_set_x, test_set_y

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


# datasets = load_data('mnist.pkl.gz')
datasets = load_data()
# test_mlp(datasets=datasets, n_hidden=200, batch_size=1)
test_DBN(datasets=datasets, pretraining_epochs=100, training_epochs=1000, batch_size=1)
