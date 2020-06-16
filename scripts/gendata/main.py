from collections import Counter

import keras.datasets as kd
import numpy as np
import pandas as pd
import keras as keras
from keras.preprocessing.sequence import pad_sequences
import scipy.io as sio
import glob
from skimage.io import imread
from sklearn.utils import shuffle
import os
import arff_utils as arff_utils


def prep_mnist():
    (x_train, y_train), (x_test, y_test) = kd.mnist.load_data()
    x = np.array([img.flatten() for img in np.concatenate((x_train, x_test))]) / 255
    y = np.concatenate((y_train, y_test))

    xy = np.array([np.append(row, label) for (row, label) in list(zip(x, y))])

    print(x.shape, y.shape, xy.shape)
    # print(xy[0])

    arff_utils.image_stream_to_arff(xy, (28, 28), 'MNIST', 'MNIST.arff')


def prep_mnist_f():
    (x_train, y_train), (x_test, y_test) = kd.fashion_mnist.load_data()
    x = np.array([img.flatten() for img in np.concatenate((x_train, x_test))]) / 255
    y = np.concatenate((y_train, y_test))

    labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boots']
    xy = np.array([np.append(row, labels[label]) for (row, label) in list(zip(x, y))])

    print(x.shape, y.shape, xy.shape)
    # print(xy[0])

    arff_utils.image_stream_to_arff(xy, (28, 28), 'MNIST_F', 'MNIST_F.arff')


def prep_cifar():
    (x_train, y_train), (x_test, y_test) = kd.cifar10.load_data()
    x_rgb = np.concatenate((x_train, x_test))
    x = np.array([[[np.average(c) / 255 for c in row] for row in img] for img in x_rgb])
    x = np.array([img.flatten() for img in x])
    y = np.concatenate((y_train, y_test)).flatten()

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    xy = np.array([np.append(row, labels[label]) for (row, label) in list(zip(x, y))])

    print(x.shape, y.shape, xy.shape)
    # print(x_train[1][0][1], xy[1][1])

    arff_utils.image_stream_to_arff(xy, (32, 32), 'CIFAR10', 'CIFAR10.arff')


def prep_cmater(root):
    path = f'{root}/raw-batch/vis/cmater/datasets/bangla-numerals/training-images.npz'
    data = np.load(path)

    x = np.array([[[np.average(c) / 255 for c in row] for row in img] for img in data.f.images])
    x = np.array([img.flatten() for img in x])
    y = data.f.labels
    xy = np.array([np.append(row, label) for _ in range(2) for (row, label) in list(zip(x, y))])

    np.random.shuffle(xy)
    print(x.shape, y.shape, xy.shape)

    arff_utils.image_stream_to_arff(xy, (32, 32), 'CMATER-BANGLA', 'CMATER-BANGLA.arff')


def prep_malaria(root):
    pos_path = f'{root}/raw-batch/vis/malaria/32/Parasitized'
    neg_path = f'{root}/raw-batch/vis/malaria/32/Uninfected'
    train_files = glob.glob(pos_path + "/*") + glob.glob(neg_path + "/*")

    xy = []
    for i, name in enumerate(train_files):
        try:
            img = imread(name)
            y = 'pos' if 'Parasitized' in name else 'neg'

            x = np.array([[np.average(c) / 255 for c in row] for row in img]).flatten()
            xy.append(np.append(x, y))
            print(i, name, x.shape, y)
        except ValueError as e:
            print(f'Could not read {name}: {e}')

    xy = np.array(xy)
    np.random.shuffle(xy)
    print(xy.shape)

    arff_utils.image_stream_to_arff(xy, (32, 32), 'MALARIA', 'MALARIA.arff')


def prep_dogs_vs_cats(root):
    path = f'{root}/raw-batch/vis/dogs-vs-cats/train_32'
    train_files = glob.glob(path + "/*")

    xy = []
    for i, name in enumerate(train_files):
        try:
            img = imread(name)
            y = 'cat' if 'cat' in name.split('/')[-1] else 'dog'

            x = np.array([[np.average(c) / 255 for c in row] for row in img]).flatten()
            xy.append(np.append(x, y))
            print(i, name, x.shape, y)
        except ValueError as e:
            print(f'Could not read {name}: {e}')

    xy = np.array(xy)
    np.random.shuffle(xy)
    print(xy.shape)

    arff_utils.image_stream_to_arff(xy, (32, 32), 'DOGS-VS-CATS', 'DOGS-VS-CATS.arff')


def prep_intel(root):
    path = f'{root}/raw-batch/vis/intel_imgs/seg_train_32'

    xy = []
    for label in os.listdir(path):
        for image_file_path in glob.glob(path + '/' + label + '/*'):
            try:
                img = imread(image_file_path)
                x = np.array([[np.average(c) / 255 for c in row] for row in img]).flatten()
                y = label
                for _ in range(2):
                    xy.append(np.append(x, y))
                print(image_file_path, x.shape, y)
            except ValueError as e:
                print(f'Could not read {image_file_path}: {e}')

    xy = np.array(xy)
    np.random.shuffle(xy)
    print(xy.shape)

    arff_utils.image_stream_to_arff(xy, (32, 32), 'INTEL-IMGS', 'INTEL-IMGS.arff')


def prep_imagenette(root):
    path = f'{root}/raw-batch/vis/imagenette/train_64'

    xy = []
    for label in os.listdir(path):
        for image_file_path in glob.glob(path + '/' + label + '/*'):
            try:
                img = imread(image_file_path)
                x = np.array([[np.average(c) / 255 for c in row] for row in img]).flatten()
                y = label
                xy.append(np.append(x, y))
                print(image_file_path, x.shape, y)
            except ValueError as e:
                print(f'Could not read {image_file_path}: {e}')

    xy = np.array(xy)
    np.random.shuffle(xy)
    print(xy.shape)

    arff_utils.image_stream_to_arff(xy, (64, 64), 'IMAGENETTE', 'IMAGENETTE.arff')


def prep_bbc_news(root):
    path = f'{root}/raw-batch/text/bbc_news/bbc-text.csv'
    data = pd.read_csv(path)
    x = data['text']
    y = np.array(data['category'])

    tokenizer = keras.preprocessing.text.Tokenizer(char_level=False)
    tokenizer.fit_on_texts(x)
    text_seqs = tokenizer.texts_to_sequences(x)

    # lens = np.array([len(seq) for seq in text_seqs])
    # plt.hist(lens, bins=25)
    # plt.show()
    max_len = 1000
    text_seqs_trim = [seq if len(seq) <= max_len else seq[:max_len] for seq in text_seqs]
    x = np.array(pad_sequences(text_seqs_trim, padding='post'))

    xy = np.array([np.append(row, label) for _ in range(15) for (row, label) in list(zip(x, y))])
    np.random.shuffle(xy)

    print(x.shape, y.shape, xy.shape)
    #print(tokenizer.sequences_to_texts(x)[:5], y[:5])

    arff_utils.text_stream_to_arff(xy, max_len, 'BBC', 'BBC.arff')


def prep_sogou_news(root):
    path = f'{root}/raw-batch/text/sogou_news/train.csv'
    data = pd.read_csv(path, header=None, names=['class', 'title', 'data'])
    x = data['data'][10000:40000]
    y = np.array(data['class'])[10000:40000]
    print(Counter(y))

    tokenizer = keras.preprocessing.text.Tokenizer(char_level=False)
    tokenizer.fit_on_texts(x)
    text_seqs = tokenizer.texts_to_sequences(x)

    # lens = np.array([len(seq) for seq in text_seqs])
    # plt.hist(lens, bins=25)
    # plt.show()
    max_len = 1500
    text_seqs_trim = [seq if len(seq) <= max_len else seq[:max_len] for seq in text_seqs]
    x = np.array(pad_sequences(text_seqs_trim, padding='post'))

    xy = np.array([np.append(row, label) for (row, label) in list(zip(x, y))])
    np.random.shuffle(xy)

    print(x.shape, y.shape, xy.shape)
    # print(tokenizer.sequences_to_texts(x)[:5], y[:5])

    arff_utils.text_stream_to_arff(xy, max_len, 'SOGOU-30K', 'SOGOU-30K.arff')


def prep_ag_news(root):
    path = f'{root}/raw-batch/text/ag_news/train.csv'
    data = shuffle(pd.read_csv(path, header=None, names=['class', 'title', 'data']), random_state=20)
    class_labels = ['World', 'Sports', 'Business', 'SciTech']
    x = data['data'][:30000]
    y = np.array([class_labels[label - 1] for label in np.array(data['class'][:30000])])
    print(Counter(y))

    tokenizer = keras.preprocessing.text.Tokenizer(char_level=False)
    tokenizer.fit_on_texts(x)
    text_seqs = tokenizer.texts_to_sequences(x)

    # lens = np.array([len(seq) for seq in text_seqs])
    # plt.hist(lens, bins=25)
    # plt.show()
    # max_len = 600
    # text_seqs_trim = [seq if len(seq) <= max_len else seq[:max_len] for seq in text_seqs]

    x = np.array(pad_sequences(text_seqs, padding='post'))
    xy = np.array([np.append(row, label) for (row, label) in list(zip(x, y))])

    print(x.shape, y.shape, xy.shape)
    # print(tokenizer.sequences_to_texts(x)[:5], y[:5])

    arff_utils.text_stream_to_arff(xy, xy.shape[1] - 1, 'AGNEWS-30K', 'AGNEWS-30K.arff')


def prep_semg(root):
    paths = [f'{root}/raw-batch/generic/semg/female_1.mat', f'{root}/raw-batch/generic/semg/female_2.mat',
             f'{root}/raw-batch/generic/semg/female_3.mat', f'{root}/raw-batch/generic/semg/male_1.mat',
             f'{root}/raw-batch/generic/semg/male_2.mat']

    x, y = [], []
    for path in paths:
        xp, yp = load_mat(path)
        for xpp in xp:
            x.append(xpp)
        for ypp in yp:
            y.append(ypp)

    x = np.array(x)
    y = np.array(y)

    xy = np.array([np.append(row, label) for _ in range(15) for (row, label) in list(zip(x, y))])
    np.random.shuffle(xy)

    print(x.shape, y.shape, xy.shape)
    #print(xy[0])

    arff_utils.generic_stream_to_arff(xy, 3000, 'SEMG', 'SEMG.arff')


def load_mat(mat_path):
    move2label = {'spher_ch1': 'spher', 'spher_ch2': 'spher', 'tip_ch1': 'tip', 'tip_ch2': 'tip', 'palm_ch1': 'palm', 'palm_ch2': 'palm',
                  'lat_ch1': 'lat', 'lat_ch2': 'lat', 'cyl_ch1': 'cyl', 'cyl_ch2': 'cyl', 'hook_ch1': 'hook', 'hook_ch2': 'hook'}
    X = None
    y = None
    data = sio.loadmat(mat_path)
    for k in sorted(move2label.keys()):
        X_cur = data[k]
        y_cur = np.full(X_cur.shape[0], move2label[k])
        if X is None:
            X, y = X_cur, y_cur
        else:
            X = np.vstack((X, X_cur))
            y = np.concatenate((y, y_cur))
    return X, y


def main():
    print('Running...')
    root_path = '.'

    prep_mnist()
    prep_mnist_f()
    prep_cifar()
    prep_cmater(root_path)
    prep_malaria(root_path)
    prep_dogs_vs_cats(root_path)
    prep_intel(root_path)
    prep_imagenette(root_path)

    prep_bbc_news(root_path)
    prep_sogou_news(root_path)
    prep_ag_news(root_path)

    prep_semg(root_path)


if __name__ == "__main__":
    main()
