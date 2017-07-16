import numpy as np
from load_cifar import extract
from load_cifar import load_batch
from load_cifar import load_meta
import matplotlib.pyplot as plt


def main():
    tarpath = "cifar-10-python.tar.gz"
    data_path = extract(tarpath)
    images, labels = load_batch(data_path[0])
    labels_strings = load_meta()
    fig = plt.figure()
    N = 16
    for i, img, label_i in zip(range(N), images[:N], labels[:N]):
        img = np.reshape(img, [32, 32, 3], order="F")
        img = np.rot90(img, k=3)
        ax = fig.add_subplot(4, 4, i+1)
        ax.imshow(img)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
        ax.set_title(labels_strings[label_i])
    plt.show()


if __name__ == "__main__":
    main()
