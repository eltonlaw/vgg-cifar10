import tensorflow as tf
import numpy as np
from load_cifar import Dataset

tarpath = "./cifar-10-python.tar.gz"
data_path = extract(tarpath)


g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, [None, n_data], name="image")
    y = tf.placeholder(tf.float32, [None, n_labels], name="label")


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
