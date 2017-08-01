""" Original VGG-19 architecture

For debugging purposes...
    * img_s set to 32
    * layer sizes decreased
"""
import tensorflow as tf
# pylint: disable=import-error,invalid-name
from load_cifar import Dataset

# Parameters taken from paper
batch_size = 2 # 250 # Changed to 250 so it's divisble by train set number
n_batches = int(50000/batch_size)
momentum = 0.9
weight_decay = 0.0005
dropout_prob = 0.5
lr = 0.01 # Initial Learning Rate

# Dataset Parameters
img_s = 64
n_labels = 10

# pylint: disable=too-few-public-methods
class LearningRateModifier:
    """ Modifies the Learning Rate"""
    def __init__(self, required_acc_change=0.01, n_decreases=3):
        self.prev_acc = 0.0000000001
        self.required_acc_change = required_acc_change
        self.n_decreases = n_decreases
        self.not_converged = True

    def stopped_improving(self, curr_acc):
        """ Check if accuracy has improved by at least a certain amt"""
        lr_stopped_improving = False
        print("curr_acc: {}".format(curr_acc))
        print("self.prev_acc: {}".format(self.prev_acc))
        delta_acc = (curr_acc - self.prev_acc)/(self.prev_acc)
        self.prev_acc = curr_acc
        if self.required_acc_change > delta_acc:
            lr_stopped_improving = True
            self.n_decreases -= 1
            if self.n_decreases < 0:
                self.not_converged = False
        return lr_stopped_improving

def conv_layer(tensor, kernel_shape, bias_shape):
    """ Convolutional-ReLU layer """
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    tf.add_to_collection(weights, "all_weights")
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    output = tf.nn.conv2d(tensor, weights, strides=[1, 1, 1, 1],
                          padding='SAME')
    return tf.nn.relu(output + biases)

# pylint: disable=too-many-arguments, redefined-outer-name
def fc_layer(vector, batch_size, n_in, n_out, activation_fn=tf.nn.relu):
    """ Fully Connected Layer"""
    weights = tf.get_variable("weights", [n_in, n_out],
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", [batch_size, n_out],
                             initializer=tf.constant_initializer(0.0))
    output = tf.add(tf.matmul(vector, weights), biases)
    if activation_fn is not None:
        output = activation_fn(output)
    return output

def max_pool_layer(tensor, image_s):
    """ 2x2 Max pooling with stride 2"""
    image_s = int(image_s/2)
    tensor = tf.nn.max_pool(tensor, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
    return tensor, image_s

g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, [None, img_s, img_s, 3], name="image")
    y = tf.placeholder(tf.float32, [None, n_labels], name="label")

    with tf.variable_scope("conv1"):
        conv1 = conv_layer(X, [3, 3, 3, 4], [img_s, img_s, 4])
    with tf.variable_scope("conv2"):
        conv2 = conv_layer(conv1, [3, 3, 4, 4], [img_s, img_s, 4])
        pool1, img_s = max_pool_layer(conv2, img_s)

    cin = 4 # Channel In
    cout = 28 # Channel Out
    with tf.variable_scope("conv3"):
        conv3 = conv_layer(pool1, [3, 3, cin, cout], [img_s, img_s, cout])
    with tf.variable_scope("conv4"):
        conv4 = conv_layer(conv3, [3, 3, cout, cout], [img_s, img_s, cout])
        pool2, img_s = max_pool_layer(conv4, img_s)

    cin = cout # 64
    cout = 56
    with tf.variable_scope("conv5"):
        conv5 = conv_layer(pool2, [3, 3, cin, cout], [img_s, img_s, cout])
    with tf.variable_scope("conv6"):
        conv6 = conv_layer(conv5, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope("conv7"):
        conv7 = conv_layer(conv6, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope("conv8"):
        conv8 = conv_layer(conv7, [3, 3, cout, cout], [img_s, img_s, cout])
        pool3, img_s = max_pool_layer(conv8, img_s)

    cin = cout # 256
    cout = 12
    with tf.variable_scope("conv9"):
        conv9 = conv_layer(pool3, [3, 3, cin, cout], [img_s, img_s, cout])
    with tf.variable_scope("conv10"):
        conv10 = conv_layer(conv9, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope("conv11"):
        conv11 = conv_layer(conv10, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope("conv12"):
        conv12 = conv_layer(conv11, [3, 3, cout, cout], [img_s, img_s, cout])
        pool4, img_s = max_pool_layer(conv12, img_s)

    cin = cout # 512
    cout = 12
    with tf.variable_scope("conv13"):
        conv13 = conv_layer(pool4, [3, 3, cin, cout], [img_s, img_s, cout])
    with tf.variable_scope("conv14"):
        conv14 = conv_layer(conv13, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope("conv15"):
        conv15 = conv_layer(conv14, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope("conv16"):
        conv16 = conv_layer(conv15, [3, 3, cout, cout], [img_s, img_s, cout])
        pool5, img_s = max_pool_layer(conv16, img_s)

    with tf.variable_scope("fc1"):
        n_in = img_s * img_s * cout # 7*7*512
        n_out = 96 # 4096
        pool5_1d = tf.reshape(pool5, [batch_size, n_in])
        fc1 = fc_layer(pool5_1d, batch_size, n_in, n_out)
        fc1_drop = tf.nn.dropout(fc1, dropout_prob)
    with tf.variable_scope("fc2"):
        n_in = n_out # 4096
        n_out = 96
        fc2 = fc_layer(fc1_drop, batch_size, n_in, n_out)
        fc2_drop = tf.nn.dropout(fc2, dropout_prob)
    with tf.variable_scope("fc3"):
        n_in = n_out
        y_ = fc_layer(fc2_drop, batch_size, n_in, n_labels, activation_fn=None)

    with tf.variable_scope('weights_norm'):
        weights_norm = tf.reduce_sum(
            input_tensor=weight_decay * tf.stack(
                [tf.nn.l2_loss(i) for i in tf.get_collection('all_weights')]
            ),
            name='weights_norm'
        )
    tf.add_to_collection('losses', weights_norm)

    with tf.variable_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                    logits=y_))
    tf.add_to_collection('losses', cross_entropy)

    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    train = tf.train.MomentumOptimizer(lr, momentum).minimize(total_loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    cifar = Dataset(batch_size=batch_size, data_path="../cifar-10-batches-py")
    X_tr, y_tr = cifar.load_n_images(n=2)
    X_val, y_val = cifar.load_n_images(n=2)

    LRM = LearningRateModifier(lr)
    while LRM.not_converged:
        for _ in range(n_batches):
            _, a_tr = sess.run([train, accuracy],
                               feed_dict={X:X_tr, y: y_tr})
        validation_acc = sess.run([accuracy], feed_dict={X:X_val, y: y_val})
        if LRM.stopped_improving(validation_acc):
            lr = lr/10.

    print("FINAL TRAIN ACCURACY: {}".format(a_tr))
    print("FINAL VALIDATION ACCURACY: {}".format(validation_acc))
