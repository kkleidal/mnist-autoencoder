import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import *

class GraphConfig:
    learning_rate = 0.002
    n_hidden = 16
    n_hidden2 = 16

def build_graph(cfg):
    inputs = tf.placeholder(tf.float32, [None, 784], name="inputs")
    labels = tf.placeholder(tf.float32, [None, 10], name="labels")

    train_summaries = []
    image_summaries = []

    with tf.name_scope('hidden'):
        with tf.name_scope('weights'):
            weights = weight_variable([784, cfg.n_hidden])
            train_summaries.extend(variable_summaries(weights))
        with tf.name_scope('biases'):
            bias = bias_variable(cfg.n_hidden)
            train_summaries.extend(variable_summaries(bias))
        with tf.name_scope('activation'):
            activation = tf.matmul(inputs, weights) + bias
            train_summaries.append(tf.summary.histogram("activation", activation))
        with tf.name_scope('output'):
            hidden = tf.nn.relu(activation)
            train_summaries.append(tf.summary.histogram("output", hidden))

    with tf.name_scope('class-output'):
        with tf.name_scope('weights'):
            weights = weight_variable([cfg.n_hidden, 10])
            train_summaries.extend(variable_summaries(weights))
        with tf.name_scope('biases'):
            bias = bias_variable(10)
            train_summaries.extend(variable_summaries(bias))
        with tf.name_scope('activation'):
            activation = tf.matmul(hidden, weights) + bias
            train_summaries.append(tf.summary.histogram("activation", activation))
            logits = activation
        with tf.name_scope('output'):
            with tf.name_scope('probabilities'):
                proba = tf.nn.softmax(logits)
            with tf.name_scope('predictions'):
                prediction = tf.argmax(proba, 1)
            digit_inputs = tf.placeholder(tf.float32, [None, 10], name="digit_inputs")
            train_summaries.append(tf.summary.histogram("proba", proba))
            train_summaries.append(tf.summary.histogram("prediction", prediction))

    with tf.name_scope('hidden2'):
        with tf.name_scope('weights'):
            weights = weight_variable([10, cfg.n_hidden2])
            train_summaries.extend(variable_summaries(weights))
        with tf.name_scope('biases'):
            bias = bias_variable(cfg.n_hidden2)
            train_summaries.extend(variable_summaries(bias))
        with tf.name_scope('activation'):
            activation = tf.matmul(proba, weights) + bias
            digit_activation = tf.matmul(digit_inputs, weights) + bias
            train_summaries.append(tf.summary.histogram("activation", activation))
        with tf.name_scope('output'):
            hidden = tf.nn.relu(activation)
            digit_hidden = tf.nn.relu(digit_activation)
            train_summaries.append(tf.summary.histogram("output", hidden))

    with tf.name_scope('auto-output'):
        with tf.name_scope('weights'):
            weights = weight_variable([cfg.n_hidden2, 784])
            train_summaries.extend(variable_summaries(weights))
        with tf.name_scope('biases'):
            bias = bias_variable(784)
            train_summaries.extend(variable_summaries(bias))
        with tf.name_scope('activation'):
            activation = tf.matmul(hidden, weights) + bias
            digit_activation = tf.matmul(digit_hidden, weights) + bias
            train_summaries.append(tf.summary.histogram("activation", activation))
        with tf.name_scope('output'):
            auto = activation  # sigmoid?
            digit_auto = digit_activation  # sigmoid?

            train_summaries.append(tf.summary.histogram("output", auto))

            images = tf.reshape(digit_auto, tf.pack([tf.shape(digit_auto)[0], 28, 28, 1]))
            image_summaries.append(tf.summary.image("image", images, max_outputs=20))

    with tf.name_scope('accuracy'):
        with tf.name_scope('accuracy'):
            actual = tf.argmax(labels, 1, name="actual")
            with tf.name_scope('num_correct'):
                correct = tf.reduce_sum(tf.to_int32(tf.equal(prediction, actual)))
                train_summaries.append(tf.summary.scalar("num_correct", correct))

    with tf.name_scope('loss'):
        with tf.name_scope('class'):
            class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
            train_summaries.append(tf.summary.scalar("loss", class_loss))
        with tf.name_scope('auto'):
            auto_loss = tf.reduce_mean(tf.square(auto - inputs))
            train_summaries.append(tf.summary.scalar("loss", auto_loss))
        with tf.name_scope('total'):
            alpha = tf.Variable(0.9, trainable=False, name="alpha")
            total_loss = alpha * class_loss + (1 - alpha) * auto_loss
            train_summaries.append(tf.summary.scalar("loss", total_loss))
        

    train = tf.train.AdamOptimizer(cfg.learning_rate).minimize(total_loss, name="train")

    g = lambda: None
    g.cfg = cfg
    g.inputs = inputs
    g.labels = labels
    g.proba = proba
    g.prediction = prediction
    g.loss = class_loss
    g.train = train
    g.correct = correct
    g.alpha = alpha
    g.summaries = tf.summary.merge(train_summaries)
    g.image_summaries = tf.summary.merge(image_summaries)
    g.digit_inputs = digit_inputs

    return g

def main():
    graph = build_graph(GraphConfig()) 
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('tflog', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(graph.alpha.assign(0.0))
        for epoch in xrange(100):
            while mnist.train.epochs_completed < epoch + 1:
                inputs, labels = mnist.train.next_batch(100)
                # Train on the batch:
                sess.run(graph.train, feed_dict={
                    graph.inputs: inputs,
                    graph.labels: labels,
                })
            # Train error:
            summary, train_acc = sess.run([graph.summaries, graph.correct], feed_dict={
                graph.inputs: mnist.train.images,
                graph.labels: mnist.train.labels,
            }) 
            train_acc /= float(mnist.train.images.shape[0])
            summary_writer.add_summary(summary, epoch)
            # Test error:
            test_acc = sess.run(graph.correct, feed_dict={
                graph.inputs: mnist.test.images,
                graph.labels: mnist.test.labels,
            }) / float(mnist.test.images.shape[0])
            print "Epoch %d train accuracy: %.4f; test accuracy: %.4f" % (epoch + 1, train_acc, test_acc)
            # Images:
            digits = np.vstack([
                np.eye(10, dtype=float),
                np.array([0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, 0]),
                np.array([0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.5]),
            ])
            image_summary = sess.run(graph.image_summaries, feed_dict={
                graph.digit_inputs: digits,
            })
            summary_writer.add_summary(image_summary, epoch)
if __name__ == "__main__":
    main()
