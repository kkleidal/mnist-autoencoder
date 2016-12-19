import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class GraphConfig:
    learning_rate = 0.002
    n_hidden = 16

def build_graph(cfg):
    inputs = tf.placeholder(tf.float32, [None, 784], name="inputs")
    labels = tf.placeholder(tf.float32, [None, 10], name="labels")

    W_in_hidden1 = tf.Variable(tf.random_uniform([784, cfg.n_hidden], -0.1, 0.1), name="weights_in_hidden1")
    b_hidden1 = tf.Variable(tf.random_uniform([cfg.n_hidden], -0.1, 0.1), name="biases_hidden1")

    hidden1 = tf.nn.relu(tf.matmul(inputs, W_in_hidden1) + b_hidden1)

    W_hidden1_out = tf.Variable(tf.random_uniform([cfg.n_hidden, 10], -0.1, 0.1), name="weights_hidden1_out")
    b_out = tf.Variable(tf.random_uniform([10], -0.1, 0.1), name="biases_out")

    logits = tf.matmul(hidden1, W_hidden1_out) + b_out

    proba = tf.nn.softmax(logits)
    prediction = tf.argmax(proba, 1)
    actual = tf.argmax(labels, 1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    correct = tf.reduce_sum(tf.to_int32(tf.equal(prediction, actual)))

    train = tf.train.AdamOptimizer(cfg.learning_rate).minimize(loss)

    g = lambda: None
    g.cfg = cfg
    g.inputs = inputs
    g.labels = labels
    g.proba = proba
    g.prediction = prediction
    g.loss = loss
    g.train = train
    g.correct = correct

    return g

def main():
    graph = build_graph(GraphConfig()) 
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in xrange(100):
            while mnist.train.epochs_completed < epoch + 1:
                inputs, labels = mnist.train.next_batch(100)
                # Train on the batch:
                sess.run(graph.train, feed_dict={
                    graph.inputs: inputs,
                    graph.labels: labels,
                })
            # Train error:
            train_acc = sess.run(graph.correct, feed_dict={
                graph.inputs: mnist.train.images,
                graph.labels: mnist.train.labels,
            }) / float(mnist.train.images.shape[0])
            # Test error:
            test_acc = sess.run(graph.correct, feed_dict={
                graph.inputs: mnist.test.images,
                graph.labels: mnist.test.labels,
            }) / float(mnist.test.images.shape[0])
            print "Epoch %d train accuracy: %.4f; test accuracy: %.4f" % (epoch + 1, train_acc, test_acc)

if __name__ == "__main__":
    main()
