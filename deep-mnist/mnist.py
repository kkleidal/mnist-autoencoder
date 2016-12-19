import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class GraphConfig:
    learning_rate = 0.01

def build_graph(cfg):
    inputs = tf.placeholder(tf.float32, [None, 784], name="inputs")
    labels = tf.placeholder(tf.float32, [None, 10], name="labels")

    W = tf.Variable(tf.random_uniform([784, 10], -1.0, 1.0), name="weights")
    b = tf.Variable(tf.random_uniform([10], -1.0, 1.0), name="biases")

    logits = tf.matmul(inputs, W) + b

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
    # g.total = total

    return g

def main():
    graph = build_graph(GraphConfig()) 
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    i = 0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in xrange(10):
            while mnist.train.epochs_completed < epoch + 1:
                inputs, labels = mnist.train.next_batch(100)
                # Train on the batch:
                correct, loss, _ = sess.run([graph.correct, graph.loss, graph.train], feed_dict={
                    graph.inputs: inputs,
                    graph.labels: labels,
                })
                # print "Batch %6d. Accuracy: %.3f. Loss: %12.5f" % (i, correct / float(inputs.shape[0]), loss)
                i += 1
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
