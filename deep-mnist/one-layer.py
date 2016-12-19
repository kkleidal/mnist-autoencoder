import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class GraphConfig:
    learning_rate = 0.002
    n_hidden = 16

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)

def weight_variable(dims):
    return tf.Variable(tf.random_uniform(dims, -0.1, 0.1), name="weights")

def bias_variable(dim):
    return tf.Variable(tf.random_uniform([dim], -0.1, 0.1), name="bias")

def build_graph(cfg):
    inputs = tf.placeholder(tf.float32, [None, 784], name="inputs")
    labels = tf.placeholder(tf.float32, [None, 10], name="labels")

    with tf.name_scope('hidden'):
        with tf.name_scope('weights'):
            weights = weight_variable([784, cfg.n_hidden])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            bias = bias_variable(cfg.n_hidden)
            variable_summaries(bias)
        with tf.name_scope('activation'):
            activation = tf.matmul(inputs, weights) + bias
            tf.summary.histogram("activation", activation)
        with tf.name_scope('output'):
            hidden = tf.nn.relu(activation)
            tf.summary.histogram("output", hidden)

    with tf.name_scope('output'):
        with tf.name_scope('weights'):
            weights = weight_variable([cfg.n_hidden, 10])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            bias = bias_variable(10)
            variable_summaries(bias)
        with tf.name_scope('activation'):
            activation = tf.matmul(hidden, weights) + bias
            tf.summary.histogram("activation", activation)
            logits = activation
        with tf.name_scope('output'):
            proba = tf.nn.softmax(logits, name="probabilities")
            prediction = tf.argmax(proba, 1, name="predictions")
            tf.summary.histogram("proba", proba)
            tf.summary.histogram("prediction", prediction)

    with tf.name_scope('accuracy'):
        actual = tf.argmax(labels, 1, name="actual")
        with tf.name_scope('num_correct'):
            correct = tf.reduce_sum(tf.to_int32(tf.equal(prediction, actual)))
            tf.summary.scalar("num_correct", correct)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels), name="loss")
        tf.summary.scalar("loss", loss)

    train = tf.train.AdamOptimizer(cfg.learning_rate).minimize(loss, name="train")

    g = lambda: None
    g.cfg = cfg
    g.inputs = inputs
    g.labels = labels
    g.proba = proba
    g.prediction = prediction
    g.loss = loss
    g.train = train
    g.correct = correct
    g.summaries = tf.summary.merge_all()

    return g

def main():
    graph = build_graph(GraphConfig()) 
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('tflog', sess.graph)
        sess.run(tf.global_variables_initializer())
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

if __name__ == "__main__":
    main()
