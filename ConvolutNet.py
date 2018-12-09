import tensorflow as tf
import numpy as np
from preproces import read_images
from imageresize import resize
#Fast settings
def ConvolutNet(testmode):
    Test_mode = testmode

    # Parameters
    learning_rate = 0.00001
    num_steps = 37000
    display_step = 10000
    n_epoch = 20
    batch_size = 16


    #Apply settings
    if Test_mode:
        display_step = 100
        



    # Network Parameters
    dropout = 0.95 # Dropout, probability to keep units

    # Build the data input
    X, Y = read_images('./dataset', 'folder', batch_size)
    X_test, Y_test = read_images('./Test', 'folder', batch_size)


    #Build the Test data
    #X_test, Y_test = read_images('test_dataset/', 'folder', batch_size)

    #
    # Create model
    def conv_net(x, n_classes, dropout, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):

            # Convolution Layer with 128 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x, 128, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 10, 5,padding='same')

            # Convolution Layer with 32 filters and a kernel size of 5
            conv2 = tf.layers.conv2d(conv1, 128*2, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 20, 10,padding='same')

            # Convolution Layer with 32 filters and a kernel size of 5
            conv3 = tf.layers.conv2d(conv2, 128*2*2, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv3 = tf.layers.max_pooling2d(conv3, 20, 10,padding='same')

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv3)

            # Fully connected layer 
            fc1 = tf.layers.dense(fc1, 2048)

            # Fully connected layer
            fc1 = tf.layers.dense(fc1, 2048)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc1, n_classes)
            # Because 'softmax_cross_entropy_with_logits' already apply softmax,
            # I only apply softmax to testing network
            out = tf.nn.softmax(out,name='out') if not is_training else out

        return out


    # Because Dropout have different behavior at training and prediction time, I
    # need to create 2 distinct computation graphs that share the same weights.
    # Create a graph for training
    logits_train = conv_net(X, 2, dropout, reuse=False, is_training=True)
    # Create another graph for testing that reuse the same weights
    logits_test = conv_net(X_test, 2, dropout, reuse=True, is_training=False)

    # Define loss and optimizer (with train logits, for dropout to take effect)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=Y))


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)


    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y_test, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Saver object
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, '.', 'model/model.pbtxt')  
        epoch = 0

        # Run the initializer
        sess.run(init)

        # Start the data queue
        tf.train.start_queue_runners()
        while epoch < n_epoch:
            for step in range(1, num_steps+1):

                if step % display_step == 0:
                    # Run optimization and calculate batch loss and accuracy
                    _, loss, acc = sess.run([train_op, loss_op, accuracy])
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))


                else:
                    # Only run the optimization op (backprop)
                    sess.run(train_op)

            print("Epoch compleart : ", epoch)
            #Save the model
            save_path = saver.save(sess, "./model/model.ckpt")
            print("Model saved in path: %s" % save_path)
            epoch+=1

#Python Programming Tutorials, pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-#machine-learning-tutorial/.


