import tensorflow as tf
import numpy as np
from preproces import read_images
import cv2
import matplotlib.pyplot as plt
from imageresize import resize




def UseConvNet(image):
    
    #define image path
    image = str(image)
    #resize the image to 277 by 277 so it fits our model
    resize(image)
    #Import Image
    image = cv2.imread(image)

    # cast image, which is now an array of intergers to an array of floats, intergers where giving a error
    X = image.astype('float')
    print("dtype is : ",X.dtype)
    #preprosses image to have all values on a scale from 0-1
    X = X * (1.0/255)
    #define placeholders for model
    x = tf.placeholder('float')
    y = tf.placeholder('int64')


    #
    # Create model that the weights from teh save file will be aplleided to 
    def conv_net(x, n_classes, is_training):
       
        x = tf.reshape(x, shape=[-1,227,227,1])


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
        #fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # I only apply softmax to testing network
        #out = tf.nn.softmax(out)

        return out
    saver = tf.train.import_meta_graph('./model/model.ckpt.meta')







    def use_neural_network(input_data):
        prediction = conv_net(x, 2, is_training=False)
        
            
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,"./model/model.ckpt")
      
           
            # pos: [1,0] , argmax: 0
            # neg: [0,1] , argmax: 1 
            
            result = sess.run(tf.argmax(prediction.eval(feed_dict={x:input_data}),1))
            print(result)
            if result[0] < result[1]:
                print('Positive:')
            elif result[0] > result[1]:
                print('Negative:')
            elif result[0] == result[1]:
            	print("Can not tell")
            else:
            	print("Ops somthing went wrong ¯/_(ツ)_/¯")
            

    use_neural_network(X)





#source : Aymericdamien. “Aymericdamien/TensorFlow-Examples.” GitHub, github.com/aymericdamien/#TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py. 
