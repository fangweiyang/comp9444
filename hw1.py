"""
    All tensorflow objects, if not otherwise specified, should be explicity
    created with tf.float32 datatypes. Not specifying this datatype for variables and
    placeholders will cause your code to fail some tests.
    
    For the specified functionality in this assignment, there are generally high
    level Tensorflow library calls that can be used. As we are assessing tensorflow,
    functionality that is technically correct but implemented manually, using a
    library such as numpy, will fail tests. If you find yourself writing 50+ line
    methods, it may be a good idea to look for a simpler solution.
    
    Along with the provided functional prototypes, there is another file,
    "train.py" which calls the functions listed in this file. It trains the
    specified network on the MNIST dataset, and then optimizes the loss using a
    standard gradient decent optimizer. You can run this code to check the models
    you create.
    
    """


count=1
import tensorflow as tf
class Average:
    array=[]
    
    def __init__(self,value):
        Average.array.append(int(value))
    
    def __repr(self):
        return  (sum(Average.array)/len(Average.array))

def input_placeholder():
    """
        This placeholder serves as the input to the model, and will be populated
        with the raw images, flattened into single row vectors of length 784.
        
        The number of images to be stored in the placeholder for each minibatch,
        i.e. the minibatch size, may vary during training and testing, so your
        placeholder must allow for a varying number of rows.
        
        :return: A tensorflow placeholder of type float32 and correct shape
        """
    return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")

def target_placeholder():
    """
        This placeholder serves as the output for the model, and will be
        populated with targets for training, and testing. Each output will
        be a single one-hot row vector, of length equal to the number of
        classes to be classified (hint: there's one class for each digit)
        
        The number of target rows to be stored in the placeholder for each
        minibatch, i.e. the minibatch size, may vary during training and
        testing, so your placeholder must allow for a varying number of
        rows.
        
        :return: A tensorflow placeholder of type float32 and correct shape
        """
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")

def onelayer(X, Y, layersize=10):
    input_size =X.get_shape().as_list()[1]
    w=tf.Variable(tf.zeros([input_size,layersize]))
    b=tf.Variable(tf.zeros([1,layersize])+0.1)
    logits=tf.matmul(X,w)+b
    preds=tf.nn.softmax(logits)
    batch_xentropy=-tf.reduce_sum(Y*tf.log(preds),reduction_indices=[1])
    batch_loss=tf.reduce_mean(batch_xentropy)
    return w, b, logits, preds, batch_xentropy, batch_loss


def twolayer(X, Y, hiddensize=30, outputsize=10):
    input_size =X.get_shape().as_list()[1]
    w1=tf.Variable(tf.zeros([input_size,hiddensize]))
    b1=tf.Variable(tf.zeros([1,hiddensize])+0.1)
    matrixInputs=tf.matmul(X,w1)+b1
    
    w2=tf.Variable(tf.random_normal([hiddensize,outputsize]))
    b2=tf.Variable(tf.random_normal([1,outputsize])+0.1)
    logits=tf.matmul(matrixInputs,w2)+b2
    preds=tf.nn.softmax(logits)
    batch_xentropy=-tf.reduce_sum(Y*tf.log(preds),reduction_indices=[1])
    batch_xentropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    batch_loss=tf.reduce_mean(batch_xentropy)
    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss

def convnet(X, Y, convlayer_sizes=[10, 10], \
            filter_shape=[3, 3], outputsize=10, padding="same"):
    conv1=tf.layers.conv2d(inputs=X,filters=convlayer_sizes[0],kernel_size=filter_shape,padding="same",activation=tf.nn.relu)
    conv2=tf.layers.conv2d(inputs=conv1,filters=convlayer_sizes[1],kernel_size=filter_shape, padding="same",activation=tf.nn.relu)
    
    height=conv2.get_shape().as_list()[1]
    width = conv2.get_shape().as_list()[2]
    fix=tf.reshape(conv2, [-1, width * height * convlayer_sizes[1]])
    
    w=tf.Variable(tf.random_normal([width*height*convlayer_sizes[1],outputsize],stddev=0.1))
    b=tf.Variable(tf.random_normal([1,outputsize])+0.1)
    
    logits=tf.matmul(fix,w)+b
    preds=tf.nn.softmax(logits)
    batch_xentropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    batch_loss=tf.reduce_mean(batch_xentropy)
    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss
def wei_variable(shape):
    initial=tf.truncated_normal()

def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
        Run one step of training.
        
        :param sess: the current session
        :param batch: holds the inputs and target outputs for the current minibatch
        batch[0] - array of shape [minibatch_size, 784] with each row holding the
        input images
        batch[1] - array of shape [minibatch_size, 10] with each row holding the
        one-hot encoded targets
        :param X: the input placeholder
        :param Y: the output target placeholder
        :param train_op: the tensorflow operation that will run one step of training
        :param loss_op: the tensorflow operation that will return the loss of your
        model on the batch input/output
        
        :return: a 3-tuple: train_op_result, loss, summary
        which are the results of running the train_op, loss_op and summaries_op
        respectively.
        """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary

