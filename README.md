
# Image Classification
In this project, you'll classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).  The dataset consists of airplanes, dogs, cats, and other objects. You'll preprocess the images, then train a convolutional neural network on all the samples. The images need to be normalized and the labels need to be one-hot encoded.  You'll get to apply what you learned and build a convolutional, max pooling, dropout, and fully connected layers.  At the end, you'll get to see your neural network's predictions on the sample images.
## Get the Data
Run the following cell to download the [CIFAR-10 dataset for python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)
```

    CIFAR-10 Dataset: 171MB [00:40, 4.18MB/s]                              


    All files found!


## Explore the Data
The dataset is broken into batches to prevent your machine from running out of memory.  The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.. Each batch contains the labels and images that are one of the following:
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

Understanding a dataset is part of making predictions on the data.  Play around with the code cell below by changing the `batch_id` and `sample_id`. The `batch_id` is the id for a batch (1-5). The `sample_id` is the id for a image and label pair in the batch.

Ask yourself "What are all possible labels?", "What is the range of values for the image data?", "Are the labels in order or random?".  Answers to questions like these will help you preprocess the data and end up with better predictions.


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper
import numpy as np

# Explore the dataset
batch_id = 5
sample_id = 12
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
```

    
    Stats of batch 5:
    Samples: 10000
    Label Counts: {0: 1014, 1: 1014, 2: 952, 3: 1016, 4: 997, 5: 1025, 6: 980, 7: 977, 8: 1003, 9: 1022}
    First 20 Labels: [1, 8, 5, 1, 5, 7, 4, 3, 8, 2, 7, 2, 0, 1, 5, 9, 6, 2, 0, 8]
    
    Example of Image 12:
    Image - Min Value: 47 Max Value: 255
    Image - Shape: (32, 32, 3)
    Label - Label Id: 0 Name: airplane



![png](img/output_3_1.png)


## Implement Preprocess Functions
### Normalize
In the cell below, implement the `normalize` function to take in image data, `x`, and return it as a normalized Numpy array. The values should be in the range of 0 to 1, inclusive.  The return object should be the same shape as `x`.


```python
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    x_norm = (x/255)
    return x_norm


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)
```

    Tests Passed


### One-hot encode
Just like the previous code cell, you'll be implementing a function for preprocessing.  This time, you'll implement the `one_hot_encode` function. The input, `x`, are a list of labels.  Implement the function to return the list of labels as One-Hot encoded Numpy array.  The possible values for labels are 0 to 9. The one-hot encoding function should return the same encoding for each value between each call to `one_hot_encode`.  Make sure to save the map of encodings outside the function.

Hint: Don't reinvent the wheel.


```python
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    all = []
    
    for label in x:
#        print(label)
        
        # create vector with zeros
        lbl_vec = np.zeros(10)
        
        # set the appropriate value to 1.
        lbl_vec[label] = 1.
        
#        print(lbl_vec)
        
        all.append(lbl_vec)
    
#    print(all)
    return np.array(all)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)
```

    Tests Passed


### Randomize Data
As you saw from exploring the data above, the order of the samples are randomized.  It doesn't hurt to randomize it again, but you don't need to for this dataset.

## Preprocess all the data and save it
Running the code cell below will preprocess all the CIFAR-10 data and save it to file. The code below also uses 10% of the training data for validation.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
```

# Check Point
This is your first checkpoint.  If you ever decide to come back to this notebook or have to restart the notebook, you can start from here.  The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
```

## Build the network
For the neural network, you'll build each layer into a function.  Most of the code you've seen has been outside of functions. To test your code more thoroughly, we require that you put each layer in a function.  This allows us to give you better feedback and test for simple mistakes using our unittests before you submit your project.

>**Note:** If you're finding it hard to dedicate enough time for this course each week, we've provided a small shortcut to this part of the project. In the next couple of problems, you'll have the option to use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages to build each layer, except the layers you build in the "Convolutional and Max Pooling Layer" section.  TF Layers is similar to Keras's and TFLearn's abstraction to layers, so it's easy to pickup.

>However, if you would like to get the most out of this course, try to solve all the problems _without_ using anything from the TF Layers packages. You **can** still use classes from other packages that happen to have the same name as ones you find in TF Layers! For example, instead of using the TF Layers version of the `conv2d` class, [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d), you would want to use the TF Neural Network version of `conv2d`, [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d). 

Let's begin!

### Input
The neural network needs to read the image data, one-hot encoded labels, and dropout keep probability. Implement the following functions
* Implement `neural_net_image_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
 * Set the shape using `image_shape` with batch size set to `None`.
 * Name the TensorFlow placeholder "x" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
* Implement `neural_net_label_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
 * Set the shape using `n_classes` with batch size set to `None`.
 * Name the TensorFlow placeholder "y" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
* Implement `neural_net_keep_prob_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) for dropout keep probability.
 * Name the TensorFlow placeholder "keep_prob" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).

These names will be used at the end of the project to load your saved model.

Note: `None` for shapes in TensorFlow allow for a dynamic size.


```python
import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
#    print(image_shape)
    x = tf.placeholder(dtype=tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]], name='x')
    return x


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
#    print(n_classes)
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    return y


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    return keep_prob


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
```

    Image Input Tests Passed.
    Label Input Tests Passed.
    Keep Prob Tests Passed.


### Convolution and Max Pooling Layer
Convolution layers have a lot of success with images. For this code cell, you should implement the function `conv2d_maxpool` to apply convolution then max pooling:
* Create the weight and bias using `conv_ksize`, `conv_num_outputs` and the shape of `x_tensor`.
* Apply a convolution to `x_tensor` using weight and `conv_strides`.
 * We recommend you use same padding, but you're welcome to use any padding.
* Add bias
* Add a nonlinear activation to the convolution.
* Apply Max Pooling using `pool_ksize` and `pool_strides`.
 * We recommend you use same padding, but you're welcome to use any padding.

**Note:** You **can't** use [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) for **this** layer, but you can still use TensorFlow's [Neural Network](https://www.tensorflow.org/api_docs/python/tf/nn) package. You may still use the shortcut option for all the **other** layers.


```python
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    weights = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[0], x_tensor.get_shape().as_list()[3], conv_num_outputs], stddev=0.1))
    bias = tf.Variable(tf.random_normal([conv_num_outputs]))
    
    # conv2d
    x = tf.nn.conv2d(x_tensor, weights, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    x = tf.nn.bias_add(x, bias)
    x = tf.nn.relu(x)
    
    # max_pool
    x = tf.nn.max_pool(x, ksize=[1, pool_ksize[0], pool_ksize[1], 1], strides=[1, pool_strides[0], pool_strides[1], 1], padding='SAME')
    
    return x 


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_con_pool(conv2d_maxpool)
```

    Tests Passed


### Flatten Layer
Implement the `flatten` function to change the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor.  The output should be the shape (*Batch Size*, *Flattened Image Size*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.


```python
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    shape = x_tensor.get_shape().as_list()
    x_reshaped = tf.reshape(x_tensor, [-1, shape[1] * shape[2] * shape[3]])
    
#    print(x_reshaped.get_shape().as_list())
    
    return x_reshaped


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_flatten(flatten)
```

    Tests Passed


### Fully-Connected Layer
Implement the `fully_conn` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.


```python
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    shape = x_tensor.get_shape().as_list()

    weights = tf.Variable(tf.truncated_normal([shape[1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.random_normal([num_outputs]))
    
    x = tf.add(tf.matmul(x_tensor, weights), bias)
    
    x = tf.nn.relu(x)
    
    return x


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)
```

    Tests Passed


### Output Layer
Implement the `output` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.

**Note:** Activation, softmax, or cross entropy should **not** be applied to this.


```python
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    
    shape = x_tensor.get_shape().as_list()

    weights = tf.Variable(tf.truncated_normal([shape[1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.random_normal([num_outputs]))
    
    x = tf.add(tf.matmul(x_tensor, weights), bias)
    
    return x


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)
```

    Tests Passed


### Create Convolutional Model
Implement the function `conv_net` to create a convolutional neural network model. The function takes in a batch of images, `x`, and outputs logits.  Use the layers you created above to create this model:

* Apply 1, 2, or 3 Convolution and Max Pool layers
* Apply a Flatten Layer
* Apply 1, 2, or 3 Fully Connected Layers
* Apply an Output Layer
* Return the output
* Apply [TensorFlow's Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) to one or more layers in the model using `keep_prob`. 


```python
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv1 = conv2d_maxpool(x, 32, [5, 5], [1, 1], [2, 2], [2, 2])
    conv2 = conv2d_maxpool(conv1, 64, [3, 3], [1, 1], [2, 2], [2, 2])
    conv3 = conv2d_maxpool(conv2, 128, [2, 2], [1, 1], [2, 2], [2, 2])

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flat = flatten(conv3)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fc1 = fully_conn(flat, 1024)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    
    
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    out = output(fc1, 10)
    
    # TODO: return output
    return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)
```

    Neural Network Built!


## Train the Neural Network
### Single Optimization
Implement the function `train_neural_network` to do a single optimization.  The optimization should use `optimizer` to optimize in `session` with a `feed_dict` of the following:
* `x` for image input
* `y` for labels
* `keep_prob` for keep probability for dropout

This function will be called for each batch, so `tf.global_variables_initializer()` has already been called.

Note: Nothing needs to be returned. This function is only optimizing the neural network.


```python
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    session.run(optimizer, feed_dict={ x: feature_batch, y: label_batch, keep_prob: keep_probability })
    pass


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_train_nn(train_neural_network)
```

    Tests Passed


### Show Stats
Implement the function `print_stats` to print loss and validation accuracy.  Use the global variables `valid_features` and `valid_labels` to calculate validation accuracy.  Use a keep probability of `1.0` to calculate the loss and validation accuracy.


```python
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    loss = session.run(cost, feed_dict={ x: feature_batch, y: label_batch, keep_prob: 1. })
    valid_acc = session.run(accuracy, feed_dict={ x: valid_features, y: valid_labels, keep_prob: 1. })
    
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
    pass
```

### Hyperparameters
Tune the following parameters:
* Set `epochs` to the number of iterations until the network stops learning or start overfitting
* Set `batch_size` to the highest number that your machine has memory for.  Most people set them to common sizes of memory:
 * 64
 * 128
 * 256
 * ...
* Set `keep_probability` to the probability of keeping a node using dropout


```python
# TODO: Tune Parameters
epochs = 30
batch_size = 256
keep_probability = 0.8
```

### Train on a Single CIFAR-10 Batch
Instead of training the neural network on all the CIFAR-10 batches of data, let's use a single batch. This should save time while you iterate on the model to get a better accuracy.  Once the final validation accuracy is 50% or greater, run the model on all the data in the next section.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
```

    Checking the Training on a Single Batch...
    Epoch  1, CIFAR-10 Batch 1:  Loss:     2.2305 Validation Accuracy: 0.231800
    Epoch  2, CIFAR-10 Batch 1:  Loss:     2.1441 Validation Accuracy: 0.281000
    Epoch  3, CIFAR-10 Batch 1:  Loss:     1.9610 Validation Accuracy: 0.357600
    Epoch  4, CIFAR-10 Batch 1:  Loss:     1.9908 Validation Accuracy: 0.371600
    Epoch  5, CIFAR-10 Batch 1:  Loss:     1.7397 Validation Accuracy: 0.398000
    Epoch  6, CIFAR-10 Batch 1:  Loss:     1.6426 Validation Accuracy: 0.404400
    Epoch  7, CIFAR-10 Batch 1:  Loss:     1.5272 Validation Accuracy: 0.447200
    Epoch  8, CIFAR-10 Batch 1:  Loss:     1.5220 Validation Accuracy: 0.464200
    Epoch  9, CIFAR-10 Batch 1:  Loss:     1.4705 Validation Accuracy: 0.463800
    Epoch 10, CIFAR-10 Batch 1:  Loss:     1.3629 Validation Accuracy: 0.483000
    Epoch 11, CIFAR-10 Batch 1:  Loss:     1.3521 Validation Accuracy: 0.479000
    Epoch 12, CIFAR-10 Batch 1:  Loss:     1.3169 Validation Accuracy: 0.484600
    Epoch 13, CIFAR-10 Batch 1:  Loss:     1.2463 Validation Accuracy: 0.504600
    Epoch 14, CIFAR-10 Batch 1:  Loss:     1.1138 Validation Accuracy: 0.506400
    Epoch 15, CIFAR-10 Batch 1:  Loss:     1.0747 Validation Accuracy: 0.500600
    Epoch 16, CIFAR-10 Batch 1:  Loss:     1.0729 Validation Accuracy: 0.517000
    Epoch 17, CIFAR-10 Batch 1:  Loss:     0.9559 Validation Accuracy: 0.518400
    Epoch 18, CIFAR-10 Batch 1:  Loss:     0.8956 Validation Accuracy: 0.520400
    Epoch 19, CIFAR-10 Batch 1:  Loss:     0.8173 Validation Accuracy: 0.507200
    Epoch 20, CIFAR-10 Batch 1:  Loss:     0.8655 Validation Accuracy: 0.509800
    Epoch 21, CIFAR-10 Batch 1:  Loss:     0.7226 Validation Accuracy: 0.518600
    Epoch 22, CIFAR-10 Batch 1:  Loss:     0.6942 Validation Accuracy: 0.524600
    Epoch 23, CIFAR-10 Batch 1:  Loss:     0.6572 Validation Accuracy: 0.527600
    Epoch 24, CIFAR-10 Batch 1:  Loss:     0.6508 Validation Accuracy: 0.509600
    Epoch 25, CIFAR-10 Batch 1:  Loss:     0.6870 Validation Accuracy: 0.517600


### Fully Train the Model
Now that you got a good accuracy with a single CIFAR-10 batch, try it with all five batches.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
```

    Training...
    Epoch  1, CIFAR-10 Batch 1:  Loss:     2.2201 Validation Accuracy: 0.267800
    Epoch  1, CIFAR-10 Batch 2:  Loss:     1.9677 Validation Accuracy: 0.342200
    Epoch  1, CIFAR-10 Batch 3:  Loss:     1.5462 Validation Accuracy: 0.373800
    Epoch  1, CIFAR-10 Batch 4:  Loss:     1.5947 Validation Accuracy: 0.413600
    Epoch  1, CIFAR-10 Batch 5:  Loss:     1.5226 Validation Accuracy: 0.452600
    Epoch  2, CIFAR-10 Batch 1:  Loss:     1.7198 Validation Accuracy: 0.446800
    Epoch  2, CIFAR-10 Batch 2:  Loss:     1.5747 Validation Accuracy: 0.471800
    Epoch  2, CIFAR-10 Batch 3:  Loss:     1.1214 Validation Accuracy: 0.459800
    Epoch  2, CIFAR-10 Batch 4:  Loss:     1.2990 Validation Accuracy: 0.472800
    Epoch  2, CIFAR-10 Batch 5:  Loss:     1.4149 Validation Accuracy: 0.482200
    Epoch  3, CIFAR-10 Batch 1:  Loss:     1.4585 Validation Accuracy: 0.490000
    Epoch  3, CIFAR-10 Batch 2:  Loss:     1.4134 Validation Accuracy: 0.500600
    Epoch  3, CIFAR-10 Batch 3:  Loss:     0.9723 Validation Accuracy: 0.506200
    Epoch  3, CIFAR-10 Batch 4:  Loss:     1.1021 Validation Accuracy: 0.535400
    Epoch  3, CIFAR-10 Batch 5:  Loss:     1.2372 Validation Accuracy: 0.528800
    Epoch  4, CIFAR-10 Batch 1:  Loss:     1.2625 Validation Accuracy: 0.527200
    Epoch  4, CIFAR-10 Batch 2:  Loss:     1.1580 Validation Accuracy: 0.547800
    Epoch  4, CIFAR-10 Batch 3:  Loss:     0.8398 Validation Accuracy: 0.533600
    Epoch  4, CIFAR-10 Batch 4:  Loss:     0.9282 Validation Accuracy: 0.552400
    Epoch  4, CIFAR-10 Batch 5:  Loss:     1.0547 Validation Accuracy: 0.550200
    Epoch  5, CIFAR-10 Batch 1:  Loss:     1.0746 Validation Accuracy: 0.550800
    Epoch  5, CIFAR-10 Batch 2:  Loss:     0.9737 Validation Accuracy: 0.567800
    Epoch  5, CIFAR-10 Batch 3:  Loss:     0.6614 Validation Accuracy: 0.538400
    Epoch  5, CIFAR-10 Batch 4:  Loss:     0.7823 Validation Accuracy: 0.576400
    Epoch  5, CIFAR-10 Batch 5:  Loss:     0.8947 Validation Accuracy: 0.564200
    Epoch  6, CIFAR-10 Batch 1:  Loss:     0.9927 Validation Accuracy: 0.578600
    Epoch  6, CIFAR-10 Batch 2:  Loss:     0.7872 Validation Accuracy: 0.585200
    Epoch  6, CIFAR-10 Batch 3:  Loss:     0.4959 Validation Accuracy: 0.578800
    Epoch  6, CIFAR-10 Batch 4:  Loss:     0.6571 Validation Accuracy: 0.597400
    Epoch  6, CIFAR-10 Batch 5:  Loss:     0.7065 Validation Accuracy: 0.592000
    Epoch  7, CIFAR-10 Batch 1:  Loss:     0.8494 Validation Accuracy: 0.574800
    Epoch  7, CIFAR-10 Batch 2:  Loss:     0.6360 Validation Accuracy: 0.590200
    Epoch  7, CIFAR-10 Batch 3:  Loss:     0.4003 Validation Accuracy: 0.588000
    Epoch  7, CIFAR-10 Batch 4:  Loss:     0.5589 Validation Accuracy: 0.594200
    Epoch  7, CIFAR-10 Batch 5:  Loss:     0.5781 Validation Accuracy: 0.606800
    Epoch  8, CIFAR-10 Batch 1:  Loss:     0.7249 Validation Accuracy: 0.593000
    Epoch  8, CIFAR-10 Batch 2:  Loss:     0.5055 Validation Accuracy: 0.610200
    Epoch  8, CIFAR-10 Batch 3:  Loss:     0.3497 Validation Accuracy: 0.577400
    Epoch  8, CIFAR-10 Batch 4:  Loss:     0.4441 Validation Accuracy: 0.608400
    Epoch  8, CIFAR-10 Batch 5:  Loss:     0.5030 Validation Accuracy: 0.603000
    Epoch  9, CIFAR-10 Batch 1:  Loss:     0.5523 Validation Accuracy: 0.610000
    Epoch  9, CIFAR-10 Batch 2:  Loss:     0.4271 Validation Accuracy: 0.612000
    Epoch  9, CIFAR-10 Batch 3:  Loss:     0.2784 Validation Accuracy: 0.589200
    Epoch  9, CIFAR-10 Batch 4:  Loss:     0.3592 Validation Accuracy: 0.616800
    Epoch  9, CIFAR-10 Batch 5:  Loss:     0.3812 Validation Accuracy: 0.620000
    Epoch 10, CIFAR-10 Batch 1:  Loss:     0.4753 Validation Accuracy: 0.620400
    Epoch 10, CIFAR-10 Batch 2:  Loss:     0.3398 Validation Accuracy: 0.606600
    Epoch 10, CIFAR-10 Batch 3:  Loss:     0.2027 Validation Accuracy: 0.603000
    Epoch 10, CIFAR-10 Batch 4:  Loss:     0.2578 Validation Accuracy: 0.621000
    Epoch 10, CIFAR-10 Batch 5:  Loss:     0.2830 Validation Accuracy: 0.620000
    Epoch 11, CIFAR-10 Batch 1:  Loss:     0.4022 Validation Accuracy: 0.632200
    Epoch 11, CIFAR-10 Batch 2:  Loss:     0.2558 Validation Accuracy: 0.624600
    Epoch 11, CIFAR-10 Batch 3:  Loss:     0.1926 Validation Accuracy: 0.606400
    Epoch 11, CIFAR-10 Batch 4:  Loss:     0.2317 Validation Accuracy: 0.633400
    Epoch 11, CIFAR-10 Batch 5:  Loss:     0.2139 Validation Accuracy: 0.632600
    Epoch 12, CIFAR-10 Batch 1:  Loss:     0.3607 Validation Accuracy: 0.627400
    Epoch 12, CIFAR-10 Batch 2:  Loss:     0.2359 Validation Accuracy: 0.628800
    Epoch 12, CIFAR-10 Batch 3:  Loss:     0.1581 Validation Accuracy: 0.625400
    Epoch 12, CIFAR-10 Batch 4:  Loss:     0.1576 Validation Accuracy: 0.631000
    Epoch 12, CIFAR-10 Batch 5:  Loss:     0.1714 Validation Accuracy: 0.634600
    Epoch 13, CIFAR-10 Batch 1:  Loss:     0.3082 Validation Accuracy: 0.621000
    Epoch 13, CIFAR-10 Batch 2:  Loss:     0.1438 Validation Accuracy: 0.627600
    Epoch 13, CIFAR-10 Batch 3:  Loss:     0.1138 Validation Accuracy: 0.638400
    Epoch 13, CIFAR-10 Batch 4:  Loss:     0.1287 Validation Accuracy: 0.635200
    Epoch 13, CIFAR-10 Batch 5:  Loss:     0.1014 Validation Accuracy: 0.635400
    Epoch 14, CIFAR-10 Batch 1:  Loss:     0.1898 Validation Accuracy: 0.639600
    Epoch 14, CIFAR-10 Batch 2:  Loss:     0.1324 Validation Accuracy: 0.638600
    Epoch 14, CIFAR-10 Batch 3:  Loss:     0.1061 Validation Accuracy: 0.640400
    Epoch 14, CIFAR-10 Batch 4:  Loss:     0.1274 Validation Accuracy: 0.625000
    Epoch 14, CIFAR-10 Batch 5:  Loss:     0.1091 Validation Accuracy: 0.620400
    Epoch 15, CIFAR-10 Batch 1:  Loss:     0.1400 Validation Accuracy: 0.648000
    Epoch 15, CIFAR-10 Batch 2:  Loss:     0.1019 Validation Accuracy: 0.652000
    Epoch 15, CIFAR-10 Batch 3:  Loss:     0.0778 Validation Accuracy: 0.639000
    Epoch 15, CIFAR-10 Batch 4:  Loss:     0.1032 Validation Accuracy: 0.632600
    Epoch 15, CIFAR-10 Batch 5:  Loss:     0.0877 Validation Accuracy: 0.636600
    Epoch 16, CIFAR-10 Batch 1:  Loss:     0.1323 Validation Accuracy: 0.653800
    Epoch 16, CIFAR-10 Batch 2:  Loss:     0.1034 Validation Accuracy: 0.644800
    Epoch 16, CIFAR-10 Batch 3:  Loss:     0.0749 Validation Accuracy: 0.632800
    Epoch 16, CIFAR-10 Batch 4:  Loss:     0.1005 Validation Accuracy: 0.639600
    Epoch 16, CIFAR-10 Batch 5:  Loss:     0.0648 Validation Accuracy: 0.654600
    Epoch 17, CIFAR-10 Batch 1:  Loss:     0.0824 Validation Accuracy: 0.649800
    Epoch 17, CIFAR-10 Batch 2:  Loss:     0.0813 Validation Accuracy: 0.638000
    Epoch 17, CIFAR-10 Batch 3:  Loss:     0.0807 Validation Accuracy: 0.617600
    Epoch 17, CIFAR-10 Batch 4:  Loss:     0.0568 Validation Accuracy: 0.635400
    Epoch 17, CIFAR-10 Batch 5:  Loss:     0.0619 Validation Accuracy: 0.655200
    Epoch 18, CIFAR-10 Batch 1:  Loss:     0.0641 Validation Accuracy: 0.652400
    Epoch 18, CIFAR-10 Batch 2:  Loss:     0.0461 Validation Accuracy: 0.645800
    Epoch 18, CIFAR-10 Batch 3:  Loss:     0.0689 Validation Accuracy: 0.638400
    Epoch 18, CIFAR-10 Batch 4:  Loss:     0.0577 Validation Accuracy: 0.610200
    Epoch 18, CIFAR-10 Batch 5:  Loss:     0.0563 Validation Accuracy: 0.664000
    Epoch 19, CIFAR-10 Batch 1:  Loss:     0.0676 Validation Accuracy: 0.661600
    Epoch 19, CIFAR-10 Batch 2:  Loss:     0.0512 Validation Accuracy: 0.640400
    Epoch 19, CIFAR-10 Batch 3:  Loss:     0.0511 Validation Accuracy: 0.641200
    Epoch 19, CIFAR-10 Batch 4:  Loss:     0.0827 Validation Accuracy: 0.604800
    Epoch 19, CIFAR-10 Batch 5:  Loss:     0.0363 Validation Accuracy: 0.660600
    Epoch 20, CIFAR-10 Batch 1:  Loss:     0.0570 Validation Accuracy: 0.654600
    Epoch 20, CIFAR-10 Batch 2:  Loss:     0.0327 Validation Accuracy: 0.632200
    Epoch 20, CIFAR-10 Batch 3:  Loss:     0.0522 Validation Accuracy: 0.649200
    Epoch 20, CIFAR-10 Batch 4:  Loss:     0.0342 Validation Accuracy: 0.633200
    Epoch 20, CIFAR-10 Batch 5:  Loss:     0.0356 Validation Accuracy: 0.654400
    Epoch 21, CIFAR-10 Batch 1:  Loss:     0.0466 Validation Accuracy: 0.661400
    Epoch 21, CIFAR-10 Batch 2:  Loss:     0.0295 Validation Accuracy: 0.625400
    Epoch 21, CIFAR-10 Batch 3:  Loss:     0.0321 Validation Accuracy: 0.634200
    Epoch 21, CIFAR-10 Batch 4:  Loss:     0.0269 Validation Accuracy: 0.645800
    Epoch 21, CIFAR-10 Batch 5:  Loss:     0.0330 Validation Accuracy: 0.654000
    Epoch 22, CIFAR-10 Batch 1:  Loss:     0.0438 Validation Accuracy: 0.638400
    Epoch 22, CIFAR-10 Batch 2:  Loss:     0.0404 Validation Accuracy: 0.630600
    Epoch 22, CIFAR-10 Batch 3:  Loss:     0.0265 Validation Accuracy: 0.629600
    Epoch 22, CIFAR-10 Batch 4:  Loss:     0.0266 Validation Accuracy: 0.644600
    Epoch 22, CIFAR-10 Batch 5:  Loss:     0.0243 Validation Accuracy: 0.652200
    Epoch 23, CIFAR-10 Batch 1:  Loss:     0.0224 Validation Accuracy: 0.637400
    Epoch 23, CIFAR-10 Batch 2:  Loss:     0.0150 Validation Accuracy: 0.640000
    Epoch 23, CIFAR-10 Batch 3:  Loss:     0.0136 Validation Accuracy: 0.653800
    Epoch 23, CIFAR-10 Batch 4:  Loss:     0.0235 Validation Accuracy: 0.658400
    Epoch 23, CIFAR-10 Batch 5:  Loss:     0.0217 Validation Accuracy: 0.647200
    Epoch 24, CIFAR-10 Batch 1:  Loss:     0.0257 Validation Accuracy: 0.633000
    Epoch 24, CIFAR-10 Batch 2:  Loss:     0.0146 Validation Accuracy: 0.641800
    Epoch 24, CIFAR-10 Batch 3:  Loss:     0.0148 Validation Accuracy: 0.650200
    Epoch 24, CIFAR-10 Batch 4:  Loss:     0.0121 Validation Accuracy: 0.656200
    Epoch 24, CIFAR-10 Batch 5:  Loss:     0.0103 Validation Accuracy: 0.656000
    Epoch 25, CIFAR-10 Batch 1:  Loss:     0.0217 Validation Accuracy: 0.636800
    Epoch 25, CIFAR-10 Batch 2:  Loss:     0.0141 Validation Accuracy: 0.631600
    Epoch 25, CIFAR-10 Batch 3:  Loss:     0.0106 Validation Accuracy: 0.646800
    Epoch 25, CIFAR-10 Batch 4:  Loss:     0.0169 Validation Accuracy: 0.638400
    Epoch 25, CIFAR-10 Batch 5:  Loss:     0.0306 Validation Accuracy: 0.630400
    Epoch 26, CIFAR-10 Batch 1:  Loss:     0.0251 Validation Accuracy: 0.635200
    Epoch 26, CIFAR-10 Batch 2:  Loss:     0.0178 Validation Accuracy: 0.645000
    Epoch 26, CIFAR-10 Batch 3:  Loss:     0.0121 Validation Accuracy: 0.655400
    Epoch 26, CIFAR-10 Batch 4:  Loss:     0.0099 Validation Accuracy: 0.640800
    Epoch 26, CIFAR-10 Batch 5:  Loss:     0.0090 Validation Accuracy: 0.639000
    Epoch 27, CIFAR-10 Batch 1:  Loss:     0.0123 Validation Accuracy: 0.652400
    Epoch 27, CIFAR-10 Batch 2:  Loss:     0.0067 Validation Accuracy: 0.649000
    Epoch 27, CIFAR-10 Batch 3:  Loss:     0.0101 Validation Accuracy: 0.654000
    Epoch 27, CIFAR-10 Batch 4:  Loss:     0.0135 Validation Accuracy: 0.640600
    Epoch 27, CIFAR-10 Batch 5:  Loss:     0.0081 Validation Accuracy: 0.644400
    Epoch 28, CIFAR-10 Batch 1:  Loss:     0.0119 Validation Accuracy: 0.633400
    Epoch 28, CIFAR-10 Batch 2:  Loss:     0.0096 Validation Accuracy: 0.651800
    Epoch 28, CIFAR-10 Batch 3:  Loss:     0.0022 Validation Accuracy: 0.658000
    Epoch 28, CIFAR-10 Batch 4:  Loss:     0.0060 Validation Accuracy: 0.641000
    Epoch 28, CIFAR-10 Batch 5:  Loss:     0.0049 Validation Accuracy: 0.639600
    Epoch 29, CIFAR-10 Batch 1:  Loss:     0.0120 Validation Accuracy: 0.637600
    Epoch 29, CIFAR-10 Batch 2:  Loss:     0.0053 Validation Accuracy: 0.656600
    Epoch 29, CIFAR-10 Batch 3:  Loss:     0.0034 Validation Accuracy: 0.655600
    Epoch 29, CIFAR-10 Batch 4:  Loss:     0.0039 Validation Accuracy: 0.649600
    Epoch 29, CIFAR-10 Batch 5:  Loss:     0.0067 Validation Accuracy: 0.637200
    Epoch 30, CIFAR-10 Batch 1:  Loss:     0.0149 Validation Accuracy: 0.640000
    Epoch 30, CIFAR-10 Batch 2:  Loss:     0.0122 Validation Accuracy: 0.641800
    Epoch 30, CIFAR-10 Batch 3:  Loss:     0.0040 Validation Accuracy: 0.660800
    Epoch 30, CIFAR-10 Batch 4:  Loss:     0.0070 Validation Accuracy: 0.642200
    Epoch 30, CIFAR-10 Batch 5:  Loss:     0.0042 Validation Accuracy: 0.642000


# Checkpoint
The model has been saved to disk.
## Test Model
Test your model against the test dataset.  This will be your final accuracy. You should have an accuracy greater than 50%. If you don't, keep tweaking the model architecture and parameters.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
```

    Testing Accuracy: 0.63154296875
    



![png](img/output_36_1.png)


## Why 50-70% Accuracy?
You might be wondering why you can't get an accuracy any higher. First things first, 50% isn't bad for a simple CNN.  Pure guessing would get you 10% accuracy. However, you might notice people are getting scores [well above 70%](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130).  That's because we haven't taught you all there is to know about neural networks. We still need to cover a few more techniques.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook.  Save the notebook file as "dlnd_image_classification.ipynb" and save it as a HTML file under "File" -> "Download as".  Include the "helper.py" and "problem_unittests.py" files in your submission.
