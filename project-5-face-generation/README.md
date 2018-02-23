
# Face Generation
In this project, you'll use generative adversarial networks to generate new images of faces.
### Get the Data
You'll be using two datasets in this project:
- MNIST
- CelebA

Since the celebA dataset is complex and you're doing GANs in a project for the first time, we want you to test your neural network on MNIST before CelebA.  Running the GANs on MNIST will allow you to see how well your model trains sooner.

If you're using [FloydHub](https://www.floydhub.com/), set `data_dir` to "/input" and use the [FloydHub data ID](http://docs.floydhub.com/home/using_datasets/) "R5KrjnANiKVhLWAkpXhNBe".


```python
data_dir = './data'

# FloydHub - Use with data ID "R5KrjnANiKVhLWAkpXhNBe"
#data_dir = '/input'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)
```

    Downloading mnist: 9.92MB [00:00, 32.9MB/s]                            
    Extracting mnist: 100%|██████████| 60.0K/60.0K [00:19<00:00, 3.15KFile/s] 
    Downloading celeba: 1.44GB [01:13, 19.7MB/s]                               


    Extracting celeba...


## Explore the Data
### MNIST
As you're aware, the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains images of handwritten digits. You can view the first number of examples by changing `show_n_images`. 


```python
show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
%matplotlib inline
import os
from glob import glob
from matplotlib import pyplot

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f53711914e0>




![png](readme_media/output_3_1.png)


### CelebA
The [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations.  Since you're going to be generating faces, you won't need the annotations.  You can view the first number of examples by changing `show_n_images`.


```python
show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))
```




    <matplotlib.image.AxesImage at 0x7f53710be470>




![png](readme_media/output_5_1.png)


## Preprocess the Data
Since the project's main focus is on building the GANs, we'll preprocess the data for you.  The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images.  The CelebA images will be cropped to remove parts of the image that don't include a face, then resized down to 28x28.

The MNIST images are black and white images with a single [color channel](https://en.wikipedia.org/wiki/Channel_(digital_image%29) while the CelebA images have [3 color channels (RGB color channel)](https://en.wikipedia.org/wiki/Channel_(digital_image%29#RGB_Images).
## Build the Neural Network
You'll build the components necessary to build a GANs by implementing the following functions below:
- `model_inputs`
- `discriminator`
- `generator`
- `model_loss`
- `model_opt`
- `train`

### Check the Version of TensorFlow and Access to GPU
This will check to make sure you have the correct version of TensorFlow and access to a GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    Default GPU Device: /gpu:0


### Input
Implement the `model_inputs` function to create TF Placeholders for the Neural Network. It should create the following placeholders:
- Real input images placeholder with rank 4 using `image_width`, `image_height`, and `image_channels`.
- Z input placeholder with rank 2 using `z_dim`.
- Learning rate placeholder with rank 0.

Return the placeholders in the following the tuple (tensor of real input images, tensor of z data)


```python
import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    real_inputs = tf.placeholder(tf.float32, (None,image_width, image_height, image_channels), name='real_inputs')
    z_inputs = tf.placeholder(tf.float32,(None,z_dim), name='z_inputs')
    lr = tf.placeholder(tf.float32, name='lr')

    return real_inputs, z_inputs, lr


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed


### Discriminator
Implement `discriminator` to create a discriminator neural network that discriminates on `images`.  This function should be able to reuse the variabes in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "discriminator" to allow the variables to be reused.  The function should return a tuple of (tensor output of the generator, tensor logits of the generator).


```python
def discriminator(images, reuse=False, alpha=0.25):
    """
    Create the discriminator network
    :param image: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    
    with tf.variable_scope('discriminator', reuse=reuse):
        
        # Input layer is 28 x 28 x ???
        x_1 = tf.layers.conv2d(images, 64, 3, strides=2, padding='same')
        relu_1 = tf.maximum(alpha * x_1, x_1)
        
        # 14 x 14 x 64
        print ('Discriminator Layer 1 shape:', relu_1.shape)
      
        x_2 = tf.layers.conv2d(relu_1, 128, 3, strides=2, padding='same')
        bn_2 = tf.layers.batch_normalization(x_2, training=True)
        relu_2 = tf.maximum(alpha * bn_2, bn_2)
        
        # 7 x 7 x 128
        print ('Discriminator Layer 2 shape:', relu_2.shape)
        
        x_3 = tf.layers.conv2d(relu_2, 256, 3, strides=2, padding='same')
        bn_3 = tf.layers.batch_normalization(x_3, training=True)
        relu_3 = tf.maximum(alpha * bn_3, bn_3)

        # 4 x 4 x 256
        print ('Discriminator Layer 3 shape:',relu_3.shape)
        
        x_4 = tf.layers.conv2d(relu_3, 512, 3, strides=2, padding='same')
        bn_4 = tf.layers.batch_normalization(x_4, training=True)
        relu_4 = tf.maximum(alpha * bn_4, bn_4)

        # 2 x 2 x 512
        print ('Discriminator Layer 4 shape:',relu_4.shape)
        
        # Flatten it
        flat = tf.reshape(relu_4, (-1, 2*2*512))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)
        
        return out, logits

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(discriminator, tf)
```

    Discriminator Layer 1 shape: (?, 14, 14, 64)
    Discriminator Layer 2 shape: (?, 7, 7, 128)
    Discriminator Layer 3 shape: (?, 4, 4, 256)
    Discriminator Layer 4 shape: (?, 2, 2, 512)
    Discriminator Layer 1 shape: (?, 14, 14, 64)
    Discriminator Layer 2 shape: (?, 7, 7, 128)
    Discriminator Layer 3 shape: (?, 4, 4, 256)
    Discriminator Layer 4 shape: (?, 2, 2, 512)
    Tests Passed


### Generator
Implement `generator` to generate an image using `z`. This function should be able to reuse the variabes in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "generator" to allow the variables to be reused. The function should return the generated 28 x 28 x `out_channel_dim` images.


```python
def generator(z, out_channel_dim, is_train=True, alpha=0.25):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    
    if is_train==True:
        reuse = False
    else:
        reuse = True

    with tf.variable_scope('generator', reuse=reuse):
        
        # 1) Fully Connected Layer
        x_1 = tf.layers.dense(z, 2*2*512)
        
        # Reshape FC
        x_1 = tf.reshape(x_1, (-1, 2, 2, 512))
        x_1 = tf.layers.batch_normalization(x_1, training=is_train)
        x_1 = tf.maximum(alpha * x_1, x_1)

        # 2 x 2 x 512 now
        print ('Generator Layer 1 shape:', x_1.shape)
        
        x_2 = tf.layers.conv2d_transpose(x_1, 256, 3, strides=2, padding='same')
        x_2 = tf.layers.batch_normalization(x_2, training=is_train)
        x_2 = tf.maximum(alpha * x_2, x_2)

        # 4 x 4 x 256 now
        print ('Generator Layer 2 shape:', x_2.shape)
        
        x_3 = tf.layers.conv2d_transpose(x_2, 128, 4, strides=1, padding='valid')
        x_3 = tf.layers.batch_normalization(x_3, training=is_train)
        x_3 = tf.maximum(alpha * x_3, x_3)

        # 7 x 7 x 128 now
        print ('Generator Layer 3 shape::',x_3.shape)
        
        x_4 = tf.layers.conv2d_transpose(x_3, 64, 3, strides=2, padding='same')
        x_4 = tf.layers.batch_normalization(x_4, training=is_train)
        x_4 = tf.maximum(alpha * x_4, x_4)

        # 14 x 14 x 64 now
        print ('Generator Layer 4 shape:',x_4.shape)
        
        # Logits layer
        logits = tf.layers.conv2d_transpose(x_4, out_channel_dim, 5, strides=2, padding='same')

        # 28 x 28 x out_channel now
        print ('Generator Logits Layer shape:',logits.shape)
        output = tf.tanh(logits) 
        print ('Generator Output Layer shape:',output.shape)
        
        return output    


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(generator, tf)
```

    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 5)
    Generator Output Layer shape: (?, 28, 28, 5)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 5)
    Generator Output Layer shape: (?, 28, 28, 5)
    Tests Passed


### Loss
Implement `model_loss` to build the GANs for training and calculate the loss.  The function should return a tuple of (discriminator loss, generator loss).  Use the following functions you implemented:
- `discriminator(images, reuse=False)`
- `generator(z, out_channel_dim, is_train=True)`


```python
def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    alpha = 0.25
    
    gen_model = generator(input_z, out_channel_dim, alpha=alpha)
    dis_model_real, dis_logits_real = discriminator(input_real, reuse=False, alpha=alpha)
    dis_model_fake, dis_logits_fake = discriminator(gen_model, reuse=True, alpha=alpha)

    dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_real, labels=tf.ones_like(dis_model_real)))
    dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_fake, labels=tf.zeros_like(dis_model_fake)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_fake, labels=tf.ones_like(dis_model_fake)))

    dis_loss = dis_loss_real + dis_loss_fake

    return dis_loss, gen_loss


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_loss(model_loss)
```

    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 4)
    Generator Output Layer shape: (?, 28, 28, 4)
    Discriminator Layer 1 shape: (?, 14, 14, 64)
    Discriminator Layer 2 shape: (?, 7, 7, 128)
    Discriminator Layer 3 shape: (?, 4, 4, 256)
    Discriminator Layer 4 shape: (?, 2, 2, 512)
    Discriminator Layer 1 shape: (?, 14, 14, 64)
    Discriminator Layer 2 shape: (?, 7, 7, 128)
    Discriminator Layer 3 shape: (?, 4, 4, 256)
    Discriminator Layer 4 shape: (?, 2, 2, 512)
    Tests Passed


### Optimization
Implement `model_opt` to create the optimization operations for the GANs. Use [`tf.trainable_variables`](https://www.tensorflow.org/api_docs/python/tf/trainable_variables) to get all the trainable variables.  Filter the variables with names that are in the discriminator and generator scope names.  The function should return a tuple of (discriminator training operation, generator training operation).


```python
def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """

    # Get the trainable_variables, split into G and D parts
    trainable_vars = tf.trainable_variables()
    generator_vars = [var for var in trainable_vars if var.name.startswith('generator')]
    discriminator_vars = [var for var in trainable_vars if var.name.startswith('discriminator')]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gen_updates = [opt for opt in update_ops if opt.name.startswith('generator')]
    dis_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=discriminator_vars)
    with tf.control_dependencies(gen_updates):
        gen_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=generator_vars)
    
    return dis_train_opt, gen_train_opt


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_opt(model_opt, tf)
```

    Tests Passed


## Neural Network Training
### Show Output
Use this function to show the current output of the generator during training. It will help you determine how well the GANs is training.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()
```

### Train
Implement `train` to build and train the GANs.  Use the following functions you implemented:
- `model_inputs(image_width, image_height, image_channels, z_dim)`
- `model_loss(input_real, input_z, out_channel_dim)`
- `model_opt(d_loss, g_loss, learning_rate, beta1)`

Use the `show_generator_output` to show `generator` output while you train. Running `show_generator_output` for every batch will drastically increase training time and increase the size of the notebook.  It's recommended to print the `generator` output every 100 batches.


```python
def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    
    # determine color channels
    if(data_image_mode == 'RGB'):
        image_channels = 3
    else:
        image_channels = 1
    
    # determine width / height
    image_width = data_shape[1]
    image_height = data_shape[2]
    
    # inputs
    input_real, input_z, lr = model_inputs(image_width,
                                           image_height,
                                           image_channels,
                                           z_dim)
    
    # losses
    dis_loss, gen_loss = model_loss(input_real, input_z, image_channels)
    
    
    dis_train_opt, gen_train_opt = model_opt(dis_loss, gen_loss, learning_rate, beta1)
    batch_num = 0
    losses = []
    samples = []
    
    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    saver = tf.train.Saver(var_list = gen_vars)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                
                # normalize images -1 <-> 1
                batch_images = batch_images * 2.0
                
                # sample random noise for generator
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
                # run optimizers
                _ = sess.run(dis_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(gen_train_opt, feed_dict={input_z: batch_z})
                
                batch_num += 1
                
                if (batch_num%100 == 0):
                    show_generator_output(sess=sess,
                                          image_mode=data_image_mode,
                                          input_z=input_z,
                                          n_images=10,
                                          out_channel_dim=image_channels)
            
            # every epoch: calc losses
            train_loss_dis = sess.run(dis_loss, {input_z: batch_z, input_real: batch_images})
            train_loss_gen = gen_loss.eval({input_z: batch_z})

            # print losses
            print("Epoch {}/{}...".format(epoch_i+1, epochs),
                  "Discriminator Loss: {:.4f}...".format(train_loss_dis),
                  "Generator Loss: {:.4f}".format(train_loss_gen))    
            
            # save losses
            losses.append((train_loss_dis, train_loss_gen))

            # sample from generator
            sample_z = np.random.uniform(-1, 1, size=(16, z_dim))
            gen_samples = sess.run(
                           generator(input_z, image_channels, is_train=False),
                           feed_dict={input_z: sample_z})
            
            samples.append(gen_samples)
            saver.save(sess, './checkpoints/generator.ckpt')
                
                
```

### MNIST
Test your GANs architecture on MNIST.  After 2 epochs, the GANs should be able to generate images that look like handwritten digits.  Make sure the loss of the generator is lower than the loss of the discriminator or close to 0.


```python
batch_size = 128
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 10

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)
```

    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)
    Discriminator Layer 1 shape: (?, 14, 14, 64)
    Discriminator Layer 2 shape: (?, 7, 7, 128)
    Discriminator Layer 3 shape: (?, 4, 4, 256)
    Discriminator Layer 4 shape: (?, 2, 2, 512)
    Discriminator Layer 1 shape: (?, 14, 14, 64)
    Discriminator Layer 2 shape: (?, 7, 7, 128)
    Discriminator Layer 3 shape: (?, 4, 4, 256)
    Discriminator Layer 4 shape: (?, 2, 2, 512)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_1.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_3.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_5.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_7.png)


    Epoch 1/10... Discriminator Loss: 0.4488... Generator Loss: 1.7337
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_9.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_11.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_13.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_15.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_17.png)


    Epoch 2/10... Discriminator Loss: 0.3643... Generator Loss: 2.1198
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_19.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_21.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_23.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_25.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_27.png)


    Epoch 3/10... Discriminator Loss: 1.2061... Generator Loss: 0.4271
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_29.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_31.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_33.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_35.png)


    Epoch 4/10... Discriminator Loss: 0.4858... Generator Loss: 1.2835
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_37.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_39.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_41.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_43.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_45.png)


    Epoch 5/10... Discriminator Loss: 0.1805... Generator Loss: 2.5960
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_47.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_49.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_51.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_53.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_55.png)


    Epoch 6/10... Discriminator Loss: 0.3461... Generator Loss: 1.5465
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_57.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_59.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_61.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_63.png)


    Epoch 7/10... Discriminator Loss: 1.5497... Generator Loss: 0.3835
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_65.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_67.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_69.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_71.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_73.png)


    Epoch 8/10... Discriminator Loss: 0.4954... Generator Loss: 1.2082
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_75.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_77.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_79.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_81.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_83.png)


    Epoch 9/10... Discriminator Loss: 0.2450... Generator Loss: 1.8977
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_85.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_87.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_89.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)



![png](readme_media/output_23_91.png)


    Epoch 10/10... Discriminator Loss: 0.2850... Generator Loss: 1.8280
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 1)
    Generator Output Layer shape: (?, 28, 28, 1)


### CelebA
Run your GANs on CelebA.  It will take around 20 minutes on the average GPU to run one epoch.  You can run the whole epoch or stop when it starts to generate realistic faces.


```python
batch_size = 128
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)
```

    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)
    Discriminator Layer 1 shape: (?, 14, 14, 64)
    Discriminator Layer 2 shape: (?, 7, 7, 128)
    Discriminator Layer 3 shape: (?, 4, 4, 256)
    Discriminator Layer 4 shape: (?, 2, 2, 512)
    Discriminator Layer 1 shape: (?, 14, 14, 64)
    Discriminator Layer 2 shape: (?, 7, 7, 128)
    Discriminator Layer 3 shape: (?, 4, 4, 256)
    Discriminator Layer 4 shape: (?, 2, 2, 512)
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_1.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_3.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_5.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_7.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_9.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_11.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_13.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_15.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_17.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_19.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_21.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_23.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_25.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_27.png)


    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)



![png](readme_media/output_25_29.png)


    Epoch 1/1... Discriminator Loss: 0.7133... Generator Loss: 1.6946
    Generator Layer 1 shape: (?, 2, 2, 512)
    Generator Layer 2 shape: (?, 4, 4, 256)
    Generator Layer 3 shape:: (?, 7, 7, 128)
    Generator Layer 4 shape: (?, 14, 14, 64)
    Generator Logits Layer shape: (?, 28, 28, 3)
    Generator Output Layer shape: (?, 28, 28, 3)


### Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
