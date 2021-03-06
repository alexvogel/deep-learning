
# Language Translation
In this project, you’re going to take a peek into the realm of neural network machine translation.  You’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
## Get the Data
Since translating the whole language of English to French will take lots of time to train, we have provided you with a small portion of the English corpus.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)
```

## Explore the Data
Play around with view_sentence_range to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 227
    Number of sentences: 137861
    Average number of words in a sentence: 13.225277634719028
    
    English sentences 0 to 10:
    new jersey is sometimes quiet during autumn , and it is snowy in april .
    the united states is usually chilly during july , and it is usually freezing in november .
    california is usually quiet during march , and it is usually hot in june .
    the united states is sometimes mild during june , and it is cold in september .
    your least liked fruit is the grape , but my least liked is the apple .
    his favorite fruit is the orange , but my favorite is the grape .
    paris is relaxing during december , but it is usually chilly in july .
    new jersey is busy during spring , and it is never hot in march .
    our least liked fruit is the lemon , but my least liked is the grape .
    the united states is sometimes busy during january , and it is sometimes warm in november .
    
    French sentences 0 to 10:
    new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
    les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
    california est généralement calme en mars , et il est généralement chaud en juin .
    les états-unis est parfois légère en juin , et il fait froid en septembre .
    votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .
    son fruit préféré est l'orange , mais mon préféré est le raisin .
    paris est relaxant en décembre , mais il est généralement froid en juillet .
    new jersey est occupé au printemps , et il est jamais chaude en mars .
    notre fruit est moins aimé le citron , mais mon moins aimé est le raisin .
    les états-unis est parfois occupé en janvier , et il est parfois chaud en novembre .


## Implement Preprocessing Function
### Text to Word Ids
As you did with other RNNs, you must turn the text into a number so the computer can understand it. In the function `text_to_ids()`, you'll turn `source_text` and `target_text` from words to ids.  However, you need to add the `<EOS>` word id at the end of each sentence from `target_text`.  This will help the neural network predict when the sentence should end.

You can get the `<EOS>` word id by doing:
```python
target_vocab_to_int['<EOS>']
```
You can get other word ids using `source_vocab_to_int` and `target_vocab_to_int`.


```python
def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    source_id_text = []
    target_id_text = []
    
    #print(source_text)
    
    source_lines = source_text.split('\n')
    target_lines = target_text.split('\n')
    
#    print(source_vocab_to_int)
    
    for line in source_lines:
#        print('line:',type(line), line)
        encoded_line = []
        for word in line.split():
#            print('type word', type(word), word)
            encoded_line.append(source_vocab_to_int[word])
#        print('encoded line', encoded_line)
        source_id_text.append(encoded_line)
    
    for line in target_lines:
#        print('line:',type(line), line)
        encoded_line = []
        for word in line.split():
#            print('type word', type(word), word)
            encoded_line.append(target_vocab_to_int[word])
        encoded_line.append(target_vocab_to_int['<EOS>'])
#        print('encoded line', encoded_line)
        target_id_text.append(encoded_line)
    
    print('length of source_id_text is', len(source_id_text))
    return source_id_text, target_id_text

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)
```

    length of source_id_text is 4
    Tests Passed


### Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
```

    length of source_id_text is 137861


# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
```

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
assert LooseVersion(tf.__version__) in [LooseVersion('1.0.0'), LooseVersion('1.0.1')], 'This project requires TensorFlow version 1.0  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.1


    /localhome/avoge/miniconda3/envs/tensorflow/lib/python3.6/site-packages/ipykernel_launcher.py:14: UserWarning: No GPU found. Please use a GPU to train your neural network.
      


## Build the Neural Network
You'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
- `model_inputs`
- `process_decoding_input`
- `encoding_layer`
- `decoding_layer_train`
- `decoding_layer_infer`
- `decoding_layer`
- `seq2seq_model`

### Input
Implement the `model_inputs()` function to create TF Placeholders for the Neural Network. It should create the following placeholders:

- Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
- Targets placeholder with rank 2.
- Learning rate placeholder with rank 0.
- Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.

Return the placeholders in the following the tuple (Input, Targets, Learing Rate, Keep Probability)


```python
def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    # TODO: Implement Function
    input = tf.placeholder(tf.int32, shape=[None, None], name='input')
    target = tf.placeholder(tf.int32, shape=[None, None], name='target')
    lr = tf.placeholder(tf.float32, name='lr')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return input, target, lr, keep_prob

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed


### Process Decoding Input
Implement `process_decoding_input` using TensorFlow to remove the last word id from each batch in `target_data` and concat the GO ID to the begining of each batch.


```python
def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for dencoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    
    # tf.strided_slice(input_, begin, end, strides=None, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, var=None, name=None)
    all_but_eos = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    
    # tf.concat(values, axis, name='concat')
    # tf.fill(dims, value, name=None)
    mod_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), all_but_eos], 1)    
    
    return mod_input

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_decoding_input(process_decoding_input)
```

    Tests Passed


### Encoding
Implement `encoding_layer()` to create a Encoder RNN layer using [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn).


```python
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """
    # TODO: Implement Function
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    
    cells = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
    # tf.nn.dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None)
    _, cells_state = tf.nn.dynamic_rnn(cells, rnn_inputs, dtype=tf.float32)

    return cells_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)
```

    Tests Passed


### Decoding - Training
Create training logits using [`tf.contrib.seq2seq.simple_decoder_fn_train()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_train) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).  Apply the `output_fn` to the [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) outputs.


```python
def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """
    tr_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
    tr_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, tr_decoder_fn, dec_embed_input, sequence_length, scope=decoding_scope)
    tr_logits =  output_fn(tr_pred)
    return tr_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)
```

    Tests Passed


### Decoding - Inference
Create inference logits using [`tf.contrib.seq2seq.simple_decoder_fn_inference()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_inference) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder). 


```python
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: The maximum allowed time steps to decode
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    inference_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn, encoder_state, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size)
    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, inference_decoder_fn, scope=decoding_scope)
    return inference_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)
```

    Tests Passed


### Build the Decoding Layer
Implement `decoding_layer()` to create a Decoder RNN layer.

- Create RNN cell for decoding using `rnn_size` and `num_layers`.
- Create the output fuction using [`lambda`](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) to transform it's input, logits, to class logits.
- Use the your `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)` function to get the training logits.
- Use your `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size, decoding_scope, output_fn, keep_prob)` function to get the inference logits.

Note: You'll need to use [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) to share variables between training and inference.


```python
def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """
    decoding_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    decoding_cell = tf.contrib.rnn.DropoutWrapper(decoding_cell, output_keep_prob=keep_prob)
    decoding_cell = tf.contrib.rnn.MultiRNNCell([decoding_cell] * num_layers)
    
    begin_with_id = target_vocab_to_int['<GO>']
    end_with_id = target_vocab_to_int['<EOS>']
    
    max_length = sequence_length - 1
    
    with tf.variable_scope("decoding") as decoding_scope:
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
        train_logits = decoding_layer_train(encoder_state, decoding_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)
        decoding_scope.reuse_variables()
        inference_logits = decoding_layer_infer(encoder_state, decoding_cell, dec_embeddings, begin_with_id, end_with_id, max_length, vocab_size, decoding_scope, output_fn, keep_prob)
    
    return train_logits, inference_logits

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)
```

    Tests Passed


### Build the Neural Network
Apply the functions you implemented above to:

- Apply embedding to the input data for the encoder.
- Encode the input using your `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob)`.
- Process target data using your `process_decoding_input(target_data, target_vocab_to_int, batch_size)` function.
- Apply embedding to the target data for the decoder.
- Decode the encoded input using your `decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)`.


```python
def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """
    rnn_inputs = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)
    encoder_state = encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob)
    target_data = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, dec_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, target_data)
    train_logits, infer_logits = decoding_layer(dec_embed_input, dec_embeddings, encoder_state, target_vocab_size, sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)    
    return train_logits, infer_logits

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)
```

    Tests Passed


## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `num_layers` to the number of layers.
- Set `encoding_embedding_size` to the size of the embedding for the encoder.
- Set `decoding_embedding_size` to the size of the embedding for the decoder.
- Set `learning_rate` to the learning rate.
- Set `keep_probability` to the Dropout keep probability


```python
# Number of Epochs
epochs = 10
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 128
# Number of Layers
num_layers = 2
# Embedding Size
# should be aprox. the size of unique words
encoding_embedding_size = 227
decoding_embedding_size = 227
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.75
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_source_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(max_source_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)
    
    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
```

### Train
Train the neural network on the preprocessed data. If you have a hard time getting a good loss, check the forms to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import time

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1]), (0,0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = helper.pad_sentence_batch(source_int_text[:batch_size])
valid_target = helper.pad_sentence_batch(target_int_text[:batch_size])

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            start_time = time.time()
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})
            
            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})
                
            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            end_time = time.time()
            print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                  .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/538 - Train Accuracy:  0.234, Validation Accuracy:  0.316, Loss:  5.902
    Epoch   0 Batch    1/538 - Train Accuracy:  0.231, Validation Accuracy:  0.316, Loss:  5.615
    Epoch   0 Batch    2/538 - Train Accuracy:  0.252, Validation Accuracy:  0.316, Loss:  5.466
    Epoch   0 Batch    3/538 - Train Accuracy:  0.229, Validation Accuracy:  0.316, Loss:  5.355
    Epoch   0 Batch    4/538 - Train Accuracy:  0.237, Validation Accuracy:  0.316, Loss:  5.188
    Epoch   0 Batch    5/538 - Train Accuracy:  0.275, Validation Accuracy:  0.327, Loss:  4.968
    Epoch   0 Batch    6/538 - Train Accuracy:  0.279, Validation Accuracy:  0.328, Loss:  4.797
    Epoch   0 Batch    7/538 - Train Accuracy:  0.276, Validation Accuracy:  0.345, Loss:  4.711
    Epoch   0 Batch    8/538 - Train Accuracy:  0.277, Validation Accuracy:  0.346, Loss:  4.577
    Epoch   0 Batch    9/538 - Train Accuracy:  0.278, Validation Accuracy:  0.346, Loss:  4.421
    Epoch   0 Batch   10/538 - Train Accuracy:  0.258, Validation Accuracy:  0.346, Loss:  4.376
    Epoch   0 Batch   11/538 - Train Accuracy:  0.271, Validation Accuracy:  0.346, Loss:  4.203
    Epoch   0 Batch   12/538 - Train Accuracy:  0.266, Validation Accuracy:  0.346, Loss:  4.124
    Epoch   0 Batch   13/538 - Train Accuracy:  0.318, Validation Accuracy:  0.346, Loss:  3.808
    Epoch   0 Batch   14/538 - Train Accuracy:  0.273, Validation Accuracy:  0.346, Loss:  3.891
    Epoch   0 Batch   15/538 - Train Accuracy:  0.320, Validation Accuracy:  0.347, Loss:  3.623
    Epoch   0 Batch   16/538 - Train Accuracy:  0.302, Validation Accuracy:  0.347, Loss:  3.600
    Epoch   0 Batch   17/538 - Train Accuracy:  0.285, Validation Accuracy:  0.347, Loss:  3.612
    Epoch   0 Batch   18/538 - Train Accuracy:  0.272, Validation Accuracy:  0.347, Loss:  3.609
    Epoch   0 Batch   19/538 - Train Accuracy:  0.272, Validation Accuracy:  0.347, Loss:  3.542
    Epoch   0 Batch   20/538 - Train Accuracy:  0.305, Validation Accuracy:  0.352, Loss:  3.352
    Epoch   0 Batch   21/538 - Train Accuracy:  0.231, Validation Accuracy:  0.354, Loss:  3.594
    Epoch   0 Batch   22/538 - Train Accuracy:  0.291, Validation Accuracy:  0.358, Loss:  3.360
    Epoch   0 Batch   23/538 - Train Accuracy:  0.298, Validation Accuracy:  0.362, Loss:  3.309
    Epoch   0 Batch   24/538 - Train Accuracy:  0.315, Validation Accuracy:  0.369, Loss:  3.241
    Epoch   0 Batch   25/538 - Train Accuracy:  0.304, Validation Accuracy:  0.368, Loss:  3.271
    Epoch   0 Batch   26/538 - Train Accuracy:  0.313, Validation Accuracy:  0.379, Loss:  3.237
    Epoch   0 Batch   27/538 - Train Accuracy:  0.323, Validation Accuracy:  0.387, Loss:  3.173
    Epoch   0 Batch   28/538 - Train Accuracy:  0.386, Validation Accuracy:  0.392, Loss:  2.901
    Epoch   0 Batch   29/538 - Train Accuracy:  0.348, Validation Accuracy:  0.394, Loss:  3.057
    Epoch   0 Batch   30/538 - Train Accuracy:  0.325, Validation Accuracy:  0.396, Loss:  3.142
    Epoch   0 Batch   31/538 - Train Accuracy:  0.359, Validation Accuracy:  0.398, Loss:  2.981
    Epoch   0 Batch   32/538 - Train Accuracy:  0.337, Validation Accuracy:  0.399, Loss:  3.038
    Epoch   0 Batch   33/538 - Train Accuracy:  0.357, Validation Accuracy:  0.399, Loss:  2.956
    Epoch   0 Batch   34/538 - Train Accuracy:  0.344, Validation Accuracy:  0.401, Loss:  3.026
    Epoch   0 Batch   35/538 - Train Accuracy:  0.327, Validation Accuracy:  0.403, Loss:  3.029
    Epoch   0 Batch   36/538 - Train Accuracy:  0.359, Validation Accuracy:  0.402, Loss:  2.897
    Epoch   0 Batch   37/538 - Train Accuracy:  0.337, Validation Accuracy:  0.403, Loss:  2.954
    Epoch   0 Batch   38/538 - Train Accuracy:  0.330, Validation Accuracy:  0.404, Loss:  3.006
    Epoch   0 Batch   39/538 - Train Accuracy:  0.334, Validation Accuracy:  0.404, Loss:  2.971
    Epoch   0 Batch   40/538 - Train Accuracy:  0.397, Validation Accuracy:  0.405, Loss:  2.695
    Epoch   0 Batch   41/538 - Train Accuracy:  0.343, Validation Accuracy:  0.411, Loss:  2.915
    Epoch   0 Batch   42/538 - Train Accuracy:  0.361, Validation Accuracy:  0.420, Loss:  2.871
    Epoch   0 Batch   43/538 - Train Accuracy:  0.360, Validation Accuracy:  0.418, Loss:  2.878
    Epoch   0 Batch   44/538 - Train Accuracy:  0.344, Validation Accuracy:  0.419, Loss:  2.905
    Epoch   0 Batch   45/538 - Train Accuracy:  0.394, Validation Accuracy:  0.422, Loss:  2.731
    Epoch   0 Batch   46/538 - Train Accuracy:  0.363, Validation Accuracy:  0.422, Loss:  2.834
    Epoch   0 Batch   47/538 - Train Accuracy:  0.391, Validation Accuracy:  0.423, Loss:  2.709
    Epoch   0 Batch   48/538 - Train Accuracy:  0.409, Validation Accuracy:  0.426, Loss:  2.660
    Epoch   0 Batch   49/538 - Train Accuracy:  0.351, Validation Accuracy:  0.426, Loss:  2.843
    Epoch   0 Batch   50/538 - Train Accuracy:  0.374, Validation Accuracy:  0.434, Loss:  2.752
    Epoch   0 Batch   51/538 - Train Accuracy:  0.327, Validation Accuracy:  0.441, Loss:  2.968
    Epoch   0 Batch   52/538 - Train Accuracy:  0.394, Validation Accuracy:  0.445, Loss:  2.733
    Epoch   0 Batch   53/538 - Train Accuracy:  0.441, Validation Accuracy:  0.454, Loss:  2.538
    Epoch   0 Batch   54/538 - Train Accuracy:  0.413, Validation Accuracy:  0.459, Loss:  2.695
    Epoch   0 Batch   55/538 - Train Accuracy:  0.391, Validation Accuracy:  0.459, Loss:  2.748
    Epoch   0 Batch   56/538 - Train Accuracy:  0.419, Validation Accuracy:  0.456, Loss:  2.622
    Epoch   0 Batch   57/538 - Train Accuracy:  0.382, Validation Accuracy:  0.456, Loss:  2.775
    Epoch   0 Batch   58/538 - Train Accuracy:  0.375, Validation Accuracy:  0.454, Loss:  2.763
    Epoch   0 Batch   59/538 - Train Accuracy:  0.384, Validation Accuracy:  0.454, Loss:  2.714
    Epoch   0 Batch   60/538 - Train Accuracy:  0.401, Validation Accuracy:  0.461, Loss:  2.679
    Epoch   0 Batch   61/538 - Train Accuracy:  0.409, Validation Accuracy:  0.463, Loss:  2.649
    Epoch   0 Batch   62/538 - Train Accuracy:  0.415, Validation Accuracy:  0.463, Loss:  2.609
    Epoch   0 Batch   63/538 - Train Accuracy:  0.427, Validation Accuracy:  0.456, Loss:  2.507
    Epoch   0 Batch   64/538 - Train Accuracy:  0.431, Validation Accuracy:  0.462, Loss:  2.515
    Epoch   0 Batch   65/538 - Train Accuracy:  0.390, Validation Accuracy:  0.459, Loss:  2.656
    Epoch   0 Batch   66/538 - Train Accuracy:  0.430, Validation Accuracy:  0.468, Loss:  2.515
    Epoch   0 Batch   67/538 - Train Accuracy:  0.402, Validation Accuracy:  0.455, Loss:  2.544
    Epoch   0 Batch   68/538 - Train Accuracy:  0.443, Validation Accuracy:  0.469, Loss:  2.483
    Epoch   0 Batch   69/538 - Train Accuracy:  0.417, Validation Accuracy:  0.469, Loss:  2.577
    Epoch   0 Batch   70/538 - Train Accuracy:  0.439, Validation Accuracy:  0.472, Loss:  2.469
    Epoch   0 Batch   71/538 - Train Accuracy:  0.412, Validation Accuracy:  0.472, Loss:  2.558
    Epoch   0 Batch   72/538 - Train Accuracy:  0.447, Validation Accuracy:  0.474, Loss:  2.454
    Epoch   0 Batch   73/538 - Train Accuracy:  0.420, Validation Accuracy:  0.473, Loss:  2.550
    Epoch   0 Batch   74/538 - Train Accuracy:  0.452, Validation Accuracy:  0.472, Loss:  2.432
    Epoch   0 Batch   75/538 - Train Accuracy:  0.454, Validation Accuracy:  0.474, Loss:  2.398
    Epoch   0 Batch   76/538 - Train Accuracy:  0.416, Validation Accuracy:  0.476, Loss:  2.549
    Epoch   0 Batch   77/538 - Train Accuracy:  0.424, Validation Accuracy:  0.480, Loss:  2.520
    Epoch   0 Batch   78/538 - Train Accuracy:  0.456, Validation Accuracy:  0.483, Loss:  2.397
    Epoch   0 Batch   79/538 - Train Accuracy:  0.461, Validation Accuracy:  0.487, Loss:  2.376
    Epoch   0 Batch   80/538 - Train Accuracy:  0.434, Validation Accuracy:  0.491, Loss:  2.506
    Epoch   0 Batch   81/538 - Train Accuracy:  0.445, Validation Accuracy:  0.497, Loss:  2.505
    Epoch   0 Batch   82/538 - Train Accuracy:  0.457, Validation Accuracy:  0.495, Loss:  2.441
    Epoch   0 Batch   83/538 - Train Accuracy:  0.452, Validation Accuracy:  0.495, Loss:  2.469
    Epoch   0 Batch   84/538 - Train Accuracy:  0.464, Validation Accuracy:  0.493, Loss:  2.373
    Epoch   0 Batch   85/538 - Train Accuracy:  0.486, Validation Accuracy:  0.489, Loss:  2.252
    Epoch   0 Batch   86/538 - Train Accuracy:  0.446, Validation Accuracy:  0.494, Loss:  2.449
    Epoch   0 Batch   87/538 - Train Accuracy:  0.455, Validation Accuracy:  0.498, Loss:  2.404
    Epoch   0 Batch   88/538 - Train Accuracy:  0.469, Validation Accuracy:  0.504, Loss:  2.378
    Epoch   0 Batch   89/538 - Train Accuracy:  0.477, Validation Accuracy:  0.507, Loss:  2.381
    Epoch   0 Batch   90/538 - Train Accuracy:  0.485, Validation Accuracy:  0.505, Loss:  2.301
    Epoch   0 Batch   91/538 - Train Accuracy:  0.450, Validation Accuracy:  0.502, Loss:  2.425
    Epoch   0 Batch   92/538 - Train Accuracy:  0.470, Validation Accuracy:  0.508, Loss:  2.373
    Epoch   0 Batch   93/538 - Train Accuracy:  0.465, Validation Accuracy:  0.515, Loss:  2.398
    Epoch   0 Batch   94/538 - Train Accuracy:  0.474, Validation Accuracy:  0.514, Loss:  2.400
    Epoch   0 Batch   95/538 - Train Accuracy:  0.510, Validation Accuracy:  0.508, Loss:  2.171
    Epoch   0 Batch   96/538 - Train Accuracy:  0.499, Validation Accuracy:  0.512, Loss:  2.212
    Epoch   0 Batch   97/538 - Train Accuracy:  0.462, Validation Accuracy:  0.503, Loss:  2.348
    Epoch   0 Batch   98/538 - Train Accuracy:  0.499, Validation Accuracy:  0.507, Loss:  2.184
    Epoch   0 Batch   99/538 - Train Accuracy:  0.457, Validation Accuracy:  0.506, Loss:  2.343
    Epoch   0 Batch  100/538 - Train Accuracy:  0.492, Validation Accuracy:  0.519, Loss:  2.304
    Epoch   0 Batch  101/538 - Train Accuracy:  0.472, Validation Accuracy:  0.512, Loss:  2.289
    Epoch   0 Batch  102/538 - Train Accuracy:  0.472, Validation Accuracy:  0.517, Loss:  2.348
    Epoch   0 Batch  103/538 - Train Accuracy:  0.481, Validation Accuracy:  0.516, Loss:  2.266
    Epoch   0 Batch  104/538 - Train Accuracy:  0.494, Validation Accuracy:  0.512, Loss:  2.216
    Epoch   0 Batch  105/538 - Train Accuracy:  0.497, Validation Accuracy:  0.513, Loss:  2.193
    Epoch   0 Batch  106/538 - Train Accuracy:  0.476, Validation Accuracy:  0.515, Loss:  2.281
    Epoch   0 Batch  107/538 - Train Accuracy:  0.467, Validation Accuracy:  0.522, Loss:  2.302
    Epoch   0 Batch  108/538 - Train Accuracy:  0.498, Validation Accuracy:  0.524, Loss:  2.235
    Epoch   0 Batch  109/538 - Train Accuracy:  0.493, Validation Accuracy:  0.526, Loss:  2.238
    Epoch   0 Batch  110/538 - Train Accuracy:  0.486, Validation Accuracy:  0.528, Loss:  2.269
    Epoch   0 Batch  111/538 - Train Accuracy:  0.520, Validation Accuracy:  0.525, Loss:  2.139
    Epoch   0 Batch  112/538 - Train Accuracy:  0.482, Validation Accuracy:  0.524, Loss:  2.261
    Epoch   0 Batch  113/538 - Train Accuracy:  0.480, Validation Accuracy:  0.524, Loss:  2.247
    Epoch   0 Batch  114/538 - Train Accuracy:  0.515, Validation Accuracy:  0.529, Loss:  2.105
    Epoch   0 Batch  115/538 - Train Accuracy:  0.499, Validation Accuracy:  0.533, Loss:  2.209
    Epoch   0 Batch  116/538 - Train Accuracy:  0.520, Validation Accuracy:  0.539, Loss:  2.137
    Epoch   0 Batch  117/538 - Train Accuracy:  0.517, Validation Accuracy:  0.539, Loss:  2.112
    Epoch   0 Batch  118/538 - Train Accuracy:  0.517, Validation Accuracy:  0.541, Loss:  2.122
    Epoch   0 Batch  119/538 - Train Accuracy:  0.528, Validation Accuracy:  0.536, Loss:  2.073
    Epoch   0 Batch  120/538 - Train Accuracy:  0.487, Validation Accuracy:  0.534, Loss:  2.179
    Epoch   0 Batch  121/538 - Train Accuracy:  0.523, Validation Accuracy:  0.538, Loss:  2.081
    Epoch   0 Batch  122/538 - Train Accuracy:  0.513, Validation Accuracy:  0.537, Loss:  2.103
    Epoch   0 Batch  123/538 - Train Accuracy:  0.530, Validation Accuracy:  0.542, Loss:  2.051
    Epoch   0 Batch  124/538 - Train Accuracy:  0.539, Validation Accuracy:  0.544, Loss:  2.008
    Epoch   0 Batch  125/538 - Train Accuracy:  0.523, Validation Accuracy:  0.544, Loss:  2.078
    Epoch   0 Batch  126/538 - Train Accuracy:  0.540, Validation Accuracy:  0.552, Loss:  2.008
    Epoch   0 Batch  127/538 - Train Accuracy:  0.506, Validation Accuracy:  0.550, Loss:  2.173
    Epoch   0 Batch  128/538 - Train Accuracy:  0.534, Validation Accuracy:  0.555, Loss:  2.044
    Epoch   0 Batch  129/538 - Train Accuracy:  0.529, Validation Accuracy:  0.552, Loss:  2.041
    Epoch   0 Batch  130/538 - Train Accuracy:  0.525, Validation Accuracy:  0.551, Loss:  2.065
    Epoch   0 Batch  131/538 - Train Accuracy:  0.514, Validation Accuracy:  0.554, Loss:  2.122
    Epoch   0 Batch  132/538 - Train Accuracy:  0.532, Validation Accuracy:  0.550, Loss:  2.025
    Epoch   0 Batch  133/538 - Train Accuracy:  0.556, Validation Accuracy:  0.556, Loss:  1.952
    Epoch   0 Batch  134/538 - Train Accuracy:  0.497, Validation Accuracy:  0.544, Loss:  2.164
    Epoch   0 Batch  135/538 - Train Accuracy:  0.534, Validation Accuracy:  0.553, Loss:  2.031
    Epoch   0 Batch  136/538 - Train Accuracy:  0.522, Validation Accuracy:  0.556, Loss:  2.039
    Epoch   0 Batch  137/538 - Train Accuracy:  0.531, Validation Accuracy:  0.552, Loss:  2.029
    Epoch   0 Batch  138/538 - Train Accuracy:  0.535, Validation Accuracy:  0.553, Loss:  2.018
    Epoch   0 Batch  139/538 - Train Accuracy:  0.501, Validation Accuracy:  0.546, Loss:  2.151
    Epoch   0 Batch  140/538 - Train Accuracy:  0.520, Validation Accuracy:  0.554, Loss:  2.138
    Epoch   0 Batch  141/538 - Train Accuracy:  0.512, Validation Accuracy:  0.553, Loss:  2.101
    Epoch   0 Batch  142/538 - Train Accuracy:  0.557, Validation Accuracy:  0.559, Loss:  1.938
    Epoch   0 Batch  143/538 - Train Accuracy:  0.501, Validation Accuracy:  0.542, Loss:  2.098
    Epoch   0 Batch  144/538 - Train Accuracy:  0.529, Validation Accuracy:  0.563, Loss:  2.070
    Epoch   0 Batch  145/538 - Train Accuracy:  0.540, Validation Accuracy:  0.546, Loss:  1.987
    Epoch   0 Batch  146/538 - Train Accuracy:  0.547, Validation Accuracy:  0.553, Loss:  1.927
    Epoch   0 Batch  147/538 - Train Accuracy:  0.545, Validation Accuracy:  0.556, Loss:  1.929
    Epoch   0 Batch  148/538 - Train Accuracy:  0.516, Validation Accuracy:  0.563, Loss:  2.108
    Epoch   0 Batch  149/538 - Train Accuracy:  0.526, Validation Accuracy:  0.553, Loss:  1.991
    Epoch   0 Batch  150/538 - Train Accuracy:  0.531, Validation Accuracy:  0.559, Loss:  2.034
    Epoch   0 Batch  151/538 - Train Accuracy:  0.541, Validation Accuracy:  0.563, Loss:  1.940
    Epoch   0 Batch  152/538 - Train Accuracy:  0.548, Validation Accuracy:  0.564, Loss:  1.937
    Epoch   0 Batch  153/538 - Train Accuracy:  0.523, Validation Accuracy:  0.552, Loss:  2.026
    Epoch   0 Batch  154/538 - Train Accuracy:  0.546, Validation Accuracy:  0.559, Loss:  1.963
    Epoch   0 Batch  155/538 - Train Accuracy:  0.536, Validation Accuracy:  0.557, Loss:  1.926
    Epoch   0 Batch  156/538 - Train Accuracy:  0.522, Validation Accuracy:  0.562, Loss:  2.005
    Epoch   0 Batch  157/538 - Train Accuracy:  0.548, Validation Accuracy:  0.563, Loss:  1.925
    Epoch   0 Batch  158/538 - Train Accuracy:  0.517, Validation Accuracy:  0.556, Loss:  2.027
    Epoch   0 Batch  159/538 - Train Accuracy:  0.534, Validation Accuracy:  0.568, Loss:  2.011
    Epoch   0 Batch  160/538 - Train Accuracy:  0.535, Validation Accuracy:  0.567, Loss:  1.918
    Epoch   0 Batch  161/538 - Train Accuracy:  0.543, Validation Accuracy:  0.574, Loss:  1.962
    Epoch   0 Batch  162/538 - Train Accuracy:  0.562, Validation Accuracy:  0.568, Loss:  1.855
    Epoch   0 Batch  163/538 - Train Accuracy:  0.548, Validation Accuracy:  0.574, Loss:  1.933
    Epoch   0 Batch  164/538 - Train Accuracy:  0.527, Validation Accuracy:  0.577, Loss:  1.998
    Epoch   0 Batch  165/538 - Train Accuracy:  0.572, Validation Accuracy:  0.580, Loss:  1.834
    Epoch   0 Batch  166/538 - Train Accuracy:  0.544, Validation Accuracy:  0.573, Loss:  1.912
    Epoch   0 Batch  167/538 - Train Accuracy:  0.570, Validation Accuracy:  0.567, Loss:  1.818
    Epoch   0 Batch  168/538 - Train Accuracy:  0.538, Validation Accuracy:  0.577, Loss:  1.985
    Epoch   0 Batch  169/538 - Train Accuracy:  0.538, Validation Accuracy:  0.575, Loss:  1.912
    Epoch   0 Batch  170/538 - Train Accuracy:  0.551, Validation Accuracy:  0.579, Loss:  1.875
    Epoch   0 Batch  171/538 - Train Accuracy:  0.521, Validation Accuracy:  0.567, Loss:  1.987
    Epoch   0 Batch  172/538 - Train Accuracy:  0.555, Validation Accuracy:  0.576, Loss:  1.867
    Epoch   0 Batch  173/538 - Train Accuracy:  0.561, Validation Accuracy:  0.579, Loss:  1.854
    Epoch   0 Batch  174/538 - Train Accuracy:  0.530, Validation Accuracy:  0.584, Loss:  1.980
    Epoch   0 Batch  175/538 - Train Accuracy:  0.528, Validation Accuracy:  0.579, Loss:  1.976
    Epoch   0 Batch  176/538 - Train Accuracy:  0.547, Validation Accuracy:  0.585, Loss:  1.957
    Epoch   0 Batch  177/538 - Train Accuracy:  0.560, Validation Accuracy:  0.590, Loss:  1.863
    Epoch   0 Batch  178/538 - Train Accuracy:  0.572, Validation Accuracy:  0.589, Loss:  1.798
    Epoch   0 Batch  179/538 - Train Accuracy:  0.554, Validation Accuracy:  0.592, Loss:  1.886
    Epoch   0 Batch  180/538 - Train Accuracy:  0.574, Validation Accuracy:  0.593, Loss:  1.815
    Epoch   0 Batch  181/538 - Train Accuracy:  0.535, Validation Accuracy:  0.591, Loss:  1.940
    Epoch   0 Batch  182/538 - Train Accuracy:  0.546, Validation Accuracy:  0.586, Loss:  1.920
    Epoch   0 Batch  183/538 - Train Accuracy:  0.572, Validation Accuracy:  0.585, Loss:  1.789
    Epoch   0 Batch  184/538 - Train Accuracy:  0.574, Validation Accuracy:  0.584, Loss:  1.793
    Epoch   0 Batch  185/538 - Train Accuracy:  0.571, Validation Accuracy:  0.590, Loss:  1.836
    Epoch   0 Batch  186/538 - Train Accuracy:  0.573, Validation Accuracy:  0.593, Loss:  1.825
    Epoch   0 Batch  187/538 - Train Accuracy:  0.589, Validation Accuracy:  0.597, Loss:  1.746
    Epoch   0 Batch  188/538 - Train Accuracy:  0.547, Validation Accuracy:  0.594, Loss:  1.875
    Epoch   0 Batch  189/538 - Train Accuracy:  0.562, Validation Accuracy:  0.588, Loss:  1.858
    Epoch   0 Batch  190/538 - Train Accuracy:  0.576, Validation Accuracy:  0.588, Loss:  1.807
    Epoch   0 Batch  191/538 - Train Accuracy:  0.578, Validation Accuracy:  0.589, Loss:  1.766
    Epoch   0 Batch  192/538 - Train Accuracy:  0.572, Validation Accuracy:  0.594, Loss:  1.785
    Epoch   0 Batch  193/538 - Train Accuracy:  0.584, Validation Accuracy:  0.595, Loss:  1.743
    Epoch   0 Batch  194/538 - Train Accuracy:  0.553, Validation Accuracy:  0.597, Loss:  1.880
    Epoch   0 Batch  195/538 - Train Accuracy:  0.588, Validation Accuracy:  0.603, Loss:  1.746
    Epoch   0 Batch  196/538 - Train Accuracy:  0.583, Validation Accuracy:  0.606, Loss:  1.801
    Epoch   0 Batch  197/538 - Train Accuracy:  0.597, Validation Accuracy:  0.601, Loss:  1.733
    Epoch   0 Batch  198/538 - Train Accuracy:  0.596, Validation Accuracy:  0.597, Loss:  1.731
    Epoch   0 Batch  199/538 - Train Accuracy:  0.554, Validation Accuracy:  0.598, Loss:  1.870
    Epoch   0 Batch  200/538 - Train Accuracy:  0.561, Validation Accuracy:  0.597, Loss:  1.815
    Epoch   0 Batch  201/538 - Train Accuracy:  0.593, Validation Accuracy:  0.593, Loss:  1.717
    Epoch   0 Batch  202/538 - Train Accuracy:  0.563, Validation Accuracy:  0.593, Loss:  1.840
    Epoch   0 Batch  203/538 - Train Accuracy:  0.556, Validation Accuracy:  0.598, Loss:  1.847
    Epoch   0 Batch  204/538 - Train Accuracy:  0.564, Validation Accuracy:  0.602, Loss:  1.817
    Epoch   0 Batch  205/538 - Train Accuracy:  0.595, Validation Accuracy:  0.605, Loss:  1.706
    Epoch   0 Batch  206/538 - Train Accuracy:  0.563, Validation Accuracy:  0.604, Loss:  1.831
    Epoch   0 Batch  207/538 - Train Accuracy:  0.593, Validation Accuracy:  0.604, Loss:  1.691
    Epoch   0 Batch  208/538 - Train Accuracy:  0.582, Validation Accuracy:  0.611, Loss:  1.772
    Epoch   0 Batch  209/538 - Train Accuracy:  0.572, Validation Accuracy:  0.604, Loss:  1.800
    Epoch   0 Batch  210/538 - Train Accuracy:  0.576, Validation Accuracy:  0.604, Loss:  1.747
    Epoch   0 Batch  211/538 - Train Accuracy:  0.562, Validation Accuracy:  0.611, Loss:  1.842
    Epoch   0 Batch  212/538 - Train Accuracy:  0.582, Validation Accuracy:  0.605, Loss:  1.731
    Epoch   0 Batch  213/538 - Train Accuracy:  0.574, Validation Accuracy:  0.603, Loss:  1.750
    Epoch   0 Batch  214/538 - Train Accuracy:  0.572, Validation Accuracy:  0.601, Loss:  1.771
    Epoch   0 Batch  215/538 - Train Accuracy:  0.566, Validation Accuracy:  0.610, Loss:  1.790
    Epoch   0 Batch  216/538 - Train Accuracy:  0.560, Validation Accuracy:  0.607, Loss:  1.820
    Epoch   0 Batch  217/538 - Train Accuracy:  0.594, Validation Accuracy:  0.606, Loss:  1.703
    Epoch   0 Batch  218/538 - Train Accuracy:  0.572, Validation Accuracy:  0.610, Loss:  1.787
    Epoch   0 Batch  219/538 - Train Accuracy:  0.576, Validation Accuracy:  0.610, Loss:  1.787
    Epoch   0 Batch  220/538 - Train Accuracy:  0.582, Validation Accuracy:  0.604, Loss:  1.699
    Epoch   0 Batch  221/538 - Train Accuracy:  0.596, Validation Accuracy:  0.611, Loss:  1.682
    Epoch   0 Batch  222/538 - Train Accuracy:  0.596, Validation Accuracy:  0.611, Loss:  1.664
    Epoch   0 Batch  223/538 - Train Accuracy:  0.574, Validation Accuracy:  0.611, Loss:  1.768
    Epoch   0 Batch  224/538 - Train Accuracy:  0.572, Validation Accuracy:  0.610, Loss:  1.786
    Epoch   0 Batch  225/538 - Train Accuracy:  0.611, Validation Accuracy:  0.613, Loss:  1.650
    Epoch   0 Batch  226/538 - Train Accuracy:  0.594, Validation Accuracy:  0.613, Loss:  1.682
    Epoch   0 Batch  227/538 - Train Accuracy:  0.623, Validation Accuracy:  0.609, Loss:  1.600
    Epoch   0 Batch  228/538 - Train Accuracy:  0.584, Validation Accuracy:  0.612, Loss:  1.688
    Epoch   0 Batch  229/538 - Train Accuracy:  0.594, Validation Accuracy:  0.612, Loss:  1.662
    Epoch   0 Batch  230/538 - Train Accuracy:  0.580, Validation Accuracy:  0.615, Loss:  1.722
    Epoch   0 Batch  231/538 - Train Accuracy:  0.589, Validation Accuracy:  0.608, Loss:  1.711
    Epoch   0 Batch  232/538 - Train Accuracy:  0.580, Validation Accuracy:  0.611, Loss:  1.746
    Epoch   0 Batch  233/538 - Train Accuracy:  0.601, Validation Accuracy:  0.609, Loss:  1.629
    Epoch   0 Batch  234/538 - Train Accuracy:  0.573, Validation Accuracy:  0.604, Loss:  1.725
    Epoch   0 Batch  235/538 - Train Accuracy:  0.595, Validation Accuracy:  0.612, Loss:  1.656
    Epoch   0 Batch  236/538 - Train Accuracy:  0.564, Validation Accuracy:  0.612, Loss:  1.734
    Epoch   0 Batch  237/538 - Train Accuracy:  0.601, Validation Accuracy:  0.618, Loss:  1.651
    Epoch   0 Batch  238/538 - Train Accuracy:  0.610, Validation Accuracy:  0.613, Loss:  1.597
    Epoch   0 Batch  239/538 - Train Accuracy:  0.582, Validation Accuracy:  0.615, Loss:  1.715
    Epoch   0 Batch  240/538 - Train Accuracy:  0.584, Validation Accuracy:  0.619, Loss:  1.687
    Epoch   0 Batch  241/538 - Train Accuracy:  0.573, Validation Accuracy:  0.620, Loss:  1.718
    Epoch   0 Batch  242/538 - Train Accuracy:  0.595, Validation Accuracy:  0.615, Loss:  1.664
    Epoch   0 Batch  243/538 - Train Accuracy:  0.574, Validation Accuracy:  0.616, Loss:  1.726
    Epoch   0 Batch  244/538 - Train Accuracy:  0.590, Validation Accuracy:  0.618, Loss:  1.643
    Epoch   0 Batch  245/538 - Train Accuracy:  0.586, Validation Accuracy:  0.617, Loss:  1.690
    Epoch   0 Batch  246/538 - Train Accuracy:  0.610, Validation Accuracy:  0.624, Loss:  1.584
    Epoch   0 Batch  247/538 - Train Accuracy:  0.569, Validation Accuracy:  0.614, Loss:  1.708
    Epoch   0 Batch  248/538 - Train Accuracy:  0.591, Validation Accuracy:  0.616, Loss:  1.666
    Epoch   0 Batch  249/538 - Train Accuracy:  0.605, Validation Accuracy:  0.623, Loss:  1.609
    Epoch   0 Batch  250/538 - Train Accuracy:  0.604, Validation Accuracy:  0.628, Loss:  1.665
    Epoch   0 Batch  251/538 - Train Accuracy:  0.595, Validation Accuracy:  0.628, Loss:  1.655
    Epoch   0 Batch  252/538 - Train Accuracy:  0.615, Validation Accuracy:  0.627, Loss:  1.568
    Epoch   0 Batch  253/538 - Train Accuracy:  0.605, Validation Accuracy:  0.619, Loss:  1.584
    Epoch   0 Batch  254/538 - Train Accuracy:  0.586, Validation Accuracy:  0.620, Loss:  1.639
    Epoch   0 Batch  255/538 - Train Accuracy:  0.596, Validation Accuracy:  0.624, Loss:  1.642
    Epoch   0 Batch  256/538 - Train Accuracy:  0.582, Validation Accuracy:  0.620, Loss:  1.671
    Epoch   0 Batch  257/538 - Train Accuracy:  0.612, Validation Accuracy:  0.624, Loss:  1.569
    Epoch   0 Batch  258/538 - Train Accuracy:  0.623, Validation Accuracy:  0.626, Loss:  1.561
    Epoch   0 Batch  259/538 - Train Accuracy:  0.607, Validation Accuracy:  0.626, Loss:  1.564
    Epoch   0 Batch  260/538 - Train Accuracy:  0.592, Validation Accuracy:  0.628, Loss:  1.622
    Epoch   0 Batch  261/538 - Train Accuracy:  0.586, Validation Accuracy:  0.628, Loss:  1.645
    Epoch   0 Batch  262/538 - Train Accuracy:  0.580, Validation Accuracy:  0.625, Loss:  1.677
    Epoch   0 Batch  263/538 - Train Accuracy:  0.598, Validation Accuracy:  0.627, Loss:  1.619
    Epoch   0 Batch  264/538 - Train Accuracy:  0.594, Validation Accuracy:  0.623, Loss:  1.623
    Epoch   0 Batch  265/538 - Train Accuracy:  0.574, Validation Accuracy:  0.620, Loss:  1.676
    Epoch   0 Batch  266/538 - Train Accuracy:  0.611, Validation Accuracy:  0.617, Loss:  1.570
    Epoch   0 Batch  267/538 - Train Accuracy:  0.592, Validation Accuracy:  0.619, Loss:  1.599
    Epoch   0 Batch  268/538 - Train Accuracy:  0.614, Validation Accuracy:  0.624, Loss:  1.529
    Epoch   0 Batch  269/538 - Train Accuracy:  0.594, Validation Accuracy:  0.622, Loss:  1.589
    Epoch   0 Batch  270/538 - Train Accuracy:  0.592, Validation Accuracy:  0.621, Loss:  1.612
    Epoch   0 Batch  271/538 - Train Accuracy:  0.601, Validation Accuracy:  0.627, Loss:  1.609
    Epoch   0 Batch  272/538 - Train Accuracy:  0.579, Validation Accuracy:  0.624, Loss:  1.684
    Epoch   0 Batch  273/538 - Train Accuracy:  0.601, Validation Accuracy:  0.622, Loss:  1.569
    Epoch   0 Batch  274/538 - Train Accuracy:  0.564, Validation Accuracy:  0.619, Loss:  1.689
    Epoch   0 Batch  275/538 - Train Accuracy:  0.585, Validation Accuracy:  0.623, Loss:  1.621
    Epoch   0 Batch  276/538 - Train Accuracy:  0.605, Validation Accuracy:  0.626, Loss:  1.586
    Epoch   0 Batch  277/538 - Train Accuracy:  0.605, Validation Accuracy:  0.627, Loss:  1.574
    Epoch   0 Batch  278/538 - Train Accuracy:  0.603, Validation Accuracy:  0.618, Loss:  1.566
    Epoch   0 Batch  279/538 - Train Accuracy:  0.599, Validation Accuracy:  0.624, Loss:  1.572
    Epoch   0 Batch  280/538 - Train Accuracy:  0.607, Validation Accuracy:  0.615, Loss:  1.486
    Epoch   0 Batch  281/538 - Train Accuracy:  0.590, Validation Accuracy:  0.629, Loss:  1.616
    Epoch   0 Batch  282/538 - Train Accuracy:  0.608, Validation Accuracy:  0.625, Loss:  1.528
    Epoch   0 Batch  283/538 - Train Accuracy:  0.612, Validation Accuracy:  0.628, Loss:  1.534
    Epoch   0 Batch  284/538 - Train Accuracy:  0.605, Validation Accuracy:  0.623, Loss:  1.550
    Epoch   0 Batch  285/538 - Train Accuracy:  0.604, Validation Accuracy:  0.619, Loss:  1.489
    Epoch   0 Batch  286/538 - Train Accuracy:  0.595, Validation Accuracy:  0.627, Loss:  1.575
    Epoch   0 Batch  287/538 - Train Accuracy:  0.626, Validation Accuracy:  0.623, Loss:  1.467
    Epoch   0 Batch  288/538 - Train Accuracy:  0.596, Validation Accuracy:  0.626, Loss:  1.566
    Epoch   0 Batch  289/538 - Train Accuracy:  0.611, Validation Accuracy:  0.626, Loss:  1.462
    Epoch   0 Batch  290/538 - Train Accuracy:  0.600, Validation Accuracy:  0.630, Loss:  1.549
    Epoch   0 Batch  291/538 - Train Accuracy:  0.607, Validation Accuracy:  0.629, Loss:  1.507
    Epoch   0 Batch  292/538 - Train Accuracy:  0.622, Validation Accuracy:  0.628, Loss:  1.461
    Epoch   0 Batch  293/538 - Train Accuracy:  0.610, Validation Accuracy:  0.626, Loss:  1.481
    Epoch   0 Batch  294/538 - Train Accuracy:  0.575, Validation Accuracy:  0.623, Loss:  1.610
    Epoch   0 Batch  295/538 - Train Accuracy:  0.631, Validation Accuracy:  0.628, Loss:  1.420
    Epoch   0 Batch  296/538 - Train Accuracy:  0.612, Validation Accuracy:  0.631, Loss:  1.477
    Epoch   0 Batch  297/538 - Train Accuracy:  0.601, Validation Accuracy:  0.634, Loss:  1.573
    Epoch   0 Batch  298/538 - Train Accuracy:  0.608, Validation Accuracy:  0.635, Loss:  1.515
    Epoch   0 Batch  299/538 - Train Accuracy:  0.611, Validation Accuracy:  0.629, Loss:  1.502
    Epoch   0 Batch  300/538 - Train Accuracy:  0.615, Validation Accuracy:  0.625, Loss:  1.439
    Epoch   0 Batch  301/538 - Train Accuracy:  0.587, Validation Accuracy:  0.624, Loss:  1.507
    Epoch   0 Batch  302/538 - Train Accuracy:  0.621, Validation Accuracy:  0.625, Loss:  1.433
    Epoch   0 Batch  303/538 - Train Accuracy:  0.644, Validation Accuracy:  0.631, Loss:  1.378
    Epoch   0 Batch  304/538 - Train Accuracy:  0.602, Validation Accuracy:  0.635, Loss:  1.522
    Epoch   0 Batch  305/538 - Train Accuracy:  0.622, Validation Accuracy:  0.632, Loss:  1.468
    Epoch   0 Batch  306/538 - Train Accuracy:  0.621, Validation Accuracy:  0.632, Loss:  1.453
    Epoch   0 Batch  307/538 - Train Accuracy:  0.602, Validation Accuracy:  0.628, Loss:  1.481
    Epoch   0 Batch  308/538 - Train Accuracy:  0.621, Validation Accuracy:  0.629, Loss:  1.449
    Epoch   0 Batch  309/538 - Train Accuracy:  0.593, Validation Accuracy:  0.628, Loss:  1.499
    Epoch   0 Batch  310/538 - Train Accuracy:  0.609, Validation Accuracy:  0.629, Loss:  1.471
    Epoch   0 Batch  311/538 - Train Accuracy:  0.624, Validation Accuracy:  0.631, Loss:  1.430
    Epoch   0 Batch  312/538 - Train Accuracy:  0.637, Validation Accuracy:  0.633, Loss:  1.360
    Epoch   0 Batch  313/538 - Train Accuracy:  0.598, Validation Accuracy:  0.634, Loss:  1.498
    Epoch   0 Batch  314/538 - Train Accuracy:  0.610, Validation Accuracy:  0.637, Loss:  1.448
    Epoch   0 Batch  315/538 - Train Accuracy:  0.599, Validation Accuracy:  0.640, Loss:  1.475
    Epoch   0 Batch  316/538 - Train Accuracy:  0.625, Validation Accuracy:  0.640, Loss:  1.427
    Epoch   0 Batch  317/538 - Train Accuracy:  0.616, Validation Accuracy:  0.640, Loss:  1.427
    Epoch   0 Batch  318/538 - Train Accuracy:  0.616, Validation Accuracy:  0.633, Loss:  1.429
    Epoch   0 Batch  319/538 - Train Accuracy:  0.610, Validation Accuracy:  0.632, Loss:  1.437
    Epoch   0 Batch  320/538 - Train Accuracy:  0.618, Validation Accuracy:  0.631, Loss:  1.435
    Epoch   0 Batch  321/538 - Train Accuracy:  0.629, Validation Accuracy:  0.643, Loss:  1.397
    Epoch   0 Batch  322/538 - Train Accuracy:  0.616, Validation Accuracy:  0.642, Loss:  1.438
    Epoch   0 Batch  323/538 - Train Accuracy:  0.626, Validation Accuracy:  0.643, Loss:  1.412
    Epoch   0 Batch  324/538 - Train Accuracy:  0.595, Validation Accuracy:  0.635, Loss:  1.499
    Epoch   0 Batch  325/538 - Train Accuracy:  0.617, Validation Accuracy:  0.643, Loss:  1.424
    Epoch   0 Batch  326/538 - Train Accuracy:  0.613, Validation Accuracy:  0.636, Loss:  1.440
    Epoch   0 Batch  327/538 - Train Accuracy:  0.607, Validation Accuracy:  0.642, Loss:  1.443
    Epoch   0 Batch  328/538 - Train Accuracy:  0.639, Validation Accuracy:  0.639, Loss:  1.366
    Epoch   0 Batch  329/538 - Train Accuracy:  0.628, Validation Accuracy:  0.640, Loss:  1.400
    Epoch   0 Batch  330/538 - Train Accuracy:  0.624, Validation Accuracy:  0.639, Loss:  1.377
    Epoch   0 Batch  331/538 - Train Accuracy:  0.610, Validation Accuracy:  0.633, Loss:  1.423
    Epoch   0 Batch  332/538 - Train Accuracy:  0.604, Validation Accuracy:  0.638, Loss:  1.428
    Epoch   0 Batch  333/538 - Train Accuracy:  0.625, Validation Accuracy:  0.642, Loss:  1.374
    Epoch   0 Batch  334/538 - Train Accuracy:  0.652, Validation Accuracy:  0.638, Loss:  1.272
    Epoch   0 Batch  335/538 - Train Accuracy:  0.630, Validation Accuracy:  0.640, Loss:  1.379
    Epoch   0 Batch  336/538 - Train Accuracy:  0.619, Validation Accuracy:  0.638, Loss:  1.370
    Epoch   0 Batch  337/538 - Train Accuracy:  0.616, Validation Accuracy:  0.640, Loss:  1.382
    Epoch   0 Batch  338/538 - Train Accuracy:  0.617, Validation Accuracy:  0.640, Loss:  1.397
    Epoch   0 Batch  339/538 - Train Accuracy:  0.619, Validation Accuracy:  0.641, Loss:  1.388
    Epoch   0 Batch  340/538 - Train Accuracy:  0.604, Validation Accuracy:  0.643, Loss:  1.435
    Epoch   0 Batch  341/538 - Train Accuracy:  0.605, Validation Accuracy:  0.642, Loss:  1.425
    Epoch   0 Batch  342/538 - Train Accuracy:  0.626, Validation Accuracy:  0.637, Loss:  1.375
    Epoch   0 Batch  343/538 - Train Accuracy:  0.622, Validation Accuracy:  0.637, Loss:  1.413
    Epoch   0 Batch  344/538 - Train Accuracy:  0.614, Validation Accuracy:  0.638, Loss:  1.389
    Epoch   0 Batch  345/538 - Train Accuracy:  0.630, Validation Accuracy:  0.634, Loss:  1.324
    Epoch   0 Batch  346/538 - Train Accuracy:  0.623, Validation Accuracy:  0.641, Loss:  1.398
    Epoch   0 Batch  347/538 - Train Accuracy:  0.623, Validation Accuracy:  0.642, Loss:  1.385
    Epoch   0 Batch  348/538 - Train Accuracy:  0.636, Validation Accuracy:  0.648, Loss:  1.357
    Epoch   0 Batch  349/538 - Train Accuracy:  0.618, Validation Accuracy:  0.649, Loss:  1.384
    Epoch   0 Batch  350/538 - Train Accuracy:  0.633, Validation Accuracy:  0.646, Loss:  1.375
    Epoch   0 Batch  351/538 - Train Accuracy:  0.606, Validation Accuracy:  0.641, Loss:  1.431
    Epoch   0 Batch  352/538 - Train Accuracy:  0.626, Validation Accuracy:  0.640, Loss:  1.355
    Epoch   0 Batch  353/538 - Train Accuracy:  0.628, Validation Accuracy:  0.642, Loss:  1.380
    Epoch   0 Batch  354/538 - Train Accuracy:  0.609, Validation Accuracy:  0.641, Loss:  1.444
    Epoch   0 Batch  355/538 - Train Accuracy:  0.608, Validation Accuracy:  0.641, Loss:  1.404
    Epoch   0 Batch  356/538 - Train Accuracy:  0.640, Validation Accuracy:  0.644, Loss:  1.305
    Epoch   0 Batch  357/538 - Train Accuracy:  0.628, Validation Accuracy:  0.645, Loss:  1.362
    Epoch   0 Batch  358/538 - Train Accuracy:  0.628, Validation Accuracy:  0.645, Loss:  1.352
    Epoch   0 Batch  359/538 - Train Accuracy:  0.637, Validation Accuracy:  0.646, Loss:  1.322
    Epoch   0 Batch  360/538 - Train Accuracy:  0.618, Validation Accuracy:  0.647, Loss:  1.377
    Epoch   0 Batch  361/538 - Train Accuracy:  0.635, Validation Accuracy:  0.638, Loss:  1.309
    Epoch   0 Batch  362/538 - Train Accuracy:  0.634, Validation Accuracy:  0.640, Loss:  1.285
    Epoch   0 Batch  363/538 - Train Accuracy:  0.626, Validation Accuracy:  0.639, Loss:  1.311
    Epoch   0 Batch  364/538 - Train Accuracy:  0.608, Validation Accuracy:  0.643, Loss:  1.387
    Epoch   0 Batch  365/538 - Train Accuracy:  0.634, Validation Accuracy:  0.649, Loss:  1.378
    Epoch   0 Batch  366/538 - Train Accuracy:  0.642, Validation Accuracy:  0.646, Loss:  1.344
    Epoch   0 Batch  367/538 - Train Accuracy:  0.628, Validation Accuracy:  0.643, Loss:  1.345
    Epoch   0 Batch  368/538 - Train Accuracy:  0.671, Validation Accuracy:  0.643, Loss:  1.168
    Epoch   0 Batch  369/538 - Train Accuracy:  0.619, Validation Accuracy:  0.648, Loss:  1.331
    Epoch   0 Batch  370/538 - Train Accuracy:  0.625, Validation Accuracy:  0.653, Loss:  1.349
    Epoch   0 Batch  371/538 - Train Accuracy:  0.644, Validation Accuracy:  0.650, Loss:  1.273
    Epoch   0 Batch  372/538 - Train Accuracy:  0.642, Validation Accuracy:  0.651, Loss:  1.314
    Epoch   0 Batch  373/538 - Train Accuracy:  0.616, Validation Accuracy:  0.648, Loss:  1.359
    Epoch   0 Batch  374/538 - Train Accuracy:  0.623, Validation Accuracy:  0.646, Loss:  1.332
    Epoch   0 Batch  375/538 - Train Accuracy:  0.650, Validation Accuracy:  0.651, Loss:  1.243
    Epoch   0 Batch  376/538 - Train Accuracy:  0.637, Validation Accuracy:  0.652, Loss:  1.313
    Epoch   0 Batch  377/538 - Train Accuracy:  0.640, Validation Accuracy:  0.652, Loss:  1.292
    Epoch   0 Batch  378/538 - Train Accuracy:  0.644, Validation Accuracy:  0.649, Loss:  1.267
    Epoch   0 Batch  379/538 - Train Accuracy:  0.632, Validation Accuracy:  0.648, Loss:  1.297
    Epoch   0 Batch  380/538 - Train Accuracy:  0.627, Validation Accuracy:  0.647, Loss:  1.306
    Epoch   0 Batch  381/538 - Train Accuracy:  0.652, Validation Accuracy:  0.650, Loss:  1.244
    Epoch   0 Batch  382/538 - Train Accuracy:  0.625, Validation Accuracy:  0.652, Loss:  1.328
    Epoch   0 Batch  383/538 - Train Accuracy:  0.639, Validation Accuracy:  0.661, Loss:  1.315
    Epoch   0 Batch  384/538 - Train Accuracy:  0.647, Validation Accuracy:  0.667, Loss:  1.315
    Epoch   0 Batch  385/538 - Train Accuracy:  0.650, Validation Accuracy:  0.661, Loss:  1.276
    Epoch   0 Batch  386/538 - Train Accuracy:  0.637, Validation Accuracy:  0.660, Loss:  1.284
    Epoch   0 Batch  387/538 - Train Accuracy:  0.625, Validation Accuracy:  0.653, Loss:  1.308
    Epoch   0 Batch  388/538 - Train Accuracy:  0.640, Validation Accuracy:  0.646, Loss:  1.265
    Epoch   0 Batch  389/538 - Train Accuracy:  0.620, Validation Accuracy:  0.654, Loss:  1.334
    Epoch   0 Batch  390/538 - Train Accuracy:  0.650, Validation Accuracy:  0.654, Loss:  1.241
    Epoch   0 Batch  391/538 - Train Accuracy:  0.648, Validation Accuracy:  0.657, Loss:  1.277
    Epoch   0 Batch  392/538 - Train Accuracy:  0.632, Validation Accuracy:  0.652, Loss:  1.274
    Epoch   0 Batch  393/538 - Train Accuracy:  0.650, Validation Accuracy:  0.654, Loss:  1.219
    Epoch   0 Batch  394/538 - Train Accuracy:  0.600, Validation Accuracy:  0.654, Loss:  1.367
    Epoch   0 Batch  395/538 - Train Accuracy:  0.626, Validation Accuracy:  0.659, Loss:  1.315
    Epoch   0 Batch  396/538 - Train Accuracy:  0.632, Validation Accuracy:  0.660, Loss:  1.267
    Epoch   0 Batch  397/538 - Train Accuracy:  0.631, Validation Accuracy:  0.657, Loss:  1.309
    Epoch   0 Batch  398/538 - Train Accuracy:  0.638, Validation Accuracy:  0.657, Loss:  1.290
    Epoch   0 Batch  399/538 - Train Accuracy:  0.606, Validation Accuracy:  0.644, Loss:  1.349
    Epoch   0 Batch  400/538 - Train Accuracy:  0.645, Validation Accuracy:  0.653, Loss:  1.242
    Epoch   0 Batch  401/538 - Train Accuracy:  0.630, Validation Accuracy:  0.651, Loss:  1.267
    Epoch   0 Batch  402/538 - Train Accuracy:  0.636, Validation Accuracy:  0.656, Loss:  1.270
    Epoch   0 Batch  403/538 - Train Accuracy:  0.635, Validation Accuracy:  0.656, Loss:  1.290
    Epoch   0 Batch  404/538 - Train Accuracy:  0.652, Validation Accuracy:  0.654, Loss:  1.232
    Epoch   0 Batch  405/538 - Train Accuracy:  0.645, Validation Accuracy:  0.650, Loss:  1.229
    Epoch   0 Batch  406/538 - Train Accuracy:  0.645, Validation Accuracy:  0.645, Loss:  1.256
    Epoch   0 Batch  407/538 - Train Accuracy:  0.634, Validation Accuracy:  0.647, Loss:  1.219
    Epoch   0 Batch  408/538 - Train Accuracy:  0.620, Validation Accuracy:  0.649, Loss:  1.329
    Epoch   0 Batch  409/538 - Train Accuracy:  0.632, Validation Accuracy:  0.653, Loss:  1.287
    Epoch   0 Batch  410/538 - Train Accuracy:  0.670, Validation Accuracy:  0.664, Loss:  1.226
    Epoch   0 Batch  411/538 - Train Accuracy:  0.665, Validation Accuracy:  0.667, Loss:  1.195
    Epoch   0 Batch  412/538 - Train Accuracy:  0.655, Validation Accuracy:  0.664, Loss:  1.190
    Epoch   0 Batch  413/538 - Train Accuracy:  0.626, Validation Accuracy:  0.651, Loss:  1.279
    Epoch   0 Batch  414/538 - Train Accuracy:  0.632, Validation Accuracy:  0.657, Loss:  1.262
    Epoch   0 Batch  415/538 - Train Accuracy:  0.628, Validation Accuracy:  0.657, Loss:  1.284
    Epoch   0 Batch  416/538 - Train Accuracy:  0.676, Validation Accuracy:  0.664, Loss:  1.155
    Epoch   0 Batch  417/538 - Train Accuracy:  0.640, Validation Accuracy:  0.667, Loss:  1.233
    Epoch   0 Batch  418/538 - Train Accuracy:  0.645, Validation Accuracy:  0.666, Loss:  1.268
    Epoch   0 Batch  419/538 - Train Accuracy:  0.648, Validation Accuracy:  0.661, Loss:  1.225
    Epoch   0 Batch  420/538 - Train Accuracy:  0.651, Validation Accuracy:  0.657, Loss:  1.221
    Epoch   0 Batch  421/538 - Train Accuracy:  0.652, Validation Accuracy:  0.662, Loss:  1.202
    Epoch   0 Batch  422/538 - Train Accuracy:  0.653, Validation Accuracy:  0.669, Loss:  1.199
    Epoch   0 Batch  423/538 - Train Accuracy:  0.656, Validation Accuracy:  0.669, Loss:  1.199
    Epoch   0 Batch  424/538 - Train Accuracy:  0.645, Validation Accuracy:  0.663, Loss:  1.215
    Epoch   0 Batch  425/538 - Train Accuracy:  0.664, Validation Accuracy:  0.666, Loss:  1.152
    Epoch   0 Batch  426/538 - Train Accuracy:  0.657, Validation Accuracy:  0.669, Loss:  1.193
    Epoch   0 Batch  427/538 - Train Accuracy:  0.632, Validation Accuracy:  0.666, Loss:  1.236
    Epoch   0 Batch  428/538 - Train Accuracy:  0.669, Validation Accuracy:  0.666, Loss:  1.165
    Epoch   0 Batch  429/538 - Train Accuracy:  0.663, Validation Accuracy:  0.670, Loss:  1.178
    Epoch   0 Batch  430/538 - Train Accuracy:  0.653, Validation Accuracy:  0.681, Loss:  1.221
    Epoch   0 Batch  431/538 - Train Accuracy:  0.659, Validation Accuracy:  0.678, Loss:  1.200
    Epoch   0 Batch  432/538 - Train Accuracy:  0.688, Validation Accuracy:  0.670, Loss:  1.097
    Epoch   0 Batch  433/538 - Train Accuracy:  0.633, Validation Accuracy:  0.654, Loss:  1.228
    Epoch   0 Batch  434/538 - Train Accuracy:  0.620, Validation Accuracy:  0.655, Loss:  1.270
    Epoch   0 Batch  435/538 - Train Accuracy:  0.634, Validation Accuracy:  0.656, Loss:  1.199
    Epoch   0 Batch  436/538 - Train Accuracy:  0.640, Validation Accuracy:  0.667, Loss:  1.250
    Epoch   0 Batch  437/538 - Train Accuracy:  0.645, Validation Accuracy:  0.659, Loss:  1.226
    Epoch   0 Batch  438/538 - Train Accuracy:  0.662, Validation Accuracy:  0.661, Loss:  1.172
    Epoch   0 Batch  439/538 - Train Accuracy:  0.673, Validation Accuracy:  0.663, Loss:  1.139
    Epoch   0 Batch  440/538 - Train Accuracy:  0.628, Validation Accuracy:  0.663, Loss:  1.207
    Epoch   0 Batch  441/538 - Train Accuracy:  0.640, Validation Accuracy:  0.662, Loss:  1.235
    Epoch   0 Batch  442/538 - Train Accuracy:  0.669, Validation Accuracy:  0.652, Loss:  1.091
    Epoch   0 Batch  443/538 - Train Accuracy:  0.647, Validation Accuracy:  0.657, Loss:  1.179
    Epoch   0 Batch  444/538 - Train Accuracy:  0.687, Validation Accuracy:  0.668, Loss:  1.100
    Epoch   0 Batch  445/538 - Train Accuracy:  0.652, Validation Accuracy:  0.671, Loss:  1.156
    Epoch   0 Batch  446/538 - Train Accuracy:  0.683, Validation Accuracy:  0.677, Loss:  1.098
    Epoch   0 Batch  447/538 - Train Accuracy:  0.648, Validation Accuracy:  0.672, Loss:  1.176
    Epoch   0 Batch  448/538 - Train Accuracy:  0.664, Validation Accuracy:  0.669, Loss:  1.085
    Epoch   0 Batch  449/538 - Train Accuracy:  0.656, Validation Accuracy:  0.673, Loss:  1.171
    Epoch   0 Batch  450/538 - Train Accuracy:  0.663, Validation Accuracy:  0.673, Loss:  1.173
    Epoch   0 Batch  451/538 - Train Accuracy:  0.642, Validation Accuracy:  0.670, Loss:  1.208
    Epoch   0 Batch  452/538 - Train Accuracy:  0.635, Validation Accuracy:  0.656, Loss:  1.155
    Epoch   0 Batch  453/538 - Train Accuracy:  0.645, Validation Accuracy:  0.669, Loss:  1.176
    Epoch   0 Batch  454/538 - Train Accuracy:  0.671, Validation Accuracy:  0.671, Loss:  1.104
    Epoch   0 Batch  455/538 - Train Accuracy:  0.687, Validation Accuracy:  0.678, Loss:  1.079
    Epoch   0 Batch  456/538 - Train Accuracy:  0.720, Validation Accuracy:  0.677, Loss:  0.978
    Epoch   0 Batch  457/538 - Train Accuracy:  0.645, Validation Accuracy:  0.676, Loss:  1.192
    Epoch   0 Batch  458/538 - Train Accuracy:  0.662, Validation Accuracy:  0.671, Loss:  1.115
    Epoch   0 Batch  459/538 - Train Accuracy:  0.672, Validation Accuracy:  0.674, Loss:  1.139
    Epoch   0 Batch  460/538 - Train Accuracy:  0.664, Validation Accuracy:  0.671, Loss:  1.140
    Epoch   0 Batch  461/538 - Train Accuracy:  0.632, Validation Accuracy:  0.663, Loss:  1.227
    Epoch   0 Batch  462/538 - Train Accuracy:  0.640, Validation Accuracy:  0.661, Loss:  1.140
    Epoch   0 Batch  463/538 - Train Accuracy:  0.643, Validation Accuracy:  0.674, Loss:  1.147
    Epoch   0 Batch  464/538 - Train Accuracy:  0.665, Validation Accuracy:  0.680, Loss:  1.151
    Epoch   0 Batch  465/538 - Train Accuracy:  0.652, Validation Accuracy:  0.678, Loss:  1.152
    Epoch   0 Batch  466/538 - Train Accuracy:  0.648, Validation Accuracy:  0.670, Loss:  1.173
    Epoch   0 Batch  467/538 - Train Accuracy:  0.667, Validation Accuracy:  0.659, Loss:  1.103
    Epoch   0 Batch  468/538 - Train Accuracy:  0.665, Validation Accuracy:  0.665, Loss:  1.151
    Epoch   0 Batch  469/538 - Train Accuracy:  0.659, Validation Accuracy:  0.668, Loss:  1.140
    Epoch   0 Batch  470/538 - Train Accuracy:  0.678, Validation Accuracy:  0.675, Loss:  1.092
    Epoch   0 Batch  471/538 - Train Accuracy:  0.673, Validation Accuracy:  0.678, Loss:  1.102
    Epoch   0 Batch  472/538 - Train Accuracy:  0.679, Validation Accuracy:  0.678, Loss:  1.112
    Epoch   0 Batch  473/538 - Train Accuracy:  0.653, Validation Accuracy:  0.676, Loss:  1.131
    Epoch   0 Batch  474/538 - Train Accuracy:  0.672, Validation Accuracy:  0.672, Loss:  1.076
    Epoch   0 Batch  475/538 - Train Accuracy:  0.685, Validation Accuracy:  0.673, Loss:  1.047
    Epoch   0 Batch  476/538 - Train Accuracy:  0.661, Validation Accuracy:  0.675, Loss:  1.137
    Epoch   0 Batch  477/538 - Train Accuracy:  0.669, Validation Accuracy:  0.676, Loss:  1.109
    Epoch   0 Batch  478/538 - Train Accuracy:  0.675, Validation Accuracy:  0.678, Loss:  1.092
    Epoch   0 Batch  479/538 - Train Accuracy:  0.674, Validation Accuracy:  0.678, Loss:  1.067
    Epoch   0 Batch  480/538 - Train Accuracy:  0.675, Validation Accuracy:  0.678, Loss:  1.096
    Epoch   0 Batch  481/538 - Train Accuracy:  0.681, Validation Accuracy:  0.678, Loss:  1.074
    Epoch   0 Batch  482/538 - Train Accuracy:  0.695, Validation Accuracy:  0.675, Loss:  0.999
    Epoch   0 Batch  483/538 - Train Accuracy:  0.633, Validation Accuracy:  0.661, Loss:  1.131
    Epoch   0 Batch  484/538 - Train Accuracy:  0.664, Validation Accuracy:  0.667, Loss:  1.079
    Epoch   0 Batch  485/538 - Train Accuracy:  0.674, Validation Accuracy:  0.678, Loss:  1.065
    Epoch   0 Batch  486/538 - Train Accuracy:  0.692, Validation Accuracy:  0.680, Loss:  1.047
    Epoch   0 Batch  487/538 - Train Accuracy:  0.680, Validation Accuracy:  0.683, Loss:  1.033
    Epoch   0 Batch  488/538 - Train Accuracy:  0.678, Validation Accuracy:  0.680, Loss:  1.047
    Epoch   0 Batch  489/538 - Train Accuracy:  0.650, Validation Accuracy:  0.672, Loss:  1.130
    Epoch   0 Batch  490/538 - Train Accuracy:  0.677, Validation Accuracy:  0.678, Loss:  1.041
    Epoch   0 Batch  491/538 - Train Accuracy:  0.641, Validation Accuracy:  0.679, Loss:  1.141
    Epoch   0 Batch  492/538 - Train Accuracy:  0.666, Validation Accuracy:  0.678, Loss:  1.104
    Epoch   0 Batch  493/538 - Train Accuracy:  0.657, Validation Accuracy:  0.670, Loss:  1.063
    Epoch   0 Batch  494/538 - Train Accuracy:  0.658, Validation Accuracy:  0.670, Loss:  1.113
    Epoch   0 Batch  495/538 - Train Accuracy:  0.665, Validation Accuracy:  0.688, Loss:  1.083
    Epoch   0 Batch  496/538 - Train Accuracy:  0.676, Validation Accuracy:  0.693, Loss:  1.110
    Epoch   0 Batch  497/538 - Train Accuracy:  0.684, Validation Accuracy:  0.695, Loss:  1.061
    Epoch   0 Batch  498/538 - Train Accuracy:  0.660, Validation Accuracy:  0.671, Loss:  1.083
    Epoch   0 Batch  499/538 - Train Accuracy:  0.662, Validation Accuracy:  0.666, Loss:  1.071
    Epoch   0 Batch  500/538 - Train Accuracy:  0.693, Validation Accuracy:  0.674, Loss:  1.016
    Epoch   0 Batch  501/538 - Train Accuracy:  0.675, Validation Accuracy:  0.681, Loss:  1.067
    Epoch   0 Batch  502/538 - Train Accuracy:  0.673, Validation Accuracy:  0.680, Loss:  1.088
    Epoch   0 Batch  503/538 - Train Accuracy:  0.686, Validation Accuracy:  0.682, Loss:  1.013
    Epoch   0 Batch  504/538 - Train Accuracy:  0.669, Validation Accuracy:  0.677, Loss:  1.067
    Epoch   0 Batch  505/538 - Train Accuracy:  0.668, Validation Accuracy:  0.678, Loss:  1.071
    Epoch   0 Batch  506/538 - Train Accuracy:  0.675, Validation Accuracy:  0.683, Loss:  1.058
    Epoch   0 Batch  507/538 - Train Accuracy:  0.671, Validation Accuracy:  0.687, Loss:  1.095
    Epoch   0 Batch  508/538 - Train Accuracy:  0.672, Validation Accuracy:  0.686, Loss:  0.999
    Epoch   0 Batch  509/538 - Train Accuracy:  0.680, Validation Accuracy:  0.678, Loss:  1.059
    Epoch   0 Batch  510/538 - Train Accuracy:  0.673, Validation Accuracy:  0.675, Loss:  1.006
    Epoch   0 Batch  511/538 - Train Accuracy:  0.681, Validation Accuracy:  0.682, Loss:  0.997
    Epoch   0 Batch  512/538 - Train Accuracy:  0.698, Validation Accuracy:  0.691, Loss:  1.021
    Epoch   0 Batch  513/538 - Train Accuracy:  0.666, Validation Accuracy:  0.691, Loss:  1.085
    Epoch   0 Batch  514/538 - Train Accuracy:  0.671, Validation Accuracy:  0.688, Loss:  1.083
    Epoch   0 Batch  515/538 - Train Accuracy:  0.663, Validation Accuracy:  0.679, Loss:  1.024
    Epoch   0 Batch  516/538 - Train Accuracy:  0.638, Validation Accuracy:  0.661, Loss:  1.045
    Epoch   0 Batch  517/538 - Train Accuracy:  0.679, Validation Accuracy:  0.682, Loss:  1.024
    Epoch   0 Batch  518/538 - Train Accuracy:  0.665, Validation Accuracy:  0.693, Loss:  1.080
    Epoch   0 Batch  519/538 - Train Accuracy:  0.709, Validation Accuracy:  0.694, Loss:  0.989
    Epoch   0 Batch  520/538 - Train Accuracy:  0.675, Validation Accuracy:  0.694, Loss:  1.090
    Epoch   0 Batch  521/538 - Train Accuracy:  0.683, Validation Accuracy:  0.693, Loss:  1.055
    Epoch   0 Batch  522/538 - Train Accuracy:  0.643, Validation Accuracy:  0.676, Loss:  1.053
    Epoch   0 Batch  523/538 - Train Accuracy:  0.666, Validation Accuracy:  0.679, Loss:  1.048
    Epoch   0 Batch  524/538 - Train Accuracy:  0.652, Validation Accuracy:  0.691, Loss:  1.094
    Epoch   0 Batch  525/538 - Train Accuracy:  0.689, Validation Accuracy:  0.695, Loss:  1.020
    Epoch   0 Batch  526/538 - Train Accuracy:  0.677, Validation Accuracy:  0.690, Loss:  1.053
    Epoch   0 Batch  527/538 - Train Accuracy:  0.676, Validation Accuracy:  0.682, Loss:  1.012
    Epoch   0 Batch  528/538 - Train Accuracy:  0.651, Validation Accuracy:  0.685, Loss:  1.109
    Epoch   0 Batch  529/538 - Train Accuracy:  0.663, Validation Accuracy:  0.690, Loss:  1.063
    Epoch   0 Batch  530/538 - Train Accuracy:  0.661, Validation Accuracy:  0.689, Loss:  1.076
    Epoch   0 Batch  531/538 - Train Accuracy:  0.670, Validation Accuracy:  0.686, Loss:  1.026
    Epoch   0 Batch  532/538 - Train Accuracy:  0.644, Validation Accuracy:  0.682, Loss:  1.019
    Epoch   0 Batch  533/538 - Train Accuracy:  0.674, Validation Accuracy:  0.680, Loss:  0.997
    Epoch   0 Batch  534/538 - Train Accuracy:  0.675, Validation Accuracy:  0.690, Loss:  1.019
    Epoch   0 Batch  535/538 - Train Accuracy:  0.693, Validation Accuracy:  0.696, Loss:  0.998
    Epoch   0 Batch  536/538 - Train Accuracy:  0.705, Validation Accuracy:  0.694, Loss:  1.019
    Epoch   1 Batch    0/538 - Train Accuracy:  0.671, Validation Accuracy:  0.693, Loss:  1.019
    Epoch   1 Batch    1/538 - Train Accuracy:  0.680, Validation Accuracy:  0.680, Loss:  1.015
    Epoch   1 Batch    2/538 - Train Accuracy:  0.668, Validation Accuracy:  0.683, Loss:  1.042
    Epoch   1 Batch    3/538 - Train Accuracy:  0.678, Validation Accuracy:  0.696, Loss:  1.010
    Epoch   1 Batch    4/538 - Train Accuracy:  0.689, Validation Accuracy:  0.699, Loss:  1.010
    Epoch   1 Batch    5/538 - Train Accuracy:  0.677, Validation Accuracy:  0.693, Loss:  1.021
    Epoch   1 Batch    6/538 - Train Accuracy:  0.680, Validation Accuracy:  0.685, Loss:  0.979
    Epoch   1 Batch    7/538 - Train Accuracy:  0.657, Validation Accuracy:  0.677, Loss:  0.996
    Epoch   1 Batch    8/538 - Train Accuracy:  0.672, Validation Accuracy:  0.681, Loss:  1.003
    Epoch   1 Batch    9/538 - Train Accuracy:  0.669, Validation Accuracy:  0.676, Loss:  0.989
    Epoch   1 Batch   10/538 - Train Accuracy:  0.653, Validation Accuracy:  0.676, Loss:  1.050
    Epoch   1 Batch   11/538 - Train Accuracy:  0.662, Validation Accuracy:  0.689, Loss:  1.012
    Epoch   1 Batch   12/538 - Train Accuracy:  0.663, Validation Accuracy:  0.696, Loss:  1.018
    Epoch   1 Batch   13/538 - Train Accuracy:  0.700, Validation Accuracy:  0.699, Loss:  0.914
    Epoch   1 Batch   14/538 - Train Accuracy:  0.688, Validation Accuracy:  0.697, Loss:  0.980
    Epoch   1 Batch   15/538 - Train Accuracy:  0.690, Validation Accuracy:  0.690, Loss:  0.953
    Epoch   1 Batch   16/538 - Train Accuracy:  0.690, Validation Accuracy:  0.692, Loss:  0.959
    Epoch   1 Batch   17/538 - Train Accuracy:  0.679, Validation Accuracy:  0.687, Loss:  0.995
    Epoch   1 Batch   18/538 - Train Accuracy:  0.680, Validation Accuracy:  0.698, Loss:  1.007
    Epoch   1 Batch   19/538 - Train Accuracy:  0.675, Validation Accuracy:  0.700, Loss:  1.033
    Epoch   1 Batch   20/538 - Train Accuracy:  0.670, Validation Accuracy:  0.695, Loss:  0.963
    Epoch   1 Batch   21/538 - Train Accuracy:  0.693, Validation Accuracy:  0.703, Loss:  0.993
    Epoch   1 Batch   22/538 - Train Accuracy:  0.679, Validation Accuracy:  0.702, Loss:  0.992
    Epoch   1 Batch   23/538 - Train Accuracy:  0.682, Validation Accuracy:  0.705, Loss:  0.987
    Epoch   1 Batch   24/538 - Train Accuracy:  0.685, Validation Accuracy:  0.702, Loss:  0.978
    Epoch   1 Batch   25/538 - Train Accuracy:  0.674, Validation Accuracy:  0.694, Loss:  0.966
    Epoch   1 Batch   26/538 - Train Accuracy:  0.671, Validation Accuracy:  0.699, Loss:  1.022
    Epoch   1 Batch   27/538 - Train Accuracy:  0.688, Validation Accuracy:  0.700, Loss:  0.956
    Epoch   1 Batch   28/538 - Train Accuracy:  0.703, Validation Accuracy:  0.697, Loss:  0.893
    Epoch   1 Batch   29/538 - Train Accuracy:  0.681, Validation Accuracy:  0.699, Loss:  0.947
    Epoch   1 Batch   30/538 - Train Accuracy:  0.665, Validation Accuracy:  0.697, Loss:  0.994
    Epoch   1 Batch   31/538 - Train Accuracy:  0.691, Validation Accuracy:  0.696, Loss:  0.929
    Epoch   1 Batch   32/538 - Train Accuracy:  0.675, Validation Accuracy:  0.689, Loss:  0.956
    Epoch   1 Batch   33/538 - Train Accuracy:  0.686, Validation Accuracy:  0.688, Loss:  0.942
    Epoch   1 Batch   34/538 - Train Accuracy:  0.679, Validation Accuracy:  0.702, Loss:  0.985
    Epoch   1 Batch   35/538 - Train Accuracy:  0.689, Validation Accuracy:  0.709, Loss:  0.966
    Epoch   1 Batch   36/538 - Train Accuracy:  0.711, Validation Accuracy:  0.708, Loss:  0.916
    Epoch   1 Batch   37/538 - Train Accuracy:  0.704, Validation Accuracy:  0.699, Loss:  0.934
    Epoch   1 Batch   38/538 - Train Accuracy:  0.652, Validation Accuracy:  0.694, Loss:  0.987
    Epoch   1 Batch   39/538 - Train Accuracy:  0.699, Validation Accuracy:  0.705, Loss:  0.964
    Epoch   1 Batch   40/538 - Train Accuracy:  0.725, Validation Accuracy:  0.712, Loss:  0.877
    Epoch   1 Batch   41/538 - Train Accuracy:  0.698, Validation Accuracy:  0.713, Loss:  0.963
    Epoch   1 Batch   42/538 - Train Accuracy:  0.690, Validation Accuracy:  0.697, Loss:  0.951
    Epoch   1 Batch   43/538 - Train Accuracy:  0.674, Validation Accuracy:  0.688, Loss:  0.969
    Epoch   1 Batch   44/538 - Train Accuracy:  0.665, Validation Accuracy:  0.696, Loss:  0.986
    Epoch   1 Batch   45/538 - Train Accuracy:  0.691, Validation Accuracy:  0.703, Loss:  0.905
    Epoch   1 Batch   46/538 - Train Accuracy:  0.695, Validation Accuracy:  0.700, Loss:  0.935
    Epoch   1 Batch   47/538 - Train Accuracy:  0.682, Validation Accuracy:  0.700, Loss:  0.937
    Epoch   1 Batch   48/538 - Train Accuracy:  0.692, Validation Accuracy:  0.692, Loss:  0.903
    Epoch   1 Batch   49/538 - Train Accuracy:  0.676, Validation Accuracy:  0.691, Loss:  0.966
    Epoch   1 Batch   50/538 - Train Accuracy:  0.680, Validation Accuracy:  0.694, Loss:  0.921
    Epoch   1 Batch   51/538 - Train Accuracy:  0.650, Validation Accuracy:  0.682, Loss:  1.024
    Epoch   1 Batch   52/538 - Train Accuracy:  0.665, Validation Accuracy:  0.682, Loss:  0.945
    Epoch   1 Batch   53/538 - Train Accuracy:  0.691, Validation Accuracy:  0.696, Loss:  0.887
    Epoch   1 Batch   54/538 - Train Accuracy:  0.700, Validation Accuracy:  0.707, Loss:  0.906
    Epoch   1 Batch   55/538 - Train Accuracy:  0.689, Validation Accuracy:  0.707, Loss:  0.938
    Epoch   1 Batch   56/538 - Train Accuracy:  0.691, Validation Accuracy:  0.703, Loss:  0.899
    Epoch   1 Batch   57/538 - Train Accuracy:  0.667, Validation Accuracy:  0.706, Loss:  0.969
    Epoch   1 Batch   58/538 - Train Accuracy:  0.665, Validation Accuracy:  0.694, Loss:  0.982
    Epoch   1 Batch   59/538 - Train Accuracy:  0.673, Validation Accuracy:  0.699, Loss:  0.936
    Epoch   1 Batch   60/538 - Train Accuracy:  0.693, Validation Accuracy:  0.706, Loss:  0.912
    Epoch   1 Batch   61/538 - Train Accuracy:  0.687, Validation Accuracy:  0.700, Loss:  0.912
    Epoch   1 Batch   62/538 - Train Accuracy:  0.691, Validation Accuracy:  0.678, Loss:  0.911
    Epoch   1 Batch   63/538 - Train Accuracy:  0.682, Validation Accuracy:  0.678, Loss:  0.879
    Epoch   1 Batch   64/538 - Train Accuracy:  0.700, Validation Accuracy:  0.683, Loss:  0.881
    Epoch   1 Batch   65/538 - Train Accuracy:  0.649, Validation Accuracy:  0.689, Loss:  0.977
    Epoch   1 Batch   66/538 - Train Accuracy:  0.708, Validation Accuracy:  0.696, Loss:  0.865
    Epoch   1 Batch   67/538 - Train Accuracy:  0.691, Validation Accuracy:  0.704, Loss:  0.911
    Epoch   1 Batch   68/538 - Train Accuracy:  0.706, Validation Accuracy:  0.705, Loss:  0.855
    Epoch   1 Batch   69/538 - Train Accuracy:  0.703, Validation Accuracy:  0.707, Loss:  0.911
    Epoch   1 Batch   70/538 - Train Accuracy:  0.697, Validation Accuracy:  0.706, Loss:  0.888
    Epoch   1 Batch   71/538 - Train Accuracy:  0.680, Validation Accuracy:  0.699, Loss:  0.910
    Epoch   1 Batch   72/538 - Train Accuracy:  0.709, Validation Accuracy:  0.695, Loss:  0.871
    Epoch   1 Batch   73/538 - Train Accuracy:  0.662, Validation Accuracy:  0.698, Loss:  0.937
    Epoch   1 Batch   74/538 - Train Accuracy:  0.706, Validation Accuracy:  0.698, Loss:  0.867
    Epoch   1 Batch   75/538 - Train Accuracy:  0.693, Validation Accuracy:  0.706, Loss:  0.876
    Epoch   1 Batch   76/538 - Train Accuracy:  0.674, Validation Accuracy:  0.716, Loss:  0.945
    Epoch   1 Batch   77/538 - Train Accuracy:  0.713, Validation Accuracy:  0.722, Loss:  0.919
    Epoch   1 Batch   78/538 - Train Accuracy:  0.728, Validation Accuracy:  0.723, Loss:  0.912
    Epoch   1 Batch   79/538 - Train Accuracy:  0.695, Validation Accuracy:  0.711, Loss:  0.862
    Epoch   1 Batch   80/538 - Train Accuracy:  0.681, Validation Accuracy:  0.703, Loss:  0.926
    Epoch   1 Batch   81/538 - Train Accuracy:  0.671, Validation Accuracy:  0.705, Loss:  0.910
    Epoch   1 Batch   82/538 - Train Accuracy:  0.670, Validation Accuracy:  0.712, Loss:  0.896
    Epoch   1 Batch   83/538 - Train Accuracy:  0.689, Validation Accuracy:  0.716, Loss:  0.914
    Epoch   1 Batch   84/538 - Train Accuracy:  0.697, Validation Accuracy:  0.710, Loss:  0.885
    Epoch   1 Batch   85/538 - Train Accuracy:  0.708, Validation Accuracy:  0.697, Loss:  0.830
    Epoch   1 Batch   86/538 - Train Accuracy:  0.695, Validation Accuracy:  0.700, Loss:  0.906
    Epoch   1 Batch   87/538 - Train Accuracy:  0.697, Validation Accuracy:  0.705, Loss:  0.892
    Epoch   1 Batch   88/538 - Train Accuracy:  0.701, Validation Accuracy:  0.713, Loss:  0.885
    Epoch   1 Batch   89/538 - Train Accuracy:  0.706, Validation Accuracy:  0.714, Loss:  0.879
    Epoch   1 Batch   90/538 - Train Accuracy:  0.693, Validation Accuracy:  0.700, Loss:  0.884
    Epoch   1 Batch   91/538 - Train Accuracy:  0.678, Validation Accuracy:  0.700, Loss:  0.917
    Epoch   1 Batch   92/538 - Train Accuracy:  0.703, Validation Accuracy:  0.713, Loss:  0.890
    Epoch   1 Batch   93/538 - Train Accuracy:  0.691, Validation Accuracy:  0.719, Loss:  0.893
    Epoch   1 Batch   94/538 - Train Accuracy:  0.705, Validation Accuracy:  0.717, Loss:  0.894
    Epoch   1 Batch   95/538 - Train Accuracy:  0.720, Validation Accuracy:  0.709, Loss:  0.813
    Epoch   1 Batch   96/538 - Train Accuracy:  0.718, Validation Accuracy:  0.707, Loss:  0.824
    Epoch   1 Batch   97/538 - Train Accuracy:  0.701, Validation Accuracy:  0.711, Loss:  0.877
    Epoch   1 Batch   98/538 - Train Accuracy:  0.720, Validation Accuracy:  0.710, Loss:  0.808
    Epoch   1 Batch   99/538 - Train Accuracy:  0.678, Validation Accuracy:  0.701, Loss:  0.872
    Epoch   1 Batch  100/538 - Train Accuracy:  0.702, Validation Accuracy:  0.699, Loss:  0.851
    Epoch   1 Batch  101/538 - Train Accuracy:  0.682, Validation Accuracy:  0.690, Loss:  0.874
    Epoch   1 Batch  102/538 - Train Accuracy:  0.693, Validation Accuracy:  0.700, Loss:  0.863
    Epoch   1 Batch  103/538 - Train Accuracy:  0.713, Validation Accuracy:  0.712, Loss:  0.836
    Epoch   1 Batch  104/538 - Train Accuracy:  0.712, Validation Accuracy:  0.714, Loss:  0.846
    Epoch   1 Batch  105/538 - Train Accuracy:  0.719, Validation Accuracy:  0.710, Loss:  0.818
    Epoch   1 Batch  106/538 - Train Accuracy:  0.680, Validation Accuracy:  0.697, Loss:  0.834
    Epoch   1 Batch  107/538 - Train Accuracy:  0.678, Validation Accuracy:  0.694, Loss:  0.898
    Epoch   1 Batch  108/538 - Train Accuracy:  0.702, Validation Accuracy:  0.700, Loss:  0.840
    Epoch   1 Batch  109/538 - Train Accuracy:  0.708, Validation Accuracy:  0.709, Loss:  0.846
    Epoch   1 Batch  110/538 - Train Accuracy:  0.697, Validation Accuracy:  0.717, Loss:  0.880
    Epoch   1 Batch  111/538 - Train Accuracy:  0.732, Validation Accuracy:  0.716, Loss:  0.805
    Epoch   1 Batch  112/538 - Train Accuracy:  0.712, Validation Accuracy:  0.711, Loss:  0.857
    Epoch   1 Batch  113/538 - Train Accuracy:  0.697, Validation Accuracy:  0.710, Loss:  0.893
    Epoch   1 Batch  114/538 - Train Accuracy:  0.721, Validation Accuracy:  0.713, Loss:  0.814
    Epoch   1 Batch  115/538 - Train Accuracy:  0.705, Validation Accuracy:  0.721, Loss:  0.850
    Epoch   1 Batch  116/538 - Train Accuracy:  0.731, Validation Accuracy:  0.718, Loss:  0.868
    Epoch   1 Batch  117/538 - Train Accuracy:  0.722, Validation Accuracy:  0.720, Loss:  0.817
    Epoch   1 Batch  118/538 - Train Accuracy:  0.718, Validation Accuracy:  0.705, Loss:  0.807
    Epoch   1 Batch  119/538 - Train Accuracy:  0.731, Validation Accuracy:  0.702, Loss:  0.792
    Epoch   1 Batch  120/538 - Train Accuracy:  0.709, Validation Accuracy:  0.716, Loss:  0.843
    Epoch   1 Batch  121/538 - Train Accuracy:  0.737, Validation Accuracy:  0.734, Loss:  0.816
    Epoch   1 Batch  122/538 - Train Accuracy:  0.730, Validation Accuracy:  0.740, Loss:  0.812
    Epoch   1 Batch  123/538 - Train Accuracy:  0.723, Validation Accuracy:  0.726, Loss:  0.789
    Epoch   1 Batch  124/538 - Train Accuracy:  0.729, Validation Accuracy:  0.715, Loss:  0.772
    Epoch   1 Batch  125/538 - Train Accuracy:  0.722, Validation Accuracy:  0.714, Loss:  0.820
    Epoch   1 Batch  126/538 - Train Accuracy:  0.723, Validation Accuracy:  0.726, Loss:  0.804
    Epoch   1 Batch  127/538 - Train Accuracy:  0.724, Validation Accuracy:  0.737, Loss:  0.871
    Epoch   1 Batch  128/538 - Train Accuracy:  0.730, Validation Accuracy:  0.733, Loss:  0.815
    Epoch   1 Batch  129/538 - Train Accuracy:  0.729, Validation Accuracy:  0.727, Loss:  0.807
    Epoch   1 Batch  130/538 - Train Accuracy:  0.733, Validation Accuracy:  0.716, Loss:  0.781
    Epoch   1 Batch  131/538 - Train Accuracy:  0.716, Validation Accuracy:  0.705, Loss:  0.832
    Epoch   1 Batch  132/538 - Train Accuracy:  0.698, Validation Accuracy:  0.711, Loss:  0.823
    Epoch   1 Batch  133/538 - Train Accuracy:  0.732, Validation Accuracy:  0.720, Loss:  0.766
    Epoch   1 Batch  134/538 - Train Accuracy:  0.699, Validation Accuracy:  0.724, Loss:  0.888
    Epoch   1 Batch  135/538 - Train Accuracy:  0.728, Validation Accuracy:  0.727, Loss:  0.812
    Epoch   1 Batch  136/538 - Train Accuracy:  0.717, Validation Accuracy:  0.725, Loss:  0.818
    Epoch   1 Batch  137/538 - Train Accuracy:  0.716, Validation Accuracy:  0.723, Loss:  0.814
    Epoch   1 Batch  138/538 - Train Accuracy:  0.719, Validation Accuracy:  0.722, Loss:  0.800
    Epoch   1 Batch  139/538 - Train Accuracy:  0.713, Validation Accuracy:  0.729, Loss:  0.883
    Epoch   1 Batch  140/538 - Train Accuracy:  0.700, Validation Accuracy:  0.725, Loss:  0.873
    Epoch   1 Batch  141/538 - Train Accuracy:  0.718, Validation Accuracy:  0.719, Loss:  0.851
    Epoch   1 Batch  142/538 - Train Accuracy:  0.733, Validation Accuracy:  0.724, Loss:  0.782
    Epoch   1 Batch  143/538 - Train Accuracy:  0.725, Validation Accuracy:  0.731, Loss:  0.834
    Epoch   1 Batch  144/538 - Train Accuracy:  0.727, Validation Accuracy:  0.735, Loss:  0.825
    Epoch   1 Batch  145/538 - Train Accuracy:  0.730, Validation Accuracy:  0.738, Loss:  0.828
    Epoch   1 Batch  146/538 - Train Accuracy:  0.747, Validation Accuracy:  0.732, Loss:  0.781
    Epoch   1 Batch  147/538 - Train Accuracy:  0.732, Validation Accuracy:  0.732, Loss:  0.793
    Epoch   1 Batch  148/538 - Train Accuracy:  0.704, Validation Accuracy:  0.739, Loss:  0.880
    Epoch   1 Batch  149/538 - Train Accuracy:  0.748, Validation Accuracy:  0.732, Loss:  0.801
    Epoch   1 Batch  150/538 - Train Accuracy:  0.737, Validation Accuracy:  0.732, Loss:  0.815
    Epoch   1 Batch  151/538 - Train Accuracy:  0.733, Validation Accuracy:  0.727, Loss:  0.777
    Epoch   1 Batch  152/538 - Train Accuracy:  0.730, Validation Accuracy:  0.721, Loss:  0.790
    Epoch   1 Batch  153/538 - Train Accuracy:  0.715, Validation Accuracy:  0.725, Loss:  0.832
    Epoch   1 Batch  154/538 - Train Accuracy:  0.730, Validation Accuracy:  0.729, Loss:  0.785
    Epoch   1 Batch  155/538 - Train Accuracy:  0.720, Validation Accuracy:  0.743, Loss:  0.804
    Epoch   1 Batch  156/538 - Train Accuracy:  0.744, Validation Accuracy:  0.747, Loss:  0.808
    Epoch   1 Batch  157/538 - Train Accuracy:  0.751, Validation Accuracy:  0.741, Loss:  0.770
    Epoch   1 Batch  158/538 - Train Accuracy:  0.721, Validation Accuracy:  0.732, Loss:  0.819
    Epoch   1 Batch  159/538 - Train Accuracy:  0.730, Validation Accuracy:  0.735, Loss:  0.814
    Epoch   1 Batch  160/538 - Train Accuracy:  0.728, Validation Accuracy:  0.744, Loss:  0.777
    Epoch   1 Batch  161/538 - Train Accuracy:  0.731, Validation Accuracy:  0.740, Loss:  0.777
    Epoch   1 Batch  162/538 - Train Accuracy:  0.723, Validation Accuracy:  0.746, Loss:  0.763
    Epoch   1 Batch  163/538 - Train Accuracy:  0.727, Validation Accuracy:  0.738, Loss:  0.804
    Epoch   1 Batch  164/538 - Train Accuracy:  0.699, Validation Accuracy:  0.729, Loss:  0.841
    Epoch   1 Batch  165/538 - Train Accuracy:  0.742, Validation Accuracy:  0.729, Loss:  0.736
    Epoch   1 Batch  166/538 - Train Accuracy:  0.745, Validation Accuracy:  0.734, Loss:  0.801
    Epoch   1 Batch  167/538 - Train Accuracy:  0.757, Validation Accuracy:  0.738, Loss:  0.757
    Epoch   1 Batch  168/538 - Train Accuracy:  0.722, Validation Accuracy:  0.735, Loss:  0.834
    Epoch   1 Batch  169/538 - Train Accuracy:  0.737, Validation Accuracy:  0.735, Loss:  0.770
    Epoch   1 Batch  170/538 - Train Accuracy:  0.716, Validation Accuracy:  0.728, Loss:  0.778
    Epoch   1 Batch  171/538 - Train Accuracy:  0.699, Validation Accuracy:  0.719, Loss:  0.796
    Epoch   1 Batch  172/538 - Train Accuracy:  0.725, Validation Accuracy:  0.729, Loss:  0.788
    Epoch   1 Batch  173/538 - Train Accuracy:  0.727, Validation Accuracy:  0.732, Loss:  0.768
    Epoch   1 Batch  174/538 - Train Accuracy:  0.719, Validation Accuracy:  0.731, Loss:  0.822
    Epoch   1 Batch  175/538 - Train Accuracy:  0.738, Validation Accuracy:  0.733, Loss:  0.803
    Epoch   1 Batch  176/538 - Train Accuracy:  0.725, Validation Accuracy:  0.733, Loss:  0.807
    Epoch   1 Batch  177/538 - Train Accuracy:  0.736, Validation Accuracy:  0.737, Loss:  0.771
    Epoch   1 Batch  178/538 - Train Accuracy:  0.727, Validation Accuracy:  0.736, Loss:  0.763
    Epoch   1 Batch  179/538 - Train Accuracy:  0.738, Validation Accuracy:  0.736, Loss:  0.783
    Epoch   1 Batch  180/538 - Train Accuracy:  0.745, Validation Accuracy:  0.739, Loss:  0.767
    Epoch   1 Batch  181/538 - Train Accuracy:  0.705, Validation Accuracy:  0.739, Loss:  0.800
    Epoch   1 Batch  182/538 - Train Accuracy:  0.721, Validation Accuracy:  0.736, Loss:  0.779
    Epoch   1 Batch  183/538 - Train Accuracy:  0.759, Validation Accuracy:  0.737, Loss:  0.713
    Epoch   1 Batch  184/538 - Train Accuracy:  0.753, Validation Accuracy:  0.739, Loss:  0.729
    Epoch   1 Batch  185/538 - Train Accuracy:  0.755, Validation Accuracy:  0.743, Loss:  0.737
    Epoch   1 Batch  186/538 - Train Accuracy:  0.752, Validation Accuracy:  0.743, Loss:  0.750
    Epoch   1 Batch  187/538 - Train Accuracy:  0.754, Validation Accuracy:  0.742, Loss:  0.728
    Epoch   1 Batch  188/538 - Train Accuracy:  0.747, Validation Accuracy:  0.733, Loss:  0.753
    Epoch   1 Batch  189/538 - Train Accuracy:  0.731, Validation Accuracy:  0.736, Loss:  0.759
    Epoch   1 Batch  190/538 - Train Accuracy:  0.737, Validation Accuracy:  0.737, Loss:  0.780
    Epoch   1 Batch  191/538 - Train Accuracy:  0.750, Validation Accuracy:  0.738, Loss:  0.737
    Epoch   1 Batch  192/538 - Train Accuracy:  0.746, Validation Accuracy:  0.744, Loss:  0.748
    Epoch   1 Batch  193/538 - Train Accuracy:  0.753, Validation Accuracy:  0.746, Loss:  0.717
    Epoch   1 Batch  194/538 - Train Accuracy:  0.727, Validation Accuracy:  0.749, Loss:  0.774
    Epoch   1 Batch  195/538 - Train Accuracy:  0.760, Validation Accuracy:  0.750, Loss:  0.745
    Epoch   1 Batch  196/538 - Train Accuracy:  0.750, Validation Accuracy:  0.752, Loss:  0.755
    Epoch   1 Batch  197/538 - Train Accuracy:  0.757, Validation Accuracy:  0.750, Loss:  0.716
    Epoch   1 Batch  198/538 - Train Accuracy:  0.750, Validation Accuracy:  0.749, Loss:  0.745
    Epoch   1 Batch  199/538 - Train Accuracy:  0.732, Validation Accuracy:  0.748, Loss:  0.802
    Epoch   1 Batch  200/538 - Train Accuracy:  0.741, Validation Accuracy:  0.746, Loss:  0.753
    Epoch   1 Batch  201/538 - Train Accuracy:  0.748, Validation Accuracy:  0.739, Loss:  0.727
    Epoch   1 Batch  202/538 - Train Accuracy:  0.743, Validation Accuracy:  0.745, Loss:  0.762
    Epoch   1 Batch  203/538 - Train Accuracy:  0.713, Validation Accuracy:  0.750, Loss:  0.788
    Epoch   1 Batch  204/538 - Train Accuracy:  0.735, Validation Accuracy:  0.752, Loss:  0.746
    Epoch   1 Batch  205/538 - Train Accuracy:  0.753, Validation Accuracy:  0.750, Loss:  0.713
    Epoch   1 Batch  206/538 - Train Accuracy:  0.714, Validation Accuracy:  0.752, Loss:  0.758
    Epoch   1 Batch  207/538 - Train Accuracy:  0.762, Validation Accuracy:  0.751, Loss:  0.715
    Epoch   1 Batch  208/538 - Train Accuracy:  0.754, Validation Accuracy:  0.751, Loss:  0.745
    Epoch   1 Batch  209/538 - Train Accuracy:  0.750, Validation Accuracy:  0.746, Loss:  0.752
    Epoch   1 Batch  210/538 - Train Accuracy:  0.723, Validation Accuracy:  0.740, Loss:  0.735
    Epoch   1 Batch  211/538 - Train Accuracy:  0.721, Validation Accuracy:  0.735, Loss:  0.774
    Epoch   1 Batch  212/538 - Train Accuracy:  0.739, Validation Accuracy:  0.740, Loss:  0.725
    Epoch   1 Batch  213/538 - Train Accuracy:  0.759, Validation Accuracy:  0.749, Loss:  0.733
    Epoch   1 Batch  214/538 - Train Accuracy:  0.763, Validation Accuracy:  0.751, Loss:  0.741
    Epoch   1 Batch  215/538 - Train Accuracy:  0.747, Validation Accuracy:  0.753, Loss:  0.743
    Epoch   1 Batch  216/538 - Train Accuracy:  0.749, Validation Accuracy:  0.746, Loss:  0.742
    Epoch   1 Batch  217/538 - Train Accuracy:  0.750, Validation Accuracy:  0.738, Loss:  0.723
    Epoch   1 Batch  218/538 - Train Accuracy:  0.729, Validation Accuracy:  0.742, Loss:  0.750
    Epoch   1 Batch  219/538 - Train Accuracy:  0.738, Validation Accuracy:  0.749, Loss:  0.773
    Epoch   1 Batch  220/538 - Train Accuracy:  0.761, Validation Accuracy:  0.755, Loss:  0.720
    Epoch   1 Batch  221/538 - Train Accuracy:  0.760, Validation Accuracy:  0.751, Loss:  0.710
    Epoch   1 Batch  222/538 - Train Accuracy:  0.742, Validation Accuracy:  0.746, Loss:  0.700
    Epoch   1 Batch  223/538 - Train Accuracy:  0.734, Validation Accuracy:  0.746, Loss:  0.768
    Epoch   1 Batch  224/538 - Train Accuracy:  0.745, Validation Accuracy:  0.749, Loss:  0.755
    Epoch   1 Batch  225/538 - Train Accuracy:  0.752, Validation Accuracy:  0.746, Loss:  0.690
    Epoch   1 Batch  226/538 - Train Accuracy:  0.742, Validation Accuracy:  0.743, Loss:  0.703
    Epoch   1 Batch  227/538 - Train Accuracy:  0.759, Validation Accuracy:  0.741, Loss:  0.693
    Epoch   1 Batch  228/538 - Train Accuracy:  0.743, Validation Accuracy:  0.738, Loss:  0.705
    Epoch   1 Batch  229/538 - Train Accuracy:  0.750, Validation Accuracy:  0.743, Loss:  0.716
    Epoch   1 Batch  230/538 - Train Accuracy:  0.754, Validation Accuracy:  0.750, Loss:  0.733
    Epoch   1 Batch  231/538 - Train Accuracy:  0.752, Validation Accuracy:  0.749, Loss:  0.718
    Epoch   1 Batch  232/538 - Train Accuracy:  0.765, Validation Accuracy:  0.750, Loss:  0.717
    Epoch   1 Batch  233/538 - Train Accuracy:  0.774, Validation Accuracy:  0.746, Loss:  0.720
    Epoch   1 Batch  234/538 - Train Accuracy:  0.746, Validation Accuracy:  0.740, Loss:  0.736
    Epoch   1 Batch  235/538 - Train Accuracy:  0.755, Validation Accuracy:  0.742, Loss:  0.702
    Epoch   1 Batch  236/538 - Train Accuracy:  0.734, Validation Accuracy:  0.745, Loss:  0.749
    Epoch   1 Batch  237/538 - Train Accuracy:  0.772, Validation Accuracy:  0.749, Loss:  0.687
    Epoch   1 Batch  238/538 - Train Accuracy:  0.776, Validation Accuracy:  0.744, Loss:  0.681
    Epoch   1 Batch  239/538 - Train Accuracy:  0.739, Validation Accuracy:  0.741, Loss:  0.748
    Epoch   1 Batch  240/538 - Train Accuracy:  0.729, Validation Accuracy:  0.741, Loss:  0.738
    Epoch   1 Batch  241/538 - Train Accuracy:  0.739, Validation Accuracy:  0.740, Loss:  0.736
    Epoch   1 Batch  242/538 - Train Accuracy:  0.759, Validation Accuracy:  0.744, Loss:  0.705
    Epoch   1 Batch  243/538 - Train Accuracy:  0.751, Validation Accuracy:  0.755, Loss:  0.733
    Epoch   1 Batch  244/538 - Train Accuracy:  0.754, Validation Accuracy:  0.757, Loss:  0.709
    Epoch   1 Batch  245/538 - Train Accuracy:  0.741, Validation Accuracy:  0.754, Loss:  0.735
    Epoch   1 Batch  246/538 - Train Accuracy:  0.760, Validation Accuracy:  0.745, Loss:  0.683
    Epoch   1 Batch  247/538 - Train Accuracy:  0.719, Validation Accuracy:  0.748, Loss:  0.731
    Epoch   1 Batch  248/538 - Train Accuracy:  0.760, Validation Accuracy:  0.754, Loss:  0.721
    Epoch   1 Batch  249/538 - Train Accuracy:  0.766, Validation Accuracy:  0.757, Loss:  0.675
    Epoch   1 Batch  250/538 - Train Accuracy:  0.768, Validation Accuracy:  0.760, Loss:  0.700
    Epoch   1 Batch  251/538 - Train Accuracy:  0.747, Validation Accuracy:  0.758, Loss:  0.709
    Epoch   1 Batch  252/538 - Train Accuracy:  0.764, Validation Accuracy:  0.748, Loss:  0.655
    Epoch   1 Batch  253/538 - Train Accuracy:  0.737, Validation Accuracy:  0.741, Loss:  0.676
    Epoch   1 Batch  254/538 - Train Accuracy:  0.735, Validation Accuracy:  0.741, Loss:  0.717
    Epoch   1 Batch  255/538 - Train Accuracy:  0.751, Validation Accuracy:  0.752, Loss:  0.714
    Epoch   1 Batch  256/538 - Train Accuracy:  0.744, Validation Accuracy:  0.759, Loss:  0.720
    Epoch   1 Batch  257/538 - Train Accuracy:  0.755, Validation Accuracy:  0.758, Loss:  0.689
    Epoch   1 Batch  258/538 - Train Accuracy:  0.777, Validation Accuracy:  0.757, Loss:  0.669
    Epoch   1 Batch  259/538 - Train Accuracy:  0.764, Validation Accuracy:  0.760, Loss:  0.676
    Epoch   1 Batch  260/538 - Train Accuracy:  0.718, Validation Accuracy:  0.757, Loss:  0.708
    Epoch   1 Batch  261/538 - Train Accuracy:  0.747, Validation Accuracy:  0.752, Loss:  0.716
    Epoch   1 Batch  262/538 - Train Accuracy:  0.748, Validation Accuracy:  0.750, Loss:  0.707
    Epoch   1 Batch  263/538 - Train Accuracy:  0.744, Validation Accuracy:  0.754, Loss:  0.684
    Epoch   1 Batch  264/538 - Train Accuracy:  0.737, Validation Accuracy:  0.757, Loss:  0.711
    Epoch   1 Batch  265/538 - Train Accuracy:  0.729, Validation Accuracy:  0.755, Loss:  0.732
    Epoch   1 Batch  266/538 - Train Accuracy:  0.753, Validation Accuracy:  0.755, Loss:  0.700
    Epoch   1 Batch  267/538 - Train Accuracy:  0.761, Validation Accuracy:  0.754, Loss:  0.695
    Epoch   1 Batch  268/538 - Train Accuracy:  0.786, Validation Accuracy:  0.756, Loss:  0.638
    Epoch   1 Batch  269/538 - Train Accuracy:  0.743, Validation Accuracy:  0.755, Loss:  0.696
    Epoch   1 Batch  270/538 - Train Accuracy:  0.761, Validation Accuracy:  0.754, Loss:  0.686
    Epoch   1 Batch  271/538 - Train Accuracy:  0.751, Validation Accuracy:  0.760, Loss:  0.687
    Epoch   1 Batch  272/538 - Train Accuracy:  0.747, Validation Accuracy:  0.759, Loss:  0.750
    Epoch   1 Batch  273/538 - Train Accuracy:  0.764, Validation Accuracy:  0.760, Loss:  0.696
    Epoch   1 Batch  274/538 - Train Accuracy:  0.728, Validation Accuracy:  0.760, Loss:  0.747
    Epoch   1 Batch  275/538 - Train Accuracy:  0.738, Validation Accuracy:  0.762, Loss:  0.720
    Epoch   1 Batch  276/538 - Train Accuracy:  0.754, Validation Accuracy:  0.758, Loss:  0.705
    Epoch   1 Batch  277/538 - Train Accuracy:  0.759, Validation Accuracy:  0.754, Loss:  0.683
    Epoch   1 Batch  278/538 - Train Accuracy:  0.774, Validation Accuracy:  0.746, Loss:  0.660
    Epoch   1 Batch  279/538 - Train Accuracy:  0.741, Validation Accuracy:  0.747, Loss:  0.676
    Epoch   1 Batch  280/538 - Train Accuracy:  0.767, Validation Accuracy:  0.759, Loss:  0.653
    Epoch   1 Batch  281/538 - Train Accuracy:  0.750, Validation Accuracy:  0.767, Loss:  0.706
    Epoch   1 Batch  282/538 - Train Accuracy:  0.780, Validation Accuracy:  0.763, Loss:  0.674
    Epoch   1 Batch  283/538 - Train Accuracy:  0.772, Validation Accuracy:  0.760, Loss:  0.658
    Epoch   1 Batch  284/538 - Train Accuracy:  0.756, Validation Accuracy:  0.753, Loss:  0.691
    Epoch   1 Batch  285/538 - Train Accuracy:  0.752, Validation Accuracy:  0.750, Loss:  0.635
    Epoch   1 Batch  286/538 - Train Accuracy:  0.735, Validation Accuracy:  0.754, Loss:  0.704
    Epoch   1 Batch  287/538 - Train Accuracy:  0.777, Validation Accuracy:  0.763, Loss:  0.653
    Epoch   1 Batch  288/538 - Train Accuracy:  0.762, Validation Accuracy:  0.762, Loss:  0.694
    Epoch   1 Batch  289/538 - Train Accuracy:  0.767, Validation Accuracy:  0.765, Loss:  0.629
    Epoch   1 Batch  290/538 - Train Accuracy:  0.773, Validation Accuracy:  0.769, Loss:  0.669
    Epoch   1 Batch  291/538 - Train Accuracy:  0.766, Validation Accuracy:  0.766, Loss:  0.643
    Epoch   1 Batch  292/538 - Train Accuracy:  0.773, Validation Accuracy:  0.766, Loss:  0.647
    Epoch   1 Batch  293/538 - Train Accuracy:  0.771, Validation Accuracy:  0.762, Loss:  0.660
    Epoch   1 Batch  294/538 - Train Accuracy:  0.746, Validation Accuracy:  0.765, Loss:  0.707
    Epoch   1 Batch  295/538 - Train Accuracy:  0.771, Validation Accuracy:  0.770, Loss:  0.627
    Epoch   1 Batch  296/538 - Train Accuracy:  0.759, Validation Accuracy:  0.771, Loss:  0.653
    Epoch   1 Batch  297/538 - Train Accuracy:  0.754, Validation Accuracy:  0.771, Loss:  0.678
    Epoch   1 Batch  298/538 - Train Accuracy:  0.744, Validation Accuracy:  0.770, Loss:  0.670
    Epoch   1 Batch  299/538 - Train Accuracy:  0.761, Validation Accuracy:  0.769, Loss:  0.680
    Epoch   1 Batch  300/538 - Train Accuracy:  0.772, Validation Accuracy:  0.776, Loss:  0.643
    Epoch   1 Batch  301/538 - Train Accuracy:  0.757, Validation Accuracy:  0.778, Loss:  0.692
    Epoch   1 Batch  302/538 - Train Accuracy:  0.781, Validation Accuracy:  0.775, Loss:  0.637
    Epoch   1 Batch  303/538 - Train Accuracy:  0.791, Validation Accuracy:  0.771, Loss:  0.622
    Epoch   1 Batch  304/538 - Train Accuracy:  0.748, Validation Accuracy:  0.765, Loss:  0.677
    Epoch   1 Batch  305/538 - Train Accuracy:  0.764, Validation Accuracy:  0.762, Loss:  0.635
    Epoch   1 Batch  306/538 - Train Accuracy:  0.756, Validation Accuracy:  0.763, Loss:  0.669
    Epoch   1 Batch  307/538 - Train Accuracy:  0.767, Validation Accuracy:  0.771, Loss:  0.661
    Epoch   1 Batch  308/538 - Train Accuracy:  0.794, Validation Accuracy:  0.782, Loss:  0.643
    Epoch   1 Batch  309/538 - Train Accuracy:  0.777, Validation Accuracy:  0.781, Loss:  0.661
    Epoch   1 Batch  310/538 - Train Accuracy:  0.770, Validation Accuracy:  0.776, Loss:  0.663
    Epoch   1 Batch  311/538 - Train Accuracy:  0.783, Validation Accuracy:  0.767, Loss:  0.633
    Epoch   1 Batch  312/538 - Train Accuracy:  0.771, Validation Accuracy:  0.770, Loss:  0.607
    Epoch   1 Batch  313/538 - Train Accuracy:  0.774, Validation Accuracy:  0.772, Loss:  0.689
    Epoch   1 Batch  314/538 - Train Accuracy:  0.773, Validation Accuracy:  0.771, Loss:  0.662
    Epoch   1 Batch  315/538 - Train Accuracy:  0.771, Validation Accuracy:  0.771, Loss:  0.656
    Epoch   1 Batch  316/538 - Train Accuracy:  0.777, Validation Accuracy:  0.763, Loss:  0.644
    Epoch   1 Batch  317/538 - Train Accuracy:  0.771, Validation Accuracy:  0.758, Loss:  0.648
    Epoch   1 Batch  318/538 - Train Accuracy:  0.775, Validation Accuracy:  0.765, Loss:  0.650
    Epoch   1 Batch  319/538 - Train Accuracy:  0.762, Validation Accuracy:  0.761, Loss:  0.616
    Epoch   1 Batch  320/538 - Train Accuracy:  0.759, Validation Accuracy:  0.763, Loss:  0.658
    Epoch   1 Batch  321/538 - Train Accuracy:  0.773, Validation Accuracy:  0.767, Loss:  0.628
    Epoch   1 Batch  322/538 - Train Accuracy:  0.770, Validation Accuracy:  0.772, Loss:  0.663
    Epoch   1 Batch  323/538 - Train Accuracy:  0.774, Validation Accuracy:  0.771, Loss:  0.624
    Epoch   1 Batch  324/538 - Train Accuracy:  0.758, Validation Accuracy:  0.776, Loss:  0.693
    Epoch   1 Batch  325/538 - Train Accuracy:  0.788, Validation Accuracy:  0.781, Loss:  0.640
    Epoch   1 Batch  326/538 - Train Accuracy:  0.781, Validation Accuracy:  0.783, Loss:  0.639
    Epoch   1 Batch  327/538 - Train Accuracy:  0.779, Validation Accuracy:  0.779, Loss:  0.658
    Epoch   1 Batch  328/538 - Train Accuracy:  0.796, Validation Accuracy:  0.775, Loss:  0.611
    Epoch   1 Batch  329/538 - Train Accuracy:  0.772, Validation Accuracy:  0.775, Loss:  0.622
    Epoch   1 Batch  330/538 - Train Accuracy:  0.783, Validation Accuracy:  0.770, Loss:  0.617
    Epoch   1 Batch  331/538 - Train Accuracy:  0.764, Validation Accuracy:  0.767, Loss:  0.641
    Epoch   1 Batch  332/538 - Train Accuracy:  0.774, Validation Accuracy:  0.767, Loss:  0.647
    Epoch   1 Batch  333/538 - Train Accuracy:  0.783, Validation Accuracy:  0.772, Loss:  0.630
    Epoch   1 Batch  334/538 - Train Accuracy:  0.797, Validation Accuracy:  0.773, Loss:  0.601
    Epoch   1 Batch  335/538 - Train Accuracy:  0.773, Validation Accuracy:  0.773, Loss:  0.627
    Epoch   1 Batch  336/538 - Train Accuracy:  0.778, Validation Accuracy:  0.777, Loss:  0.630
    Epoch   1 Batch  337/538 - Train Accuracy:  0.779, Validation Accuracy:  0.775, Loss:  0.629
    Epoch   1 Batch  338/538 - Train Accuracy:  0.764, Validation Accuracy:  0.773, Loss:  0.648
    Epoch   1 Batch  339/538 - Train Accuracy:  0.757, Validation Accuracy:  0.772, Loss:  0.627
    Epoch   1 Batch  340/538 - Train Accuracy:  0.750, Validation Accuracy:  0.768, Loss:  0.645
    Epoch   1 Batch  341/538 - Train Accuracy:  0.758, Validation Accuracy:  0.767, Loss:  0.644
    Epoch   1 Batch  342/538 - Train Accuracy:  0.773, Validation Accuracy:  0.767, Loss:  0.629
    Epoch   1 Batch  343/538 - Train Accuracy:  0.776, Validation Accuracy:  0.770, Loss:  0.643
    Epoch   1 Batch  344/538 - Train Accuracy:  0.797, Validation Accuracy:  0.771, Loss:  0.617
    Epoch   1 Batch  345/538 - Train Accuracy:  0.787, Validation Accuracy:  0.773, Loss:  0.598
    Epoch   1 Batch  346/538 - Train Accuracy:  0.763, Validation Accuracy:  0.772, Loss:  0.653
    Epoch   1 Batch  347/538 - Train Accuracy:  0.776, Validation Accuracy:  0.778, Loss:  0.623
    Epoch   1 Batch  348/538 - Train Accuracy:  0.791, Validation Accuracy:  0.776, Loss:  0.608
    Epoch   1 Batch  349/538 - Train Accuracy:  0.803, Validation Accuracy:  0.772, Loss:  0.623
    Epoch   1 Batch  350/538 - Train Accuracy:  0.773, Validation Accuracy:  0.769, Loss:  0.655
    Epoch   1 Batch  351/538 - Train Accuracy:  0.766, Validation Accuracy:  0.773, Loss:  0.668
    Epoch   1 Batch  352/538 - Train Accuracy:  0.769, Validation Accuracy:  0.772, Loss:  0.637
    Epoch   1 Batch  353/538 - Train Accuracy:  0.772, Validation Accuracy:  0.767, Loss:  0.656
    Epoch   1 Batch  354/538 - Train Accuracy:  0.756, Validation Accuracy:  0.768, Loss:  0.665
    Epoch   1 Batch  355/538 - Train Accuracy:  0.767, Validation Accuracy:  0.774, Loss:  0.651
    Epoch   1 Batch  356/538 - Train Accuracy:  0.780, Validation Accuracy:  0.774, Loss:  0.594
    Epoch   1 Batch  357/538 - Train Accuracy:  0.774, Validation Accuracy:  0.775, Loss:  0.630
    Epoch   1 Batch  358/538 - Train Accuracy:  0.792, Validation Accuracy:  0.777, Loss:  0.607
    Epoch   1 Batch  359/538 - Train Accuracy:  0.769, Validation Accuracy:  0.779, Loss:  0.621
    Epoch   1 Batch  360/538 - Train Accuracy:  0.768, Validation Accuracy:  0.775, Loss:  0.655
    Epoch   1 Batch  361/538 - Train Accuracy:  0.792, Validation Accuracy:  0.773, Loss:  0.605
    Epoch   1 Batch  362/538 - Train Accuracy:  0.784, Validation Accuracy:  0.769, Loss:  0.588
    Epoch   1 Batch  363/538 - Train Accuracy:  0.779, Validation Accuracy:  0.777, Loss:  0.605
    Epoch   1 Batch  364/538 - Train Accuracy:  0.754, Validation Accuracy:  0.784, Loss:  0.660
    Epoch   1 Batch  365/538 - Train Accuracy:  0.777, Validation Accuracy:  0.784, Loss:  0.656
    Epoch   1 Batch  366/538 - Train Accuracy:  0.790, Validation Accuracy:  0.780, Loss:  0.637
    Epoch   1 Batch  367/538 - Train Accuracy:  0.790, Validation Accuracy:  0.781, Loss:  0.599
    Epoch   1 Batch  368/538 - Train Accuracy:  0.798, Validation Accuracy:  0.775, Loss:  0.555
    Epoch   1 Batch  369/538 - Train Accuracy:  0.779, Validation Accuracy:  0.784, Loss:  0.611
    Epoch   1 Batch  370/538 - Train Accuracy:  0.785, Validation Accuracy:  0.782, Loss:  0.647
    Epoch   1 Batch  371/538 - Train Accuracy:  0.789, Validation Accuracy:  0.786, Loss:  0.597
    Epoch   1 Batch  372/538 - Train Accuracy:  0.803, Validation Accuracy:  0.784, Loss:  0.603
    Epoch   1 Batch  373/538 - Train Accuracy:  0.769, Validation Accuracy:  0.774, Loss:  0.610
    Epoch   1 Batch  374/538 - Train Accuracy:  0.788, Validation Accuracy:  0.774, Loss:  0.616
    Epoch   1 Batch  375/538 - Train Accuracy:  0.800, Validation Accuracy:  0.775, Loss:  0.562
    Epoch   1 Batch  376/538 - Train Accuracy:  0.782, Validation Accuracy:  0.779, Loss:  0.621
    Epoch   1 Batch  377/538 - Train Accuracy:  0.791, Validation Accuracy:  0.780, Loss:  0.613
    Epoch   1 Batch  378/538 - Train Accuracy:  0.799, Validation Accuracy:  0.779, Loss:  0.582
    Epoch   1 Batch  379/538 - Train Accuracy:  0.786, Validation Accuracy:  0.779, Loss:  0.602
    Epoch   1 Batch  380/538 - Train Accuracy:  0.784, Validation Accuracy:  0.776, Loss:  0.599
    Epoch   1 Batch  381/538 - Train Accuracy:  0.798, Validation Accuracy:  0.781, Loss:  0.566
    Epoch   1 Batch  382/538 - Train Accuracy:  0.775, Validation Accuracy:  0.784, Loss:  0.622
    Epoch   1 Batch  383/538 - Train Accuracy:  0.783, Validation Accuracy:  0.788, Loss:  0.616
    Epoch   1 Batch  384/538 - Train Accuracy:  0.789, Validation Accuracy:  0.781, Loss:  0.612
    Epoch   1 Batch  385/538 - Train Accuracy:  0.793, Validation Accuracy:  0.781, Loss:  0.604
    Epoch   1 Batch  386/538 - Train Accuracy:  0.790, Validation Accuracy:  0.779, Loss:  0.633
    Epoch   1 Batch  387/538 - Train Accuracy:  0.790, Validation Accuracy:  0.778, Loss:  0.613
    Epoch   1 Batch  388/538 - Train Accuracy:  0.793, Validation Accuracy:  0.784, Loss:  0.591
    Epoch   1 Batch  389/538 - Train Accuracy:  0.772, Validation Accuracy:  0.790, Loss:  0.650
    Epoch   1 Batch  390/538 - Train Accuracy:  0.800, Validation Accuracy:  0.790, Loss:  0.584
    Epoch   1 Batch  391/538 - Train Accuracy:  0.790, Validation Accuracy:  0.789, Loss:  0.609
    Epoch   1 Batch  392/538 - Train Accuracy:  0.784, Validation Accuracy:  0.787, Loss:  0.593
    Epoch   1 Batch  393/538 - Train Accuracy:  0.810, Validation Accuracy:  0.781, Loss:  0.566
    Epoch   1 Batch  394/538 - Train Accuracy:  0.749, Validation Accuracy:  0.777, Loss:  0.634
    Epoch   1 Batch  395/538 - Train Accuracy:  0.784, Validation Accuracy:  0.784, Loss:  0.625
    Epoch   1 Batch  396/538 - Train Accuracy:  0.799, Validation Accuracy:  0.790, Loss:  0.596
    Epoch   1 Batch  397/538 - Train Accuracy:  0.779, Validation Accuracy:  0.791, Loss:  0.636
    Epoch   1 Batch  398/538 - Train Accuracy:  0.786, Validation Accuracy:  0.795, Loss:  0.615
    Epoch   1 Batch  399/538 - Train Accuracy:  0.772, Validation Accuracy:  0.797, Loss:  0.640
    Epoch   1 Batch  400/538 - Train Accuracy:  0.797, Validation Accuracy:  0.789, Loss:  0.596
    Epoch   1 Batch  401/538 - Train Accuracy:  0.788, Validation Accuracy:  0.787, Loss:  0.609
    Epoch   1 Batch  402/538 - Train Accuracy:  0.793, Validation Accuracy:  0.787, Loss:  0.597
    Epoch   1 Batch  403/538 - Train Accuracy:  0.795, Validation Accuracy:  0.788, Loss:  0.602
    Epoch   1 Batch  404/538 - Train Accuracy:  0.781, Validation Accuracy:  0.781, Loss:  0.584
    Epoch   1 Batch  405/538 - Train Accuracy:  0.772, Validation Accuracy:  0.780, Loss:  0.592
    Epoch   1 Batch  406/538 - Train Accuracy:  0.778, Validation Accuracy:  0.784, Loss:  0.595
    Epoch   1 Batch  407/538 - Train Accuracy:  0.797, Validation Accuracy:  0.780, Loss:  0.580
    Epoch   1 Batch  408/538 - Train Accuracy:  0.771, Validation Accuracy:  0.784, Loss:  0.645
    Epoch   1 Batch  409/538 - Train Accuracy:  0.787, Validation Accuracy:  0.791, Loss:  0.608
    Epoch   1 Batch  410/538 - Train Accuracy:  0.810, Validation Accuracy:  0.790, Loss:  0.585
    Epoch   1 Batch  411/538 - Train Accuracy:  0.817, Validation Accuracy:  0.790, Loss:  0.564
    Epoch   1 Batch  412/538 - Train Accuracy:  0.791, Validation Accuracy:  0.788, Loss:  0.563
    Epoch   1 Batch  413/538 - Train Accuracy:  0.779, Validation Accuracy:  0.782, Loss:  0.605
    Epoch   1 Batch  414/538 - Train Accuracy:  0.744, Validation Accuracy:  0.784, Loss:  0.616
    Epoch   1 Batch  415/538 - Train Accuracy:  0.769, Validation Accuracy:  0.788, Loss:  0.610
    Epoch   1 Batch  416/538 - Train Accuracy:  0.806, Validation Accuracy:  0.784, Loss:  0.559
    Epoch   1 Batch  417/538 - Train Accuracy:  0.793, Validation Accuracy:  0.785, Loss:  0.605
    Epoch   1 Batch  418/538 - Train Accuracy:  0.811, Validation Accuracy:  0.794, Loss:  0.610
    Epoch   1 Batch  419/538 - Train Accuracy:  0.794, Validation Accuracy:  0.793, Loss:  0.576
    Epoch   1 Batch  420/538 - Train Accuracy:  0.791, Validation Accuracy:  0.789, Loss:  0.586
    Epoch   1 Batch  421/538 - Train Accuracy:  0.803, Validation Accuracy:  0.793, Loss:  0.581
    Epoch   1 Batch  422/538 - Train Accuracy:  0.802, Validation Accuracy:  0.797, Loss:  0.594
    Epoch   1 Batch  423/538 - Train Accuracy:  0.806, Validation Accuracy:  0.794, Loss:  0.600
    Epoch   1 Batch  424/538 - Train Accuracy:  0.782, Validation Accuracy:  0.789, Loss:  0.594
    Epoch   1 Batch  425/538 - Train Accuracy:  0.782, Validation Accuracy:  0.788, Loss:  0.575
    Epoch   1 Batch  426/538 - Train Accuracy:  0.793, Validation Accuracy:  0.784, Loss:  0.567
    Epoch   1 Batch  427/538 - Train Accuracy:  0.785, Validation Accuracy:  0.788, Loss:  0.584
    Epoch   1 Batch  428/538 - Train Accuracy:  0.808, Validation Accuracy:  0.790, Loss:  0.553
    Epoch   1 Batch  429/538 - Train Accuracy:  0.802, Validation Accuracy:  0.796, Loss:  0.571
    Epoch   1 Batch  430/538 - Train Accuracy:  0.785, Validation Accuracy:  0.799, Loss:  0.597
    Epoch   1 Batch  431/538 - Train Accuracy:  0.802, Validation Accuracy:  0.798, Loss:  0.568
    Epoch   1 Batch  432/538 - Train Accuracy:  0.807, Validation Accuracy:  0.800, Loss:  0.537
    Epoch   1 Batch  433/538 - Train Accuracy:  0.780, Validation Accuracy:  0.796, Loss:  0.615
    Epoch   1 Batch  434/538 - Train Accuracy:  0.771, Validation Accuracy:  0.801, Loss:  0.600
    Epoch   1 Batch  435/538 - Train Accuracy:  0.800, Validation Accuracy:  0.801, Loss:  0.575
    Epoch   1 Batch  436/538 - Train Accuracy:  0.797, Validation Accuracy:  0.803, Loss:  0.594
    Epoch   1 Batch  437/538 - Train Accuracy:  0.797, Validation Accuracy:  0.807, Loss:  0.601
    Epoch   1 Batch  438/538 - Train Accuracy:  0.796, Validation Accuracy:  0.799, Loss:  0.568
    Epoch   1 Batch  439/538 - Train Accuracy:  0.820, Validation Accuracy:  0.787, Loss:  0.545
    Epoch   1 Batch  440/538 - Train Accuracy:  0.786, Validation Accuracy:  0.794, Loss:  0.597
    Epoch   1 Batch  441/538 - Train Accuracy:  0.772, Validation Accuracy:  0.792, Loss:  0.595
    Epoch   1 Batch  442/538 - Train Accuracy:  0.802, Validation Accuracy:  0.794, Loss:  0.532
    Epoch   1 Batch  443/538 - Train Accuracy:  0.802, Validation Accuracy:  0.791, Loss:  0.580
    Epoch   1 Batch  444/538 - Train Accuracy:  0.817, Validation Accuracy:  0.791, Loss:  0.538
    Epoch   1 Batch  445/538 - Train Accuracy:  0.806, Validation Accuracy:  0.795, Loss:  0.553
    Epoch   1 Batch  446/538 - Train Accuracy:  0.811, Validation Accuracy:  0.800, Loss:  0.537
    Epoch   1 Batch  447/538 - Train Accuracy:  0.780, Validation Accuracy:  0.800, Loss:  0.572
    Epoch   1 Batch  448/538 - Train Accuracy:  0.784, Validation Accuracy:  0.801, Loss:  0.542
    Epoch   1 Batch  449/538 - Train Accuracy:  0.801, Validation Accuracy:  0.807, Loss:  0.585
    Epoch   1 Batch  450/538 - Train Accuracy:  0.807, Validation Accuracy:  0.807, Loss:  0.596
    Epoch   1 Batch  451/538 - Train Accuracy:  0.783, Validation Accuracy:  0.801, Loss:  0.572
    Epoch   1 Batch  452/538 - Train Accuracy:  0.797, Validation Accuracy:  0.797, Loss:  0.548
    Epoch   1 Batch  453/538 - Train Accuracy:  0.808, Validation Accuracy:  0.795, Loss:  0.577
    Epoch   1 Batch  454/538 - Train Accuracy:  0.801, Validation Accuracy:  0.798, Loss:  0.543
    Epoch   1 Batch  455/538 - Train Accuracy:  0.813, Validation Accuracy:  0.800, Loss:  0.532
    Epoch   1 Batch  456/538 - Train Accuracy:  0.850, Validation Accuracy:  0.806, Loss:  0.503
    Epoch   1 Batch  457/538 - Train Accuracy:  0.776, Validation Accuracy:  0.800, Loss:  0.584
    Epoch   1 Batch  458/538 - Train Accuracy:  0.807, Validation Accuracy:  0.800, Loss:  0.534
    Epoch   1 Batch  459/538 - Train Accuracy:  0.806, Validation Accuracy:  0.803, Loss:  0.549
    Epoch   1 Batch  460/538 - Train Accuracy:  0.788, Validation Accuracy:  0.801, Loss:  0.572
    Epoch   1 Batch  461/538 - Train Accuracy:  0.808, Validation Accuracy:  0.796, Loss:  0.600
    Epoch   1 Batch  462/538 - Train Accuracy:  0.783, Validation Accuracy:  0.796, Loss:  0.555
    Epoch   1 Batch  463/538 - Train Accuracy:  0.777, Validation Accuracy:  0.798, Loss:  0.585
    Epoch   1 Batch  464/538 - Train Accuracy:  0.805, Validation Accuracy:  0.806, Loss:  0.567
    Epoch   1 Batch  465/538 - Train Accuracy:  0.806, Validation Accuracy:  0.801, Loss:  0.549
    Epoch   1 Batch  466/538 - Train Accuracy:  0.809, Validation Accuracy:  0.800, Loss:  0.574
    Epoch   1 Batch  467/538 - Train Accuracy:  0.806, Validation Accuracy:  0.803, Loss:  0.547
    Epoch   1 Batch  468/538 - Train Accuracy:  0.809, Validation Accuracy:  0.801, Loss:  0.577
    Epoch   1 Batch  469/538 - Train Accuracy:  0.800, Validation Accuracy:  0.801, Loss:  0.560
    Epoch   1 Batch  470/538 - Train Accuracy:  0.816, Validation Accuracy:  0.795, Loss:  0.532
    Epoch   1 Batch  471/538 - Train Accuracy:  0.796, Validation Accuracy:  0.797, Loss:  0.551
    Epoch   1 Batch  472/538 - Train Accuracy:  0.824, Validation Accuracy:  0.794, Loss:  0.546
    Epoch   1 Batch  473/538 - Train Accuracy:  0.784, Validation Accuracy:  0.800, Loss:  0.577
    Epoch   1 Batch  474/538 - Train Accuracy:  0.809, Validation Accuracy:  0.804, Loss:  0.528
    Epoch   1 Batch  475/538 - Train Accuracy:  0.825, Validation Accuracy:  0.807, Loss:  0.528
    Epoch   1 Batch  476/538 - Train Accuracy:  0.797, Validation Accuracy:  0.808, Loss:  0.551
    Epoch   1 Batch  477/538 - Train Accuracy:  0.810, Validation Accuracy:  0.802, Loss:  0.557
    Epoch   1 Batch  478/538 - Train Accuracy:  0.811, Validation Accuracy:  0.806, Loss:  0.545
    Epoch   1 Batch  479/538 - Train Accuracy:  0.815, Validation Accuracy:  0.811, Loss:  0.531
    Epoch   1 Batch  480/538 - Train Accuracy:  0.824, Validation Accuracy:  0.807, Loss:  0.539
    Epoch   1 Batch  481/538 - Train Accuracy:  0.817, Validation Accuracy:  0.808, Loss:  0.548
    Epoch   1 Batch  482/538 - Train Accuracy:  0.817, Validation Accuracy:  0.801, Loss:  0.486
    Epoch   1 Batch  483/538 - Train Accuracy:  0.777, Validation Accuracy:  0.806, Loss:  0.575
    Epoch   1 Batch  484/538 - Train Accuracy:  0.816, Validation Accuracy:  0.806, Loss:  0.570
    Epoch   1 Batch  485/538 - Train Accuracy:  0.814, Validation Accuracy:  0.811, Loss:  0.520
    Epoch   1 Batch  486/538 - Train Accuracy:  0.830, Validation Accuracy:  0.801, Loss:  0.520
    Epoch   1 Batch  487/538 - Train Accuracy:  0.818, Validation Accuracy:  0.805, Loss:  0.499
    Epoch   1 Batch  488/538 - Train Accuracy:  0.826, Validation Accuracy:  0.809, Loss:  0.525
    Epoch   1 Batch  489/538 - Train Accuracy:  0.803, Validation Accuracy:  0.810, Loss:  0.556
    Epoch   1 Batch  490/538 - Train Accuracy:  0.815, Validation Accuracy:  0.804, Loss:  0.526
    Epoch   1 Batch  491/538 - Train Accuracy:  0.776, Validation Accuracy:  0.802, Loss:  0.573
    Epoch   1 Batch  492/538 - Train Accuracy:  0.804, Validation Accuracy:  0.806, Loss:  0.556
    Epoch   1 Batch  493/538 - Train Accuracy:  0.797, Validation Accuracy:  0.807, Loss:  0.522
    Epoch   1 Batch  494/538 - Train Accuracy:  0.797, Validation Accuracy:  0.802, Loss:  0.566
    Epoch   1 Batch  495/538 - Train Accuracy:  0.806, Validation Accuracy:  0.802, Loss:  0.543
    Epoch   1 Batch  496/538 - Train Accuracy:  0.815, Validation Accuracy:  0.804, Loss:  0.532
    Epoch   1 Batch  497/538 - Train Accuracy:  0.815, Validation Accuracy:  0.806, Loss:  0.508
    Epoch   1 Batch  498/538 - Train Accuracy:  0.805, Validation Accuracy:  0.806, Loss:  0.527
    Epoch   1 Batch  499/538 - Train Accuracy:  0.801, Validation Accuracy:  0.812, Loss:  0.524
    Epoch   1 Batch  500/538 - Train Accuracy:  0.825, Validation Accuracy:  0.814, Loss:  0.492
    Epoch   1 Batch  501/538 - Train Accuracy:  0.837, Validation Accuracy:  0.818, Loss:  0.538
    Epoch   1 Batch  502/538 - Train Accuracy:  0.823, Validation Accuracy:  0.817, Loss:  0.530
    Epoch   1 Batch  503/538 - Train Accuracy:  0.826, Validation Accuracy:  0.816, Loss:  0.520
    Epoch   1 Batch  504/538 - Train Accuracy:  0.821, Validation Accuracy:  0.814, Loss:  0.514
    Epoch   1 Batch  505/538 - Train Accuracy:  0.819, Validation Accuracy:  0.810, Loss:  0.535
    Epoch   1 Batch  506/538 - Train Accuracy:  0.811, Validation Accuracy:  0.807, Loss:  0.519
    Epoch   1 Batch  507/538 - Train Accuracy:  0.786, Validation Accuracy:  0.811, Loss:  0.560
    Epoch   1 Batch  508/538 - Train Accuracy:  0.798, Validation Accuracy:  0.808, Loss:  0.510
    Epoch   1 Batch  509/538 - Train Accuracy:  0.809, Validation Accuracy:  0.806, Loss:  0.539
    Epoch   1 Batch  510/538 - Train Accuracy:  0.832, Validation Accuracy:  0.802, Loss:  0.510
    Epoch   1 Batch  511/538 - Train Accuracy:  0.808, Validation Accuracy:  0.811, Loss:  0.519
    Epoch   1 Batch  512/538 - Train Accuracy:  0.826, Validation Accuracy:  0.802, Loss:  0.514
    Epoch   1 Batch  513/538 - Train Accuracy:  0.788, Validation Accuracy:  0.806, Loss:  0.541
    Epoch   1 Batch  514/538 - Train Accuracy:  0.812, Validation Accuracy:  0.809, Loss:  0.548
    Epoch   1 Batch  515/538 - Train Accuracy:  0.826, Validation Accuracy:  0.814, Loss:  0.519
    Epoch   1 Batch  516/538 - Train Accuracy:  0.792, Validation Accuracy:  0.812, Loss:  0.550
    Epoch   1 Batch  517/538 - Train Accuracy:  0.830, Validation Accuracy:  0.805, Loss:  0.519
    Epoch   1 Batch  518/538 - Train Accuracy:  0.795, Validation Accuracy:  0.803, Loss:  0.550
    Epoch   1 Batch  519/538 - Train Accuracy:  0.824, Validation Accuracy:  0.813, Loss:  0.518
    Epoch   1 Batch  520/538 - Train Accuracy:  0.801, Validation Accuracy:  0.818, Loss:  0.561
    Epoch   1 Batch  521/538 - Train Accuracy:  0.820, Validation Accuracy:  0.820, Loss:  0.538
    Epoch   1 Batch  522/538 - Train Accuracy:  0.809, Validation Accuracy:  0.816, Loss:  0.522
    Epoch   1 Batch  523/538 - Train Accuracy:  0.815, Validation Accuracy:  0.806, Loss:  0.520
    Epoch   1 Batch  524/538 - Train Accuracy:  0.806, Validation Accuracy:  0.806, Loss:  0.546
    Epoch   1 Batch  525/538 - Train Accuracy:  0.831, Validation Accuracy:  0.812, Loss:  0.530
    Epoch   1 Batch  526/538 - Train Accuracy:  0.814, Validation Accuracy:  0.822, Loss:  0.525
    Epoch   1 Batch  527/538 - Train Accuracy:  0.836, Validation Accuracy:  0.825, Loss:  0.502
    Epoch   1 Batch  528/538 - Train Accuracy:  0.812, Validation Accuracy:  0.827, Loss:  0.574
    Epoch   1 Batch  529/538 - Train Accuracy:  0.787, Validation Accuracy:  0.822, Loss:  0.549
    Epoch   1 Batch  530/538 - Train Accuracy:  0.778, Validation Accuracy:  0.819, Loss:  0.565
    Epoch   1 Batch  531/538 - Train Accuracy:  0.807, Validation Accuracy:  0.822, Loss:  0.523
    Epoch   1 Batch  532/538 - Train Accuracy:  0.804, Validation Accuracy:  0.825, Loss:  0.508
    Epoch   1 Batch  533/538 - Train Accuracy:  0.814, Validation Accuracy:  0.831, Loss:  0.513
    Epoch   1 Batch  534/538 - Train Accuracy:  0.828, Validation Accuracy:  0.827, Loss:  0.504
    Epoch   1 Batch  535/538 - Train Accuracy:  0.820, Validation Accuracy:  0.815, Loss:  0.498
    Epoch   1 Batch  536/538 - Train Accuracy:  0.823, Validation Accuracy:  0.804, Loss:  0.526
    Epoch   2 Batch    0/538 - Train Accuracy:  0.815, Validation Accuracy:  0.804, Loss:  0.504
    Epoch   2 Batch    1/538 - Train Accuracy:  0.828, Validation Accuracy:  0.806, Loss:  0.514
    Epoch   2 Batch    2/538 - Train Accuracy:  0.808, Validation Accuracy:  0.809, Loss:  0.536
    Epoch   2 Batch    3/538 - Train Accuracy:  0.810, Validation Accuracy:  0.812, Loss:  0.509
    Epoch   2 Batch    4/538 - Train Accuracy:  0.810, Validation Accuracy:  0.810, Loss:  0.500
    Epoch   2 Batch    5/538 - Train Accuracy:  0.798, Validation Accuracy:  0.811, Loss:  0.538
    Epoch   2 Batch    6/538 - Train Accuracy:  0.811, Validation Accuracy:  0.811, Loss:  0.508
    Epoch   2 Batch    7/538 - Train Accuracy:  0.852, Validation Accuracy:  0.818, Loss:  0.509
    Epoch   2 Batch    8/538 - Train Accuracy:  0.825, Validation Accuracy:  0.823, Loss:  0.520
    Epoch   2 Batch    9/538 - Train Accuracy:  0.810, Validation Accuracy:  0.824, Loss:  0.513
    Epoch   2 Batch   10/538 - Train Accuracy:  0.820, Validation Accuracy:  0.821, Loss:  0.536
    Epoch   2 Batch   11/538 - Train Accuracy:  0.814, Validation Accuracy:  0.810, Loss:  0.493
    Epoch   2 Batch   12/538 - Train Accuracy:  0.809, Validation Accuracy:  0.807, Loss:  0.510
    Epoch   2 Batch   13/538 - Train Accuracy:  0.819, Validation Accuracy:  0.806, Loss:  0.465
    Epoch   2 Batch   14/538 - Train Accuracy:  0.802, Validation Accuracy:  0.808, Loss:  0.496
    Epoch   2 Batch   15/538 - Train Accuracy:  0.827, Validation Accuracy:  0.809, Loss:  0.497
    Epoch   2 Batch   16/538 - Train Accuracy:  0.820, Validation Accuracy:  0.814, Loss:  0.489
    Epoch   2 Batch   17/538 - Train Accuracy:  0.817, Validation Accuracy:  0.816, Loss:  0.497
    Epoch   2 Batch   18/538 - Train Accuracy:  0.812, Validation Accuracy:  0.820, Loss:  0.529
    Epoch   2 Batch   19/538 - Train Accuracy:  0.807, Validation Accuracy:  0.823, Loss:  0.545
    Epoch   2 Batch   20/538 - Train Accuracy:  0.811, Validation Accuracy:  0.823, Loss:  0.509
    Epoch   2 Batch   21/538 - Train Accuracy:  0.833, Validation Accuracy:  0.819, Loss:  0.502
    Epoch   2 Batch   22/538 - Train Accuracy:  0.810, Validation Accuracy:  0.814, Loss:  0.529
    Epoch   2 Batch   23/538 - Train Accuracy:  0.811, Validation Accuracy:  0.813, Loss:  0.522
    Epoch   2 Batch   24/538 - Train Accuracy:  0.829, Validation Accuracy:  0.817, Loss:  0.499
    Epoch   2 Batch   25/538 - Train Accuracy:  0.825, Validation Accuracy:  0.813, Loss:  0.503
    Epoch   2 Batch   26/538 - Train Accuracy:  0.824, Validation Accuracy:  0.812, Loss:  0.529
    Epoch   2 Batch   27/538 - Train Accuracy:  0.837, Validation Accuracy:  0.810, Loss:  0.482
    Epoch   2 Batch   28/538 - Train Accuracy:  0.810, Validation Accuracy:  0.809, Loss:  0.467
    Epoch   2 Batch   29/538 - Train Accuracy:  0.833, Validation Accuracy:  0.807, Loss:  0.475
    Epoch   2 Batch   30/538 - Train Accuracy:  0.818, Validation Accuracy:  0.811, Loss:  0.516
    Epoch   2 Batch   31/538 - Train Accuracy:  0.838, Validation Accuracy:  0.817, Loss:  0.470
    Epoch   2 Batch   32/538 - Train Accuracy:  0.822, Validation Accuracy:  0.820, Loss:  0.485
    Epoch   2 Batch   33/538 - Train Accuracy:  0.839, Validation Accuracy:  0.821, Loss:  0.479
    Epoch   2 Batch   34/538 - Train Accuracy:  0.816, Validation Accuracy:  0.821, Loss:  0.527
    Epoch   2 Batch   35/538 - Train Accuracy:  0.817, Validation Accuracy:  0.821, Loss:  0.491
    Epoch   2 Batch   36/538 - Train Accuracy:  0.834, Validation Accuracy:  0.819, Loss:  0.462
    Epoch   2 Batch   37/538 - Train Accuracy:  0.821, Validation Accuracy:  0.816, Loss:  0.494
    Epoch   2 Batch   38/538 - Train Accuracy:  0.793, Validation Accuracy:  0.809, Loss:  0.518
    Epoch   2 Batch   39/538 - Train Accuracy:  0.822, Validation Accuracy:  0.798, Loss:  0.499
    Epoch   2 Batch   40/538 - Train Accuracy:  0.835, Validation Accuracy:  0.799, Loss:  0.463
    Epoch   2 Batch   41/538 - Train Accuracy:  0.827, Validation Accuracy:  0.801, Loss:  0.505
    Epoch   2 Batch   42/538 - Train Accuracy:  0.822, Validation Accuracy:  0.808, Loss:  0.497
    Epoch   2 Batch   43/538 - Train Accuracy:  0.828, Validation Accuracy:  0.817, Loss:  0.524
    Epoch   2 Batch   44/538 - Train Accuracy:  0.796, Validation Accuracy:  0.822, Loss:  0.526
    Epoch   2 Batch   45/538 - Train Accuracy:  0.817, Validation Accuracy:  0.818, Loss:  0.474
    Epoch   2 Batch   46/538 - Train Accuracy:  0.833, Validation Accuracy:  0.820, Loss:  0.478
    Epoch   2 Batch   47/538 - Train Accuracy:  0.831, Validation Accuracy:  0.822, Loss:  0.500
    Epoch   2 Batch   48/538 - Train Accuracy:  0.811, Validation Accuracy:  0.824, Loss:  0.486
    Epoch   2 Batch   49/538 - Train Accuracy:  0.816, Validation Accuracy:  0.824, Loss:  0.522
    Epoch   2 Batch   50/538 - Train Accuracy:  0.817, Validation Accuracy:  0.824, Loss:  0.471
    Epoch   2 Batch   51/538 - Train Accuracy:  0.800, Validation Accuracy:  0.814, Loss:  0.534
    Epoch   2 Batch   52/538 - Train Accuracy:  0.814, Validation Accuracy:  0.816, Loss:  0.516
    Epoch   2 Batch   53/538 - Train Accuracy:  0.823, Validation Accuracy:  0.812, Loss:  0.461
    Epoch   2 Batch   54/538 - Train Accuracy:  0.818, Validation Accuracy:  0.811, Loss:  0.474
    Epoch   2 Batch   55/538 - Train Accuracy:  0.809, Validation Accuracy:  0.817, Loss:  0.489
    Epoch   2 Batch   56/538 - Train Accuracy:  0.829, Validation Accuracy:  0.819, Loss:  0.480
    Epoch   2 Batch   57/538 - Train Accuracy:  0.798, Validation Accuracy:  0.816, Loss:  0.526
    Epoch   2 Batch   58/538 - Train Accuracy:  0.797, Validation Accuracy:  0.820, Loss:  0.511
    Epoch   2 Batch   59/538 - Train Accuracy:  0.818, Validation Accuracy:  0.825, Loss:  0.502
    Epoch   2 Batch   60/538 - Train Accuracy:  0.842, Validation Accuracy:  0.829, Loss:  0.485
    Epoch   2 Batch   61/538 - Train Accuracy:  0.839, Validation Accuracy:  0.826, Loss:  0.467
    Epoch   2 Batch   62/538 - Train Accuracy:  0.829, Validation Accuracy:  0.825, Loss:  0.482
    Epoch   2 Batch   63/538 - Train Accuracy:  0.836, Validation Accuracy:  0.807, Loss:  0.452
    Epoch   2 Batch   64/538 - Train Accuracy:  0.815, Validation Accuracy:  0.799, Loss:  0.464
    Epoch   2 Batch   65/538 - Train Accuracy:  0.794, Validation Accuracy:  0.805, Loss:  0.508
    Epoch   2 Batch   66/538 - Train Accuracy:  0.827, Validation Accuracy:  0.811, Loss:  0.457
    Epoch   2 Batch   67/538 - Train Accuracy:  0.845, Validation Accuracy:  0.820, Loss:  0.470
    Epoch   2 Batch   68/538 - Train Accuracy:  0.841, Validation Accuracy:  0.826, Loss:  0.441
    Epoch   2 Batch   69/538 - Train Accuracy:  0.826, Validation Accuracy:  0.830, Loss:  0.496
    Epoch   2 Batch   70/538 - Train Accuracy:  0.829, Validation Accuracy:  0.829, Loss:  0.483
    Epoch   2 Batch   71/538 - Train Accuracy:  0.828, Validation Accuracy:  0.833, Loss:  0.498
    Epoch   2 Batch   72/538 - Train Accuracy:  0.838, Validation Accuracy:  0.832, Loss:  0.498
    Epoch   2 Batch   73/538 - Train Accuracy:  0.813, Validation Accuracy:  0.834, Loss:  0.504
    Epoch   2 Batch   74/538 - Train Accuracy:  0.824, Validation Accuracy:  0.830, Loss:  0.459
    Epoch   2 Batch   75/538 - Train Accuracy:  0.830, Validation Accuracy:  0.831, Loss:  0.460
    Epoch   2 Batch   76/538 - Train Accuracy:  0.821, Validation Accuracy:  0.830, Loss:  0.507
    Epoch   2 Batch   77/538 - Train Accuracy:  0.835, Validation Accuracy:  0.821, Loss:  0.483
    Epoch   2 Batch   78/538 - Train Accuracy:  0.830, Validation Accuracy:  0.816, Loss:  0.480
    Epoch   2 Batch   79/538 - Train Accuracy:  0.832, Validation Accuracy:  0.817, Loss:  0.450
    Epoch   2 Batch   80/538 - Train Accuracy:  0.818, Validation Accuracy:  0.816, Loss:  0.510
    Epoch   2 Batch   81/538 - Train Accuracy:  0.821, Validation Accuracy:  0.826, Loss:  0.475
    Epoch   2 Batch   82/538 - Train Accuracy:  0.822, Validation Accuracy:  0.828, Loss:  0.469
    Epoch   2 Batch   83/538 - Train Accuracy:  0.824, Validation Accuracy:  0.830, Loss:  0.490
    Epoch   2 Batch   84/538 - Train Accuracy:  0.823, Validation Accuracy:  0.833, Loss:  0.471
    Epoch   2 Batch   85/538 - Train Accuracy:  0.847, Validation Accuracy:  0.827, Loss:  0.454
    Epoch   2 Batch   86/538 - Train Accuracy:  0.848, Validation Accuracy:  0.827, Loss:  0.473
    Epoch   2 Batch   87/538 - Train Accuracy:  0.829, Validation Accuracy:  0.825, Loss:  0.470
    Epoch   2 Batch   88/538 - Train Accuracy:  0.832, Validation Accuracy:  0.832, Loss:  0.482
    Epoch   2 Batch   89/538 - Train Accuracy:  0.832, Validation Accuracy:  0.833, Loss:  0.461
    Epoch   2 Batch   90/538 - Train Accuracy:  0.826, Validation Accuracy:  0.835, Loss:  0.475
    Epoch   2 Batch   91/538 - Train Accuracy:  0.826, Validation Accuracy:  0.834, Loss:  0.484
    Epoch   2 Batch   92/538 - Train Accuracy:  0.825, Validation Accuracy:  0.830, Loss:  0.489
    Epoch   2 Batch   93/538 - Train Accuracy:  0.818, Validation Accuracy:  0.821, Loss:  0.471
    Epoch   2 Batch   94/538 - Train Accuracy:  0.831, Validation Accuracy:  0.823, Loss:  0.477
    Epoch   2 Batch   95/538 - Train Accuracy:  0.828, Validation Accuracy:  0.820, Loss:  0.434
    Epoch   2 Batch   96/538 - Train Accuracy:  0.859, Validation Accuracy:  0.821, Loss:  0.428
    Epoch   2 Batch   97/538 - Train Accuracy:  0.836, Validation Accuracy:  0.826, Loss:  0.450
    Epoch   2 Batch   98/538 - Train Accuracy:  0.851, Validation Accuracy:  0.826, Loss:  0.434
    Epoch   2 Batch   99/538 - Train Accuracy:  0.820, Validation Accuracy:  0.827, Loss:  0.459
    Epoch   2 Batch  100/538 - Train Accuracy:  0.842, Validation Accuracy:  0.827, Loss:  0.455
    Epoch   2 Batch  101/538 - Train Accuracy:  0.816, Validation Accuracy:  0.826, Loss:  0.497
    Epoch   2 Batch  102/538 - Train Accuracy:  0.827, Validation Accuracy:  0.832, Loss:  0.483
    Epoch   2 Batch  103/538 - Train Accuracy:  0.849, Validation Accuracy:  0.832, Loss:  0.449
    Epoch   2 Batch  104/538 - Train Accuracy:  0.842, Validation Accuracy:  0.836, Loss:  0.456
    Epoch   2 Batch  105/538 - Train Accuracy:  0.845, Validation Accuracy:  0.827, Loss:  0.440
    Epoch   2 Batch  106/538 - Train Accuracy:  0.817, Validation Accuracy:  0.823, Loss:  0.442
    Epoch   2 Batch  107/538 - Train Accuracy:  0.827, Validation Accuracy:  0.826, Loss:  0.483
    Epoch   2 Batch  108/538 - Train Accuracy:  0.838, Validation Accuracy:  0.830, Loss:  0.461
    Epoch   2 Batch  109/538 - Train Accuracy:  0.849, Validation Accuracy:  0.832, Loss:  0.451
    Epoch   2 Batch  110/538 - Train Accuracy:  0.833, Validation Accuracy:  0.829, Loss:  0.472
    Epoch   2 Batch  111/538 - Train Accuracy:  0.850, Validation Accuracy:  0.828, Loss:  0.425
    Epoch   2 Batch  112/538 - Train Accuracy:  0.839, Validation Accuracy:  0.823, Loss:  0.461
    Epoch   2 Batch  113/538 - Train Accuracy:  0.826, Validation Accuracy:  0.824, Loss:  0.484
    Epoch   2 Batch  114/538 - Train Accuracy:  0.834, Validation Accuracy:  0.827, Loss:  0.437
    Epoch   2 Batch  115/538 - Train Accuracy:  0.839, Validation Accuracy:  0.829, Loss:  0.462
    Epoch   2 Batch  116/538 - Train Accuracy:  0.827, Validation Accuracy:  0.829, Loss:  0.493
    Epoch   2 Batch  117/538 - Train Accuracy:  0.837, Validation Accuracy:  0.832, Loss:  0.427
    Epoch   2 Batch  118/538 - Train Accuracy:  0.827, Validation Accuracy:  0.832, Loss:  0.436
    Epoch   2 Batch  119/538 - Train Accuracy:  0.841, Validation Accuracy:  0.826, Loss:  0.413
    Epoch   2 Batch  120/538 - Train Accuracy:  0.824, Validation Accuracy:  0.825, Loss:  0.448
    Epoch   2 Batch  121/538 - Train Accuracy:  0.850, Validation Accuracy:  0.825, Loss:  0.436
    Epoch   2 Batch  122/538 - Train Accuracy:  0.837, Validation Accuracy:  0.832, Loss:  0.429
    Epoch   2 Batch  123/538 - Train Accuracy:  0.841, Validation Accuracy:  0.837, Loss:  0.425
    Epoch   2 Batch  124/538 - Train Accuracy:  0.851, Validation Accuracy:  0.830, Loss:  0.424
    Epoch   2 Batch  125/538 - Train Accuracy:  0.830, Validation Accuracy:  0.829, Loss:  0.459
    Epoch   2 Batch  126/538 - Train Accuracy:  0.836, Validation Accuracy:  0.827, Loss:  0.448
    Epoch   2 Batch  127/538 - Train Accuracy:  0.828, Validation Accuracy:  0.827, Loss:  0.476
    Epoch   2 Batch  128/538 - Train Accuracy:  0.828, Validation Accuracy:  0.829, Loss:  0.455
    Epoch   2 Batch  129/538 - Train Accuracy:  0.835, Validation Accuracy:  0.834, Loss:  0.425
    Epoch   2 Batch  130/538 - Train Accuracy:  0.852, Validation Accuracy:  0.831, Loss:  0.430
    Epoch   2 Batch  131/538 - Train Accuracy:  0.848, Validation Accuracy:  0.825, Loss:  0.455
    Epoch   2 Batch  132/538 - Train Accuracy:  0.820, Validation Accuracy:  0.822, Loss:  0.453
    Epoch   2 Batch  133/538 - Train Accuracy:  0.841, Validation Accuracy:  0.823, Loss:  0.421
    Epoch   2 Batch  134/538 - Train Accuracy:  0.810, Validation Accuracy:  0.826, Loss:  0.480
    Epoch   2 Batch  135/538 - Train Accuracy:  0.844, Validation Accuracy:  0.830, Loss:  0.463
    Epoch   2 Batch  136/538 - Train Accuracy:  0.826, Validation Accuracy:  0.829, Loss:  0.449
    Epoch   2 Batch  137/538 - Train Accuracy:  0.835, Validation Accuracy:  0.826, Loss:  0.459
    Epoch   2 Batch  138/538 - Train Accuracy:  0.823, Validation Accuracy:  0.819, Loss:  0.448
    Epoch   2 Batch  139/538 - Train Accuracy:  0.815, Validation Accuracy:  0.822, Loss:  0.498
    Epoch   2 Batch  140/538 - Train Accuracy:  0.823, Validation Accuracy:  0.829, Loss:  0.480
    Epoch   2 Batch  141/538 - Train Accuracy:  0.858, Validation Accuracy:  0.827, Loss:  0.491
    Epoch   2 Batch  142/538 - Train Accuracy:  0.849, Validation Accuracy:  0.827, Loss:  0.428
    Epoch   2 Batch  143/538 - Train Accuracy:  0.847, Validation Accuracy:  0.829, Loss:  0.451
    Epoch   2 Batch  144/538 - Train Accuracy:  0.828, Validation Accuracy:  0.830, Loss:  0.455
    Epoch   2 Batch  145/538 - Train Accuracy:  0.822, Validation Accuracy:  0.830, Loss:  0.456
    Epoch   2 Batch  146/538 - Train Accuracy:  0.846, Validation Accuracy:  0.829, Loss:  0.434
    Epoch   2 Batch  147/538 - Train Accuracy:  0.841, Validation Accuracy:  0.832, Loss:  0.451
    Epoch   2 Batch  148/538 - Train Accuracy:  0.810, Validation Accuracy:  0.830, Loss:  0.511
    Epoch   2 Batch  149/538 - Train Accuracy:  0.857, Validation Accuracy:  0.831, Loss:  0.431
    Epoch   2 Batch  150/538 - Train Accuracy:  0.847, Validation Accuracy:  0.827, Loss:  0.451
    Epoch   2 Batch  151/538 - Train Accuracy:  0.851, Validation Accuracy:  0.828, Loss:  0.426
    Epoch   2 Batch  152/538 - Train Accuracy:  0.851, Validation Accuracy:  0.833, Loss:  0.438
    Epoch   2 Batch  153/538 - Train Accuracy:  0.821, Validation Accuracy:  0.830, Loss:  0.456
    Epoch   2 Batch  154/538 - Train Accuracy:  0.844, Validation Accuracy:  0.827, Loss:  0.429
    Epoch   2 Batch  155/538 - Train Accuracy:  0.832, Validation Accuracy:  0.830, Loss:  0.452
    Epoch   2 Batch  156/538 - Train Accuracy:  0.849, Validation Accuracy:  0.834, Loss:  0.440
    Epoch   2 Batch  157/538 - Train Accuracy:  0.862, Validation Accuracy:  0.835, Loss:  0.414
    Epoch   2 Batch  158/538 - Train Accuracy:  0.830, Validation Accuracy:  0.833, Loss:  0.446
    Epoch   2 Batch  159/538 - Train Accuracy:  0.829, Validation Accuracy:  0.835, Loss:  0.463
    Epoch   2 Batch  160/538 - Train Accuracy:  0.838, Validation Accuracy:  0.831, Loss:  0.428
    Epoch   2 Batch  161/538 - Train Accuracy:  0.843, Validation Accuracy:  0.831, Loss:  0.429
    Epoch   2 Batch  162/538 - Train Accuracy:  0.847, Validation Accuracy:  0.832, Loss:  0.427
    Epoch   2 Batch  163/538 - Train Accuracy:  0.844, Validation Accuracy:  0.837, Loss:  0.468
    Epoch   2 Batch  164/538 - Train Accuracy:  0.812, Validation Accuracy:  0.836, Loss:  0.469
    Epoch   2 Batch  165/538 - Train Accuracy:  0.839, Validation Accuracy:  0.836, Loss:  0.398
    Epoch   2 Batch  166/538 - Train Accuracy:  0.862, Validation Accuracy:  0.827, Loss:  0.448
    Epoch   2 Batch  167/538 - Train Accuracy:  0.842, Validation Accuracy:  0.832, Loss:  0.432
    Epoch   2 Batch  168/538 - Train Accuracy:  0.827, Validation Accuracy:  0.831, Loss:  0.475
    Epoch   2 Batch  169/538 - Train Accuracy:  0.866, Validation Accuracy:  0.833, Loss:  0.414
    Epoch   2 Batch  170/538 - Train Accuracy:  0.837, Validation Accuracy:  0.837, Loss:  0.435
    Epoch   2 Batch  171/538 - Train Accuracy:  0.850, Validation Accuracy:  0.835, Loss:  0.457
    Epoch   2 Batch  172/538 - Train Accuracy:  0.833, Validation Accuracy:  0.830, Loss:  0.443
    Epoch   2 Batch  173/538 - Train Accuracy:  0.844, Validation Accuracy:  0.827, Loss:  0.421
    Epoch   2 Batch  174/538 - Train Accuracy:  0.831, Validation Accuracy:  0.828, Loss:  0.457
    Epoch   2 Batch  175/538 - Train Accuracy:  0.836, Validation Accuracy:  0.835, Loss:  0.458
    Epoch   2 Batch  176/538 - Train Accuracy:  0.824, Validation Accuracy:  0.841, Loss:  0.464
    Epoch   2 Batch  177/538 - Train Accuracy:  0.841, Validation Accuracy:  0.842, Loss:  0.444
    Epoch   2 Batch  178/538 - Train Accuracy:  0.836, Validation Accuracy:  0.835, Loss:  0.427
    Epoch   2 Batch  179/538 - Train Accuracy:  0.863, Validation Accuracy:  0.827, Loss:  0.435
    Epoch   2 Batch  180/538 - Train Accuracy:  0.846, Validation Accuracy:  0.830, Loss:  0.435
    Epoch   2 Batch  181/538 - Train Accuracy:  0.832, Validation Accuracy:  0.834, Loss:  0.462
    Epoch   2 Batch  182/538 - Train Accuracy:  0.837, Validation Accuracy:  0.837, Loss:  0.434
    Epoch   2 Batch  183/538 - Train Accuracy:  0.868, Validation Accuracy:  0.842, Loss:  0.397
    Epoch   2 Batch  184/538 - Train Accuracy:  0.831, Validation Accuracy:  0.841, Loss:  0.423
    Epoch   2 Batch  185/538 - Train Accuracy:  0.861, Validation Accuracy:  0.843, Loss:  0.410
    Epoch   2 Batch  186/538 - Train Accuracy:  0.853, Validation Accuracy:  0.844, Loss:  0.422
    Epoch   2 Batch  187/538 - Train Accuracy:  0.858, Validation Accuracy:  0.842, Loss:  0.423
    Epoch   2 Batch  188/538 - Train Accuracy:  0.849, Validation Accuracy:  0.837, Loss:  0.415
    Epoch   2 Batch  189/538 - Train Accuracy:  0.855, Validation Accuracy:  0.836, Loss:  0.444
    Epoch   2 Batch  190/538 - Train Accuracy:  0.839, Validation Accuracy:  0.834, Loss:  0.450
    Epoch   2 Batch  191/538 - Train Accuracy:  0.859, Validation Accuracy:  0.834, Loss:  0.414
    Epoch   2 Batch  192/538 - Train Accuracy:  0.856, Validation Accuracy:  0.832, Loss:  0.420
    Epoch   2 Batch  193/538 - Train Accuracy:  0.841, Validation Accuracy:  0.830, Loss:  0.412
    Epoch   2 Batch  194/538 - Train Accuracy:  0.833, Validation Accuracy:  0.837, Loss:  0.438
    Epoch   2 Batch  195/538 - Train Accuracy:  0.863, Validation Accuracy:  0.836, Loss:  0.411
    Epoch   2 Batch  196/538 - Train Accuracy:  0.836, Validation Accuracy:  0.835, Loss:  0.434
    Epoch   2 Batch  197/538 - Train Accuracy:  0.849, Validation Accuracy:  0.838, Loss:  0.414
    Epoch   2 Batch  198/538 - Train Accuracy:  0.857, Validation Accuracy:  0.835, Loss:  0.426
    Epoch   2 Batch  199/538 - Train Accuracy:  0.831, Validation Accuracy:  0.835, Loss:  0.450
    Epoch   2 Batch  200/538 - Train Accuracy:  0.860, Validation Accuracy:  0.834, Loss:  0.407
    Epoch   2 Batch  201/538 - Train Accuracy:  0.852, Validation Accuracy:  0.832, Loss:  0.425
    Epoch   2 Batch  202/538 - Train Accuracy:  0.860, Validation Accuracy:  0.830, Loss:  0.421
    Epoch   2 Batch  203/538 - Train Accuracy:  0.826, Validation Accuracy:  0.833, Loss:  0.455
    Epoch   2 Batch  204/538 - Train Accuracy:  0.841, Validation Accuracy:  0.829, Loss:  0.423
    Epoch   2 Batch  205/538 - Train Accuracy:  0.858, Validation Accuracy:  0.830, Loss:  0.405
    Epoch   2 Batch  206/538 - Train Accuracy:  0.832, Validation Accuracy:  0.834, Loss:  0.435
    Epoch   2 Batch  207/538 - Train Accuracy:  0.859, Validation Accuracy:  0.836, Loss:  0.413
    Epoch   2 Batch  208/538 - Train Accuracy:  0.856, Validation Accuracy:  0.839, Loss:  0.431
    Epoch   2 Batch  209/538 - Train Accuracy:  0.864, Validation Accuracy:  0.840, Loss:  0.426
    Epoch   2 Batch  210/538 - Train Accuracy:  0.829, Validation Accuracy:  0.840, Loss:  0.438
    Epoch   2 Batch  211/538 - Train Accuracy:  0.822, Validation Accuracy:  0.838, Loss:  0.441
    Epoch   2 Batch  212/538 - Train Accuracy:  0.831, Validation Accuracy:  0.843, Loss:  0.427
    Epoch   2 Batch  213/538 - Train Accuracy:  0.850, Validation Accuracy:  0.839, Loss:  0.404
    Epoch   2 Batch  214/538 - Train Accuracy:  0.853, Validation Accuracy:  0.844, Loss:  0.421
    Epoch   2 Batch  215/538 - Train Accuracy:  0.845, Validation Accuracy:  0.846, Loss:  0.438
    Epoch   2 Batch  216/538 - Train Accuracy:  0.858, Validation Accuracy:  0.844, Loss:  0.416
    Epoch   2 Batch  217/538 - Train Accuracy:  0.861, Validation Accuracy:  0.838, Loss:  0.410
    Epoch   2 Batch  218/538 - Train Accuracy:  0.835, Validation Accuracy:  0.837, Loss:  0.421
    Epoch   2 Batch  219/538 - Train Accuracy:  0.822, Validation Accuracy:  0.834, Loss:  0.456
    Epoch   2 Batch  220/538 - Train Accuracy:  0.840, Validation Accuracy:  0.836, Loss:  0.426
    Epoch   2 Batch  221/538 - Train Accuracy:  0.870, Validation Accuracy:  0.839, Loss:  0.402
    Epoch   2 Batch  222/538 - Train Accuracy:  0.836, Validation Accuracy:  0.838, Loss:  0.388
    Epoch   2 Batch  223/538 - Train Accuracy:  0.834, Validation Accuracy:  0.834, Loss:  0.444
    Epoch   2 Batch  224/538 - Train Accuracy:  0.855, Validation Accuracy:  0.836, Loss:  0.428
    Epoch   2 Batch  225/538 - Train Accuracy:  0.854, Validation Accuracy:  0.838, Loss:  0.410
    Epoch   2 Batch  226/538 - Train Accuracy:  0.849, Validation Accuracy:  0.843, Loss:  0.416
    Epoch   2 Batch  227/538 - Train Accuracy:  0.859, Validation Accuracy:  0.841, Loss:  0.398
    Epoch   2 Batch  228/538 - Train Accuracy:  0.848, Validation Accuracy:  0.843, Loss:  0.395
    Epoch   2 Batch  229/538 - Train Accuracy:  0.847, Validation Accuracy:  0.837, Loss:  0.402
    Epoch   2 Batch  230/538 - Train Accuracy:  0.862, Validation Accuracy:  0.844, Loss:  0.409
    Epoch   2 Batch  231/538 - Train Accuracy:  0.861, Validation Accuracy:  0.844, Loss:  0.414
    Epoch   2 Batch  232/538 - Train Accuracy:  0.854, Validation Accuracy:  0.842, Loss:  0.400
    Epoch   2 Batch  233/538 - Train Accuracy:  0.860, Validation Accuracy:  0.841, Loss:  0.411
    Epoch   2 Batch  234/538 - Train Accuracy:  0.858, Validation Accuracy:  0.841, Loss:  0.420
    Epoch   2 Batch  235/538 - Train Accuracy:  0.850, Validation Accuracy:  0.846, Loss:  0.401
    Epoch   2 Batch  236/538 - Train Accuracy:  0.847, Validation Accuracy:  0.845, Loss:  0.430
    Epoch   2 Batch  237/538 - Train Accuracy:  0.861, Validation Accuracy:  0.845, Loss:  0.396
    Epoch   2 Batch  238/538 - Train Accuracy:  0.870, Validation Accuracy:  0.849, Loss:  0.386
    Epoch   2 Batch  239/538 - Train Accuracy:  0.839, Validation Accuracy:  0.848, Loss:  0.439
    Epoch   2 Batch  240/538 - Train Accuracy:  0.835, Validation Accuracy:  0.844, Loss:  0.439
    Epoch   2 Batch  241/538 - Train Accuracy:  0.839, Validation Accuracy:  0.845, Loss:  0.421
    Epoch   2 Batch  242/538 - Train Accuracy:  0.849, Validation Accuracy:  0.841, Loss:  0.400
    Epoch   2 Batch  243/538 - Train Accuracy:  0.865, Validation Accuracy:  0.841, Loss:  0.411
    Epoch   2 Batch  244/538 - Train Accuracy:  0.844, Validation Accuracy:  0.846, Loss:  0.394
    Epoch   2 Batch  245/538 - Train Accuracy:  0.846, Validation Accuracy:  0.844, Loss:  0.429
    Epoch   2 Batch  246/538 - Train Accuracy:  0.860, Validation Accuracy:  0.848, Loss:  0.375
    Epoch   2 Batch  247/538 - Train Accuracy:  0.839, Validation Accuracy:  0.846, Loss:  0.402
    Epoch   2 Batch  248/538 - Train Accuracy:  0.864, Validation Accuracy:  0.856, Loss:  0.409
    Epoch   2 Batch  249/538 - Train Accuracy:  0.856, Validation Accuracy:  0.850, Loss:  0.386
    Epoch   2 Batch  250/538 - Train Accuracy:  0.851, Validation Accuracy:  0.853, Loss:  0.408
    Epoch   2 Batch  251/538 - Train Accuracy:  0.849, Validation Accuracy:  0.854, Loss:  0.413
    Epoch   2 Batch  252/538 - Train Accuracy:  0.870, Validation Accuracy:  0.853, Loss:  0.372
    Epoch   2 Batch  253/538 - Train Accuracy:  0.828, Validation Accuracy:  0.849, Loss:  0.393
    Epoch   2 Batch  254/538 - Train Accuracy:  0.825, Validation Accuracy:  0.846, Loss:  0.427
    Epoch   2 Batch  255/538 - Train Accuracy:  0.857, Validation Accuracy:  0.847, Loss:  0.408
    Epoch   2 Batch  256/538 - Train Accuracy:  0.833, Validation Accuracy:  0.846, Loss:  0.434
    Epoch   2 Batch  257/538 - Train Accuracy:  0.848, Validation Accuracy:  0.843, Loss:  0.395
    Epoch   2 Batch  258/538 - Train Accuracy:  0.857, Validation Accuracy:  0.847, Loss:  0.389
    Epoch   2 Batch  259/538 - Train Accuracy:  0.874, Validation Accuracy:  0.844, Loss:  0.389
    Epoch   2 Batch  260/538 - Train Accuracy:  0.824, Validation Accuracy:  0.843, Loss:  0.426
    Epoch   2 Batch  261/538 - Train Accuracy:  0.855, Validation Accuracy:  0.846, Loss:  0.420
    Epoch   2 Batch  262/538 - Train Accuracy:  0.864, Validation Accuracy:  0.846, Loss:  0.402
    Epoch   2 Batch  263/538 - Train Accuracy:  0.839, Validation Accuracy:  0.842, Loss:  0.398
    Epoch   2 Batch  264/538 - Train Accuracy:  0.821, Validation Accuracy:  0.840, Loss:  0.412
    Epoch   2 Batch  265/538 - Train Accuracy:  0.835, Validation Accuracy:  0.842, Loss:  0.421
    Epoch   2 Batch  266/538 - Train Accuracy:  0.855, Validation Accuracy:  0.841, Loss:  0.410
    Epoch   2 Batch  267/538 - Train Accuracy:  0.865, Validation Accuracy:  0.844, Loss:  0.405
    Epoch   2 Batch  268/538 - Train Accuracy:  0.872, Validation Accuracy:  0.844, Loss:  0.361
    Epoch   2 Batch  269/538 - Train Accuracy:  0.862, Validation Accuracy:  0.838, Loss:  0.408
    Epoch   2 Batch  270/538 - Train Accuracy:  0.863, Validation Accuracy:  0.843, Loss:  0.403
    Epoch   2 Batch  271/538 - Train Accuracy:  0.860, Validation Accuracy:  0.840, Loss:  0.392
    Epoch   2 Batch  272/538 - Train Accuracy:  0.838, Validation Accuracy:  0.838, Loss:  0.440
    Epoch   2 Batch  273/538 - Train Accuracy:  0.862, Validation Accuracy:  0.838, Loss:  0.411
    Epoch   2 Batch  274/538 - Train Accuracy:  0.815, Validation Accuracy:  0.843, Loss:  0.436
    Epoch   2 Batch  275/538 - Train Accuracy:  0.839, Validation Accuracy:  0.841, Loss:  0.417
    Epoch   2 Batch  276/538 - Train Accuracy:  0.860, Validation Accuracy:  0.837, Loss:  0.417
    Epoch   2 Batch  277/538 - Train Accuracy:  0.851, Validation Accuracy:  0.842, Loss:  0.398
    Epoch   2 Batch  278/538 - Train Accuracy:  0.869, Validation Accuracy:  0.838, Loss:  0.383
    Epoch   2 Batch  279/538 - Train Accuracy:  0.850, Validation Accuracy:  0.844, Loss:  0.395
    Epoch   2 Batch  280/538 - Train Accuracy:  0.864, Validation Accuracy:  0.847, Loss:  0.364
    Epoch   2 Batch  281/538 - Train Accuracy:  0.852, Validation Accuracy:  0.846, Loss:  0.423
    Epoch   2 Batch  282/538 - Train Accuracy:  0.864, Validation Accuracy:  0.850, Loss:  0.407
    Epoch   2 Batch  283/538 - Train Accuracy:  0.869, Validation Accuracy:  0.853, Loss:  0.392
    Epoch   2 Batch  284/538 - Train Accuracy:  0.848, Validation Accuracy:  0.855, Loss:  0.412
    Epoch   2 Batch  285/538 - Train Accuracy:  0.841, Validation Accuracy:  0.844, Loss:  0.367
    Epoch   2 Batch  286/538 - Train Accuracy:  0.841, Validation Accuracy:  0.847, Loss:  0.411
    Epoch   2 Batch  287/538 - Train Accuracy:  0.878, Validation Accuracy:  0.849, Loss:  0.376
    Epoch   2 Batch  288/538 - Train Accuracy:  0.862, Validation Accuracy:  0.853, Loss:  0.408
    Epoch   2 Batch  289/538 - Train Accuracy:  0.864, Validation Accuracy:  0.850, Loss:  0.350
    Epoch   2 Batch  290/538 - Train Accuracy:  0.861, Validation Accuracy:  0.852, Loss:  0.392
    Epoch   2 Batch  291/538 - Train Accuracy:  0.857, Validation Accuracy:  0.844, Loss:  0.380
    Epoch   2 Batch  292/538 - Train Accuracy:  0.874, Validation Accuracy:  0.846, Loss:  0.363
    Epoch   2 Batch  293/538 - Train Accuracy:  0.864, Validation Accuracy:  0.851, Loss:  0.385
    Epoch   2 Batch  294/538 - Train Accuracy:  0.846, Validation Accuracy:  0.850, Loss:  0.415
    Epoch   2 Batch  295/538 - Train Accuracy:  0.866, Validation Accuracy:  0.853, Loss:  0.363
    Epoch   2 Batch  296/538 - Train Accuracy:  0.864, Validation Accuracy:  0.852, Loss:  0.386
    Epoch   2 Batch  297/538 - Train Accuracy:  0.870, Validation Accuracy:  0.854, Loss:  0.394
    Epoch   2 Batch  298/538 - Train Accuracy:  0.838, Validation Accuracy:  0.852, Loss:  0.392
    Epoch   2 Batch  299/538 - Train Accuracy:  0.854, Validation Accuracy:  0.855, Loss:  0.411
    Epoch   2 Batch  300/538 - Train Accuracy:  0.861, Validation Accuracy:  0.855, Loss:  0.385
    Epoch   2 Batch  301/538 - Train Accuracy:  0.852, Validation Accuracy:  0.854, Loss:  0.400
    Epoch   2 Batch  302/538 - Train Accuracy:  0.862, Validation Accuracy:  0.854, Loss:  0.361
    Epoch   2 Batch  303/538 - Train Accuracy:  0.866, Validation Accuracy:  0.857, Loss:  0.362
    Epoch   2 Batch  304/538 - Train Accuracy:  0.840, Validation Accuracy:  0.855, Loss:  0.400
    Epoch   2 Batch  305/538 - Train Accuracy:  0.866, Validation Accuracy:  0.855, Loss:  0.366
    Epoch   2 Batch  306/538 - Train Accuracy:  0.856, Validation Accuracy:  0.860, Loss:  0.394
    Epoch   2 Batch  307/538 - Train Accuracy:  0.860, Validation Accuracy:  0.867, Loss:  0.402
    Epoch   2 Batch  308/538 - Train Accuracy:  0.878, Validation Accuracy:  0.868, Loss:  0.378
    Epoch   2 Batch  309/538 - Train Accuracy:  0.869, Validation Accuracy:  0.868, Loss:  0.381
    Epoch   2 Batch  310/538 - Train Accuracy:  0.870, Validation Accuracy:  0.868, Loss:  0.396
    Epoch   2 Batch  311/538 - Train Accuracy:  0.858, Validation Accuracy:  0.862, Loss:  0.378
    Epoch   2 Batch  312/538 - Train Accuracy:  0.855, Validation Accuracy:  0.857, Loss:  0.361
    Epoch   2 Batch  313/538 - Train Accuracy:  0.860, Validation Accuracy:  0.855, Loss:  0.397
    Epoch   2 Batch  314/538 - Train Accuracy:  0.865, Validation Accuracy:  0.852, Loss:  0.392
    Epoch   2 Batch  315/538 - Train Accuracy:  0.841, Validation Accuracy:  0.860, Loss:  0.377
    Epoch   2 Batch  316/538 - Train Accuracy:  0.879, Validation Accuracy:  0.862, Loss:  0.368
    Epoch   2 Batch  317/538 - Train Accuracy:  0.867, Validation Accuracy:  0.857, Loss:  0.394
    Epoch   2 Batch  318/538 - Train Accuracy:  0.862, Validation Accuracy:  0.862, Loss:  0.382
    Epoch   2 Batch  319/538 - Train Accuracy:  0.876, Validation Accuracy:  0.862, Loss:  0.377
    Epoch   2 Batch  320/538 - Train Accuracy:  0.845, Validation Accuracy:  0.864, Loss:  0.395
    Epoch   2 Batch  321/538 - Train Accuracy:  0.859, Validation Accuracy:  0.862, Loss:  0.370
    Epoch   2 Batch  322/538 - Train Accuracy:  0.863, Validation Accuracy:  0.858, Loss:  0.395
    Epoch   2 Batch  323/538 - Train Accuracy:  0.871, Validation Accuracy:  0.861, Loss:  0.360
    Epoch   2 Batch  324/538 - Train Accuracy:  0.851, Validation Accuracy:  0.865, Loss:  0.410
    Epoch   2 Batch  325/538 - Train Accuracy:  0.876, Validation Accuracy:  0.861, Loss:  0.373
    Epoch   2 Batch  326/538 - Train Accuracy:  0.875, Validation Accuracy:  0.863, Loss:  0.381
    Epoch   2 Batch  327/538 - Train Accuracy:  0.867, Validation Accuracy:  0.860, Loss:  0.393
    Epoch   2 Batch  328/538 - Train Accuracy:  0.871, Validation Accuracy:  0.857, Loss:  0.363
    Epoch   2 Batch  329/538 - Train Accuracy:  0.873, Validation Accuracy:  0.854, Loss:  0.373
    Epoch   2 Batch  330/538 - Train Accuracy:  0.891, Validation Accuracy:  0.847, Loss:  0.365
    Epoch   2 Batch  331/538 - Train Accuracy:  0.864, Validation Accuracy:  0.846, Loss:  0.372
    Epoch   2 Batch  332/538 - Train Accuracy:  0.860, Validation Accuracy:  0.852, Loss:  0.380
    Epoch   2 Batch  333/538 - Train Accuracy:  0.866, Validation Accuracy:  0.863, Loss:  0.368
    Epoch   2 Batch  334/538 - Train Accuracy:  0.871, Validation Accuracy:  0.855, Loss:  0.357
    Epoch   2 Batch  335/538 - Train Accuracy:  0.876, Validation Accuracy:  0.858, Loss:  0.371
    Epoch   2 Batch  336/538 - Train Accuracy:  0.866, Validation Accuracy:  0.861, Loss:  0.368
    Epoch   2 Batch  337/538 - Train Accuracy:  0.864, Validation Accuracy:  0.861, Loss:  0.375
    Epoch   2 Batch  338/538 - Train Accuracy:  0.852, Validation Accuracy:  0.861, Loss:  0.377
    Epoch   2 Batch  339/538 - Train Accuracy:  0.857, Validation Accuracy:  0.861, Loss:  0.374
    Epoch   2 Batch  340/538 - Train Accuracy:  0.859, Validation Accuracy:  0.859, Loss:  0.384
    Epoch   2 Batch  341/538 - Train Accuracy:  0.848, Validation Accuracy:  0.855, Loss:  0.377
    Epoch   2 Batch  342/538 - Train Accuracy:  0.863, Validation Accuracy:  0.861, Loss:  0.379
    Epoch   2 Batch  343/538 - Train Accuracy:  0.867, Validation Accuracy:  0.862, Loss:  0.385
    Epoch   2 Batch  344/538 - Train Accuracy:  0.882, Validation Accuracy:  0.863, Loss:  0.362
    Epoch   2 Batch  345/538 - Train Accuracy:  0.849, Validation Accuracy:  0.863, Loss:  0.369
    Epoch   2 Batch  346/538 - Train Accuracy:  0.843, Validation Accuracy:  0.862, Loss:  0.387
    Epoch   2 Batch  347/538 - Train Accuracy:  0.878, Validation Accuracy:  0.858, Loss:  0.370
    Epoch   2 Batch  348/538 - Train Accuracy:  0.872, Validation Accuracy:  0.859, Loss:  0.356
    Epoch   2 Batch  349/538 - Train Accuracy:  0.888, Validation Accuracy:  0.859, Loss:  0.352
    Epoch   2 Batch  350/538 - Train Accuracy:  0.859, Validation Accuracy:  0.856, Loss:  0.398
    Epoch   2 Batch  351/538 - Train Accuracy:  0.853, Validation Accuracy:  0.855, Loss:  0.393
    Epoch   2 Batch  352/538 - Train Accuracy:  0.855, Validation Accuracy:  0.852, Loss:  0.387
    Epoch   2 Batch  353/538 - Train Accuracy:  0.864, Validation Accuracy:  0.854, Loss:  0.388
    Epoch   2 Batch  354/538 - Train Accuracy:  0.846, Validation Accuracy:  0.860, Loss:  0.402
    Epoch   2 Batch  355/538 - Train Accuracy:  0.864, Validation Accuracy:  0.857, Loss:  0.393
    Epoch   2 Batch  356/538 - Train Accuracy:  0.869, Validation Accuracy:  0.860, Loss:  0.350
    Epoch   2 Batch  357/538 - Train Accuracy:  0.857, Validation Accuracy:  0.858, Loss:  0.381
    Epoch   2 Batch  358/538 - Train Accuracy:  0.875, Validation Accuracy:  0.851, Loss:  0.352
    Epoch   2 Batch  359/538 - Train Accuracy:  0.846, Validation Accuracy:  0.856, Loss:  0.376
    Epoch   2 Batch  360/538 - Train Accuracy:  0.862, Validation Accuracy:  0.858, Loss:  0.376
    Epoch   2 Batch  361/538 - Train Accuracy:  0.875, Validation Accuracy:  0.852, Loss:  0.357
    Epoch   2 Batch  362/538 - Train Accuracy:  0.882, Validation Accuracy:  0.851, Loss:  0.333
    Epoch   2 Batch  363/538 - Train Accuracy:  0.852, Validation Accuracy:  0.851, Loss:  0.359
    Epoch   2 Batch  364/538 - Train Accuracy:  0.838, Validation Accuracy:  0.849, Loss:  0.395
    Epoch   2 Batch  365/538 - Train Accuracy:  0.854, Validation Accuracy:  0.858, Loss:  0.382
    Epoch   2 Batch  366/538 - Train Accuracy:  0.875, Validation Accuracy:  0.866, Loss:  0.384
    Epoch   2 Batch  367/538 - Train Accuracy:  0.876, Validation Accuracy:  0.857, Loss:  0.352
    Epoch   2 Batch  368/538 - Train Accuracy:  0.884, Validation Accuracy:  0.858, Loss:  0.330
    Epoch   2 Batch  369/538 - Train Accuracy:  0.866, Validation Accuracy:  0.856, Loss:  0.346
    Epoch   2 Batch  370/538 - Train Accuracy:  0.865, Validation Accuracy:  0.864, Loss:  0.380
    Epoch   2 Batch  371/538 - Train Accuracy:  0.873, Validation Accuracy:  0.861, Loss:  0.371
    Epoch   2 Batch  372/538 - Train Accuracy:  0.882, Validation Accuracy:  0.854, Loss:  0.349
    Epoch   2 Batch  373/538 - Train Accuracy:  0.867, Validation Accuracy:  0.858, Loss:  0.353
    Epoch   2 Batch  374/538 - Train Accuracy:  0.871, Validation Accuracy:  0.861, Loss:  0.360
    Epoch   2 Batch  375/538 - Train Accuracy:  0.874, Validation Accuracy:  0.859, Loss:  0.329
    Epoch   2 Batch  376/538 - Train Accuracy:  0.876, Validation Accuracy:  0.860, Loss:  0.366
    Epoch   2 Batch  377/538 - Train Accuracy:  0.873, Validation Accuracy:  0.860, Loss:  0.350
    Epoch   2 Batch  378/538 - Train Accuracy:  0.871, Validation Accuracy:  0.864, Loss:  0.361
    Epoch   2 Batch  379/538 - Train Accuracy:  0.871, Validation Accuracy:  0.863, Loss:  0.359
    Epoch   2 Batch  380/538 - Train Accuracy:  0.870, Validation Accuracy:  0.865, Loss:  0.344
    Epoch   2 Batch  381/538 - Train Accuracy:  0.874, Validation Accuracy:  0.862, Loss:  0.325
    Epoch   2 Batch  382/538 - Train Accuracy:  0.864, Validation Accuracy:  0.855, Loss:  0.381
    Epoch   2 Batch  383/538 - Train Accuracy:  0.857, Validation Accuracy:  0.854, Loss:  0.372
    Epoch   2 Batch  384/538 - Train Accuracy:  0.857, Validation Accuracy:  0.860, Loss:  0.369
    Epoch   2 Batch  385/538 - Train Accuracy:  0.874, Validation Accuracy:  0.856, Loss:  0.362
    Epoch   2 Batch  386/538 - Train Accuracy:  0.856, Validation Accuracy:  0.862, Loss:  0.378
    Epoch   2 Batch  387/538 - Train Accuracy:  0.879, Validation Accuracy:  0.860, Loss:  0.356
    Epoch   2 Batch  388/538 - Train Accuracy:  0.862, Validation Accuracy:  0.859, Loss:  0.359
    Epoch   2 Batch  389/538 - Train Accuracy:  0.860, Validation Accuracy:  0.857, Loss:  0.391
    Epoch   2 Batch  390/538 - Train Accuracy:  0.881, Validation Accuracy:  0.857, Loss:  0.333
    Epoch   2 Batch  391/538 - Train Accuracy:  0.868, Validation Accuracy:  0.857, Loss:  0.361
    Epoch   2 Batch  392/538 - Train Accuracy:  0.869, Validation Accuracy:  0.853, Loss:  0.340
    Epoch   2 Batch  393/538 - Train Accuracy:  0.888, Validation Accuracy:  0.853, Loss:  0.328
    Epoch   2 Batch  394/538 - Train Accuracy:  0.832, Validation Accuracy:  0.852, Loss:  0.372
    Epoch   2 Batch  395/538 - Train Accuracy:  0.850, Validation Accuracy:  0.863, Loss:  0.392
    Epoch   2 Batch  396/538 - Train Accuracy:  0.869, Validation Accuracy:  0.867, Loss:  0.350
    Epoch   2 Batch  397/538 - Train Accuracy:  0.872, Validation Accuracy:  0.868, Loss:  0.378
    Epoch   2 Batch  398/538 - Train Accuracy:  0.869, Validation Accuracy:  0.872, Loss:  0.363
    Epoch   2 Batch  399/538 - Train Accuracy:  0.846, Validation Accuracy:  0.875, Loss:  0.388
    Epoch   2 Batch  400/538 - Train Accuracy:  0.870, Validation Accuracy:  0.871, Loss:  0.350
    Epoch   2 Batch  401/538 - Train Accuracy:  0.874, Validation Accuracy:  0.867, Loss:  0.357
    Epoch   2 Batch  402/538 - Train Accuracy:  0.862, Validation Accuracy:  0.859, Loss:  0.353
    Epoch   2 Batch  403/538 - Train Accuracy:  0.875, Validation Accuracy:  0.864, Loss:  0.356
    Epoch   2 Batch  404/538 - Train Accuracy:  0.871, Validation Accuracy:  0.862, Loss:  0.344
    Epoch   2 Batch  405/538 - Train Accuracy:  0.865, Validation Accuracy:  0.865, Loss:  0.343
    Epoch   2 Batch  406/538 - Train Accuracy:  0.860, Validation Accuracy:  0.866, Loss:  0.355
    Epoch   2 Batch  407/538 - Train Accuracy:  0.899, Validation Accuracy:  0.861, Loss:  0.346
    Epoch   2 Batch  408/538 - Train Accuracy:  0.854, Validation Accuracy:  0.858, Loss:  0.392
    Epoch   2 Batch  409/538 - Train Accuracy:  0.858, Validation Accuracy:  0.865, Loss:  0.361
    Epoch   2 Batch  410/538 - Train Accuracy:  0.879, Validation Accuracy:  0.869, Loss:  0.349
    Epoch   2 Batch  411/538 - Train Accuracy:  0.890, Validation Accuracy:  0.869, Loss:  0.333
    Epoch   2 Batch  412/538 - Train Accuracy:  0.865, Validation Accuracy:  0.866, Loss:  0.327
    Epoch   2 Batch  413/538 - Train Accuracy:  0.879, Validation Accuracy:  0.865, Loss:  0.359
    Epoch   2 Batch  414/538 - Train Accuracy:  0.848, Validation Accuracy:  0.867, Loss:  0.381
    Epoch   2 Batch  415/538 - Train Accuracy:  0.852, Validation Accuracy:  0.864, Loss:  0.360
    Epoch   2 Batch  416/538 - Train Accuracy:  0.859, Validation Accuracy:  0.861, Loss:  0.343
    Epoch   2 Batch  417/538 - Train Accuracy:  0.871, Validation Accuracy:  0.863, Loss:  0.367
    Epoch   2 Batch  418/538 - Train Accuracy:  0.876, Validation Accuracy:  0.859, Loss:  0.366
    Epoch   2 Batch  419/538 - Train Accuracy:  0.871, Validation Accuracy:  0.865, Loss:  0.338
    Epoch   2 Batch  420/538 - Train Accuracy:  0.894, Validation Accuracy:  0.864, Loss:  0.330
    Epoch   2 Batch  421/538 - Train Accuracy:  0.872, Validation Accuracy:  0.865, Loss:  0.330
    Epoch   2 Batch  422/538 - Train Accuracy:  0.882, Validation Accuracy:  0.867, Loss:  0.345
    Epoch   2 Batch  423/538 - Train Accuracy:  0.878, Validation Accuracy:  0.871, Loss:  0.363
    Epoch   2 Batch  424/538 - Train Accuracy:  0.859, Validation Accuracy:  0.871, Loss:  0.338
    Epoch   2 Batch  425/538 - Train Accuracy:  0.871, Validation Accuracy:  0.871, Loss:  0.354
    Epoch   2 Batch  426/538 - Train Accuracy:  0.879, Validation Accuracy:  0.872, Loss:  0.342
    Epoch   2 Batch  427/538 - Train Accuracy:  0.860, Validation Accuracy:  0.872, Loss:  0.347
    Epoch   2 Batch  428/538 - Train Accuracy:  0.885, Validation Accuracy:  0.865, Loss:  0.325
    Epoch   2 Batch  429/538 - Train Accuracy:  0.881, Validation Accuracy:  0.865, Loss:  0.348
    Epoch   2 Batch  430/538 - Train Accuracy:  0.863, Validation Accuracy:  0.873, Loss:  0.353
    Epoch   2 Batch  431/538 - Train Accuracy:  0.879, Validation Accuracy:  0.875, Loss:  0.332
    Epoch   2 Batch  432/538 - Train Accuracy:  0.885, Validation Accuracy:  0.876, Loss:  0.329
    Epoch   2 Batch  433/538 - Train Accuracy:  0.855, Validation Accuracy:  0.878, Loss:  0.380
    Epoch   2 Batch  434/538 - Train Accuracy:  0.843, Validation Accuracy:  0.876, Loss:  0.354
    Epoch   2 Batch  435/538 - Train Accuracy:  0.876, Validation Accuracy:  0.868, Loss:  0.326
    Epoch   2 Batch  436/538 - Train Accuracy:  0.862, Validation Accuracy:  0.871, Loss:  0.355
    Epoch   2 Batch  437/538 - Train Accuracy:  0.876, Validation Accuracy:  0.869, Loss:  0.357
    Epoch   2 Batch  438/538 - Train Accuracy:  0.886, Validation Accuracy:  0.872, Loss:  0.339
    Epoch   2 Batch  439/538 - Train Accuracy:  0.903, Validation Accuracy:  0.869, Loss:  0.315
    Epoch   2 Batch  440/538 - Train Accuracy:  0.879, Validation Accuracy:  0.865, Loss:  0.361
    Epoch   2 Batch  441/538 - Train Accuracy:  0.858, Validation Accuracy:  0.863, Loss:  0.355
    Epoch   2 Batch  442/538 - Train Accuracy:  0.877, Validation Accuracy:  0.862, Loss:  0.315
    Epoch   2 Batch  443/538 - Train Accuracy:  0.872, Validation Accuracy:  0.868, Loss:  0.354
    Epoch   2 Batch  444/538 - Train Accuracy:  0.891, Validation Accuracy:  0.869, Loss:  0.324
    Epoch   2 Batch  445/538 - Train Accuracy:  0.888, Validation Accuracy:  0.874, Loss:  0.315
    Epoch   2 Batch  446/538 - Train Accuracy:  0.891, Validation Accuracy:  0.872, Loss:  0.319
    Epoch   2 Batch  447/538 - Train Accuracy:  0.868, Validation Accuracy:  0.869, Loss:  0.339
    Epoch   2 Batch  448/538 - Train Accuracy:  0.876, Validation Accuracy:  0.873, Loss:  0.312
    Epoch   2 Batch  449/538 - Train Accuracy:  0.890, Validation Accuracy:  0.876, Loss:  0.358
    Epoch   2 Batch  450/538 - Train Accuracy:  0.876, Validation Accuracy:  0.876, Loss:  0.369
    Epoch   2 Batch  451/538 - Train Accuracy:  0.874, Validation Accuracy:  0.877, Loss:  0.339
    Epoch   2 Batch  452/538 - Train Accuracy:  0.889, Validation Accuracy:  0.881, Loss:  0.313
    Epoch   2 Batch  453/538 - Train Accuracy:  0.877, Validation Accuracy:  0.880, Loss:  0.342
    Epoch   2 Batch  454/538 - Train Accuracy:  0.872, Validation Accuracy:  0.873, Loss:  0.323
    Epoch   2 Batch  455/538 - Train Accuracy:  0.884, Validation Accuracy:  0.873, Loss:  0.319
    Epoch   2 Batch  456/538 - Train Accuracy:  0.900, Validation Accuracy:  0.870, Loss:  0.312
    Epoch   2 Batch  457/538 - Train Accuracy:  0.850, Validation Accuracy:  0.868, Loss:  0.353
    Epoch   2 Batch  458/538 - Train Accuracy:  0.877, Validation Accuracy:  0.869, Loss:  0.319
    Epoch   2 Batch  459/538 - Train Accuracy:  0.878, Validation Accuracy:  0.866, Loss:  0.324
    Epoch   2 Batch  460/538 - Train Accuracy:  0.845, Validation Accuracy:  0.865, Loss:  0.342
    Epoch   2 Batch  461/538 - Train Accuracy:  0.885, Validation Accuracy:  0.874, Loss:  0.347
    Epoch   2 Batch  462/538 - Train Accuracy:  0.882, Validation Accuracy:  0.874, Loss:  0.323
    Epoch   2 Batch  463/538 - Train Accuracy:  0.863, Validation Accuracy:  0.873, Loss:  0.341
    Epoch   2 Batch  464/538 - Train Accuracy:  0.881, Validation Accuracy:  0.870, Loss:  0.329
    Epoch   2 Batch  465/538 - Train Accuracy:  0.879, Validation Accuracy:  0.872, Loss:  0.313
    Epoch   2 Batch  466/538 - Train Accuracy:  0.885, Validation Accuracy:  0.872, Loss:  0.343
    Epoch   2 Batch  467/538 - Train Accuracy:  0.884, Validation Accuracy:  0.866, Loss:  0.326
    Epoch   2 Batch  468/538 - Train Accuracy:  0.887, Validation Accuracy:  0.858, Loss:  0.355
    Epoch   2 Batch  469/538 - Train Accuracy:  0.889, Validation Accuracy:  0.861, Loss:  0.334
    Epoch   2 Batch  470/538 - Train Accuracy:  0.882, Validation Accuracy:  0.857, Loss:  0.320
    Epoch   2 Batch  471/538 - Train Accuracy:  0.875, Validation Accuracy:  0.862, Loss:  0.326
    Epoch   2 Batch  472/538 - Train Accuracy:  0.908, Validation Accuracy:  0.867, Loss:  0.309
    Epoch   2 Batch  473/538 - Train Accuracy:  0.852, Validation Accuracy:  0.864, Loss:  0.332
    Epoch   2 Batch  474/538 - Train Accuracy:  0.883, Validation Accuracy:  0.866, Loss:  0.318
    Epoch   2 Batch  475/538 - Train Accuracy:  0.881, Validation Accuracy:  0.871, Loss:  0.322
    Epoch   2 Batch  476/538 - Train Accuracy:  0.857, Validation Accuracy:  0.870, Loss:  0.317
    Epoch   2 Batch  477/538 - Train Accuracy:  0.877, Validation Accuracy:  0.874, Loss:  0.342
    Epoch   2 Batch  478/538 - Train Accuracy:  0.879, Validation Accuracy:  0.873, Loss:  0.318
    Epoch   2 Batch  479/538 - Train Accuracy:  0.890, Validation Accuracy:  0.875, Loss:  0.299
    Epoch   2 Batch  480/538 - Train Accuracy:  0.886, Validation Accuracy:  0.877, Loss:  0.325
    Epoch   2 Batch  481/538 - Train Accuracy:  0.890, Validation Accuracy:  0.882, Loss:  0.317
    Epoch   2 Batch  482/538 - Train Accuracy:  0.881, Validation Accuracy:  0.885, Loss:  0.304
    Epoch   2 Batch  483/538 - Train Accuracy:  0.860, Validation Accuracy:  0.883, Loss:  0.352
    Epoch   2 Batch  484/538 - Train Accuracy:  0.885, Validation Accuracy:  0.877, Loss:  0.353
    Epoch   2 Batch  485/538 - Train Accuracy:  0.896, Validation Accuracy:  0.872, Loss:  0.315
    Epoch   2 Batch  486/538 - Train Accuracy:  0.906, Validation Accuracy:  0.869, Loss:  0.302
    Epoch   2 Batch  487/538 - Train Accuracy:  0.901, Validation Accuracy:  0.874, Loss:  0.298
    Epoch   2 Batch  488/538 - Train Accuracy:  0.898, Validation Accuracy:  0.874, Loss:  0.324
    Epoch   2 Batch  489/538 - Train Accuracy:  0.870, Validation Accuracy:  0.881, Loss:  0.329
    Epoch   2 Batch  490/538 - Train Accuracy:  0.891, Validation Accuracy:  0.882, Loss:  0.318
    Epoch   2 Batch  491/538 - Train Accuracy:  0.855, Validation Accuracy:  0.877, Loss:  0.350
    Epoch   2 Batch  492/538 - Train Accuracy:  0.870, Validation Accuracy:  0.882, Loss:  0.335
    Epoch   2 Batch  493/538 - Train Accuracy:  0.878, Validation Accuracy:  0.881, Loss:  0.319
    Epoch   2 Batch  494/538 - Train Accuracy:  0.882, Validation Accuracy:  0.878, Loss:  0.335
    Epoch   2 Batch  495/538 - Train Accuracy:  0.890, Validation Accuracy:  0.877, Loss:  0.356
    Epoch   2 Batch  496/538 - Train Accuracy:  0.892, Validation Accuracy:  0.878, Loss:  0.311
    Epoch   2 Batch  497/538 - Train Accuracy:  0.885, Validation Accuracy:  0.874, Loss:  0.312
    Epoch   2 Batch  498/538 - Train Accuracy:  0.881, Validation Accuracy:  0.871, Loss:  0.317
    Epoch   2 Batch  499/538 - Train Accuracy:  0.880, Validation Accuracy:  0.875, Loss:  0.314
    Epoch   2 Batch  500/538 - Train Accuracy:  0.885, Validation Accuracy:  0.877, Loss:  0.301
    Epoch   2 Batch  501/538 - Train Accuracy:  0.897, Validation Accuracy:  0.877, Loss:  0.320
    Epoch   2 Batch  502/538 - Train Accuracy:  0.886, Validation Accuracy:  0.880, Loss:  0.313
    Epoch   2 Batch  503/538 - Train Accuracy:  0.904, Validation Accuracy:  0.880, Loss:  0.315
    Epoch   2 Batch  504/538 - Train Accuracy:  0.909, Validation Accuracy:  0.879, Loss:  0.300
    Epoch   2 Batch  505/538 - Train Accuracy:  0.897, Validation Accuracy:  0.876, Loss:  0.308
    Epoch   2 Batch  506/538 - Train Accuracy:  0.896, Validation Accuracy:  0.872, Loss:  0.310
    Epoch   2 Batch  507/538 - Train Accuracy:  0.849, Validation Accuracy:  0.873, Loss:  0.339
    Epoch   2 Batch  508/538 - Train Accuracy:  0.879, Validation Accuracy:  0.878, Loss:  0.300
    Epoch   2 Batch  509/538 - Train Accuracy:  0.876, Validation Accuracy:  0.875, Loss:  0.314
    Epoch   2 Batch  510/538 - Train Accuracy:  0.892, Validation Accuracy:  0.877, Loss:  0.303
    Epoch   2 Batch  511/538 - Train Accuracy:  0.873, Validation Accuracy:  0.873, Loss:  0.321
    Epoch   2 Batch  512/538 - Train Accuracy:  0.890, Validation Accuracy:  0.877, Loss:  0.309
    Epoch   2 Batch  513/538 - Train Accuracy:  0.865, Validation Accuracy:  0.874, Loss:  0.331
    Epoch   2 Batch  514/538 - Train Accuracy:  0.881, Validation Accuracy:  0.873, Loss:  0.327
    Epoch   2 Batch  515/538 - Train Accuracy:  0.887, Validation Accuracy:  0.883, Loss:  0.317
    Epoch   2 Batch  516/538 - Train Accuracy:  0.859, Validation Accuracy:  0.883, Loss:  0.336
    Epoch   2 Batch  517/538 - Train Accuracy:  0.886, Validation Accuracy:  0.882, Loss:  0.313
    Epoch   2 Batch  518/538 - Train Accuracy:  0.873, Validation Accuracy:  0.882, Loss:  0.335
    Epoch   2 Batch  519/538 - Train Accuracy:  0.891, Validation Accuracy:  0.882, Loss:  0.307
    Epoch   2 Batch  520/538 - Train Accuracy:  0.865, Validation Accuracy:  0.882, Loss:  0.324
    Epoch   2 Batch  521/538 - Train Accuracy:  0.888, Validation Accuracy:  0.882, Loss:  0.320
    Epoch   2 Batch  522/538 - Train Accuracy:  0.886, Validation Accuracy:  0.877, Loss:  0.305
    Epoch   2 Batch  523/538 - Train Accuracy:  0.881, Validation Accuracy:  0.881, Loss:  0.307
    Epoch   2 Batch  524/538 - Train Accuracy:  0.877, Validation Accuracy:  0.874, Loss:  0.313
    Epoch   2 Batch  525/538 - Train Accuracy:  0.883, Validation Accuracy:  0.874, Loss:  0.316
    Epoch   2 Batch  526/538 - Train Accuracy:  0.886, Validation Accuracy:  0.870, Loss:  0.323
    Epoch   2 Batch  527/538 - Train Accuracy:  0.891, Validation Accuracy:  0.879, Loss:  0.307
    Epoch   2 Batch  528/538 - Train Accuracy:  0.883, Validation Accuracy:  0.878, Loss:  0.342
    Epoch   2 Batch  529/538 - Train Accuracy:  0.851, Validation Accuracy:  0.881, Loss:  0.323
    Epoch   2 Batch  530/538 - Train Accuracy:  0.862, Validation Accuracy:  0.882, Loss:  0.344
    Epoch   2 Batch  531/538 - Train Accuracy:  0.882, Validation Accuracy:  0.878, Loss:  0.300
    Epoch   2 Batch  532/538 - Train Accuracy:  0.871, Validation Accuracy:  0.878, Loss:  0.311
    Epoch   2 Batch  533/538 - Train Accuracy:  0.892, Validation Accuracy:  0.878, Loss:  0.303
    Epoch   2 Batch  534/538 - Train Accuracy:  0.899, Validation Accuracy:  0.878, Loss:  0.301
    Epoch   2 Batch  535/538 - Train Accuracy:  0.900, Validation Accuracy:  0.880, Loss:  0.288
    Epoch   2 Batch  536/538 - Train Accuracy:  0.896, Validation Accuracy:  0.876, Loss:  0.332
    Epoch   3 Batch    0/538 - Train Accuracy:  0.896, Validation Accuracy:  0.876, Loss:  0.284
    Epoch   3 Batch    1/538 - Train Accuracy:  0.897, Validation Accuracy:  0.874, Loss:  0.307
    Epoch   3 Batch    2/538 - Train Accuracy:  0.881, Validation Accuracy:  0.874, Loss:  0.336
    Epoch   3 Batch    3/538 - Train Accuracy:  0.890, Validation Accuracy:  0.877, Loss:  0.297
    Epoch   3 Batch    4/538 - Train Accuracy:  0.884, Validation Accuracy:  0.874, Loss:  0.311
    Epoch   3 Batch    5/538 - Train Accuracy:  0.878, Validation Accuracy:  0.877, Loss:  0.317
    Epoch   3 Batch    6/538 - Train Accuracy:  0.886, Validation Accuracy:  0.881, Loss:  0.307
    Epoch   3 Batch    7/538 - Train Accuracy:  0.910, Validation Accuracy:  0.881, Loss:  0.298
    Epoch   3 Batch    8/538 - Train Accuracy:  0.891, Validation Accuracy:  0.879, Loss:  0.307
    Epoch   3 Batch    9/538 - Train Accuracy:  0.866, Validation Accuracy:  0.881, Loss:  0.303
    Epoch   3 Batch   10/538 - Train Accuracy:  0.888, Validation Accuracy:  0.878, Loss:  0.324
    Epoch   3 Batch   11/538 - Train Accuracy:  0.879, Validation Accuracy:  0.876, Loss:  0.296
    Epoch   3 Batch   12/538 - Train Accuracy:  0.900, Validation Accuracy:  0.874, Loss:  0.301
    Epoch   3 Batch   13/538 - Train Accuracy:  0.906, Validation Accuracy:  0.874, Loss:  0.271
    Epoch   3 Batch   14/538 - Train Accuracy:  0.904, Validation Accuracy:  0.872, Loss:  0.290
    Epoch   3 Batch   15/538 - Train Accuracy:  0.878, Validation Accuracy:  0.873, Loss:  0.296
    Epoch   3 Batch   16/538 - Train Accuracy:  0.891, Validation Accuracy:  0.873, Loss:  0.298
    Epoch   3 Batch   17/538 - Train Accuracy:  0.879, Validation Accuracy:  0.869, Loss:  0.300
    Epoch   3 Batch   18/538 - Train Accuracy:  0.894, Validation Accuracy:  0.873, Loss:  0.317
    Epoch   3 Batch   19/538 - Train Accuracy:  0.885, Validation Accuracy:  0.878, Loss:  0.323
    Epoch   3 Batch   20/538 - Train Accuracy:  0.886, Validation Accuracy:  0.877, Loss:  0.308
    Epoch   3 Batch   21/538 - Train Accuracy:  0.900, Validation Accuracy:  0.880, Loss:  0.276
    Epoch   3 Batch   22/538 - Train Accuracy:  0.861, Validation Accuracy:  0.882, Loss:  0.314
    Epoch   3 Batch   23/538 - Train Accuracy:  0.870, Validation Accuracy:  0.882, Loss:  0.327
    Epoch   3 Batch   24/538 - Train Accuracy:  0.886, Validation Accuracy:  0.884, Loss:  0.298
    Epoch   3 Batch   25/538 - Train Accuracy:  0.894, Validation Accuracy:  0.885, Loss:  0.298
    Epoch   3 Batch   26/538 - Train Accuracy:  0.878, Validation Accuracy:  0.877, Loss:  0.328
    Epoch   3 Batch   27/538 - Train Accuracy:  0.912, Validation Accuracy:  0.879, Loss:  0.278
    Epoch   3 Batch   28/538 - Train Accuracy:  0.899, Validation Accuracy:  0.878, Loss:  0.281
    Epoch   3 Batch   29/538 - Train Accuracy:  0.897, Validation Accuracy:  0.877, Loss:  0.284
    Epoch   3 Batch   30/538 - Train Accuracy:  0.876, Validation Accuracy:  0.877, Loss:  0.313
    Epoch   3 Batch   31/538 - Train Accuracy:  0.894, Validation Accuracy:  0.878, Loss:  0.266
    Epoch   3 Batch   32/538 - Train Accuracy:  0.877, Validation Accuracy:  0.877, Loss:  0.273
    Epoch   3 Batch   33/538 - Train Accuracy:  0.890, Validation Accuracy:  0.878, Loss:  0.281
    Epoch   3 Batch   34/538 - Train Accuracy:  0.884, Validation Accuracy:  0.879, Loss:  0.334
    Epoch   3 Batch   35/538 - Train Accuracy:  0.880, Validation Accuracy:  0.881, Loss:  0.294
    Epoch   3 Batch   36/538 - Train Accuracy:  0.896, Validation Accuracy:  0.882, Loss:  0.269
    Epoch   3 Batch   37/538 - Train Accuracy:  0.904, Validation Accuracy:  0.878, Loss:  0.290
    Epoch   3 Batch   38/538 - Train Accuracy:  0.870, Validation Accuracy:  0.878, Loss:  0.313
    Epoch   3 Batch   39/538 - Train Accuracy:  0.902, Validation Accuracy:  0.878, Loss:  0.289
    Epoch   3 Batch   40/538 - Train Accuracy:  0.894, Validation Accuracy:  0.877, Loss:  0.288
    Epoch   3 Batch   41/538 - Train Accuracy:  0.895, Validation Accuracy:  0.882, Loss:  0.299
    Epoch   3 Batch   42/538 - Train Accuracy:  0.899, Validation Accuracy:  0.874, Loss:  0.299
    Epoch   3 Batch   43/538 - Train Accuracy:  0.880, Validation Accuracy:  0.876, Loss:  0.334
    Epoch   3 Batch   44/538 - Train Accuracy:  0.874, Validation Accuracy:  0.882, Loss:  0.318
    Epoch   3 Batch   45/538 - Train Accuracy:  0.884, Validation Accuracy:  0.882, Loss:  0.269
    Epoch   3 Batch   46/538 - Train Accuracy:  0.896, Validation Accuracy:  0.879, Loss:  0.288
    Epoch   3 Batch   47/538 - Train Accuracy:  0.887, Validation Accuracy:  0.883, Loss:  0.321
    Epoch   3 Batch   48/538 - Train Accuracy:  0.878, Validation Accuracy:  0.888, Loss:  0.316
    Epoch   3 Batch   49/538 - Train Accuracy:  0.896, Validation Accuracy:  0.889, Loss:  0.301
    Epoch   3 Batch   50/538 - Train Accuracy:  0.881, Validation Accuracy:  0.888, Loss:  0.278
    Epoch   3 Batch   51/538 - Train Accuracy:  0.870, Validation Accuracy:  0.885, Loss:  0.319
    Epoch   3 Batch   52/538 - Train Accuracy:  0.897, Validation Accuracy:  0.883, Loss:  0.315
    Epoch   3 Batch   53/538 - Train Accuracy:  0.888, Validation Accuracy:  0.873, Loss:  0.278
    Epoch   3 Batch   54/538 - Train Accuracy:  0.901, Validation Accuracy:  0.873, Loss:  0.284
    Epoch   3 Batch   55/538 - Train Accuracy:  0.874, Validation Accuracy:  0.877, Loss:  0.296
    Epoch   3 Batch   56/538 - Train Accuracy:  0.888, Validation Accuracy:  0.885, Loss:  0.290
    Epoch   3 Batch   57/538 - Train Accuracy:  0.867, Validation Accuracy:  0.880, Loss:  0.318
    Epoch   3 Batch   58/538 - Train Accuracy:  0.866, Validation Accuracy:  0.879, Loss:  0.302
    Epoch   3 Batch   59/538 - Train Accuracy:  0.883, Validation Accuracy:  0.880, Loss:  0.295
    Epoch   3 Batch   60/538 - Train Accuracy:  0.911, Validation Accuracy:  0.875, Loss:  0.288
    Epoch   3 Batch   61/538 - Train Accuracy:  0.905, Validation Accuracy:  0.875, Loss:  0.287
    Epoch   3 Batch   62/538 - Train Accuracy:  0.903, Validation Accuracy:  0.873, Loss:  0.281
    Epoch   3 Batch   63/538 - Train Accuracy:  0.915, Validation Accuracy:  0.871, Loss:  0.263
    Epoch   3 Batch   64/538 - Train Accuracy:  0.887, Validation Accuracy:  0.868, Loss:  0.285
    Epoch   3 Batch   65/538 - Train Accuracy:  0.875, Validation Accuracy:  0.869, Loss:  0.305
    Epoch   3 Batch   66/538 - Train Accuracy:  0.902, Validation Accuracy:  0.871, Loss:  0.264
    Epoch   3 Batch   67/538 - Train Accuracy:  0.913, Validation Accuracy:  0.874, Loss:  0.284
    Epoch   3 Batch   68/538 - Train Accuracy:  0.890, Validation Accuracy:  0.879, Loss:  0.264
    Epoch   3 Batch   69/538 - Train Accuracy:  0.901, Validation Accuracy:  0.882, Loss:  0.293
    Epoch   3 Batch   70/538 - Train Accuracy:  0.884, Validation Accuracy:  0.887, Loss:  0.292
    Epoch   3 Batch   71/538 - Train Accuracy:  0.886, Validation Accuracy:  0.890, Loss:  0.300
    Epoch   3 Batch   72/538 - Train Accuracy:  0.905, Validation Accuracy:  0.889, Loss:  0.309
    Epoch   3 Batch   73/538 - Train Accuracy:  0.876, Validation Accuracy:  0.891, Loss:  0.299
    Epoch   3 Batch   74/538 - Train Accuracy:  0.890, Validation Accuracy:  0.894, Loss:  0.275
    Epoch   3 Batch   75/538 - Train Accuracy:  0.895, Validation Accuracy:  0.894, Loss:  0.297
    Epoch   3 Batch   76/538 - Train Accuracy:  0.878, Validation Accuracy:  0.892, Loss:  0.316
    Epoch   3 Batch   77/538 - Train Accuracy:  0.895, Validation Accuracy:  0.891, Loss:  0.292
    Epoch   3 Batch   78/538 - Train Accuracy:  0.883, Validation Accuracy:  0.891, Loss:  0.299
    Epoch   3 Batch   79/538 - Train Accuracy:  0.896, Validation Accuracy:  0.890, Loss:  0.251
    Epoch   3 Batch   80/538 - Train Accuracy:  0.878, Validation Accuracy:  0.888, Loss:  0.304
    Epoch   3 Batch   81/538 - Train Accuracy:  0.895, Validation Accuracy:  0.891, Loss:  0.284
    Epoch   3 Batch   82/538 - Train Accuracy:  0.892, Validation Accuracy:  0.892, Loss:  0.279
    Epoch   3 Batch   83/538 - Train Accuracy:  0.894, Validation Accuracy:  0.894, Loss:  0.289
    Epoch   3 Batch   84/538 - Train Accuracy:  0.880, Validation Accuracy:  0.898, Loss:  0.301
    Epoch   3 Batch   85/538 - Train Accuracy:  0.903, Validation Accuracy:  0.892, Loss:  0.265
    Epoch   3 Batch   86/538 - Train Accuracy:  0.896, Validation Accuracy:  0.890, Loss:  0.285
    Epoch   3 Batch   87/538 - Train Accuracy:  0.876, Validation Accuracy:  0.885, Loss:  0.292
    Epoch   3 Batch   88/538 - Train Accuracy:  0.879, Validation Accuracy:  0.888, Loss:  0.301
    Epoch   3 Batch   89/538 - Train Accuracy:  0.894, Validation Accuracy:  0.893, Loss:  0.273
    Epoch   3 Batch   90/538 - Train Accuracy:  0.894, Validation Accuracy:  0.896, Loss:  0.284
    Epoch   3 Batch   91/538 - Train Accuracy:  0.892, Validation Accuracy:  0.897, Loss:  0.287
    Epoch   3 Batch   92/538 - Train Accuracy:  0.888, Validation Accuracy:  0.887, Loss:  0.294
    Epoch   3 Batch   93/538 - Train Accuracy:  0.890, Validation Accuracy:  0.887, Loss:  0.281
    Epoch   3 Batch   94/538 - Train Accuracy:  0.900, Validation Accuracy:  0.889, Loss:  0.278
    Epoch   3 Batch   95/538 - Train Accuracy:  0.888, Validation Accuracy:  0.889, Loss:  0.266
    Epoch   3 Batch   96/538 - Train Accuracy:  0.908, Validation Accuracy:  0.889, Loss:  0.258
    Epoch   3 Batch   97/538 - Train Accuracy:  0.902, Validation Accuracy:  0.897, Loss:  0.274
    Epoch   3 Batch   98/538 - Train Accuracy:  0.906, Validation Accuracy:  0.899, Loss:  0.276
    Epoch   3 Batch   99/538 - Train Accuracy:  0.896, Validation Accuracy:  0.896, Loss:  0.280
    Epoch   3 Batch  100/538 - Train Accuracy:  0.901, Validation Accuracy:  0.890, Loss:  0.266
    Epoch   3 Batch  101/538 - Train Accuracy:  0.879, Validation Accuracy:  0.886, Loss:  0.304
    Epoch   3 Batch  102/538 - Train Accuracy:  0.872, Validation Accuracy:  0.886, Loss:  0.299
    Epoch   3 Batch  103/538 - Train Accuracy:  0.912, Validation Accuracy:  0.886, Loss:  0.266
    Epoch   3 Batch  104/538 - Train Accuracy:  0.892, Validation Accuracy:  0.890, Loss:  0.268
    Epoch   3 Batch  105/538 - Train Accuracy:  0.905, Validation Accuracy:  0.892, Loss:  0.255
    Epoch   3 Batch  106/538 - Train Accuracy:  0.876, Validation Accuracy:  0.886, Loss:  0.255
    Epoch   3 Batch  107/538 - Train Accuracy:  0.889, Validation Accuracy:  0.887, Loss:  0.302
    Epoch   3 Batch  108/538 - Train Accuracy:  0.911, Validation Accuracy:  0.890, Loss:  0.277
    Epoch   3 Batch  109/538 - Train Accuracy:  0.911, Validation Accuracy:  0.888, Loss:  0.265
    Epoch   3 Batch  110/538 - Train Accuracy:  0.889, Validation Accuracy:  0.884, Loss:  0.289
    Epoch   3 Batch  111/538 - Train Accuracy:  0.907, Validation Accuracy:  0.884, Loss:  0.246
    Epoch   3 Batch  112/538 - Train Accuracy:  0.897, Validation Accuracy:  0.885, Loss:  0.281
    Epoch   3 Batch  113/538 - Train Accuracy:  0.872, Validation Accuracy:  0.891, Loss:  0.318
    Epoch   3 Batch  114/538 - Train Accuracy:  0.897, Validation Accuracy:  0.890, Loss:  0.270
    Epoch   3 Batch  115/538 - Train Accuracy:  0.906, Validation Accuracy:  0.894, Loss:  0.269
    Epoch   3 Batch  116/538 - Train Accuracy:  0.892, Validation Accuracy:  0.888, Loss:  0.304
    Epoch   3 Batch  117/538 - Train Accuracy:  0.885, Validation Accuracy:  0.889, Loss:  0.270
    Epoch   3 Batch  118/538 - Train Accuracy:  0.905, Validation Accuracy:  0.889, Loss:  0.261
    Epoch   3 Batch  119/538 - Train Accuracy:  0.924, Validation Accuracy:  0.893, Loss:  0.241
    Epoch   3 Batch  120/538 - Train Accuracy:  0.905, Validation Accuracy:  0.895, Loss:  0.271
    Epoch   3 Batch  121/538 - Train Accuracy:  0.903, Validation Accuracy:  0.895, Loss:  0.259
    Epoch   3 Batch  122/538 - Train Accuracy:  0.895, Validation Accuracy:  0.891, Loss:  0.249
    Epoch   3 Batch  123/538 - Train Accuracy:  0.907, Validation Accuracy:  0.892, Loss:  0.257
    Epoch   3 Batch  124/538 - Train Accuracy:  0.912, Validation Accuracy:  0.897, Loss:  0.255
    Epoch   3 Batch  125/538 - Train Accuracy:  0.904, Validation Accuracy:  0.896, Loss:  0.286
    Epoch   3 Batch  126/538 - Train Accuracy:  0.885, Validation Accuracy:  0.894, Loss:  0.273
    Epoch   3 Batch  127/538 - Train Accuracy:  0.876, Validation Accuracy:  0.896, Loss:  0.303
    Epoch   3 Batch  128/538 - Train Accuracy:  0.893, Validation Accuracy:  0.897, Loss:  0.276
    Epoch   3 Batch  129/538 - Train Accuracy:  0.901, Validation Accuracy:  0.898, Loss:  0.250
    Epoch   3 Batch  130/538 - Train Accuracy:  0.914, Validation Accuracy:  0.891, Loss:  0.251
    Epoch   3 Batch  131/538 - Train Accuracy:  0.916, Validation Accuracy:  0.890, Loss:  0.265
    Epoch   3 Batch  132/538 - Train Accuracy:  0.872, Validation Accuracy:  0.889, Loss:  0.277
    Epoch   3 Batch  133/538 - Train Accuracy:  0.900, Validation Accuracy:  0.891, Loss:  0.249
    Epoch   3 Batch  134/538 - Train Accuracy:  0.869, Validation Accuracy:  0.890, Loss:  0.300
    Epoch   3 Batch  135/538 - Train Accuracy:  0.908, Validation Accuracy:  0.889, Loss:  0.294
    Epoch   3 Batch  136/538 - Train Accuracy:  0.899, Validation Accuracy:  0.888, Loss:  0.277
    Epoch   3 Batch  137/538 - Train Accuracy:  0.888, Validation Accuracy:  0.891, Loss:  0.287
    Epoch   3 Batch  138/538 - Train Accuracy:  0.899, Validation Accuracy:  0.898, Loss:  0.271
    Epoch   3 Batch  139/538 - Train Accuracy:  0.883, Validation Accuracy:  0.896, Loss:  0.307
    Epoch   3 Batch  140/538 - Train Accuracy:  0.874, Validation Accuracy:  0.895, Loss:  0.300
    Epoch   3 Batch  141/538 - Train Accuracy:  0.904, Validation Accuracy:  0.895, Loss:  0.304
    Epoch   3 Batch  142/538 - Train Accuracy:  0.907, Validation Accuracy:  0.895, Loss:  0.264
    Epoch   3 Batch  143/538 - Train Accuracy:  0.910, Validation Accuracy:  0.897, Loss:  0.278
    Epoch   3 Batch  144/538 - Train Accuracy:  0.896, Validation Accuracy:  0.895, Loss:  0.279
    Epoch   3 Batch  145/538 - Train Accuracy:  0.875, Validation Accuracy:  0.898, Loss:  0.303
    Epoch   3 Batch  146/538 - Train Accuracy:  0.900, Validation Accuracy:  0.900, Loss:  0.280
    Epoch   3 Batch  147/538 - Train Accuracy:  0.898, Validation Accuracy:  0.896, Loss:  0.280
    Epoch   3 Batch  148/538 - Train Accuracy:  0.874, Validation Accuracy:  0.896, Loss:  0.335
    Epoch   3 Batch  149/538 - Train Accuracy:  0.916, Validation Accuracy:  0.897, Loss:  0.263
    Epoch   3 Batch  150/538 - Train Accuracy:  0.906, Validation Accuracy:  0.896, Loss:  0.267
    Epoch   3 Batch  151/538 - Train Accuracy:  0.908, Validation Accuracy:  0.897, Loss:  0.265
    Epoch   3 Batch  152/538 - Train Accuracy:  0.892, Validation Accuracy:  0.900, Loss:  0.278
    Epoch   3 Batch  153/538 - Train Accuracy:  0.882, Validation Accuracy:  0.892, Loss:  0.289
    Epoch   3 Batch  154/538 - Train Accuracy:  0.892, Validation Accuracy:  0.890, Loss:  0.259
    Epoch   3 Batch  155/538 - Train Accuracy:  0.879, Validation Accuracy:  0.896, Loss:  0.295
    Epoch   3 Batch  156/538 - Train Accuracy:  0.908, Validation Accuracy:  0.895, Loss:  0.276
    Epoch   3 Batch  157/538 - Train Accuracy:  0.904, Validation Accuracy:  0.894, Loss:  0.253
    Epoch   3 Batch  158/538 - Train Accuracy:  0.883, Validation Accuracy:  0.896, Loss:  0.280
    Epoch   3 Batch  159/538 - Train Accuracy:  0.882, Validation Accuracy:  0.893, Loss:  0.285
    Epoch   3 Batch  160/538 - Train Accuracy:  0.887, Validation Accuracy:  0.893, Loss:  0.254
    Epoch   3 Batch  161/538 - Train Accuracy:  0.897, Validation Accuracy:  0.893, Loss:  0.259
    Epoch   3 Batch  162/538 - Train Accuracy:  0.887, Validation Accuracy:  0.894, Loss:  0.266
    Epoch   3 Batch  163/538 - Train Accuracy:  0.891, Validation Accuracy:  0.894, Loss:  0.289
    Epoch   3 Batch  164/538 - Train Accuracy:  0.879, Validation Accuracy:  0.898, Loss:  0.310
    Epoch   3 Batch  165/538 - Train Accuracy:  0.898, Validation Accuracy:  0.894, Loss:  0.242
    Epoch   3 Batch  166/538 - Train Accuracy:  0.911, Validation Accuracy:  0.890, Loss:  0.282
    Epoch   3 Batch  167/538 - Train Accuracy:  0.890, Validation Accuracy:  0.886, Loss:  0.267
    Epoch   3 Batch  168/538 - Train Accuracy:  0.871, Validation Accuracy:  0.892, Loss:  0.298
    Epoch   3 Batch  169/538 - Train Accuracy:  0.912, Validation Accuracy:  0.894, Loss:  0.244
    Epoch   3 Batch  170/538 - Train Accuracy:  0.893, Validation Accuracy:  0.892, Loss:  0.277
    Epoch   3 Batch  171/538 - Train Accuracy:  0.901, Validation Accuracy:  0.889, Loss:  0.262
    Epoch   3 Batch  172/538 - Train Accuracy:  0.892, Validation Accuracy:  0.889, Loss:  0.266
    Epoch   3 Batch  173/538 - Train Accuracy:  0.904, Validation Accuracy:  0.881, Loss:  0.265
    Epoch   3 Batch  174/538 - Train Accuracy:  0.889, Validation Accuracy:  0.888, Loss:  0.263
    Epoch   3 Batch  175/538 - Train Accuracy:  0.902, Validation Accuracy:  0.890, Loss:  0.270
    Epoch   3 Batch  176/538 - Train Accuracy:  0.872, Validation Accuracy:  0.887, Loss:  0.303
    Epoch   3 Batch  177/538 - Train Accuracy:  0.899, Validation Accuracy:  0.891, Loss:  0.275
    Epoch   3 Batch  178/538 - Train Accuracy:  0.876, Validation Accuracy:  0.890, Loss:  0.268
    Epoch   3 Batch  179/538 - Train Accuracy:  0.928, Validation Accuracy:  0.897, Loss:  0.269
    Epoch   3 Batch  180/538 - Train Accuracy:  0.904, Validation Accuracy:  0.898, Loss:  0.260
    Epoch   3 Batch  181/538 - Train Accuracy:  0.906, Validation Accuracy:  0.898, Loss:  0.286
    Epoch   3 Batch  182/538 - Train Accuracy:  0.891, Validation Accuracy:  0.895, Loss:  0.263
    Epoch   3 Batch  183/538 - Train Accuracy:  0.924, Validation Accuracy:  0.891, Loss:  0.238
    Epoch   3 Batch  184/538 - Train Accuracy:  0.908, Validation Accuracy:  0.895, Loss:  0.258
    Epoch   3 Batch  185/538 - Train Accuracy:  0.919, Validation Accuracy:  0.902, Loss:  0.238
    Epoch   3 Batch  186/538 - Train Accuracy:  0.909, Validation Accuracy:  0.904, Loss:  0.251
    Epoch   3 Batch  187/538 - Train Accuracy:  0.913, Validation Accuracy:  0.903, Loss:  0.265
    Epoch   3 Batch  188/538 - Train Accuracy:  0.906, Validation Accuracy:  0.898, Loss:  0.267
    Epoch   3 Batch  189/538 - Train Accuracy:  0.895, Validation Accuracy:  0.897, Loss:  0.255
    Epoch   3 Batch  190/538 - Train Accuracy:  0.884, Validation Accuracy:  0.895, Loss:  0.279
    Epoch   3 Batch  191/538 - Train Accuracy:  0.912, Validation Accuracy:  0.897, Loss:  0.255
    Epoch   3 Batch  192/538 - Train Accuracy:  0.909, Validation Accuracy:  0.895, Loss:  0.272
    Epoch   3 Batch  193/538 - Train Accuracy:  0.896, Validation Accuracy:  0.891, Loss:  0.244
    Epoch   3 Batch  194/538 - Train Accuracy:  0.888, Validation Accuracy:  0.892, Loss:  0.270
    Epoch   3 Batch  195/538 - Train Accuracy:  0.919, Validation Accuracy:  0.889, Loss:  0.255
    Epoch   3 Batch  196/538 - Train Accuracy:  0.892, Validation Accuracy:  0.889, Loss:  0.264
    Epoch   3 Batch  197/538 - Train Accuracy:  0.906, Validation Accuracy:  0.889, Loss:  0.255
    Epoch   3 Batch  198/538 - Train Accuracy:  0.903, Validation Accuracy:  0.891, Loss:  0.261
    Epoch   3 Batch  199/538 - Train Accuracy:  0.886, Validation Accuracy:  0.893, Loss:  0.291
    Epoch   3 Batch  200/538 - Train Accuracy:  0.918, Validation Accuracy:  0.896, Loss:  0.243
    Epoch   3 Batch  201/538 - Train Accuracy:  0.907, Validation Accuracy:  0.893, Loss:  0.258
    Epoch   3 Batch  202/538 - Train Accuracy:  0.916, Validation Accuracy:  0.896, Loss:  0.254
    Epoch   3 Batch  203/538 - Train Accuracy:  0.895, Validation Accuracy:  0.899, Loss:  0.281
    Epoch   3 Batch  204/538 - Train Accuracy:  0.889, Validation Accuracy:  0.895, Loss:  0.270
    Epoch   3 Batch  205/538 - Train Accuracy:  0.912, Validation Accuracy:  0.894, Loss:  0.242
    Epoch   3 Batch  206/538 - Train Accuracy:  0.899, Validation Accuracy:  0.894, Loss:  0.260
    Epoch   3 Batch  207/538 - Train Accuracy:  0.921, Validation Accuracy:  0.896, Loss:  0.258
    Epoch   3 Batch  208/538 - Train Accuracy:  0.904, Validation Accuracy:  0.898, Loss:  0.281
    Epoch   3 Batch  209/538 - Train Accuracy:  0.922, Validation Accuracy:  0.900, Loss:  0.250
    Epoch   3 Batch  210/538 - Train Accuracy:  0.896, Validation Accuracy:  0.900, Loss:  0.261
    Epoch   3 Batch  211/538 - Train Accuracy:  0.892, Validation Accuracy:  0.899, Loss:  0.277
    Epoch   3 Batch  212/538 - Train Accuracy:  0.892, Validation Accuracy:  0.896, Loss:  0.246
    Epoch   3 Batch  213/538 - Train Accuracy:  0.906, Validation Accuracy:  0.897, Loss:  0.252
    Epoch   3 Batch  214/538 - Train Accuracy:  0.910, Validation Accuracy:  0.892, Loss:  0.255
    Epoch   3 Batch  215/538 - Train Accuracy:  0.899, Validation Accuracy:  0.891, Loss:  0.240
    Epoch   3 Batch  216/538 - Train Accuracy:  0.920, Validation Accuracy:  0.887, Loss:  0.260
    Epoch   3 Batch  217/538 - Train Accuracy:  0.913, Validation Accuracy:  0.892, Loss:  0.251
    Epoch   3 Batch  218/538 - Train Accuracy:  0.903, Validation Accuracy:  0.896, Loss:  0.253
    Epoch   3 Batch  219/538 - Train Accuracy:  0.898, Validation Accuracy:  0.898, Loss:  0.281
    Epoch   3 Batch  220/538 - Train Accuracy:  0.884, Validation Accuracy:  0.897, Loss:  0.254
    Epoch   3 Batch  221/538 - Train Accuracy:  0.924, Validation Accuracy:  0.895, Loss:  0.249
    Epoch   3 Batch  222/538 - Train Accuracy:  0.890, Validation Accuracy:  0.891, Loss:  0.248
    Epoch   3 Batch  223/538 - Train Accuracy:  0.880, Validation Accuracy:  0.891, Loss:  0.271
    Epoch   3 Batch  224/538 - Train Accuracy:  0.894, Validation Accuracy:  0.890, Loss:  0.262
    Epoch   3 Batch  225/538 - Train Accuracy:  0.915, Validation Accuracy:  0.891, Loss:  0.247
    Epoch   3 Batch  226/538 - Train Accuracy:  0.886, Validation Accuracy:  0.891, Loss:  0.248
    Epoch   3 Batch  227/538 - Train Accuracy:  0.908, Validation Accuracy:  0.891, Loss:  0.237
    Epoch   3 Batch  228/538 - Train Accuracy:  0.891, Validation Accuracy:  0.894, Loss:  0.240
    Epoch   3 Batch  229/538 - Train Accuracy:  0.885, Validation Accuracy:  0.888, Loss:  0.247
    Epoch   3 Batch  230/538 - Train Accuracy:  0.908, Validation Accuracy:  0.888, Loss:  0.248
    Epoch   3 Batch  231/538 - Train Accuracy:  0.908, Validation Accuracy:  0.892, Loss:  0.241
    Epoch   3 Batch  232/538 - Train Accuracy:  0.919, Validation Accuracy:  0.896, Loss:  0.235
    Epoch   3 Batch  233/538 - Train Accuracy:  0.901, Validation Accuracy:  0.898, Loss:  0.268
    Epoch   3 Batch  234/538 - Train Accuracy:  0.909, Validation Accuracy:  0.901, Loss:  0.244
    Epoch   3 Batch  235/538 - Train Accuracy:  0.897, Validation Accuracy:  0.907, Loss:  0.233
    Epoch   3 Batch  236/538 - Train Accuracy:  0.892, Validation Accuracy:  0.906, Loss:  0.263
    Epoch   3 Batch  237/538 - Train Accuracy:  0.901, Validation Accuracy:  0.906, Loss:  0.239
    Epoch   3 Batch  238/538 - Train Accuracy:  0.922, Validation Accuracy:  0.904, Loss:  0.229
    Epoch   3 Batch  239/538 - Train Accuracy:  0.893, Validation Accuracy:  0.904, Loss:  0.274
    Epoch   3 Batch  240/538 - Train Accuracy:  0.891, Validation Accuracy:  0.902, Loss:  0.272
    Epoch   3 Batch  241/538 - Train Accuracy:  0.895, Validation Accuracy:  0.899, Loss:  0.246
    Epoch   3 Batch  242/538 - Train Accuracy:  0.916, Validation Accuracy:  0.897, Loss:  0.235
    Epoch   3 Batch  243/538 - Train Accuracy:  0.922, Validation Accuracy:  0.895, Loss:  0.240
    Epoch   3 Batch  244/538 - Train Accuracy:  0.893, Validation Accuracy:  0.901, Loss:  0.248
    Epoch   3 Batch  245/538 - Train Accuracy:  0.892, Validation Accuracy:  0.893, Loss:  0.274
    Epoch   3 Batch  246/538 - Train Accuracy:  0.905, Validation Accuracy:  0.895, Loss:  0.229
    Epoch   3 Batch  247/538 - Train Accuracy:  0.899, Validation Accuracy:  0.895, Loss:  0.260
    Epoch   3 Batch  248/538 - Train Accuracy:  0.920, Validation Accuracy:  0.896, Loss:  0.253
    Epoch   3 Batch  249/538 - Train Accuracy:  0.900, Validation Accuracy:  0.900, Loss:  0.217
    Epoch   3 Batch  250/538 - Train Accuracy:  0.920, Validation Accuracy:  0.900, Loss:  0.239
    Epoch   3 Batch  251/538 - Train Accuracy:  0.917, Validation Accuracy:  0.906, Loss:  0.246
    Epoch   3 Batch  252/538 - Train Accuracy:  0.913, Validation Accuracy:  0.910, Loss:  0.224
    Epoch   3 Batch  253/538 - Train Accuracy:  0.892, Validation Accuracy:  0.910, Loss:  0.224
    Epoch   3 Batch  254/538 - Train Accuracy:  0.883, Validation Accuracy:  0.909, Loss:  0.281
    Epoch   3 Batch  255/538 - Train Accuracy:  0.913, Validation Accuracy:  0.907, Loss:  0.243
    Epoch   3 Batch  256/538 - Train Accuracy:  0.892, Validation Accuracy:  0.906, Loss:  0.279
    Epoch   3 Batch  257/538 - Train Accuracy:  0.907, Validation Accuracy:  0.896, Loss:  0.239
    Epoch   3 Batch  258/538 - Train Accuracy:  0.907, Validation Accuracy:  0.897, Loss:  0.252
    Epoch   3 Batch  259/538 - Train Accuracy:  0.914, Validation Accuracy:  0.899, Loss:  0.243
    Epoch   3 Batch  260/538 - Train Accuracy:  0.880, Validation Accuracy:  0.895, Loss:  0.267
    Epoch   3 Batch  261/538 - Train Accuracy:  0.905, Validation Accuracy:  0.896, Loss:  0.261
    Epoch   3 Batch  262/538 - Train Accuracy:  0.914, Validation Accuracy:  0.892, Loss:  0.263
    Epoch   3 Batch  263/538 - Train Accuracy:  0.885, Validation Accuracy:  0.889, Loss:  0.242
    Epoch   3 Batch  264/538 - Train Accuracy:  0.879, Validation Accuracy:  0.891, Loss:  0.248
    Epoch   3 Batch  265/538 - Train Accuracy:  0.889, Validation Accuracy:  0.894, Loss:  0.260
    Epoch   3 Batch  266/538 - Train Accuracy:  0.891, Validation Accuracy:  0.896, Loss:  0.260
    Epoch   3 Batch  267/538 - Train Accuracy:  0.911, Validation Accuracy:  0.894, Loss:  0.258
    Epoch   3 Batch  268/538 - Train Accuracy:  0.916, Validation Accuracy:  0.901, Loss:  0.218
    Epoch   3 Batch  269/538 - Train Accuracy:  0.898, Validation Accuracy:  0.899, Loss:  0.250
    Epoch   3 Batch  270/538 - Train Accuracy:  0.895, Validation Accuracy:  0.897, Loss:  0.236
    Epoch   3 Batch  271/538 - Train Accuracy:  0.919, Validation Accuracy:  0.899, Loss:  0.240
    Epoch   3 Batch  272/538 - Train Accuracy:  0.894, Validation Accuracy:  0.896, Loss:  0.269
    Epoch   3 Batch  273/538 - Train Accuracy:  0.903, Validation Accuracy:  0.899, Loss:  0.255
    Epoch   3 Batch  274/538 - Train Accuracy:  0.875, Validation Accuracy:  0.899, Loss:  0.285
    Epoch   3 Batch  275/538 - Train Accuracy:  0.895, Validation Accuracy:  0.898, Loss:  0.258
    Epoch   3 Batch  276/538 - Train Accuracy:  0.899, Validation Accuracy:  0.895, Loss:  0.264
    Epoch   3 Batch  277/538 - Train Accuracy:  0.900, Validation Accuracy:  0.898, Loss:  0.230
    Epoch   3 Batch  278/538 - Train Accuracy:  0.920, Validation Accuracy:  0.892, Loss:  0.231
    Epoch   3 Batch  279/538 - Train Accuracy:  0.898, Validation Accuracy:  0.894, Loss:  0.238
    Epoch   3 Batch  280/538 - Train Accuracy:  0.906, Validation Accuracy:  0.890, Loss:  0.227
    Epoch   3 Batch  281/538 - Train Accuracy:  0.902, Validation Accuracy:  0.895, Loss:  0.272
    Epoch   3 Batch  282/538 - Train Accuracy:  0.912, Validation Accuracy:  0.898, Loss:  0.247
    Epoch   3 Batch  283/538 - Train Accuracy:  0.908, Validation Accuracy:  0.900, Loss:  0.238
    Epoch   3 Batch  284/538 - Train Accuracy:  0.899, Validation Accuracy:  0.897, Loss:  0.251
    Epoch   3 Batch  285/538 - Train Accuracy:  0.914, Validation Accuracy:  0.899, Loss:  0.225
    Epoch   3 Batch  286/538 - Train Accuracy:  0.889, Validation Accuracy:  0.900, Loss:  0.268
    Epoch   3 Batch  287/538 - Train Accuracy:  0.935, Validation Accuracy:  0.899, Loss:  0.227
    Epoch   3 Batch  288/538 - Train Accuracy:  0.915, Validation Accuracy:  0.898, Loss:  0.245
    Epoch   3 Batch  289/538 - Train Accuracy:  0.906, Validation Accuracy:  0.900, Loss:  0.217
    Epoch   3 Batch  290/538 - Train Accuracy:  0.910, Validation Accuracy:  0.903, Loss:  0.251
    Epoch   3 Batch  291/538 - Train Accuracy:  0.919, Validation Accuracy:  0.895, Loss:  0.235
    Epoch   3 Batch  292/538 - Train Accuracy:  0.911, Validation Accuracy:  0.893, Loss:  0.218
    Epoch   3 Batch  293/538 - Train Accuracy:  0.904, Validation Accuracy:  0.895, Loss:  0.238
    Epoch   3 Batch  294/538 - Train Accuracy:  0.902, Validation Accuracy:  0.894, Loss:  0.238
    Epoch   3 Batch  295/538 - Train Accuracy:  0.915, Validation Accuracy:  0.893, Loss:  0.225
    Epoch   3 Batch  296/538 - Train Accuracy:  0.907, Validation Accuracy:  0.896, Loss:  0.236
    Epoch   3 Batch  297/538 - Train Accuracy:  0.918, Validation Accuracy:  0.902, Loss:  0.239
    Epoch   3 Batch  298/538 - Train Accuracy:  0.892, Validation Accuracy:  0.901, Loss:  0.250
    Epoch   3 Batch  299/538 - Train Accuracy:  0.906, Validation Accuracy:  0.904, Loss:  0.250
    Epoch   3 Batch  300/538 - Train Accuracy:  0.902, Validation Accuracy:  0.897, Loss:  0.231
    Epoch   3 Batch  301/538 - Train Accuracy:  0.895, Validation Accuracy:  0.897, Loss:  0.258
    Epoch   3 Batch  302/538 - Train Accuracy:  0.918, Validation Accuracy:  0.894, Loss:  0.228
    Epoch   3 Batch  303/538 - Train Accuracy:  0.918, Validation Accuracy:  0.896, Loss:  0.226
    Epoch   3 Batch  304/538 - Train Accuracy:  0.902, Validation Accuracy:  0.894, Loss:  0.252
    Epoch   3 Batch  305/538 - Train Accuracy:  0.926, Validation Accuracy:  0.900, Loss:  0.228
    Epoch   3 Batch  306/538 - Train Accuracy:  0.910, Validation Accuracy:  0.900, Loss:  0.240
    Epoch   3 Batch  307/538 - Train Accuracy:  0.912, Validation Accuracy:  0.900, Loss:  0.234
    Epoch   3 Batch  308/538 - Train Accuracy:  0.919, Validation Accuracy:  0.904, Loss:  0.235
    Epoch   3 Batch  309/538 - Train Accuracy:  0.904, Validation Accuracy:  0.903, Loss:  0.222
    Epoch   3 Batch  310/538 - Train Accuracy:  0.922, Validation Accuracy:  0.901, Loss:  0.237
    Epoch   3 Batch  311/538 - Train Accuracy:  0.904, Validation Accuracy:  0.904, Loss:  0.232
    Epoch   3 Batch  312/538 - Train Accuracy:  0.908, Validation Accuracy:  0.902, Loss:  0.225
    Epoch   3 Batch  313/538 - Train Accuracy:  0.915, Validation Accuracy:  0.904, Loss:  0.257
    Epoch   3 Batch  314/538 - Train Accuracy:  0.909, Validation Accuracy:  0.907, Loss:  0.232
    Epoch   3 Batch  315/538 - Train Accuracy:  0.893, Validation Accuracy:  0.907, Loss:  0.236
    Epoch   3 Batch  316/538 - Train Accuracy:  0.908, Validation Accuracy:  0.907, Loss:  0.221
    Epoch   3 Batch  317/538 - Train Accuracy:  0.914, Validation Accuracy:  0.904, Loss:  0.236
    Epoch   3 Batch  318/538 - Train Accuracy:  0.900, Validation Accuracy:  0.895, Loss:  0.238
    Epoch   3 Batch  319/538 - Train Accuracy:  0.909, Validation Accuracy:  0.892, Loss:  0.224
    Epoch   3 Batch  320/538 - Train Accuracy:  0.898, Validation Accuracy:  0.903, Loss:  0.239
    Epoch   3 Batch  321/538 - Train Accuracy:  0.902, Validation Accuracy:  0.908, Loss:  0.220
    Epoch   3 Batch  322/538 - Train Accuracy:  0.909, Validation Accuracy:  0.912, Loss:  0.245
    Epoch   3 Batch  323/538 - Train Accuracy:  0.917, Validation Accuracy:  0.915, Loss:  0.220
    Epoch   3 Batch  324/538 - Train Accuracy:  0.916, Validation Accuracy:  0.916, Loss:  0.241
    Epoch   3 Batch  325/538 - Train Accuracy:  0.915, Validation Accuracy:  0.915, Loss:  0.228
    Epoch   3 Batch  326/538 - Train Accuracy:  0.914, Validation Accuracy:  0.915, Loss:  0.218
    Epoch   3 Batch  327/538 - Train Accuracy:  0.911, Validation Accuracy:  0.912, Loss:  0.235
    Epoch   3 Batch  328/538 - Train Accuracy:  0.921, Validation Accuracy:  0.913, Loss:  0.215
    Epoch   3 Batch  329/538 - Train Accuracy:  0.918, Validation Accuracy:  0.912, Loss:  0.230
    Epoch   3 Batch  330/538 - Train Accuracy:  0.918, Validation Accuracy:  0.906, Loss:  0.216
    Epoch   3 Batch  331/538 - Train Accuracy:  0.911, Validation Accuracy:  0.906, Loss:  0.228
    Epoch   3 Batch  332/538 - Train Accuracy:  0.915, Validation Accuracy:  0.902, Loss:  0.237
    Epoch   3 Batch  333/538 - Train Accuracy:  0.915, Validation Accuracy:  0.903, Loss:  0.221
    Epoch   3 Batch  334/538 - Train Accuracy:  0.919, Validation Accuracy:  0.903, Loss:  0.226
    Epoch   3 Batch  335/538 - Train Accuracy:  0.916, Validation Accuracy:  0.906, Loss:  0.227
    Epoch   3 Batch  336/538 - Train Accuracy:  0.915, Validation Accuracy:  0.911, Loss:  0.229
    Epoch   3 Batch  337/538 - Train Accuracy:  0.895, Validation Accuracy:  0.912, Loss:  0.232
    Epoch   3 Batch  338/538 - Train Accuracy:  0.898, Validation Accuracy:  0.913, Loss:  0.242
    Epoch   3 Batch  339/538 - Train Accuracy:  0.901, Validation Accuracy:  0.912, Loss:  0.217
    Epoch   3 Batch  340/538 - Train Accuracy:  0.904, Validation Accuracy:  0.915, Loss:  0.224
    Epoch   3 Batch  341/538 - Train Accuracy:  0.904, Validation Accuracy:  0.917, Loss:  0.238
    Epoch   3 Batch  342/538 - Train Accuracy:  0.917, Validation Accuracy:  0.914, Loss:  0.244
    Epoch   3 Batch  343/538 - Train Accuracy:  0.928, Validation Accuracy:  0.915, Loss:  0.230
    Epoch   3 Batch  344/538 - Train Accuracy:  0.922, Validation Accuracy:  0.914, Loss:  0.216
    Epoch   3 Batch  345/538 - Train Accuracy:  0.910, Validation Accuracy:  0.914, Loss:  0.220
    Epoch   3 Batch  346/538 - Train Accuracy:  0.893, Validation Accuracy:  0.911, Loss:  0.254
    Epoch   3 Batch  347/538 - Train Accuracy:  0.920, Validation Accuracy:  0.903, Loss:  0.229
    Epoch   3 Batch  348/538 - Train Accuracy:  0.905, Validation Accuracy:  0.900, Loss:  0.216
    Epoch   3 Batch  349/538 - Train Accuracy:  0.929, Validation Accuracy:  0.898, Loss:  0.216
    Epoch   3 Batch  350/538 - Train Accuracy:  0.916, Validation Accuracy:  0.897, Loss:  0.246
    Epoch   3 Batch  351/538 - Train Accuracy:  0.901, Validation Accuracy:  0.897, Loss:  0.258
    Epoch   3 Batch  352/538 - Train Accuracy:  0.894, Validation Accuracy:  0.897, Loss:  0.258
    Epoch   3 Batch  353/538 - Train Accuracy:  0.901, Validation Accuracy:  0.897, Loss:  0.230
    Epoch   3 Batch  354/538 - Train Accuracy:  0.890, Validation Accuracy:  0.897, Loss:  0.240
    Epoch   3 Batch  355/538 - Train Accuracy:  0.896, Validation Accuracy:  0.903, Loss:  0.251
    Epoch   3 Batch  356/538 - Train Accuracy:  0.911, Validation Accuracy:  0.902, Loss:  0.220
    Epoch   3 Batch  357/538 - Train Accuracy:  0.911, Validation Accuracy:  0.904, Loss:  0.234
    Epoch   3 Batch  358/538 - Train Accuracy:  0.916, Validation Accuracy:  0.906, Loss:  0.211
    Epoch   3 Batch  359/538 - Train Accuracy:  0.887, Validation Accuracy:  0.910, Loss:  0.221
    Epoch   3 Batch  360/538 - Train Accuracy:  0.909, Validation Accuracy:  0.913, Loss:  0.240
    Epoch   3 Batch  361/538 - Train Accuracy:  0.918, Validation Accuracy:  0.911, Loss:  0.219
    Epoch   3 Batch  362/538 - Train Accuracy:  0.927, Validation Accuracy:  0.912, Loss:  0.201
    Epoch   3 Batch  363/538 - Train Accuracy:  0.904, Validation Accuracy:  0.913, Loss:  0.216
    Epoch   3 Batch  364/538 - Train Accuracy:  0.889, Validation Accuracy:  0.916, Loss:  0.241
    Epoch   3 Batch  365/538 - Train Accuracy:  0.889, Validation Accuracy:  0.919, Loss:  0.237
    Epoch   3 Batch  366/538 - Train Accuracy:  0.918, Validation Accuracy:  0.917, Loss:  0.233
    Epoch   3 Batch  367/538 - Train Accuracy:  0.918, Validation Accuracy:  0.914, Loss:  0.209
    Epoch   3 Batch  368/538 - Train Accuracy:  0.917, Validation Accuracy:  0.914, Loss:  0.204
    Epoch   3 Batch  369/538 - Train Accuracy:  0.918, Validation Accuracy:  0.915, Loss:  0.222
    Epoch   3 Batch  370/538 - Train Accuracy:  0.904, Validation Accuracy:  0.912, Loss:  0.230
    Epoch   3 Batch  371/538 - Train Accuracy:  0.919, Validation Accuracy:  0.905, Loss:  0.226
    Epoch   3 Batch  372/538 - Train Accuracy:  0.926, Validation Accuracy:  0.899, Loss:  0.219
    Epoch   3 Batch  373/538 - Train Accuracy:  0.906, Validation Accuracy:  0.903, Loss:  0.216
    Epoch   3 Batch  374/538 - Train Accuracy:  0.918, Validation Accuracy:  0.909, Loss:  0.234
    Epoch   3 Batch  375/538 - Train Accuracy:  0.919, Validation Accuracy:  0.915, Loss:  0.200
    Epoch   3 Batch  376/538 - Train Accuracy:  0.907, Validation Accuracy:  0.915, Loss:  0.232
    Epoch   3 Batch  377/538 - Train Accuracy:  0.921, Validation Accuracy:  0.914, Loss:  0.226
    Epoch   3 Batch  378/538 - Train Accuracy:  0.921, Validation Accuracy:  0.904, Loss:  0.217
    Epoch   3 Batch  379/538 - Train Accuracy:  0.913, Validation Accuracy:  0.907, Loss:  0.228
    Epoch   3 Batch  380/538 - Train Accuracy:  0.912, Validation Accuracy:  0.908, Loss:  0.216
    Epoch   3 Batch  381/538 - Train Accuracy:  0.910, Validation Accuracy:  0.904, Loss:  0.198
    Epoch   3 Batch  382/538 - Train Accuracy:  0.894, Validation Accuracy:  0.900, Loss:  0.224
    Epoch   3 Batch  383/538 - Train Accuracy:  0.897, Validation Accuracy:  0.901, Loss:  0.219
    Epoch   3 Batch  384/538 - Train Accuracy:  0.897, Validation Accuracy:  0.906, Loss:  0.236
    Epoch   3 Batch  385/538 - Train Accuracy:  0.896, Validation Accuracy:  0.913, Loss:  0.225
    Epoch   3 Batch  386/538 - Train Accuracy:  0.915, Validation Accuracy:  0.910, Loss:  0.232
    Epoch   3 Batch  387/538 - Train Accuracy:  0.909, Validation Accuracy:  0.914, Loss:  0.218
    Epoch   3 Batch  388/538 - Train Accuracy:  0.912, Validation Accuracy:  0.911, Loss:  0.225
    Epoch   3 Batch  389/538 - Train Accuracy:  0.897, Validation Accuracy:  0.904, Loss:  0.247
    Epoch   3 Batch  390/538 - Train Accuracy:  0.929, Validation Accuracy:  0.903, Loss:  0.209
    Epoch   3 Batch  391/538 - Train Accuracy:  0.902, Validation Accuracy:  0.903, Loss:  0.227
    Epoch   3 Batch  392/538 - Train Accuracy:  0.912, Validation Accuracy:  0.902, Loss:  0.213
    Epoch   3 Batch  393/538 - Train Accuracy:  0.918, Validation Accuracy:  0.900, Loss:  0.196
    Epoch   3 Batch  394/538 - Train Accuracy:  0.893, Validation Accuracy:  0.903, Loss:  0.230
    Epoch   3 Batch  395/538 - Train Accuracy:  0.915, Validation Accuracy:  0.902, Loss:  0.257
    Epoch   3 Batch  396/538 - Train Accuracy:  0.917, Validation Accuracy:  0.905, Loss:  0.223
    Epoch   3 Batch  397/538 - Train Accuracy:  0.914, Validation Accuracy:  0.904, Loss:  0.236
    Epoch   3 Batch  398/538 - Train Accuracy:  0.910, Validation Accuracy:  0.907, Loss:  0.223
    Epoch   3 Batch  399/538 - Train Accuracy:  0.888, Validation Accuracy:  0.902, Loss:  0.247
    Epoch   3 Batch  400/538 - Train Accuracy:  0.914, Validation Accuracy:  0.901, Loss:  0.233
    Epoch   3 Batch  401/538 - Train Accuracy:  0.907, Validation Accuracy:  0.913, Loss:  0.226
    Epoch   3 Batch  402/538 - Train Accuracy:  0.900, Validation Accuracy:  0.916, Loss:  0.230
    Epoch   3 Batch  403/538 - Train Accuracy:  0.926, Validation Accuracy:  0.917, Loss:  0.213
    Epoch   3 Batch  404/538 - Train Accuracy:  0.913, Validation Accuracy:  0.908, Loss:  0.217
    Epoch   3 Batch  405/538 - Train Accuracy:  0.909, Validation Accuracy:  0.909, Loss:  0.224
    Epoch   3 Batch  406/538 - Train Accuracy:  0.894, Validation Accuracy:  0.912, Loss:  0.224
    Epoch   3 Batch  407/538 - Train Accuracy:  0.926, Validation Accuracy:  0.905, Loss:  0.220
    Epoch   3 Batch  408/538 - Train Accuracy:  0.891, Validation Accuracy:  0.907, Loss:  0.258
    Epoch   3 Batch  409/538 - Train Accuracy:  0.910, Validation Accuracy:  0.913, Loss:  0.227
    Epoch   3 Batch  410/538 - Train Accuracy:  0.925, Validation Accuracy:  0.912, Loss:  0.214
    Epoch   3 Batch  411/538 - Train Accuracy:  0.923, Validation Accuracy:  0.915, Loss:  0.209
    Epoch   3 Batch  412/538 - Train Accuracy:  0.909, Validation Accuracy:  0.914, Loss:  0.209
    Epoch   3 Batch  413/538 - Train Accuracy:  0.912, Validation Accuracy:  0.912, Loss:  0.236
    Epoch   3 Batch  414/538 - Train Accuracy:  0.879, Validation Accuracy:  0.901, Loss:  0.246
    Epoch   3 Batch  415/538 - Train Accuracy:  0.888, Validation Accuracy:  0.902, Loss:  0.227
    Epoch   3 Batch  416/538 - Train Accuracy:  0.907, Validation Accuracy:  0.907, Loss:  0.231
    Epoch   3 Batch  417/538 - Train Accuracy:  0.910, Validation Accuracy:  0.911, Loss:  0.231
    Epoch   3 Batch  418/538 - Train Accuracy:  0.906, Validation Accuracy:  0.909, Loss:  0.247
    Epoch   3 Batch  419/538 - Train Accuracy:  0.915, Validation Accuracy:  0.908, Loss:  0.218
    Epoch   3 Batch  420/538 - Train Accuracy:  0.933, Validation Accuracy:  0.909, Loss:  0.214
    Epoch   3 Batch  421/538 - Train Accuracy:  0.916, Validation Accuracy:  0.911, Loss:  0.219
    Epoch   3 Batch  422/538 - Train Accuracy:  0.903, Validation Accuracy:  0.910, Loss:  0.220
    Epoch   3 Batch  423/538 - Train Accuracy:  0.904, Validation Accuracy:  0.908, Loss:  0.239
    Epoch   3 Batch  424/538 - Train Accuracy:  0.898, Validation Accuracy:  0.914, Loss:  0.240
    Epoch   3 Batch  425/538 - Train Accuracy:  0.901, Validation Accuracy:  0.915, Loss:  0.224
    Epoch   3 Batch  426/538 - Train Accuracy:  0.912, Validation Accuracy:  0.914, Loss:  0.217
    Epoch   3 Batch  427/538 - Train Accuracy:  0.891, Validation Accuracy:  0.912, Loss:  0.222
    Epoch   3 Batch  428/538 - Train Accuracy:  0.909, Validation Accuracy:  0.909, Loss:  0.201
    Epoch   3 Batch  429/538 - Train Accuracy:  0.911, Validation Accuracy:  0.910, Loss:  0.219
    Epoch   3 Batch  430/538 - Train Accuracy:  0.905, Validation Accuracy:  0.906, Loss:  0.225
    Epoch   3 Batch  431/538 - Train Accuracy:  0.911, Validation Accuracy:  0.911, Loss:  0.217
    Epoch   3 Batch  432/538 - Train Accuracy:  0.908, Validation Accuracy:  0.919, Loss:  0.222
    Epoch   3 Batch  433/538 - Train Accuracy:  0.892, Validation Accuracy:  0.916, Loss:  0.256
    Epoch   3 Batch  434/538 - Train Accuracy:  0.891, Validation Accuracy:  0.915, Loss:  0.226
    Epoch   3 Batch  435/538 - Train Accuracy:  0.914, Validation Accuracy:  0.918, Loss:  0.220
    Epoch   3 Batch  436/538 - Train Accuracy:  0.907, Validation Accuracy:  0.915, Loss:  0.225
    Epoch   3 Batch  437/538 - Train Accuracy:  0.921, Validation Accuracy:  0.916, Loss:  0.220
    Epoch   3 Batch  438/538 - Train Accuracy:  0.927, Validation Accuracy:  0.914, Loss:  0.209
    Epoch   3 Batch  439/538 - Train Accuracy:  0.937, Validation Accuracy:  0.911, Loss:  0.204
    Epoch   3 Batch  440/538 - Train Accuracy:  0.905, Validation Accuracy:  0.909, Loss:  0.231
    Epoch   3 Batch  441/538 - Train Accuracy:  0.902, Validation Accuracy:  0.904, Loss:  0.221
    Epoch   3 Batch  442/538 - Train Accuracy:  0.914, Validation Accuracy:  0.905, Loss:  0.195
    Epoch   3 Batch  443/538 - Train Accuracy:  0.908, Validation Accuracy:  0.904, Loss:  0.222
    Epoch   3 Batch  444/538 - Train Accuracy:  0.925, Validation Accuracy:  0.909, Loss:  0.202
    Epoch   3 Batch  445/538 - Train Accuracy:  0.929, Validation Accuracy:  0.910, Loss:  0.191
    Epoch   3 Batch  446/538 - Train Accuracy:  0.928, Validation Accuracy:  0.912, Loss:  0.207
    Epoch   3 Batch  447/538 - Train Accuracy:  0.897, Validation Accuracy:  0.913, Loss:  0.217
    Epoch   3 Batch  448/538 - Train Accuracy:  0.910, Validation Accuracy:  0.915, Loss:  0.191
    Epoch   3 Batch  449/538 - Train Accuracy:  0.921, Validation Accuracy:  0.921, Loss:  0.224
    Epoch   3 Batch  450/538 - Train Accuracy:  0.910, Validation Accuracy:  0.920, Loss:  0.227
    Epoch   3 Batch  451/538 - Train Accuracy:  0.896, Validation Accuracy:  0.917, Loss:  0.222
    Epoch   3 Batch  452/538 - Train Accuracy:  0.914, Validation Accuracy:  0.912, Loss:  0.194
    Epoch   3 Batch  453/538 - Train Accuracy:  0.917, Validation Accuracy:  0.907, Loss:  0.225
    Epoch   3 Batch  454/538 - Train Accuracy:  0.908, Validation Accuracy:  0.909, Loss:  0.207
    Epoch   3 Batch  455/538 - Train Accuracy:  0.913, Validation Accuracy:  0.907, Loss:  0.210
    Epoch   3 Batch  456/538 - Train Accuracy:  0.931, Validation Accuracy:  0.915, Loss:  0.209
    Epoch   3 Batch  457/538 - Train Accuracy:  0.895, Validation Accuracy:  0.912, Loss:  0.218
    Epoch   3 Batch  458/538 - Train Accuracy:  0.914, Validation Accuracy:  0.912, Loss:  0.194
    Epoch   3 Batch  459/538 - Train Accuracy:  0.914, Validation Accuracy:  0.912, Loss:  0.202
    Epoch   3 Batch  460/538 - Train Accuracy:  0.885, Validation Accuracy:  0.908, Loss:  0.229
    Epoch   3 Batch  461/538 - Train Accuracy:  0.931, Validation Accuracy:  0.910, Loss:  0.209
    Epoch   3 Batch  462/538 - Train Accuracy:  0.915, Validation Accuracy:  0.909, Loss:  0.213
    Epoch   3 Batch  463/538 - Train Accuracy:  0.900, Validation Accuracy:  0.911, Loss:  0.226
    Epoch   3 Batch  464/538 - Train Accuracy:  0.921, Validation Accuracy:  0.914, Loss:  0.203
    Epoch   3 Batch  465/538 - Train Accuracy:  0.915, Validation Accuracy:  0.910, Loss:  0.197
    Epoch   3 Batch  466/538 - Train Accuracy:  0.913, Validation Accuracy:  0.911, Loss:  0.216
    Epoch   3 Batch  467/538 - Train Accuracy:  0.922, Validation Accuracy:  0.905, Loss:  0.207
    Epoch   3 Batch  468/538 - Train Accuracy:  0.929, Validation Accuracy:  0.910, Loss:  0.221
    Epoch   3 Batch  469/538 - Train Accuracy:  0.907, Validation Accuracy:  0.909, Loss:  0.213
    Epoch   3 Batch  470/538 - Train Accuracy:  0.921, Validation Accuracy:  0.905, Loss:  0.196
    Epoch   3 Batch  471/538 - Train Accuracy:  0.920, Validation Accuracy:  0.906, Loss:  0.196
    Epoch   3 Batch  472/538 - Train Accuracy:  0.948, Validation Accuracy:  0.906, Loss:  0.195
    Epoch   3 Batch  473/538 - Train Accuracy:  0.901, Validation Accuracy:  0.914, Loss:  0.215
    Epoch   3 Batch  474/538 - Train Accuracy:  0.926, Validation Accuracy:  0.912, Loss:  0.201
    Epoch   3 Batch  475/538 - Train Accuracy:  0.922, Validation Accuracy:  0.910, Loss:  0.208
    Epoch   3 Batch  476/538 - Train Accuracy:  0.922, Validation Accuracy:  0.908, Loss:  0.207
    Epoch   3 Batch  477/538 - Train Accuracy:  0.918, Validation Accuracy:  0.911, Loss:  0.219
    Epoch   3 Batch  478/538 - Train Accuracy:  0.916, Validation Accuracy:  0.909, Loss:  0.202
    Epoch   3 Batch  479/538 - Train Accuracy:  0.919, Validation Accuracy:  0.909, Loss:  0.189
    Epoch   3 Batch  480/538 - Train Accuracy:  0.921, Validation Accuracy:  0.907, Loss:  0.195
    Epoch   3 Batch  481/538 - Train Accuracy:  0.933, Validation Accuracy:  0.909, Loss:  0.220
    Epoch   3 Batch  482/538 - Train Accuracy:  0.912, Validation Accuracy:  0.909, Loss:  0.192
    Epoch   3 Batch  483/538 - Train Accuracy:  0.883, Validation Accuracy:  0.912, Loss:  0.221
    Epoch   3 Batch  484/538 - Train Accuracy:  0.912, Validation Accuracy:  0.911, Loss:  0.246
    Epoch   3 Batch  485/538 - Train Accuracy:  0.909, Validation Accuracy:  0.905, Loss:  0.207
    Epoch   3 Batch  486/538 - Train Accuracy:  0.934, Validation Accuracy:  0.908, Loss:  0.185
    Epoch   3 Batch  487/538 - Train Accuracy:  0.922, Validation Accuracy:  0.912, Loss:  0.179
    Epoch   3 Batch  488/538 - Train Accuracy:  0.925, Validation Accuracy:  0.910, Loss:  0.199
    Epoch   3 Batch  489/538 - Train Accuracy:  0.908, Validation Accuracy:  0.905, Loss:  0.206
    Epoch   3 Batch  490/538 - Train Accuracy:  0.907, Validation Accuracy:  0.911, Loss:  0.200
    Epoch   3 Batch  491/538 - Train Accuracy:  0.880, Validation Accuracy:  0.916, Loss:  0.217
    Epoch   3 Batch  492/538 - Train Accuracy:  0.918, Validation Accuracy:  0.914, Loss:  0.213
    Epoch   3 Batch  493/538 - Train Accuracy:  0.909, Validation Accuracy:  0.914, Loss:  0.200
    Epoch   3 Batch  494/538 - Train Accuracy:  0.916, Validation Accuracy:  0.910, Loss:  0.235
    Epoch   3 Batch  495/538 - Train Accuracy:  0.927, Validation Accuracy:  0.912, Loss:  0.216
    Epoch   3 Batch  496/538 - Train Accuracy:  0.923, Validation Accuracy:  0.911, Loss:  0.200
    Epoch   3 Batch  497/538 - Train Accuracy:  0.926, Validation Accuracy:  0.912, Loss:  0.191
    Epoch   3 Batch  498/538 - Train Accuracy:  0.924, Validation Accuracy:  0.914, Loss:  0.193
    Epoch   3 Batch  499/538 - Train Accuracy:  0.910, Validation Accuracy:  0.909, Loss:  0.205
    Epoch   3 Batch  500/538 - Train Accuracy:  0.919, Validation Accuracy:  0.905, Loss:  0.185
    Epoch   3 Batch  501/538 - Train Accuracy:  0.922, Validation Accuracy:  0.905, Loss:  0.216
    Epoch   3 Batch  502/538 - Train Accuracy:  0.910, Validation Accuracy:  0.911, Loss:  0.207
    Epoch   3 Batch  503/538 - Train Accuracy:  0.936, Validation Accuracy:  0.919, Loss:  0.200
    Epoch   3 Batch  504/538 - Train Accuracy:  0.937, Validation Accuracy:  0.924, Loss:  0.194
    Epoch   3 Batch  505/538 - Train Accuracy:  0.936, Validation Accuracy:  0.925, Loss:  0.198
    Epoch   3 Batch  506/538 - Train Accuracy:  0.920, Validation Accuracy:  0.923, Loss:  0.187
    Epoch   3 Batch  507/538 - Train Accuracy:  0.898, Validation Accuracy:  0.923, Loss:  0.226
    Epoch   3 Batch  508/538 - Train Accuracy:  0.902, Validation Accuracy:  0.918, Loss:  0.198
    Epoch   3 Batch  509/538 - Train Accuracy:  0.912, Validation Accuracy:  0.912, Loss:  0.200
    Epoch   3 Batch  510/538 - Train Accuracy:  0.921, Validation Accuracy:  0.906, Loss:  0.198
    Epoch   3 Batch  511/538 - Train Accuracy:  0.905, Validation Accuracy:  0.909, Loss:  0.209
    Epoch   3 Batch  512/538 - Train Accuracy:  0.922, Validation Accuracy:  0.901, Loss:  0.207
    Epoch   3 Batch  513/538 - Train Accuracy:  0.905, Validation Accuracy:  0.899, Loss:  0.199
    Epoch   3 Batch  514/538 - Train Accuracy:  0.913, Validation Accuracy:  0.896, Loss:  0.208
    Epoch   3 Batch  515/538 - Train Accuracy:  0.925, Validation Accuracy:  0.898, Loss:  0.204
    Epoch   3 Batch  516/538 - Train Accuracy:  0.905, Validation Accuracy:  0.898, Loss:  0.212
    Epoch   3 Batch  517/538 - Train Accuracy:  0.909, Validation Accuracy:  0.896, Loss:  0.207
    Epoch   3 Batch  518/538 - Train Accuracy:  0.923, Validation Accuracy:  0.899, Loss:  0.221
    Epoch   3 Batch  519/538 - Train Accuracy:  0.913, Validation Accuracy:  0.900, Loss:  0.208
    Epoch   3 Batch  520/538 - Train Accuracy:  0.900, Validation Accuracy:  0.914, Loss:  0.226
    Epoch   3 Batch  521/538 - Train Accuracy:  0.925, Validation Accuracy:  0.920, Loss:  0.212
    Epoch   3 Batch  522/538 - Train Accuracy:  0.932, Validation Accuracy:  0.925, Loss:  0.189
    Epoch   3 Batch  523/538 - Train Accuracy:  0.915, Validation Accuracy:  0.921, Loss:  0.195
    Epoch   3 Batch  524/538 - Train Accuracy:  0.915, Validation Accuracy:  0.912, Loss:  0.214
    Epoch   3 Batch  525/538 - Train Accuracy:  0.908, Validation Accuracy:  0.907, Loss:  0.204
    Epoch   3 Batch  526/538 - Train Accuracy:  0.928, Validation Accuracy:  0.904, Loss:  0.211
    Epoch   3 Batch  527/538 - Train Accuracy:  0.918, Validation Accuracy:  0.913, Loss:  0.194
    Epoch   3 Batch  528/538 - Train Accuracy:  0.919, Validation Accuracy:  0.924, Loss:  0.239
    Epoch   3 Batch  529/538 - Train Accuracy:  0.886, Validation Accuracy:  0.924, Loss:  0.206
    Epoch   3 Batch  530/538 - Train Accuracy:  0.900, Validation Accuracy:  0.926, Loss:  0.230
    Epoch   3 Batch  531/538 - Train Accuracy:  0.908, Validation Accuracy:  0.927, Loss:  0.210
    Epoch   3 Batch  532/538 - Train Accuracy:  0.914, Validation Accuracy:  0.926, Loss:  0.200
    Epoch   3 Batch  533/538 - Train Accuracy:  0.930, Validation Accuracy:  0.920, Loss:  0.202
    Epoch   3 Batch  534/538 - Train Accuracy:  0.913, Validation Accuracy:  0.914, Loss:  0.190
    Epoch   3 Batch  535/538 - Train Accuracy:  0.926, Validation Accuracy:  0.914, Loss:  0.192
    Epoch   3 Batch  536/538 - Train Accuracy:  0.926, Validation Accuracy:  0.913, Loss:  0.224
    Epoch   4 Batch    0/538 - Train Accuracy:  0.930, Validation Accuracy:  0.917, Loss:  0.178
    Epoch   4 Batch    1/538 - Train Accuracy:  0.929, Validation Accuracy:  0.915, Loss:  0.211
    Epoch   4 Batch    2/538 - Train Accuracy:  0.913, Validation Accuracy:  0.915, Loss:  0.213
    Epoch   4 Batch    3/538 - Train Accuracy:  0.917, Validation Accuracy:  0.914, Loss:  0.190
    Epoch   4 Batch    4/538 - Train Accuracy:  0.924, Validation Accuracy:  0.915, Loss:  0.196
    Epoch   4 Batch    5/538 - Train Accuracy:  0.907, Validation Accuracy:  0.913, Loss:  0.208
    Epoch   4 Batch    6/538 - Train Accuracy:  0.900, Validation Accuracy:  0.912, Loss:  0.193
    Epoch   4 Batch    7/538 - Train Accuracy:  0.933, Validation Accuracy:  0.912, Loss:  0.206
    Epoch   4 Batch    8/538 - Train Accuracy:  0.926, Validation Accuracy:  0.913, Loss:  0.188
    Epoch   4 Batch    9/538 - Train Accuracy:  0.910, Validation Accuracy:  0.910, Loss:  0.192
    Epoch   4 Batch   10/538 - Train Accuracy:  0.906, Validation Accuracy:  0.912, Loss:  0.209
    Epoch   4 Batch   11/538 - Train Accuracy:  0.918, Validation Accuracy:  0.915, Loss:  0.190
    Epoch   4 Batch   12/538 - Train Accuracy:  0.917, Validation Accuracy:  0.918, Loss:  0.189
    Epoch   4 Batch   13/538 - Train Accuracy:  0.928, Validation Accuracy:  0.922, Loss:  0.166
    Epoch   4 Batch   14/538 - Train Accuracy:  0.923, Validation Accuracy:  0.920, Loss:  0.185
    Epoch   4 Batch   15/538 - Train Accuracy:  0.914, Validation Accuracy:  0.918, Loss:  0.200
    Epoch   4 Batch   16/538 - Train Accuracy:  0.921, Validation Accuracy:  0.920, Loss:  0.200
    Epoch   4 Batch   17/538 - Train Accuracy:  0.919, Validation Accuracy:  0.919, Loss:  0.187
    Epoch   4 Batch   18/538 - Train Accuracy:  0.924, Validation Accuracy:  0.919, Loss:  0.207
    Epoch   4 Batch   19/538 - Train Accuracy:  0.913, Validation Accuracy:  0.917, Loss:  0.210
    Epoch   4 Batch   20/538 - Train Accuracy:  0.914, Validation Accuracy:  0.917, Loss:  0.196
    Epoch   4 Batch   21/538 - Train Accuracy:  0.924, Validation Accuracy:  0.916, Loss:  0.177
    Epoch   4 Batch   22/538 - Train Accuracy:  0.900, Validation Accuracy:  0.914, Loss:  0.199
    Epoch   4 Batch   23/538 - Train Accuracy:  0.909, Validation Accuracy:  0.915, Loss:  0.225
    Epoch   4 Batch   24/538 - Train Accuracy:  0.922, Validation Accuracy:  0.916, Loss:  0.197
    Epoch   4 Batch   25/538 - Train Accuracy:  0.925, Validation Accuracy:  0.909, Loss:  0.200
    Epoch   4 Batch   26/538 - Train Accuracy:  0.901, Validation Accuracy:  0.907, Loss:  0.210
    Epoch   4 Batch   27/538 - Train Accuracy:  0.934, Validation Accuracy:  0.909, Loss:  0.165
    Epoch   4 Batch   28/538 - Train Accuracy:  0.917, Validation Accuracy:  0.909, Loss:  0.191
    Epoch   4 Batch   29/538 - Train Accuracy:  0.914, Validation Accuracy:  0.905, Loss:  0.180
    Epoch   4 Batch   30/538 - Train Accuracy:  0.916, Validation Accuracy:  0.911, Loss:  0.211
    Epoch   4 Batch   31/538 - Train Accuracy:  0.925, Validation Accuracy:  0.910, Loss:  0.175
    Epoch   4 Batch   32/538 - Train Accuracy:  0.901, Validation Accuracy:  0.908, Loss:  0.172
    Epoch   4 Batch   33/538 - Train Accuracy:  0.930, Validation Accuracy:  0.907, Loss:  0.192
    Epoch   4 Batch   34/538 - Train Accuracy:  0.913, Validation Accuracy:  0.912, Loss:  0.218
    Epoch   4 Batch   35/538 - Train Accuracy:  0.927, Validation Accuracy:  0.912, Loss:  0.171
    Epoch   4 Batch   36/538 - Train Accuracy:  0.925, Validation Accuracy:  0.910, Loss:  0.166
    Epoch   4 Batch   37/538 - Train Accuracy:  0.932, Validation Accuracy:  0.909, Loss:  0.189
    Epoch   4 Batch   38/538 - Train Accuracy:  0.894, Validation Accuracy:  0.911, Loss:  0.201
    Epoch   4 Batch   39/538 - Train Accuracy:  0.929, Validation Accuracy:  0.912, Loss:  0.194
    Epoch   4 Batch   40/538 - Train Accuracy:  0.918, Validation Accuracy:  0.917, Loss:  0.182
    Epoch   4 Batch   41/538 - Train Accuracy:  0.932, Validation Accuracy:  0.915, Loss:  0.193
    Epoch   4 Batch   42/538 - Train Accuracy:  0.922, Validation Accuracy:  0.912, Loss:  0.191
    Epoch   4 Batch   43/538 - Train Accuracy:  0.904, Validation Accuracy:  0.913, Loss:  0.215
    Epoch   4 Batch   44/538 - Train Accuracy:  0.908, Validation Accuracy:  0.917, Loss:  0.202
    Epoch   4 Batch   45/538 - Train Accuracy:  0.917, Validation Accuracy:  0.917, Loss:  0.180
    Epoch   4 Batch   46/538 - Train Accuracy:  0.928, Validation Accuracy:  0.916, Loss:  0.178
    Epoch   4 Batch   47/538 - Train Accuracy:  0.909, Validation Accuracy:  0.916, Loss:  0.215
    Epoch   4 Batch   48/538 - Train Accuracy:  0.905, Validation Accuracy:  0.915, Loss:  0.200
    Epoch   4 Batch   49/538 - Train Accuracy:  0.930, Validation Accuracy:  0.919, Loss:  0.199
    Epoch   4 Batch   50/538 - Train Accuracy:  0.919, Validation Accuracy:  0.918, Loss:  0.179
    Epoch   4 Batch   51/538 - Train Accuracy:  0.893, Validation Accuracy:  0.916, Loss:  0.196
    Epoch   4 Batch   52/538 - Train Accuracy:  0.913, Validation Accuracy:  0.912, Loss:  0.194
    Epoch   4 Batch   53/538 - Train Accuracy:  0.913, Validation Accuracy:  0.912, Loss:  0.178
    Epoch   4 Batch   54/538 - Train Accuracy:  0.923, Validation Accuracy:  0.913, Loss:  0.182
    Epoch   4 Batch   55/538 - Train Accuracy:  0.899, Validation Accuracy:  0.915, Loss:  0.198
    Epoch   4 Batch   56/538 - Train Accuracy:  0.909, Validation Accuracy:  0.914, Loss:  0.184
    Epoch   4 Batch   57/538 - Train Accuracy:  0.897, Validation Accuracy:  0.913, Loss:  0.204
    Epoch   4 Batch   58/538 - Train Accuracy:  0.900, Validation Accuracy:  0.913, Loss:  0.194
    Epoch   4 Batch   59/538 - Train Accuracy:  0.917, Validation Accuracy:  0.913, Loss:  0.188
    Epoch   4 Batch   60/538 - Train Accuracy:  0.922, Validation Accuracy:  0.917, Loss:  0.184
    Epoch   4 Batch   61/538 - Train Accuracy:  0.926, Validation Accuracy:  0.915, Loss:  0.174
    Epoch   4 Batch   62/538 - Train Accuracy:  0.922, Validation Accuracy:  0.916, Loss:  0.195
    Epoch   4 Batch   63/538 - Train Accuracy:  0.935, Validation Accuracy:  0.915, Loss:  0.174
    Epoch   4 Batch   64/538 - Train Accuracy:  0.908, Validation Accuracy:  0.916, Loss:  0.190
    Epoch   4 Batch   65/538 - Train Accuracy:  0.907, Validation Accuracy:  0.919, Loss:  0.205
    Epoch   4 Batch   66/538 - Train Accuracy:  0.920, Validation Accuracy:  0.911, Loss:  0.185
    Epoch   4 Batch   67/538 - Train Accuracy:  0.938, Validation Accuracy:  0.908, Loss:  0.185
    Epoch   4 Batch   68/538 - Train Accuracy:  0.912, Validation Accuracy:  0.907, Loss:  0.169
    Epoch   4 Batch   69/538 - Train Accuracy:  0.926, Validation Accuracy:  0.912, Loss:  0.187
    Epoch   4 Batch   70/538 - Train Accuracy:  0.917, Validation Accuracy:  0.912, Loss:  0.179
    Epoch   4 Batch   71/538 - Train Accuracy:  0.923, Validation Accuracy:  0.917, Loss:  0.204
    Epoch   4 Batch   72/538 - Train Accuracy:  0.930, Validation Accuracy:  0.917, Loss:  0.228
    Epoch   4 Batch   73/538 - Train Accuracy:  0.915, Validation Accuracy:  0.917, Loss:  0.202
    Epoch   4 Batch   74/538 - Train Accuracy:  0.925, Validation Accuracy:  0.915, Loss:  0.168
    Epoch   4 Batch   75/538 - Train Accuracy:  0.922, Validation Accuracy:  0.917, Loss:  0.178
    Epoch   4 Batch   76/538 - Train Accuracy:  0.918, Validation Accuracy:  0.922, Loss:  0.203
    Epoch   4 Batch   77/538 - Train Accuracy:  0.925, Validation Accuracy:  0.924, Loss:  0.183
    Epoch   4 Batch   78/538 - Train Accuracy:  0.922, Validation Accuracy:  0.924, Loss:  0.191
    Epoch   4 Batch   79/538 - Train Accuracy:  0.922, Validation Accuracy:  0.925, Loss:  0.165
    Epoch   4 Batch   80/538 - Train Accuracy:  0.923, Validation Accuracy:  0.925, Loss:  0.191
    Epoch   4 Batch   81/538 - Train Accuracy:  0.919, Validation Accuracy:  0.926, Loss:  0.197
    Epoch   4 Batch   82/538 - Train Accuracy:  0.915, Validation Accuracy:  0.925, Loss:  0.192
    Epoch   4 Batch   83/538 - Train Accuracy:  0.929, Validation Accuracy:  0.916, Loss:  0.187
    Epoch   4 Batch   84/538 - Train Accuracy:  0.900, Validation Accuracy:  0.917, Loss:  0.208
    Epoch   4 Batch   85/538 - Train Accuracy:  0.926, Validation Accuracy:  0.919, Loss:  0.179
    Epoch   4 Batch   86/538 - Train Accuracy:  0.933, Validation Accuracy:  0.922, Loss:  0.186
    Epoch   4 Batch   87/538 - Train Accuracy:  0.917, Validation Accuracy:  0.919, Loss:  0.202
    Epoch   4 Batch   88/538 - Train Accuracy:  0.921, Validation Accuracy:  0.920, Loss:  0.192
    Epoch   4 Batch   89/538 - Train Accuracy:  0.926, Validation Accuracy:  0.918, Loss:  0.191
    Epoch   4 Batch   90/538 - Train Accuracy:  0.915, Validation Accuracy:  0.913, Loss:  0.193
    Epoch   4 Batch   91/538 - Train Accuracy:  0.924, Validation Accuracy:  0.914, Loss:  0.180
    Epoch   4 Batch   92/538 - Train Accuracy:  0.919, Validation Accuracy:  0.925, Loss:  0.190
    Epoch   4 Batch   93/538 - Train Accuracy:  0.920, Validation Accuracy:  0.926, Loss:  0.176
    Epoch   4 Batch   94/538 - Train Accuracy:  0.914, Validation Accuracy:  0.924, Loss:  0.183
    Epoch   4 Batch   95/538 - Train Accuracy:  0.917, Validation Accuracy:  0.924, Loss:  0.163
    Epoch   4 Batch   96/538 - Train Accuracy:  0.934, Validation Accuracy:  0.922, Loss:  0.170
    Epoch   4 Batch   97/538 - Train Accuracy:  0.927, Validation Accuracy:  0.923, Loss:  0.167
    Epoch   4 Batch   98/538 - Train Accuracy:  0.933, Validation Accuracy:  0.921, Loss:  0.179
    Epoch   4 Batch   99/538 - Train Accuracy:  0.913, Validation Accuracy:  0.919, Loss:  0.181
    Epoch   4 Batch  100/538 - Train Accuracy:  0.934, Validation Accuracy:  0.914, Loss:  0.181
    Epoch   4 Batch  101/538 - Train Accuracy:  0.912, Validation Accuracy:  0.911, Loss:  0.222
    Epoch   4 Batch  102/538 - Train Accuracy:  0.895, Validation Accuracy:  0.914, Loss:  0.212
    Epoch   4 Batch  103/538 - Train Accuracy:  0.937, Validation Accuracy:  0.920, Loss:  0.177
    Epoch   4 Batch  104/538 - Train Accuracy:  0.914, Validation Accuracy:  0.921, Loss:  0.184
    Epoch   4 Batch  105/538 - Train Accuracy:  0.921, Validation Accuracy:  0.921, Loss:  0.163
    Epoch   4 Batch  106/538 - Train Accuracy:  0.907, Validation Accuracy:  0.924, Loss:  0.176
    Epoch   4 Batch  107/538 - Train Accuracy:  0.913, Validation Accuracy:  0.922, Loss:  0.203
    Epoch   4 Batch  108/538 - Train Accuracy:  0.921, Validation Accuracy:  0.920, Loss:  0.178
    Epoch   4 Batch  109/538 - Train Accuracy:  0.942, Validation Accuracy:  0.917, Loss:  0.167
    Epoch   4 Batch  110/538 - Train Accuracy:  0.914, Validation Accuracy:  0.919, Loss:  0.184
    Epoch   4 Batch  111/538 - Train Accuracy:  0.928, Validation Accuracy:  0.918, Loss:  0.156
    Epoch   4 Batch  112/538 - Train Accuracy:  0.925, Validation Accuracy:  0.921, Loss:  0.196
    Epoch   4 Batch  113/538 - Train Accuracy:  0.900, Validation Accuracy:  0.916, Loss:  0.204
    Epoch   4 Batch  114/538 - Train Accuracy:  0.918, Validation Accuracy:  0.918, Loss:  0.182
    Epoch   4 Batch  115/538 - Train Accuracy:  0.922, Validation Accuracy:  0.918, Loss:  0.176
    Epoch   4 Batch  116/538 - Train Accuracy:  0.920, Validation Accuracy:  0.922, Loss:  0.211
    Epoch   4 Batch  117/538 - Train Accuracy:  0.905, Validation Accuracy:  0.923, Loss:  0.173
    Epoch   4 Batch  118/538 - Train Accuracy:  0.930, Validation Accuracy:  0.925, Loss:  0.164
    Epoch   4 Batch  119/538 - Train Accuracy:  0.940, Validation Accuracy:  0.922, Loss:  0.154
    Epoch   4 Batch  120/538 - Train Accuracy:  0.925, Validation Accuracy:  0.924, Loss:  0.163
    Epoch   4 Batch  121/538 - Train Accuracy:  0.931, Validation Accuracy:  0.918, Loss:  0.172
    Epoch   4 Batch  122/538 - Train Accuracy:  0.915, Validation Accuracy:  0.919, Loss:  0.167
    Epoch   4 Batch  123/538 - Train Accuracy:  0.927, Validation Accuracy:  0.920, Loss:  0.182
    Epoch   4 Batch  124/538 - Train Accuracy:  0.917, Validation Accuracy:  0.916, Loss:  0.160
    Epoch   4 Batch  125/538 - Train Accuracy:  0.911, Validation Accuracy:  0.914, Loss:  0.197
    Epoch   4 Batch  126/538 - Train Accuracy:  0.897, Validation Accuracy:  0.915, Loss:  0.241
    Epoch   4 Batch  127/538 - Train Accuracy:  0.903, Validation Accuracy:  0.908, Loss:  0.216
    Epoch   4 Batch  128/538 - Train Accuracy:  0.903, Validation Accuracy:  0.911, Loss:  0.208
    Epoch   4 Batch  129/538 - Train Accuracy:  0.905, Validation Accuracy:  0.907, Loss:  0.185
    Epoch   4 Batch  130/538 - Train Accuracy:  0.927, Validation Accuracy:  0.909, Loss:  0.182
    Epoch   4 Batch  131/538 - Train Accuracy:  0.935, Validation Accuracy:  0.908, Loss:  0.191
    Epoch   4 Batch  132/538 - Train Accuracy:  0.907, Validation Accuracy:  0.911, Loss:  0.206
    Epoch   4 Batch  133/538 - Train Accuracy:  0.919, Validation Accuracy:  0.911, Loss:  0.175
    Epoch   4 Batch  134/538 - Train Accuracy:  0.912, Validation Accuracy:  0.912, Loss:  0.210
    Epoch   4 Batch  135/538 - Train Accuracy:  0.920, Validation Accuracy:  0.910, Loss:  0.212
    Epoch   4 Batch  136/538 - Train Accuracy:  0.916, Validation Accuracy:  0.906, Loss:  0.194
    Epoch   4 Batch  137/538 - Train Accuracy:  0.914, Validation Accuracy:  0.904, Loss:  0.229
    Epoch   4 Batch  138/538 - Train Accuracy:  0.906, Validation Accuracy:  0.903, Loss:  0.200
    Epoch   4 Batch  139/538 - Train Accuracy:  0.909, Validation Accuracy:  0.903, Loss:  0.218
    Epoch   4 Batch  140/538 - Train Accuracy:  0.905, Validation Accuracy:  0.905, Loss:  0.231
    Epoch   4 Batch  141/538 - Train Accuracy:  0.928, Validation Accuracy:  0.905, Loss:  0.217
    Epoch   4 Batch  142/538 - Train Accuracy:  0.908, Validation Accuracy:  0.906, Loss:  0.189
    Epoch   4 Batch  143/538 - Train Accuracy:  0.908, Validation Accuracy:  0.907, Loss:  0.192
    Epoch   4 Batch  144/538 - Train Accuracy:  0.916, Validation Accuracy:  0.907, Loss:  0.206
    Epoch   4 Batch  145/538 - Train Accuracy:  0.904, Validation Accuracy:  0.908, Loss:  0.217
    Epoch   4 Batch  146/538 - Train Accuracy:  0.926, Validation Accuracy:  0.909, Loss:  0.195
    Epoch   4 Batch  147/538 - Train Accuracy:  0.922, Validation Accuracy:  0.913, Loss:  0.200
    Epoch   4 Batch  148/538 - Train Accuracy:  0.900, Validation Accuracy:  0.920, Loss:  0.237
    Epoch   4 Batch  149/538 - Train Accuracy:  0.943, Validation Accuracy:  0.925, Loss:  0.181
    Epoch   4 Batch  150/538 - Train Accuracy:  0.934, Validation Accuracy:  0.921, Loss:  0.173
    Epoch   4 Batch  151/538 - Train Accuracy:  0.923, Validation Accuracy:  0.922, Loss:  0.191
    Epoch   4 Batch  152/538 - Train Accuracy:  0.915, Validation Accuracy:  0.918, Loss:  0.186
    Epoch   4 Batch  153/538 - Train Accuracy:  0.899, Validation Accuracy:  0.919, Loss:  0.185
    Epoch   4 Batch  154/538 - Train Accuracy:  0.923, Validation Accuracy:  0.914, Loss:  0.181
    Epoch   4 Batch  155/538 - Train Accuracy:  0.911, Validation Accuracy:  0.915, Loss:  0.188
    Epoch   4 Batch  156/538 - Train Accuracy:  0.918, Validation Accuracy:  0.918, Loss:  0.180
    Epoch   4 Batch  157/538 - Train Accuracy:  0.927, Validation Accuracy:  0.920, Loss:  0.175
    Epoch   4 Batch  158/538 - Train Accuracy:  0.926, Validation Accuracy:  0.920, Loss:  0.181
    Epoch   4 Batch  159/538 - Train Accuracy:  0.918, Validation Accuracy:  0.918, Loss:  0.192
    Epoch   4 Batch  160/538 - Train Accuracy:  0.914, Validation Accuracy:  0.921, Loss:  0.162
    Epoch   4 Batch  161/538 - Train Accuracy:  0.928, Validation Accuracy:  0.924, Loss:  0.176
    Epoch   4 Batch  162/538 - Train Accuracy:  0.925, Validation Accuracy:  0.927, Loss:  0.182
    Epoch   4 Batch  163/538 - Train Accuracy:  0.909, Validation Accuracy:  0.926, Loss:  0.207
    Epoch   4 Batch  164/538 - Train Accuracy:  0.907, Validation Accuracy:  0.923, Loss:  0.206
    Epoch   4 Batch  165/538 - Train Accuracy:  0.925, Validation Accuracy:  0.923, Loss:  0.162
    Epoch   4 Batch  166/538 - Train Accuracy:  0.926, Validation Accuracy:  0.923, Loss:  0.188
    Epoch   4 Batch  167/538 - Train Accuracy:  0.918, Validation Accuracy:  0.919, Loss:  0.188
    Epoch   4 Batch  168/538 - Train Accuracy:  0.895, Validation Accuracy:  0.920, Loss:  0.206
    Epoch   4 Batch  169/538 - Train Accuracy:  0.940, Validation Accuracy:  0.917, Loss:  0.154
    Epoch   4 Batch  170/538 - Train Accuracy:  0.919, Validation Accuracy:  0.914, Loss:  0.183
    Epoch   4 Batch  171/538 - Train Accuracy:  0.921, Validation Accuracy:  0.915, Loss:  0.174
    Epoch   4 Batch  172/538 - Train Accuracy:  0.917, Validation Accuracy:  0.919, Loss:  0.183
    Epoch   4 Batch  173/538 - Train Accuracy:  0.938, Validation Accuracy:  0.928, Loss:  0.160
    Epoch   4 Batch  174/538 - Train Accuracy:  0.917, Validation Accuracy:  0.925, Loss:  0.183
    Epoch   4 Batch  175/538 - Train Accuracy:  0.919, Validation Accuracy:  0.926, Loss:  0.180
    Epoch   4 Batch  176/538 - Train Accuracy:  0.912, Validation Accuracy:  0.926, Loss:  0.196
    Epoch   4 Batch  177/538 - Train Accuracy:  0.922, Validation Accuracy:  0.927, Loss:  0.190
    Epoch   4 Batch  178/538 - Train Accuracy:  0.914, Validation Accuracy:  0.922, Loss:  0.191
    Epoch   4 Batch  179/538 - Train Accuracy:  0.941, Validation Accuracy:  0.922, Loss:  0.178
    Epoch   4 Batch  180/538 - Train Accuracy:  0.923, Validation Accuracy:  0.923, Loss:  0.166
    Epoch   4 Batch  181/538 - Train Accuracy:  0.916, Validation Accuracy:  0.921, Loss:  0.199
    Epoch   4 Batch  182/538 - Train Accuracy:  0.934, Validation Accuracy:  0.919, Loss:  0.156
    Epoch   4 Batch  183/538 - Train Accuracy:  0.936, Validation Accuracy:  0.920, Loss:  0.156
    Epoch   4 Batch  184/538 - Train Accuracy:  0.929, Validation Accuracy:  0.917, Loss:  0.172
    Epoch   4 Batch  185/538 - Train Accuracy:  0.935, Validation Accuracy:  0.914, Loss:  0.162
    Epoch   4 Batch  186/538 - Train Accuracy:  0.919, Validation Accuracy:  0.917, Loss:  0.165
    Epoch   4 Batch  187/538 - Train Accuracy:  0.932, Validation Accuracy:  0.912, Loss:  0.170
    Epoch   4 Batch  188/538 - Train Accuracy:  0.924, Validation Accuracy:  0.912, Loss:  0.168
    Epoch   4 Batch  189/538 - Train Accuracy:  0.922, Validation Accuracy:  0.913, Loss:  0.173
    Epoch   4 Batch  190/538 - Train Accuracy:  0.917, Validation Accuracy:  0.913, Loss:  0.210
    Epoch   4 Batch  191/538 - Train Accuracy:  0.924, Validation Accuracy:  0.914, Loss:  0.166
    Epoch   4 Batch  192/538 - Train Accuracy:  0.930, Validation Accuracy:  0.914, Loss:  0.167
    Epoch   4 Batch  193/538 - Train Accuracy:  0.908, Validation Accuracy:  0.916, Loss:  0.167
    Epoch   4 Batch  194/538 - Train Accuracy:  0.910, Validation Accuracy:  0.917, Loss:  0.197
    Epoch   4 Batch  195/538 - Train Accuracy:  0.942, Validation Accuracy:  0.921, Loss:  0.176
    Epoch   4 Batch  196/538 - Train Accuracy:  0.909, Validation Accuracy:  0.920, Loss:  0.170
    Epoch   4 Batch  197/538 - Train Accuracy:  0.926, Validation Accuracy:  0.923, Loss:  0.179
    Epoch   4 Batch  198/538 - Train Accuracy:  0.938, Validation Accuracy:  0.926, Loss:  0.180
    Epoch   4 Batch  199/538 - Train Accuracy:  0.911, Validation Accuracy:  0.925, Loss:  0.192
    Epoch   4 Batch  200/538 - Train Accuracy:  0.941, Validation Accuracy:  0.924, Loss:  0.166
    Epoch   4 Batch  201/538 - Train Accuracy:  0.927, Validation Accuracy:  0.925, Loss:  0.180
    Epoch   4 Batch  202/538 - Train Accuracy:  0.931, Validation Accuracy:  0.924, Loss:  0.177
    Epoch   4 Batch  203/538 - Train Accuracy:  0.926, Validation Accuracy:  0.925, Loss:  0.207
    Epoch   4 Batch  204/538 - Train Accuracy:  0.912, Validation Accuracy:  0.921, Loss:  0.208
    Epoch   4 Batch  205/538 - Train Accuracy:  0.939, Validation Accuracy:  0.918, Loss:  0.162
    Epoch   4 Batch  206/538 - Train Accuracy:  0.925, Validation Accuracy:  0.920, Loss:  0.176
    Epoch   4 Batch  207/538 - Train Accuracy:  0.927, Validation Accuracy:  0.917, Loss:  0.174
    Epoch   4 Batch  208/538 - Train Accuracy:  0.912, Validation Accuracy:  0.917, Loss:  0.201
    Epoch   4 Batch  209/538 - Train Accuracy:  0.930, Validation Accuracy:  0.915, Loss:  0.166
    Epoch   4 Batch  210/538 - Train Accuracy:  0.918, Validation Accuracy:  0.917, Loss:  0.188
    Epoch   4 Batch  211/538 - Train Accuracy:  0.923, Validation Accuracy:  0.918, Loss:  0.203
    Epoch   4 Batch  212/538 - Train Accuracy:  0.922, Validation Accuracy:  0.914, Loss:  0.176
    Epoch   4 Batch  213/538 - Train Accuracy:  0.929, Validation Accuracy:  0.917, Loss:  0.166
    Epoch   4 Batch  214/538 - Train Accuracy:  0.923, Validation Accuracy:  0.915, Loss:  0.176
    Epoch   4 Batch  215/538 - Train Accuracy:  0.931, Validation Accuracy:  0.916, Loss:  0.172
    Epoch   4 Batch  216/538 - Train Accuracy:  0.941, Validation Accuracy:  0.917, Loss:  0.169
    Epoch   4 Batch  217/538 - Train Accuracy:  0.931, Validation Accuracy:  0.918, Loss:  0.180
    Epoch   4 Batch  218/538 - Train Accuracy:  0.928, Validation Accuracy:  0.920, Loss:  0.173
    Epoch   4 Batch  219/538 - Train Accuracy:  0.915, Validation Accuracy:  0.919, Loss:  0.189
    Epoch   4 Batch  220/538 - Train Accuracy:  0.906, Validation Accuracy:  0.918, Loss:  0.184
    Epoch   4 Batch  221/538 - Train Accuracy:  0.938, Validation Accuracy:  0.920, Loss:  0.170
    Epoch   4 Batch  222/538 - Train Accuracy:  0.912, Validation Accuracy:  0.916, Loss:  0.184
    Epoch   4 Batch  223/538 - Train Accuracy:  0.907, Validation Accuracy:  0.915, Loss:  0.195
    Epoch   4 Batch  224/538 - Train Accuracy:  0.920, Validation Accuracy:  0.916, Loss:  0.194
    Epoch   4 Batch  225/538 - Train Accuracy:  0.937, Validation Accuracy:  0.921, Loss:  0.170
    Epoch   4 Batch  226/538 - Train Accuracy:  0.909, Validation Accuracy:  0.915, Loss:  0.177
    Epoch   4 Batch  227/538 - Train Accuracy:  0.924, Validation Accuracy:  0.916, Loss:  0.173
    Epoch   4 Batch  228/538 - Train Accuracy:  0.916, Validation Accuracy:  0.919, Loss:  0.158
    Epoch   4 Batch  229/538 - Train Accuracy:  0.914, Validation Accuracy:  0.914, Loss:  0.174
    Epoch   4 Batch  230/538 - Train Accuracy:  0.926, Validation Accuracy:  0.914, Loss:  0.174
    Epoch   4 Batch  231/538 - Train Accuracy:  0.924, Validation Accuracy:  0.910, Loss:  0.174
    Epoch   4 Batch  232/538 - Train Accuracy:  0.927, Validation Accuracy:  0.910, Loss:  0.168
    Epoch   4 Batch  233/538 - Train Accuracy:  0.927, Validation Accuracy:  0.914, Loss:  0.187
    Epoch   4 Batch  234/538 - Train Accuracy:  0.925, Validation Accuracy:  0.916, Loss:  0.173
    Epoch   4 Batch  235/538 - Train Accuracy:  0.930, Validation Accuracy:  0.913, Loss:  0.162
    Epoch   4 Batch  236/538 - Train Accuracy:  0.906, Validation Accuracy:  0.919, Loss:  0.181
    Epoch   4 Batch  237/538 - Train Accuracy:  0.916, Validation Accuracy:  0.916, Loss:  0.156
    Epoch   4 Batch  238/538 - Train Accuracy:  0.934, Validation Accuracy:  0.912, Loss:  0.167
    Epoch   4 Batch  239/538 - Train Accuracy:  0.904, Validation Accuracy:  0.917, Loss:  0.192
    Epoch   4 Batch  240/538 - Train Accuracy:  0.915, Validation Accuracy:  0.917, Loss:  0.188
    Epoch   4 Batch  241/538 - Train Accuracy:  0.916, Validation Accuracy:  0.917, Loss:  0.175
    Epoch   4 Batch  242/538 - Train Accuracy:  0.927, Validation Accuracy:  0.919, Loss:  0.164
    Epoch   4 Batch  243/538 - Train Accuracy:  0.949, Validation Accuracy:  0.923, Loss:  0.167
    Epoch   4 Batch  244/538 - Train Accuracy:  0.917, Validation Accuracy:  0.922, Loss:  0.176
    Epoch   4 Batch  245/538 - Train Accuracy:  0.909, Validation Accuracy:  0.918, Loss:  0.202
    Epoch   4 Batch  246/538 - Train Accuracy:  0.935, Validation Accuracy:  0.921, Loss:  0.143
    Epoch   4 Batch  247/538 - Train Accuracy:  0.906, Validation Accuracy:  0.920, Loss:  0.174
    Epoch   4 Batch  248/538 - Train Accuracy:  0.930, Validation Accuracy:  0.920, Loss:  0.168
    Epoch   4 Batch  249/538 - Train Accuracy:  0.917, Validation Accuracy:  0.919, Loss:  0.173
    Epoch   4 Batch  250/538 - Train Accuracy:  0.934, Validation Accuracy:  0.919, Loss:  0.173
    Epoch   4 Batch  251/538 - Train Accuracy:  0.944, Validation Accuracy:  0.918, Loss:  0.168
    Epoch   4 Batch  252/538 - Train Accuracy:  0.922, Validation Accuracy:  0.918, Loss:  0.152
    Epoch   4 Batch  253/538 - Train Accuracy:  0.907, Validation Accuracy:  0.916, Loss:  0.162
    Epoch   4 Batch  254/538 - Train Accuracy:  0.910, Validation Accuracy:  0.916, Loss:  0.170
    Epoch   4 Batch  255/538 - Train Accuracy:  0.932, Validation Accuracy:  0.916, Loss:  0.163
    Epoch   4 Batch  256/538 - Train Accuracy:  0.918, Validation Accuracy:  0.914, Loss:  0.181
    Epoch   4 Batch  257/538 - Train Accuracy:  0.922, Validation Accuracy:  0.914, Loss:  0.171
    Epoch   4 Batch  258/538 - Train Accuracy:  0.926, Validation Accuracy:  0.919, Loss:  0.170
    Epoch   4 Batch  259/538 - Train Accuracy:  0.937, Validation Accuracy:  0.920, Loss:  0.166
    Epoch   4 Batch  260/538 - Train Accuracy:  0.903, Validation Accuracy:  0.920, Loss:  0.187
    Epoch   4 Batch  261/538 - Train Accuracy:  0.927, Validation Accuracy:  0.919, Loss:  0.180
    Epoch   4 Batch  262/538 - Train Accuracy:  0.930, Validation Accuracy:  0.921, Loss:  0.169
    Epoch   4 Batch  263/538 - Train Accuracy:  0.916, Validation Accuracy:  0.919, Loss:  0.174
    Epoch   4 Batch  264/538 - Train Accuracy:  0.895, Validation Accuracy:  0.919, Loss:  0.166
    Epoch   4 Batch  265/538 - Train Accuracy:  0.903, Validation Accuracy:  0.916, Loss:  0.190
    Epoch   4 Batch  266/538 - Train Accuracy:  0.906, Validation Accuracy:  0.915, Loss:  0.178
    Epoch   4 Batch  267/538 - Train Accuracy:  0.925, Validation Accuracy:  0.917, Loss:  0.176
    Epoch   4 Batch  268/538 - Train Accuracy:  0.938, Validation Accuracy:  0.917, Loss:  0.145
    Epoch   4 Batch  269/538 - Train Accuracy:  0.923, Validation Accuracy:  0.919, Loss:  0.178
    Epoch   4 Batch  270/538 - Train Accuracy:  0.928, Validation Accuracy:  0.921, Loss:  0.161
    Epoch   4 Batch  271/538 - Train Accuracy:  0.930, Validation Accuracy:  0.920, Loss:  0.167
    Epoch   4 Batch  272/538 - Train Accuracy:  0.925, Validation Accuracy:  0.917, Loss:  0.182
    Epoch   4 Batch  273/538 - Train Accuracy:  0.920, Validation Accuracy:  0.917, Loss:  0.179
    Epoch   4 Batch  274/538 - Train Accuracy:  0.891, Validation Accuracy:  0.921, Loss:  0.195
    Epoch   4 Batch  275/538 - Train Accuracy:  0.920, Validation Accuracy:  0.922, Loss:  0.194
    Epoch   4 Batch  276/538 - Train Accuracy:  0.917, Validation Accuracy:  0.924, Loss:  0.179
    Epoch   4 Batch  277/538 - Train Accuracy:  0.919, Validation Accuracy:  0.923, Loss:  0.169
    Epoch   4 Batch  278/538 - Train Accuracy:  0.935, Validation Accuracy:  0.921, Loss:  0.155
    Epoch   4 Batch  279/538 - Train Accuracy:  0.920, Validation Accuracy:  0.921, Loss:  0.168
    Epoch   4 Batch  280/538 - Train Accuracy:  0.922, Validation Accuracy:  0.921, Loss:  0.155
    Epoch   4 Batch  281/538 - Train Accuracy:  0.922, Validation Accuracy:  0.923, Loss:  0.187
    Epoch   4 Batch  282/538 - Train Accuracy:  0.933, Validation Accuracy:  0.922, Loss:  0.174
    Epoch   4 Batch  283/538 - Train Accuracy:  0.921, Validation Accuracy:  0.923, Loss:  0.160
    Epoch   4 Batch  284/538 - Train Accuracy:  0.923, Validation Accuracy:  0.928, Loss:  0.172
    Epoch   4 Batch  285/538 - Train Accuracy:  0.935, Validation Accuracy:  0.928, Loss:  0.147
    Epoch   4 Batch  286/538 - Train Accuracy:  0.914, Validation Accuracy:  0.926, Loss:  0.186
    Epoch   4 Batch  287/538 - Train Accuracy:  0.940, Validation Accuracy:  0.923, Loss:  0.150
    Epoch   4 Batch  288/538 - Train Accuracy:  0.922, Validation Accuracy:  0.923, Loss:  0.178
    Epoch   4 Batch  289/538 - Train Accuracy:  0.925, Validation Accuracy:  0.920, Loss:  0.147
    Epoch   4 Batch  290/538 - Train Accuracy:  0.931, Validation Accuracy:  0.920, Loss:  0.162
    Epoch   4 Batch  291/538 - Train Accuracy:  0.930, Validation Accuracy:  0.916, Loss:  0.165
    Epoch   4 Batch  292/538 - Train Accuracy:  0.935, Validation Accuracy:  0.921, Loss:  0.152
    Epoch   4 Batch  293/538 - Train Accuracy:  0.926, Validation Accuracy:  0.922, Loss:  0.170
    Epoch   4 Batch  294/538 - Train Accuracy:  0.937, Validation Accuracy:  0.920, Loss:  0.168
    Epoch   4 Batch  295/538 - Train Accuracy:  0.930, Validation Accuracy:  0.919, Loss:  0.170
    Epoch   4 Batch  296/538 - Train Accuracy:  0.921, Validation Accuracy:  0.920, Loss:  0.171
    Epoch   4 Batch  297/538 - Train Accuracy:  0.941, Validation Accuracy:  0.922, Loss:  0.159
    Epoch   4 Batch  298/538 - Train Accuracy:  0.925, Validation Accuracy:  0.917, Loss:  0.176
    Epoch   4 Batch  299/538 - Train Accuracy:  0.930, Validation Accuracy:  0.923, Loss:  0.192
    Epoch   4 Batch  300/538 - Train Accuracy:  0.921, Validation Accuracy:  0.921, Loss:  0.164
    Epoch   4 Batch  301/538 - Train Accuracy:  0.917, Validation Accuracy:  0.920, Loss:  0.183
    Epoch   4 Batch  302/538 - Train Accuracy:  0.935, Validation Accuracy:  0.922, Loss:  0.159
    Epoch   4 Batch  303/538 - Train Accuracy:  0.928, Validation Accuracy:  0.921, Loss:  0.162
    Epoch   4 Batch  304/538 - Train Accuracy:  0.906, Validation Accuracy:  0.923, Loss:  0.172
    Epoch   4 Batch  305/538 - Train Accuracy:  0.936, Validation Accuracy:  0.930, Loss:  0.155
    Epoch   4 Batch  306/538 - Train Accuracy:  0.919, Validation Accuracy:  0.928, Loss:  0.162
    Epoch   4 Batch  307/538 - Train Accuracy:  0.926, Validation Accuracy:  0.930, Loss:  0.167
    Epoch   4 Batch  308/538 - Train Accuracy:  0.936, Validation Accuracy:  0.927, Loss:  0.165
    Epoch   4 Batch  309/538 - Train Accuracy:  0.926, Validation Accuracy:  0.927, Loss:  0.144
    Epoch   4 Batch  310/538 - Train Accuracy:  0.942, Validation Accuracy:  0.924, Loss:  0.162
    Epoch   4 Batch  311/538 - Train Accuracy:  0.914, Validation Accuracy:  0.926, Loss:  0.171
    Epoch   4 Batch  312/538 - Train Accuracy:  0.927, Validation Accuracy:  0.925, Loss:  0.161
    Epoch   4 Batch  313/538 - Train Accuracy:  0.922, Validation Accuracy:  0.929, Loss:  0.185
    Epoch   4 Batch  314/538 - Train Accuracy:  0.926, Validation Accuracy:  0.931, Loss:  0.169
    Epoch   4 Batch  315/538 - Train Accuracy:  0.920, Validation Accuracy:  0.928, Loss:  0.162
    Epoch   4 Batch  316/538 - Train Accuracy:  0.923, Validation Accuracy:  0.923, Loss:  0.155
    Epoch   4 Batch  317/538 - Train Accuracy:  0.923, Validation Accuracy:  0.922, Loss:  0.170
    Epoch   4 Batch  318/538 - Train Accuracy:  0.904, Validation Accuracy:  0.919, Loss:  0.167
    Epoch   4 Batch  319/538 - Train Accuracy:  0.927, Validation Accuracy:  0.919, Loss:  0.181
    Epoch   4 Batch  320/538 - Train Accuracy:  0.916, Validation Accuracy:  0.917, Loss:  0.173
    Epoch   4 Batch  321/538 - Train Accuracy:  0.923, Validation Accuracy:  0.914, Loss:  0.156
    Epoch   4 Batch  322/538 - Train Accuracy:  0.929, Validation Accuracy:  0.915, Loss:  0.186
    Epoch   4 Batch  323/538 - Train Accuracy:  0.920, Validation Accuracy:  0.922, Loss:  0.153
    Epoch   4 Batch  324/538 - Train Accuracy:  0.928, Validation Accuracy:  0.929, Loss:  0.174
    Epoch   4 Batch  325/538 - Train Accuracy:  0.933, Validation Accuracy:  0.925, Loss:  0.159
    Epoch   4 Batch  326/538 - Train Accuracy:  0.927, Validation Accuracy:  0.924, Loss:  0.146
    Epoch   4 Batch  327/538 - Train Accuracy:  0.922, Validation Accuracy:  0.924, Loss:  0.170
    Epoch   4 Batch  328/538 - Train Accuracy:  0.935, Validation Accuracy:  0.928, Loss:  0.155
    Epoch   4 Batch  329/538 - Train Accuracy:  0.938, Validation Accuracy:  0.923, Loss:  0.160
    Epoch   4 Batch  330/538 - Train Accuracy:  0.931, Validation Accuracy:  0.927, Loss:  0.151
    Epoch   4 Batch  331/538 - Train Accuracy:  0.931, Validation Accuracy:  0.923, Loss:  0.171
    Epoch   4 Batch  332/538 - Train Accuracy:  0.917, Validation Accuracy:  0.921, Loss:  0.165
    Epoch   4 Batch  333/538 - Train Accuracy:  0.924, Validation Accuracy:  0.914, Loss:  0.171
    Epoch   4 Batch  334/538 - Train Accuracy:  0.931, Validation Accuracy:  0.915, Loss:  0.154
    Epoch   4 Batch  335/538 - Train Accuracy:  0.935, Validation Accuracy:  0.913, Loss:  0.167
    Epoch   4 Batch  336/538 - Train Accuracy:  0.927, Validation Accuracy:  0.916, Loss:  0.163
    Epoch   4 Batch  337/538 - Train Accuracy:  0.918, Validation Accuracy:  0.915, Loss:  0.175
    Epoch   4 Batch  338/538 - Train Accuracy:  0.924, Validation Accuracy:  0.915, Loss:  0.175
    Epoch   4 Batch  339/538 - Train Accuracy:  0.914, Validation Accuracy:  0.913, Loss:  0.163
    Epoch   4 Batch  340/538 - Train Accuracy:  0.909, Validation Accuracy:  0.914, Loss:  0.169
    Epoch   4 Batch  341/538 - Train Accuracy:  0.920, Validation Accuracy:  0.921, Loss:  0.169
    Epoch   4 Batch  342/538 - Train Accuracy:  0.931, Validation Accuracy:  0.925, Loss:  0.168
    Epoch   4 Batch  343/538 - Train Accuracy:  0.945, Validation Accuracy:  0.925, Loss:  0.167
    Epoch   4 Batch  344/538 - Train Accuracy:  0.938, Validation Accuracy:  0.925, Loss:  0.151
    Epoch   4 Batch  345/538 - Train Accuracy:  0.922, Validation Accuracy:  0.927, Loss:  0.165
    Epoch   4 Batch  346/538 - Train Accuracy:  0.909, Validation Accuracy:  0.931, Loss:  0.182
    Epoch   4 Batch  347/538 - Train Accuracy:  0.930, Validation Accuracy:  0.928, Loss:  0.159
    Epoch   4 Batch  348/538 - Train Accuracy:  0.926, Validation Accuracy:  0.925, Loss:  0.156
    Epoch   4 Batch  349/538 - Train Accuracy:  0.945, Validation Accuracy:  0.919, Loss:  0.144
    Epoch   4 Batch  350/538 - Train Accuracy:  0.925, Validation Accuracy:  0.915, Loss:  0.181
    Epoch   4 Batch  351/538 - Train Accuracy:  0.918, Validation Accuracy:  0.915, Loss:  0.172
    Epoch   4 Batch  352/538 - Train Accuracy:  0.905, Validation Accuracy:  0.918, Loss:  0.187
    Epoch   4 Batch  353/538 - Train Accuracy:  0.910, Validation Accuracy:  0.915, Loss:  0.164
    Epoch   4 Batch  354/538 - Train Accuracy:  0.918, Validation Accuracy:  0.920, Loss:  0.177
    Epoch   4 Batch  355/538 - Train Accuracy:  0.923, Validation Accuracy:  0.921, Loss:  0.169
    Epoch   4 Batch  356/538 - Train Accuracy:  0.926, Validation Accuracy:  0.921, Loss:  0.156
    Epoch   4 Batch  357/538 - Train Accuracy:  0.930, Validation Accuracy:  0.925, Loss:  0.168
    Epoch   4 Batch  358/538 - Train Accuracy:  0.929, Validation Accuracy:  0.922, Loss:  0.147
    Epoch   4 Batch  359/538 - Train Accuracy:  0.911, Validation Accuracy:  0.924, Loss:  0.157
    Epoch   4 Batch  360/538 - Train Accuracy:  0.923, Validation Accuracy:  0.926, Loss:  0.166
    Epoch   4 Batch  361/538 - Train Accuracy:  0.934, Validation Accuracy:  0.927, Loss:  0.154
    Epoch   4 Batch  362/538 - Train Accuracy:  0.946, Validation Accuracy:  0.925, Loss:  0.139
    Epoch   4 Batch  363/538 - Train Accuracy:  0.926, Validation Accuracy:  0.923, Loss:  0.150
    Epoch   4 Batch  364/538 - Train Accuracy:  0.906, Validation Accuracy:  0.923, Loss:  0.185
    Epoch   4 Batch  365/538 - Train Accuracy:  0.913, Validation Accuracy:  0.929, Loss:  0.162
    Epoch   4 Batch  366/538 - Train Accuracy:  0.934, Validation Accuracy:  0.933, Loss:  0.161
    Epoch   4 Batch  367/538 - Train Accuracy:  0.929, Validation Accuracy:  0.929, Loss:  0.139
    Epoch   4 Batch  368/538 - Train Accuracy:  0.934, Validation Accuracy:  0.929, Loss:  0.141
    Epoch   4 Batch  369/538 - Train Accuracy:  0.926, Validation Accuracy:  0.927, Loss:  0.153
    Epoch   4 Batch  370/538 - Train Accuracy:  0.915, Validation Accuracy:  0.926, Loss:  0.166
    Epoch   4 Batch  371/538 - Train Accuracy:  0.938, Validation Accuracy:  0.928, Loss:  0.173
    Epoch   4 Batch  372/538 - Train Accuracy:  0.943, Validation Accuracy:  0.934, Loss:  0.153
    Epoch   4 Batch  373/538 - Train Accuracy:  0.921, Validation Accuracy:  0.933, Loss:  0.153
    Epoch   4 Batch  374/538 - Train Accuracy:  0.937, Validation Accuracy:  0.934, Loss:  0.167
    Epoch   4 Batch  375/538 - Train Accuracy:  0.929, Validation Accuracy:  0.929, Loss:  0.144
    Epoch   4 Batch  376/538 - Train Accuracy:  0.910, Validation Accuracy:  0.931, Loss:  0.152
    Epoch   4 Batch  377/538 - Train Accuracy:  0.941, Validation Accuracy:  0.933, Loss:  0.155
    Epoch   4 Batch  378/538 - Train Accuracy:  0.930, Validation Accuracy:  0.927, Loss:  0.146
    Epoch   4 Batch  379/538 - Train Accuracy:  0.932, Validation Accuracy:  0.926, Loss:  0.164
    Epoch   4 Batch  380/538 - Train Accuracy:  0.937, Validation Accuracy:  0.926, Loss:  0.148
    Epoch   4 Batch  381/538 - Train Accuracy:  0.932, Validation Accuracy:  0.928, Loss:  0.141
    Epoch   4 Batch  382/538 - Train Accuracy:  0.917, Validation Accuracy:  0.930, Loss:  0.153
    Epoch   4 Batch  383/538 - Train Accuracy:  0.927, Validation Accuracy:  0.929, Loss:  0.147
    Epoch   4 Batch  384/538 - Train Accuracy:  0.908, Validation Accuracy:  0.930, Loss:  0.164
    Epoch   4 Batch  385/538 - Train Accuracy:  0.928, Validation Accuracy:  0.930, Loss:  0.156
    Epoch   4 Batch  386/538 - Train Accuracy:  0.919, Validation Accuracy:  0.930, Loss:  0.172
    Epoch   4 Batch  387/538 - Train Accuracy:  0.936, Validation Accuracy:  0.926, Loss:  0.160
    Epoch   4 Batch  388/538 - Train Accuracy:  0.928, Validation Accuracy:  0.928, Loss:  0.154
    Epoch   4 Batch  389/538 - Train Accuracy:  0.910, Validation Accuracy:  0.929, Loss:  0.180
    Epoch   4 Batch  390/538 - Train Accuracy:  0.935, Validation Accuracy:  0.928, Loss:  0.144
    Epoch   4 Batch  391/538 - Train Accuracy:  0.923, Validation Accuracy:  0.926, Loss:  0.162
    Epoch   4 Batch  392/538 - Train Accuracy:  0.929, Validation Accuracy:  0.928, Loss:  0.153
    Epoch   4 Batch  393/538 - Train Accuracy:  0.937, Validation Accuracy:  0.928, Loss:  0.141
    Epoch   4 Batch  394/538 - Train Accuracy:  0.916, Validation Accuracy:  0.927, Loss:  0.168
    Epoch   4 Batch  395/538 - Train Accuracy:  0.923, Validation Accuracy:  0.932, Loss:  0.171
    Epoch   4 Batch  396/538 - Train Accuracy:  0.928, Validation Accuracy:  0.932, Loss:  0.158
    Epoch   4 Batch  397/538 - Train Accuracy:  0.928, Validation Accuracy:  0.931, Loss:  0.171
    Epoch   4 Batch  398/538 - Train Accuracy:  0.929, Validation Accuracy:  0.930, Loss:  0.156
    Epoch   4 Batch  399/538 - Train Accuracy:  0.917, Validation Accuracy:  0.936, Loss:  0.180
    Epoch   4 Batch  400/538 - Train Accuracy:  0.933, Validation Accuracy:  0.936, Loss:  0.162
    Epoch   4 Batch  401/538 - Train Accuracy:  0.939, Validation Accuracy:  0.936, Loss:  0.161
    Epoch   4 Batch  402/538 - Train Accuracy:  0.915, Validation Accuracy:  0.933, Loss:  0.154
    Epoch   4 Batch  403/538 - Train Accuracy:  0.942, Validation Accuracy:  0.932, Loss:  0.144
    Epoch   4 Batch  404/538 - Train Accuracy:  0.931, Validation Accuracy:  0.929, Loss:  0.159
    Epoch   4 Batch  405/538 - Train Accuracy:  0.924, Validation Accuracy:  0.925, Loss:  0.160
    Epoch   4 Batch  406/538 - Train Accuracy:  0.918, Validation Accuracy:  0.927, Loss:  0.158
    Epoch   4 Batch  407/538 - Train Accuracy:  0.945, Validation Accuracy:  0.927, Loss:  0.155
    Epoch   4 Batch  408/538 - Train Accuracy:  0.913, Validation Accuracy:  0.927, Loss:  0.187
    Epoch   4 Batch  409/538 - Train Accuracy:  0.919, Validation Accuracy:  0.924, Loss:  0.162
    Epoch   4 Batch  410/538 - Train Accuracy:  0.940, Validation Accuracy:  0.923, Loss:  0.152
    Epoch   4 Batch  411/538 - Train Accuracy:  0.940, Validation Accuracy:  0.920, Loss:  0.155
    Epoch   4 Batch  412/538 - Train Accuracy:  0.925, Validation Accuracy:  0.919, Loss:  0.141
    Epoch   4 Batch  413/538 - Train Accuracy:  0.932, Validation Accuracy:  0.917, Loss:  0.154
    Epoch   4 Batch  414/538 - Train Accuracy:  0.898, Validation Accuracy:  0.917, Loss:  0.172
    Epoch   4 Batch  415/538 - Train Accuracy:  0.916, Validation Accuracy:  0.916, Loss:  0.159
    Epoch   4 Batch  416/538 - Train Accuracy:  0.935, Validation Accuracy:  0.917, Loss:  0.156
    Epoch   4 Batch  417/538 - Train Accuracy:  0.929, Validation Accuracy:  0.918, Loss:  0.156
    Epoch   4 Batch  418/538 - Train Accuracy:  0.928, Validation Accuracy:  0.920, Loss:  0.174
    Epoch   4 Batch  419/538 - Train Accuracy:  0.930, Validation Accuracy:  0.922, Loss:  0.143
    Epoch   4 Batch  420/538 - Train Accuracy:  0.936, Validation Accuracy:  0.922, Loss:  0.148
    Epoch   4 Batch  421/538 - Train Accuracy:  0.931, Validation Accuracy:  0.929, Loss:  0.153
    Epoch   4 Batch  422/538 - Train Accuracy:  0.924, Validation Accuracy:  0.930, Loss:  0.161
    Epoch   4 Batch  423/538 - Train Accuracy:  0.924, Validation Accuracy:  0.934, Loss:  0.170
    Epoch   4 Batch  424/538 - Train Accuracy:  0.912, Validation Accuracy:  0.934, Loss:  0.167
    Epoch   4 Batch  425/538 - Train Accuracy:  0.913, Validation Accuracy:  0.937, Loss:  0.174
    Epoch   4 Batch  426/538 - Train Accuracy:  0.931, Validation Accuracy:  0.938, Loss:  0.151
    Epoch   4 Batch  427/538 - Train Accuracy:  0.911, Validation Accuracy:  0.936, Loss:  0.164
    Epoch   4 Batch  428/538 - Train Accuracy:  0.934, Validation Accuracy:  0.934, Loss:  0.137
    Epoch   4 Batch  429/538 - Train Accuracy:  0.923, Validation Accuracy:  0.933, Loss:  0.157
    Epoch   4 Batch  430/538 - Train Accuracy:  0.921, Validation Accuracy:  0.927, Loss:  0.160
    Epoch   4 Batch  431/538 - Train Accuracy:  0.925, Validation Accuracy:  0.924, Loss:  0.150
    Epoch   4 Batch  432/538 - Train Accuracy:  0.931, Validation Accuracy:  0.922, Loss:  0.157
    Epoch   4 Batch  433/538 - Train Accuracy:  0.905, Validation Accuracy:  0.921, Loss:  0.179
    Epoch   4 Batch  434/538 - Train Accuracy:  0.910, Validation Accuracy:  0.917, Loss:  0.158
    Epoch   4 Batch  435/538 - Train Accuracy:  0.930, Validation Accuracy:  0.918, Loss:  0.158
    Epoch   4 Batch  436/538 - Train Accuracy:  0.922, Validation Accuracy:  0.917, Loss:  0.159
    Epoch   4 Batch  437/538 - Train Accuracy:  0.921, Validation Accuracy:  0.919, Loss:  0.165
    Epoch   4 Batch  438/538 - Train Accuracy:  0.931, Validation Accuracy:  0.921, Loss:  0.154
    Epoch   4 Batch  439/538 - Train Accuracy:  0.944, Validation Accuracy:  0.919, Loss:  0.138
    Epoch   4 Batch  440/538 - Train Accuracy:  0.922, Validation Accuracy:  0.911, Loss:  0.170
    Epoch   4 Batch  441/538 - Train Accuracy:  0.919, Validation Accuracy:  0.917, Loss:  0.176
    Epoch   4 Batch  442/538 - Train Accuracy:  0.930, Validation Accuracy:  0.922, Loss:  0.136
    Epoch   4 Batch  443/538 - Train Accuracy:  0.921, Validation Accuracy:  0.922, Loss:  0.163
    Epoch   4 Batch  444/538 - Train Accuracy:  0.939, Validation Accuracy:  0.917, Loss:  0.143
    Epoch   4 Batch  445/538 - Train Accuracy:  0.937, Validation Accuracy:  0.918, Loss:  0.143
    Epoch   4 Batch  446/538 - Train Accuracy:  0.949, Validation Accuracy:  0.922, Loss:  0.151
    Epoch   4 Batch  447/538 - Train Accuracy:  0.902, Validation Accuracy:  0.921, Loss:  0.160
    Epoch   4 Batch  448/538 - Train Accuracy:  0.922, Validation Accuracy:  0.927, Loss:  0.132
    Epoch   4 Batch  449/538 - Train Accuracy:  0.939, Validation Accuracy:  0.930, Loss:  0.160
    Epoch   4 Batch  450/538 - Train Accuracy:  0.918, Validation Accuracy:  0.929, Loss:  0.189
    Epoch   4 Batch  451/538 - Train Accuracy:  0.912, Validation Accuracy:  0.931, Loss:  0.150
    Epoch   4 Batch  452/538 - Train Accuracy:  0.922, Validation Accuracy:  0.930, Loss:  0.135
    Epoch   4 Batch  453/538 - Train Accuracy:  0.932, Validation Accuracy:  0.925, Loss:  0.156
    Epoch   4 Batch  454/538 - Train Accuracy:  0.922, Validation Accuracy:  0.924, Loss:  0.148
    Epoch   4 Batch  455/538 - Train Accuracy:  0.935, Validation Accuracy:  0.927, Loss:  0.152
    Epoch   4 Batch  456/538 - Train Accuracy:  0.938, Validation Accuracy:  0.927, Loss:  0.185
    Epoch   4 Batch  457/538 - Train Accuracy:  0.926, Validation Accuracy:  0.928, Loss:  0.163
    Epoch   4 Batch  458/538 - Train Accuracy:  0.924, Validation Accuracy:  0.931, Loss:  0.145
    Epoch   4 Batch  459/538 - Train Accuracy:  0.925, Validation Accuracy:  0.926, Loss:  0.154
    Epoch   4 Batch  460/538 - Train Accuracy:  0.917, Validation Accuracy:  0.926, Loss:  0.165
    Epoch   4 Batch  461/538 - Train Accuracy:  0.943, Validation Accuracy:  0.927, Loss:  0.156
    Epoch   4 Batch  462/538 - Train Accuracy:  0.936, Validation Accuracy:  0.923, Loss:  0.149
    Epoch   4 Batch  463/538 - Train Accuracy:  0.911, Validation Accuracy:  0.920, Loss:  0.167
    Epoch   4 Batch  464/538 - Train Accuracy:  0.929, Validation Accuracy:  0.918, Loss:  0.152
    Epoch   4 Batch  465/538 - Train Accuracy:  0.920, Validation Accuracy:  0.909, Loss:  0.138
    Epoch   4 Batch  466/538 - Train Accuracy:  0.920, Validation Accuracy:  0.911, Loss:  0.164
    Epoch   4 Batch  467/538 - Train Accuracy:  0.944, Validation Accuracy:  0.912, Loss:  0.159
    Epoch   4 Batch  468/538 - Train Accuracy:  0.933, Validation Accuracy:  0.913, Loss:  0.166
    Epoch   4 Batch  469/538 - Train Accuracy:  0.923, Validation Accuracy:  0.917, Loss:  0.161
    Epoch   4 Batch  470/538 - Train Accuracy:  0.941, Validation Accuracy:  0.918, Loss:  0.157
    Epoch   4 Batch  471/538 - Train Accuracy:  0.932, Validation Accuracy:  0.912, Loss:  0.143
    Epoch   4 Batch  472/538 - Train Accuracy:  0.959, Validation Accuracy:  0.915, Loss:  0.139
    Epoch   4 Batch  473/538 - Train Accuracy:  0.910, Validation Accuracy:  0.915, Loss:  0.160
    Epoch   4 Batch  474/538 - Train Accuracy:  0.941, Validation Accuracy:  0.911, Loss:  0.145
    Epoch   4 Batch  475/538 - Train Accuracy:  0.927, Validation Accuracy:  0.916, Loss:  0.152
    Epoch   4 Batch  476/538 - Train Accuracy:  0.929, Validation Accuracy:  0.912, Loss:  0.149
    Epoch   4 Batch  477/538 - Train Accuracy:  0.939, Validation Accuracy:  0.916, Loss:  0.163
    Epoch   4 Batch  478/538 - Train Accuracy:  0.927, Validation Accuracy:  0.916, Loss:  0.148
    Epoch   4 Batch  479/538 - Train Accuracy:  0.922, Validation Accuracy:  0.920, Loss:  0.149
    Epoch   4 Batch  480/538 - Train Accuracy:  0.934, Validation Accuracy:  0.920, Loss:  0.147
    Epoch   4 Batch  481/538 - Train Accuracy:  0.941, Validation Accuracy:  0.923, Loss:  0.152
    Epoch   4 Batch  482/538 - Train Accuracy:  0.918, Validation Accuracy:  0.923, Loss:  0.150
    Epoch   4 Batch  483/538 - Train Accuracy:  0.900, Validation Accuracy:  0.926, Loss:  0.166
    Epoch   4 Batch  484/538 - Train Accuracy:  0.930, Validation Accuracy:  0.927, Loss:  0.181
    Epoch   4 Batch  485/538 - Train Accuracy:  0.926, Validation Accuracy:  0.925, Loss:  0.155
    Epoch   4 Batch  486/538 - Train Accuracy:  0.934, Validation Accuracy:  0.924, Loss:  0.128
    Epoch   4 Batch  487/538 - Train Accuracy:  0.941, Validation Accuracy:  0.925, Loss:  0.125
    Epoch   4 Batch  488/538 - Train Accuracy:  0.934, Validation Accuracy:  0.922, Loss:  0.138
    Epoch   4 Batch  489/538 - Train Accuracy:  0.921, Validation Accuracy:  0.922, Loss:  0.148
    Epoch   4 Batch  490/538 - Train Accuracy:  0.919, Validation Accuracy:  0.917, Loss:  0.143
    Epoch   4 Batch  491/538 - Train Accuracy:  0.907, Validation Accuracy:  0.913, Loss:  0.152
    Epoch   4 Batch  492/538 - Train Accuracy:  0.925, Validation Accuracy:  0.915, Loss:  0.150
    Epoch   4 Batch  493/538 - Train Accuracy:  0.930, Validation Accuracy:  0.916, Loss:  0.141
    Epoch   4 Batch  494/538 - Train Accuracy:  0.937, Validation Accuracy:  0.918, Loss:  0.156
    Epoch   4 Batch  495/538 - Train Accuracy:  0.939, Validation Accuracy:  0.922, Loss:  0.156
    Epoch   4 Batch  496/538 - Train Accuracy:  0.930, Validation Accuracy:  0.923, Loss:  0.134
    Epoch   4 Batch  497/538 - Train Accuracy:  0.932, Validation Accuracy:  0.923, Loss:  0.136
    Epoch   4 Batch  498/538 - Train Accuracy:  0.941, Validation Accuracy:  0.922, Loss:  0.144
    Epoch   4 Batch  499/538 - Train Accuracy:  0.923, Validation Accuracy:  0.926, Loss:  0.165
    Epoch   4 Batch  500/538 - Train Accuracy:  0.937, Validation Accuracy:  0.923, Loss:  0.132
    Epoch   4 Batch  501/538 - Train Accuracy:  0.940, Validation Accuracy:  0.924, Loss:  0.166
    Epoch   4 Batch  502/538 - Train Accuracy:  0.929, Validation Accuracy:  0.917, Loss:  0.137
    Epoch   4 Batch  503/538 - Train Accuracy:  0.948, Validation Accuracy:  0.917, Loss:  0.147
    Epoch   4 Batch  504/538 - Train Accuracy:  0.946, Validation Accuracy:  0.920, Loss:  0.138
    Epoch   4 Batch  505/538 - Train Accuracy:  0.942, Validation Accuracy:  0.920, Loss:  0.126
    Epoch   4 Batch  506/538 - Train Accuracy:  0.934, Validation Accuracy:  0.919, Loss:  0.142
    Epoch   4 Batch  507/538 - Train Accuracy:  0.924, Validation Accuracy:  0.919, Loss:  0.165
    Epoch   4 Batch  508/538 - Train Accuracy:  0.922, Validation Accuracy:  0.918, Loss:  0.148
    Epoch   4 Batch  509/538 - Train Accuracy:  0.924, Validation Accuracy:  0.920, Loss:  0.149
    Epoch   4 Batch  510/538 - Train Accuracy:  0.934, Validation Accuracy:  0.915, Loss:  0.147
    Epoch   4 Batch  511/538 - Train Accuracy:  0.920, Validation Accuracy:  0.916, Loss:  0.150
    Epoch   4 Batch  512/538 - Train Accuracy:  0.940, Validation Accuracy:  0.915, Loss:  0.147
    Epoch   4 Batch  513/538 - Train Accuracy:  0.926, Validation Accuracy:  0.918, Loss:  0.158
    Epoch   4 Batch  514/538 - Train Accuracy:  0.939, Validation Accuracy:  0.918, Loss:  0.155
    Epoch   4 Batch  515/538 - Train Accuracy:  0.934, Validation Accuracy:  0.916, Loss:  0.157
    Epoch   4 Batch  516/538 - Train Accuracy:  0.918, Validation Accuracy:  0.917, Loss:  0.152
    Epoch   4 Batch  517/538 - Train Accuracy:  0.924, Validation Accuracy:  0.915, Loss:  0.166
    Epoch   4 Batch  518/538 - Train Accuracy:  0.930, Validation Accuracy:  0.913, Loss:  0.161
    Epoch   4 Batch  519/538 - Train Accuracy:  0.922, Validation Accuracy:  0.915, Loss:  0.169
    Epoch   4 Batch  520/538 - Train Accuracy:  0.921, Validation Accuracy:  0.911, Loss:  0.163
    Epoch   4 Batch  521/538 - Train Accuracy:  0.929, Validation Accuracy:  0.912, Loss:  0.155
    Epoch   4 Batch  522/538 - Train Accuracy:  0.939, Validation Accuracy:  0.915, Loss:  0.130
    Epoch   4 Batch  523/538 - Train Accuracy:  0.931, Validation Accuracy:  0.915, Loss:  0.140
    Epoch   4 Batch  524/538 - Train Accuracy:  0.929, Validation Accuracy:  0.920, Loss:  0.155
    Epoch   4 Batch  525/538 - Train Accuracy:  0.933, Validation Accuracy:  0.920, Loss:  0.144
    Epoch   4 Batch  526/538 - Train Accuracy:  0.936, Validation Accuracy:  0.916, Loss:  0.159
    Epoch   4 Batch  527/538 - Train Accuracy:  0.929, Validation Accuracy:  0.918, Loss:  0.142
    Epoch   4 Batch  528/538 - Train Accuracy:  0.923, Validation Accuracy:  0.917, Loss:  0.162
    Epoch   4 Batch  529/538 - Train Accuracy:  0.901, Validation Accuracy:  0.928, Loss:  0.167
    Epoch   4 Batch  530/538 - Train Accuracy:  0.926, Validation Accuracy:  0.930, Loss:  0.170
    Epoch   4 Batch  531/538 - Train Accuracy:  0.921, Validation Accuracy:  0.929, Loss:  0.152
    Epoch   4 Batch  532/538 - Train Accuracy:  0.941, Validation Accuracy:  0.932, Loss:  0.141
    Epoch   4 Batch  533/538 - Train Accuracy:  0.932, Validation Accuracy:  0.931, Loss:  0.148
    Epoch   4 Batch  534/538 - Train Accuracy:  0.931, Validation Accuracy:  0.929, Loss:  0.133
    Epoch   4 Batch  535/538 - Train Accuracy:  0.945, Validation Accuracy:  0.926, Loss:  0.135
    Epoch   4 Batch  536/538 - Train Accuracy:  0.933, Validation Accuracy:  0.927, Loss:  0.163
    Epoch   5 Batch    0/538 - Train Accuracy:  0.942, Validation Accuracy:  0.931, Loss:  0.126
    Epoch   5 Batch    1/538 - Train Accuracy:  0.942, Validation Accuracy:  0.923, Loss:  0.146
    Epoch   5 Batch    2/538 - Train Accuracy:  0.944, Validation Accuracy:  0.922, Loss:  0.163
    Epoch   5 Batch    3/538 - Train Accuracy:  0.933, Validation Accuracy:  0.920, Loss:  0.142
    Epoch   5 Batch    4/538 - Train Accuracy:  0.919, Validation Accuracy:  0.919, Loss:  0.142
    Epoch   5 Batch    5/538 - Train Accuracy:  0.916, Validation Accuracy:  0.916, Loss:  0.147
    Epoch   5 Batch    6/538 - Train Accuracy:  0.927, Validation Accuracy:  0.916, Loss:  0.135
    Epoch   5 Batch    7/538 - Train Accuracy:  0.935, Validation Accuracy:  0.916, Loss:  0.150
    Epoch   5 Batch    8/538 - Train Accuracy:  0.929, Validation Accuracy:  0.924, Loss:  0.137
    Epoch   5 Batch    9/538 - Train Accuracy:  0.933, Validation Accuracy:  0.927, Loss:  0.134
    Epoch   5 Batch   10/538 - Train Accuracy:  0.909, Validation Accuracy:  0.928, Loss:  0.155
    Epoch   5 Batch   11/538 - Train Accuracy:  0.942, Validation Accuracy:  0.925, Loss:  0.137
    Epoch   5 Batch   12/538 - Train Accuracy:  0.935, Validation Accuracy:  0.929, Loss:  0.138
    Epoch   5 Batch   13/538 - Train Accuracy:  0.942, Validation Accuracy:  0.928, Loss:  0.130
    Epoch   5 Batch   14/538 - Train Accuracy:  0.930, Validation Accuracy:  0.927, Loss:  0.131
    Epoch   5 Batch   15/538 - Train Accuracy:  0.926, Validation Accuracy:  0.928, Loss:  0.154
    Epoch   5 Batch   16/538 - Train Accuracy:  0.938, Validation Accuracy:  0.926, Loss:  0.128
    Epoch   5 Batch   17/538 - Train Accuracy:  0.931, Validation Accuracy:  0.927, Loss:  0.138
    Epoch   5 Batch   18/538 - Train Accuracy:  0.939, Validation Accuracy:  0.926, Loss:  0.150
    Epoch   5 Batch   19/538 - Train Accuracy:  0.926, Validation Accuracy:  0.923, Loss:  0.154
    Epoch   5 Batch   20/538 - Train Accuracy:  0.924, Validation Accuracy:  0.920, Loss:  0.149
    Epoch   5 Batch   21/538 - Train Accuracy:  0.942, Validation Accuracy:  0.924, Loss:  0.124
    Epoch   5 Batch   22/538 - Train Accuracy:  0.910, Validation Accuracy:  0.919, Loss:  0.143
    Epoch   5 Batch   23/538 - Train Accuracy:  0.912, Validation Accuracy:  0.918, Loss:  0.165
    Epoch   5 Batch   24/538 - Train Accuracy:  0.934, Validation Accuracy:  0.919, Loss:  0.145
    Epoch   5 Batch   25/538 - Train Accuracy:  0.938, Validation Accuracy:  0.925, Loss:  0.142
    Epoch   5 Batch   26/538 - Train Accuracy:  0.927, Validation Accuracy:  0.933, Loss:  0.161
    Epoch   5 Batch   27/538 - Train Accuracy:  0.943, Validation Accuracy:  0.932, Loss:  0.122
    Epoch   5 Batch   28/538 - Train Accuracy:  0.938, Validation Accuracy:  0.928, Loss:  0.140
    Epoch   5 Batch   29/538 - Train Accuracy:  0.924, Validation Accuracy:  0.926, Loss:  0.133
    Epoch   5 Batch   30/538 - Train Accuracy:  0.928, Validation Accuracy:  0.925, Loss:  0.154
    Epoch   5 Batch   31/538 - Train Accuracy:  0.946, Validation Accuracy:  0.926, Loss:  0.124
    Epoch   5 Batch   32/538 - Train Accuracy:  0.925, Validation Accuracy:  0.927, Loss:  0.123
    Epoch   5 Batch   33/538 - Train Accuracy:  0.941, Validation Accuracy:  0.927, Loss:  0.137
    Epoch   5 Batch   34/538 - Train Accuracy:  0.921, Validation Accuracy:  0.931, Loss:  0.165
    Epoch   5 Batch   35/538 - Train Accuracy:  0.932, Validation Accuracy:  0.931, Loss:  0.132
    Epoch   5 Batch   36/538 - Train Accuracy:  0.939, Validation Accuracy:  0.929, Loss:  0.119
    Epoch   5 Batch   37/538 - Train Accuracy:  0.938, Validation Accuracy:  0.927, Loss:  0.144
    Epoch   5 Batch   38/538 - Train Accuracy:  0.912, Validation Accuracy:  0.924, Loss:  0.161
    Epoch   5 Batch   39/538 - Train Accuracy:  0.937, Validation Accuracy:  0.923, Loss:  0.140
    Epoch   5 Batch   40/538 - Train Accuracy:  0.931, Validation Accuracy:  0.923, Loss:  0.138
    Epoch   5 Batch   41/538 - Train Accuracy:  0.941, Validation Accuracy:  0.923, Loss:  0.145
    Epoch   5 Batch   42/538 - Train Accuracy:  0.934, Validation Accuracy:  0.928, Loss:  0.131
    Epoch   5 Batch   43/538 - Train Accuracy:  0.913, Validation Accuracy:  0.928, Loss:  0.171
    Epoch   5 Batch   44/538 - Train Accuracy:  0.925, Validation Accuracy:  0.931, Loss:  0.155
    Epoch   5 Batch   45/538 - Train Accuracy:  0.933, Validation Accuracy:  0.931, Loss:  0.146
    Epoch   5 Batch   46/538 - Train Accuracy:  0.943, Validation Accuracy:  0.936, Loss:  0.146
    Epoch   5 Batch   47/538 - Train Accuracy:  0.924, Validation Accuracy:  0.937, Loss:  0.154
    Epoch   5 Batch   48/538 - Train Accuracy:  0.924, Validation Accuracy:  0.934, Loss:  0.153
    Epoch   5 Batch   49/538 - Train Accuracy:  0.928, Validation Accuracy:  0.933, Loss:  0.159
    Epoch   5 Batch   50/538 - Train Accuracy:  0.931, Validation Accuracy:  0.934, Loss:  0.134
    Epoch   5 Batch   51/538 - Train Accuracy:  0.917, Validation Accuracy:  0.931, Loss:  0.160
    Epoch   5 Batch   52/538 - Train Accuracy:  0.936, Validation Accuracy:  0.931, Loss:  0.143
    Epoch   5 Batch   53/538 - Train Accuracy:  0.926, Validation Accuracy:  0.928, Loss:  0.141
    Epoch   5 Batch   54/538 - Train Accuracy:  0.935, Validation Accuracy:  0.930, Loss:  0.138
    Epoch   5 Batch   55/538 - Train Accuracy:  0.916, Validation Accuracy:  0.934, Loss:  0.150
    Epoch   5 Batch   56/538 - Train Accuracy:  0.917, Validation Accuracy:  0.932, Loss:  0.143
    Epoch   5 Batch   57/538 - Train Accuracy:  0.907, Validation Accuracy:  0.925, Loss:  0.169
    Epoch   5 Batch   58/538 - Train Accuracy:  0.922, Validation Accuracy:  0.930, Loss:  0.145
    Epoch   5 Batch   59/538 - Train Accuracy:  0.923, Validation Accuracy:  0.926, Loss:  0.147
    Epoch   5 Batch   60/538 - Train Accuracy:  0.930, Validation Accuracy:  0.924, Loss:  0.145
    Epoch   5 Batch   61/538 - Train Accuracy:  0.928, Validation Accuracy:  0.922, Loss:  0.135
    Epoch   5 Batch   62/538 - Train Accuracy:  0.935, Validation Accuracy:  0.924, Loss:  0.144
    Epoch   5 Batch   63/538 - Train Accuracy:  0.939, Validation Accuracy:  0.925, Loss:  0.132
    Epoch   5 Batch   64/538 - Train Accuracy:  0.914, Validation Accuracy:  0.927, Loss:  0.162
    Epoch   5 Batch   65/538 - Train Accuracy:  0.908, Validation Accuracy:  0.923, Loss:  0.162
    Epoch   5 Batch   66/538 - Train Accuracy:  0.941, Validation Accuracy:  0.918, Loss:  0.128
    Epoch   5 Batch   67/538 - Train Accuracy:  0.934, Validation Accuracy:  0.918, Loss:  0.139
    Epoch   5 Batch   68/538 - Train Accuracy:  0.911, Validation Accuracy:  0.922, Loss:  0.135
    Epoch   5 Batch   69/538 - Train Accuracy:  0.933, Validation Accuracy:  0.927, Loss:  0.143
    Epoch   5 Batch   70/538 - Train Accuracy:  0.929, Validation Accuracy:  0.924, Loss:  0.133
    Epoch   5 Batch   71/538 - Train Accuracy:  0.935, Validation Accuracy:  0.930, Loss:  0.152
    Epoch   5 Batch   72/538 - Train Accuracy:  0.933, Validation Accuracy:  0.929, Loss:  0.174
    Epoch   5 Batch   73/538 - Train Accuracy:  0.925, Validation Accuracy:  0.932, Loss:  0.154
    Epoch   5 Batch   74/538 - Train Accuracy:  0.937, Validation Accuracy:  0.932, Loss:  0.135
    Epoch   5 Batch   75/538 - Train Accuracy:  0.941, Validation Accuracy:  0.930, Loss:  0.137
    Epoch   5 Batch   76/538 - Train Accuracy:  0.938, Validation Accuracy:  0.931, Loss:  0.162
    Epoch   5 Batch   77/538 - Train Accuracy:  0.931, Validation Accuracy:  0.928, Loss:  0.151
    Epoch   5 Batch   78/538 - Train Accuracy:  0.922, Validation Accuracy:  0.929, Loss:  0.151
    Epoch   5 Batch   79/538 - Train Accuracy:  0.931, Validation Accuracy:  0.927, Loss:  0.115
    Epoch   5 Batch   80/538 - Train Accuracy:  0.933, Validation Accuracy:  0.927, Loss:  0.143
    Epoch   5 Batch   81/538 - Train Accuracy:  0.926, Validation Accuracy:  0.925, Loss:  0.146
    Epoch   5 Batch   82/538 - Train Accuracy:  0.926, Validation Accuracy:  0.924, Loss:  0.140
    Epoch   5 Batch   83/538 - Train Accuracy:  0.933, Validation Accuracy:  0.924, Loss:  0.142
    Epoch   5 Batch   84/538 - Train Accuracy:  0.916, Validation Accuracy:  0.924, Loss:  0.149
    Epoch   5 Batch   85/538 - Train Accuracy:  0.932, Validation Accuracy:  0.922, Loss:  0.130
    Epoch   5 Batch   86/538 - Train Accuracy:  0.936, Validation Accuracy:  0.922, Loss:  0.137
    Epoch   5 Batch   87/538 - Train Accuracy:  0.919, Validation Accuracy:  0.922, Loss:  0.150
    Epoch   5 Batch   88/538 - Train Accuracy:  0.928, Validation Accuracy:  0.922, Loss:  0.138
    Epoch   5 Batch   89/538 - Train Accuracy:  0.938, Validation Accuracy:  0.928, Loss:  0.132
    Epoch   5 Batch   90/538 - Train Accuracy:  0.935, Validation Accuracy:  0.933, Loss:  0.148
    Epoch   5 Batch   91/538 - Train Accuracy:  0.931, Validation Accuracy:  0.933, Loss:  0.127
    Epoch   5 Batch   92/538 - Train Accuracy:  0.926, Validation Accuracy:  0.933, Loss:  0.145
    Epoch   5 Batch   93/538 - Train Accuracy:  0.944, Validation Accuracy:  0.933, Loss:  0.129
    Epoch   5 Batch   94/538 - Train Accuracy:  0.930, Validation Accuracy:  0.931, Loss:  0.142
    Epoch   5 Batch   95/538 - Train Accuracy:  0.930, Validation Accuracy:  0.933, Loss:  0.124
    Epoch   5 Batch   96/538 - Train Accuracy:  0.940, Validation Accuracy:  0.934, Loss:  0.126
    Epoch   5 Batch   97/538 - Train Accuracy:  0.930, Validation Accuracy:  0.933, Loss:  0.126
    Epoch   5 Batch   98/538 - Train Accuracy:  0.937, Validation Accuracy:  0.930, Loss:  0.137
    Epoch   5 Batch   99/538 - Train Accuracy:  0.932, Validation Accuracy:  0.931, Loss:  0.139
    Epoch   5 Batch  100/538 - Train Accuracy:  0.937, Validation Accuracy:  0.924, Loss:  0.131
    Epoch   5 Batch  101/538 - Train Accuracy:  0.912, Validation Accuracy:  0.918, Loss:  0.149
    Epoch   5 Batch  102/538 - Train Accuracy:  0.917, Validation Accuracy:  0.916, Loss:  0.148
    Epoch   5 Batch  103/538 - Train Accuracy:  0.944, Validation Accuracy:  0.918, Loss:  0.132
    Epoch   5 Batch  104/538 - Train Accuracy:  0.931, Validation Accuracy:  0.926, Loss:  0.128
    Epoch   5 Batch  105/538 - Train Accuracy:  0.931, Validation Accuracy:  0.925, Loss:  0.128
    Epoch   5 Batch  106/538 - Train Accuracy:  0.936, Validation Accuracy:  0.925, Loss:  0.114
    Epoch   5 Batch  107/538 - Train Accuracy:  0.928, Validation Accuracy:  0.924, Loss:  0.145
    Epoch   5 Batch  108/538 - Train Accuracy:  0.946, Validation Accuracy:  0.922, Loss:  0.145
    Epoch   5 Batch  109/538 - Train Accuracy:  0.946, Validation Accuracy:  0.922, Loss:  0.125
    Epoch   5 Batch  110/538 - Train Accuracy:  0.931, Validation Accuracy:  0.925, Loss:  0.149
    Epoch   5 Batch  111/538 - Train Accuracy:  0.934, Validation Accuracy:  0.924, Loss:  0.126
    Epoch   5 Batch  112/538 - Train Accuracy:  0.937, Validation Accuracy:  0.930, Loss:  0.142
    Epoch   5 Batch  113/538 - Train Accuracy:  0.914, Validation Accuracy:  0.928, Loss:  0.159
    Epoch   5 Batch  114/538 - Train Accuracy:  0.926, Validation Accuracy:  0.930, Loss:  0.132
    Epoch   5 Batch  115/538 - Train Accuracy:  0.948, Validation Accuracy:  0.934, Loss:  0.135
    Epoch   5 Batch  116/538 - Train Accuracy:  0.929, Validation Accuracy:  0.933, Loss:  0.164
    Epoch   5 Batch  117/538 - Train Accuracy:  0.938, Validation Accuracy:  0.933, Loss:  0.138
    Epoch   5 Batch  118/538 - Train Accuracy:  0.941, Validation Accuracy:  0.930, Loss:  0.119
    Epoch   5 Batch  119/538 - Train Accuracy:  0.946, Validation Accuracy:  0.928, Loss:  0.114
    Epoch   5 Batch  120/538 - Train Accuracy:  0.934, Validation Accuracy:  0.926, Loss:  0.130
    Epoch   5 Batch  121/538 - Train Accuracy:  0.943, Validation Accuracy:  0.929, Loss:  0.126
    Epoch   5 Batch  122/538 - Train Accuracy:  0.928, Validation Accuracy:  0.927, Loss:  0.126
    Epoch   5 Batch  123/538 - Train Accuracy:  0.933, Validation Accuracy:  0.930, Loss:  0.133
    Epoch   5 Batch  124/538 - Train Accuracy:  0.944, Validation Accuracy:  0.929, Loss:  0.121
    Epoch   5 Batch  125/538 - Train Accuracy:  0.930, Validation Accuracy:  0.933, Loss:  0.143
    Epoch   5 Batch  126/538 - Train Accuracy:  0.918, Validation Accuracy:  0.934, Loss:  0.138
    Epoch   5 Batch  127/538 - Train Accuracy:  0.909, Validation Accuracy:  0.934, Loss:  0.164
    Epoch   5 Batch  128/538 - Train Accuracy:  0.926, Validation Accuracy:  0.932, Loss:  0.147
    Epoch   5 Batch  129/538 - Train Accuracy:  0.943, Validation Accuracy:  0.930, Loss:  0.121
    Epoch   5 Batch  130/538 - Train Accuracy:  0.943, Validation Accuracy:  0.929, Loss:  0.122
    Epoch   5 Batch  131/538 - Train Accuracy:  0.945, Validation Accuracy:  0.928, Loss:  0.122
    Epoch   5 Batch  132/538 - Train Accuracy:  0.924, Validation Accuracy:  0.925, Loss:  0.146
    Epoch   5 Batch  133/538 - Train Accuracy:  0.931, Validation Accuracy:  0.925, Loss:  0.130
    Epoch   5 Batch  134/538 - Train Accuracy:  0.921, Validation Accuracy:  0.926, Loss:  0.159
    Epoch   5 Batch  135/538 - Train Accuracy:  0.933, Validation Accuracy:  0.930, Loss:  0.151
    Epoch   5 Batch  136/538 - Train Accuracy:  0.928, Validation Accuracy:  0.931, Loss:  0.148
    Epoch   5 Batch  137/538 - Train Accuracy:  0.926, Validation Accuracy:  0.932, Loss:  0.149
    Epoch   5 Batch  138/538 - Train Accuracy:  0.922, Validation Accuracy:  0.933, Loss:  0.139
    Epoch   5 Batch  139/538 - Train Accuracy:  0.919, Validation Accuracy:  0.931, Loss:  0.159
    Epoch   5 Batch  140/538 - Train Accuracy:  0.924, Validation Accuracy:  0.927, Loss:  0.171
    Epoch   5 Batch  141/538 - Train Accuracy:  0.945, Validation Accuracy:  0.928, Loss:  0.161
    Epoch   5 Batch  142/538 - Train Accuracy:  0.929, Validation Accuracy:  0.928, Loss:  0.126
    Epoch   5 Batch  143/538 - Train Accuracy:  0.937, Validation Accuracy:  0.929, Loss:  0.131
    Epoch   5 Batch  144/538 - Train Accuracy:  0.933, Validation Accuracy:  0.934, Loss:  0.141
    Epoch   5 Batch  145/538 - Train Accuracy:  0.919, Validation Accuracy:  0.931, Loss:  0.166
    Epoch   5 Batch  146/538 - Train Accuracy:  0.933, Validation Accuracy:  0.931, Loss:  0.144
    Epoch   5 Batch  147/538 - Train Accuracy:  0.934, Validation Accuracy:  0.935, Loss:  0.145
    Epoch   5 Batch  148/538 - Train Accuracy:  0.909, Validation Accuracy:  0.938, Loss:  0.167
    Epoch   5 Batch  149/538 - Train Accuracy:  0.946, Validation Accuracy:  0.937, Loss:  0.126
    Epoch   5 Batch  150/538 - Train Accuracy:  0.946, Validation Accuracy:  0.937, Loss:  0.127
    Epoch   5 Batch  151/538 - Train Accuracy:  0.941, Validation Accuracy:  0.936, Loss:  0.138
    Epoch   5 Batch  152/538 - Train Accuracy:  0.930, Validation Accuracy:  0.930, Loss:  0.147
    Epoch   5 Batch  153/538 - Train Accuracy:  0.924, Validation Accuracy:  0.928, Loss:  0.143
    Epoch   5 Batch  154/538 - Train Accuracy:  0.934, Validation Accuracy:  0.931, Loss:  0.129
    Epoch   5 Batch  155/538 - Train Accuracy:  0.936, Validation Accuracy:  0.929, Loss:  0.154
    Epoch   5 Batch  156/538 - Train Accuracy:  0.944, Validation Accuracy:  0.928, Loss:  0.129
    Epoch   5 Batch  157/538 - Train Accuracy:  0.942, Validation Accuracy:  0.933, Loss:  0.132
    Epoch   5 Batch  158/538 - Train Accuracy:  0.945, Validation Accuracy:  0.935, Loss:  0.137
    Epoch   5 Batch  159/538 - Train Accuracy:  0.933, Validation Accuracy:  0.934, Loss:  0.141
    Epoch   5 Batch  160/538 - Train Accuracy:  0.924, Validation Accuracy:  0.936, Loss:  0.128
    Epoch   5 Batch  161/538 - Train Accuracy:  0.940, Validation Accuracy:  0.935, Loss:  0.129
    Epoch   5 Batch  162/538 - Train Accuracy:  0.930, Validation Accuracy:  0.927, Loss:  0.142
    Epoch   5 Batch  163/538 - Train Accuracy:  0.935, Validation Accuracy:  0.931, Loss:  0.153
    Epoch   5 Batch  164/538 - Train Accuracy:  0.929, Validation Accuracy:  0.934, Loss:  0.157
    Epoch   5 Batch  165/538 - Train Accuracy:  0.942, Validation Accuracy:  0.936, Loss:  0.118
    Epoch   5 Batch  166/538 - Train Accuracy:  0.938, Validation Accuracy:  0.939, Loss:  0.143
    Epoch   5 Batch  167/538 - Train Accuracy:  0.930, Validation Accuracy:  0.934, Loss:  0.150
    Epoch   5 Batch  168/538 - Train Accuracy:  0.911, Validation Accuracy:  0.936, Loss:  0.162
    Epoch   5 Batch  169/538 - Train Accuracy:  0.955, Validation Accuracy:  0.931, Loss:  0.118
    Epoch   5 Batch  170/538 - Train Accuracy:  0.935, Validation Accuracy:  0.927, Loss:  0.148
    Epoch   5 Batch  171/538 - Train Accuracy:  0.936, Validation Accuracy:  0.924, Loss:  0.128
    Epoch   5 Batch  172/538 - Train Accuracy:  0.924, Validation Accuracy:  0.917, Loss:  0.131
    Epoch   5 Batch  173/538 - Train Accuracy:  0.940, Validation Accuracy:  0.922, Loss:  0.126
    Epoch   5 Batch  174/538 - Train Accuracy:  0.920, Validation Accuracy:  0.925, Loss:  0.133
    Epoch   5 Batch  175/538 - Train Accuracy:  0.941, Validation Accuracy:  0.925, Loss:  0.134
    Epoch   5 Batch  176/538 - Train Accuracy:  0.932, Validation Accuracy:  0.921, Loss:  0.152
    Epoch   5 Batch  177/538 - Train Accuracy:  0.928, Validation Accuracy:  0.919, Loss:  0.141
    Epoch   5 Batch  178/538 - Train Accuracy:  0.916, Validation Accuracy:  0.919, Loss:  0.137
    Epoch   5 Batch  179/538 - Train Accuracy:  0.938, Validation Accuracy:  0.921, Loss:  0.124
    Epoch   5 Batch  180/538 - Train Accuracy:  0.937, Validation Accuracy:  0.925, Loss:  0.130
    Epoch   5 Batch  181/538 - Train Accuracy:  0.928, Validation Accuracy:  0.923, Loss:  0.163
    Epoch   5 Batch  182/538 - Train Accuracy:  0.948, Validation Accuracy:  0.922, Loss:  0.123
    Epoch   5 Batch  183/538 - Train Accuracy:  0.950, Validation Accuracy:  0.923, Loss:  0.118
    Epoch   5 Batch  184/538 - Train Accuracy:  0.934, Validation Accuracy:  0.924, Loss:  0.134
    Epoch   5 Batch  185/538 - Train Accuracy:  0.949, Validation Accuracy:  0.926, Loss:  0.122
    Epoch   5 Batch  186/538 - Train Accuracy:  0.937, Validation Accuracy:  0.930, Loss:  0.126
    Epoch   5 Batch  187/538 - Train Accuracy:  0.949, Validation Accuracy:  0.933, Loss:  0.136
    Epoch   5 Batch  188/538 - Train Accuracy:  0.934, Validation Accuracy:  0.932, Loss:  0.128
    Epoch   5 Batch  189/538 - Train Accuracy:  0.930, Validation Accuracy:  0.933, Loss:  0.140
    Epoch   5 Batch  190/538 - Train Accuracy:  0.926, Validation Accuracy:  0.931, Loss:  0.156
    Epoch   5 Batch  191/538 - Train Accuracy:  0.936, Validation Accuracy:  0.923, Loss:  0.122
    Epoch   5 Batch  192/538 - Train Accuracy:  0.938, Validation Accuracy:  0.922, Loss:  0.135
    Epoch   5 Batch  193/538 - Train Accuracy:  0.934, Validation Accuracy:  0.923, Loss:  0.117
    Epoch   5 Batch  194/538 - Train Accuracy:  0.924, Validation Accuracy:  0.929, Loss:  0.136
    Epoch   5 Batch  195/538 - Train Accuracy:  0.940, Validation Accuracy:  0.928, Loss:  0.132
    Epoch   5 Batch  196/538 - Train Accuracy:  0.917, Validation Accuracy:  0.929, Loss:  0.127
    Epoch   5 Batch  197/538 - Train Accuracy:  0.925, Validation Accuracy:  0.924, Loss:  0.141
    Epoch   5 Batch  198/538 - Train Accuracy:  0.935, Validation Accuracy:  0.923, Loss:  0.137
    Epoch   5 Batch  199/538 - Train Accuracy:  0.930, Validation Accuracy:  0.923, Loss:  0.149
    Epoch   5 Batch  200/538 - Train Accuracy:  0.952, Validation Accuracy:  0.920, Loss:  0.110
    Epoch   5 Batch  201/538 - Train Accuracy:  0.940, Validation Accuracy:  0.923, Loss:  0.145
    Epoch   5 Batch  202/538 - Train Accuracy:  0.938, Validation Accuracy:  0.924, Loss:  0.131
    Epoch   5 Batch  203/538 - Train Accuracy:  0.928, Validation Accuracy:  0.927, Loss:  0.157
    Epoch   5 Batch  204/538 - Train Accuracy:  0.921, Validation Accuracy:  0.928, Loss:  0.146
    Epoch   5 Batch  205/538 - Train Accuracy:  0.938, Validation Accuracy:  0.929, Loss:  0.126
    Epoch   5 Batch  206/538 - Train Accuracy:  0.919, Validation Accuracy:  0.934, Loss:  0.129
    Epoch   5 Batch  207/538 - Train Accuracy:  0.943, Validation Accuracy:  0.933, Loss:  0.128
    Epoch   5 Batch  208/538 - Train Accuracy:  0.931, Validation Accuracy:  0.930, Loss:  0.154
    Epoch   5 Batch  209/538 - Train Accuracy:  0.939, Validation Accuracy:  0.931, Loss:  0.120
    Epoch   5 Batch  210/538 - Train Accuracy:  0.933, Validation Accuracy:  0.931, Loss:  0.146
    Epoch   5 Batch  211/538 - Train Accuracy:  0.936, Validation Accuracy:  0.925, Loss:  0.155
    Epoch   5 Batch  212/538 - Train Accuracy:  0.933, Validation Accuracy:  0.924, Loss:  0.130
    Epoch   5 Batch  213/538 - Train Accuracy:  0.940, Validation Accuracy:  0.926, Loss:  0.127
    Epoch   5 Batch  214/538 - Train Accuracy:  0.946, Validation Accuracy:  0.927, Loss:  0.120
    Epoch   5 Batch  215/538 - Train Accuracy:  0.945, Validation Accuracy:  0.925, Loss:  0.128
    Epoch   5 Batch  216/538 - Train Accuracy:  0.937, Validation Accuracy:  0.923, Loss:  0.133
    Epoch   5 Batch  217/538 - Train Accuracy:  0.944, Validation Accuracy:  0.921, Loss:  0.124
    Epoch   5 Batch  218/538 - Train Accuracy:  0.938, Validation Accuracy:  0.921, Loss:  0.132
    Epoch   5 Batch  219/538 - Train Accuracy:  0.933, Validation Accuracy:  0.923, Loss:  0.154
    Epoch   5 Batch  220/538 - Train Accuracy:  0.924, Validation Accuracy:  0.922, Loss:  0.147
    Epoch   5 Batch  221/538 - Train Accuracy:  0.942, Validation Accuracy:  0.921, Loss:  0.117
    Epoch   5 Batch  222/538 - Train Accuracy:  0.922, Validation Accuracy:  0.922, Loss:  0.132
    Epoch   5 Batch  223/538 - Train Accuracy:  0.912, Validation Accuracy:  0.922, Loss:  0.143
    Epoch   5 Batch  224/538 - Train Accuracy:  0.926, Validation Accuracy:  0.922, Loss:  0.144
    Epoch   5 Batch  225/538 - Train Accuracy:  0.949, Validation Accuracy:  0.924, Loss:  0.129
    Epoch   5 Batch  226/538 - Train Accuracy:  0.928, Validation Accuracy:  0.926, Loss:  0.138
    Epoch   5 Batch  227/538 - Train Accuracy:  0.940, Validation Accuracy:  0.930, Loss:  0.126
    Epoch   5 Batch  228/538 - Train Accuracy:  0.926, Validation Accuracy:  0.931, Loss:  0.130
    Epoch   5 Batch  229/538 - Train Accuracy:  0.927, Validation Accuracy:  0.929, Loss:  0.134
    Epoch   5 Batch  230/538 - Train Accuracy:  0.935, Validation Accuracy:  0.930, Loss:  0.126
    Epoch   5 Batch  231/538 - Train Accuracy:  0.945, Validation Accuracy:  0.932, Loss:  0.135
    Epoch   5 Batch  232/538 - Train Accuracy:  0.929, Validation Accuracy:  0.930, Loss:  0.119
    Epoch   5 Batch  233/538 - Train Accuracy:  0.940, Validation Accuracy:  0.930, Loss:  0.150
    Epoch   5 Batch  234/538 - Train Accuracy:  0.940, Validation Accuracy:  0.931, Loss:  0.140
    Epoch   5 Batch  235/538 - Train Accuracy:  0.938, Validation Accuracy:  0.932, Loss:  0.114
    Epoch   5 Batch  236/538 - Train Accuracy:  0.924, Validation Accuracy:  0.934, Loss:  0.139
    Epoch   5 Batch  237/538 - Train Accuracy:  0.932, Validation Accuracy:  0.931, Loss:  0.125
    Epoch   5 Batch  238/538 - Train Accuracy:  0.935, Validation Accuracy:  0.932, Loss:  0.131
    Epoch   5 Batch  239/538 - Train Accuracy:  0.920, Validation Accuracy:  0.932, Loss:  0.149
    Epoch   5 Batch  240/538 - Train Accuracy:  0.925, Validation Accuracy:  0.933, Loss:  0.151
    Epoch   5 Batch  241/538 - Train Accuracy:  0.919, Validation Accuracy:  0.931, Loss:  0.139
    Epoch   5 Batch  242/538 - Train Accuracy:  0.937, Validation Accuracy:  0.929, Loss:  0.126
    Epoch   5 Batch  243/538 - Train Accuracy:  0.953, Validation Accuracy:  0.923, Loss:  0.131
    Epoch   5 Batch  244/538 - Train Accuracy:  0.932, Validation Accuracy:  0.927, Loss:  0.129
    Epoch   5 Batch  245/538 - Train Accuracy:  0.930, Validation Accuracy:  0.928, Loss:  0.163
    Epoch   5 Batch  246/538 - Train Accuracy:  0.943, Validation Accuracy:  0.928, Loss:  0.118
    Epoch   5 Batch  247/538 - Train Accuracy:  0.931, Validation Accuracy:  0.930, Loss:  0.129
    Epoch   5 Batch  248/538 - Train Accuracy:  0.937, Validation Accuracy:  0.931, Loss:  0.138
    Epoch   5 Batch  249/538 - Train Accuracy:  0.939, Validation Accuracy:  0.932, Loss:  0.106
    Epoch   5 Batch  250/538 - Train Accuracy:  0.937, Validation Accuracy:  0.930, Loss:  0.126
    Epoch   5 Batch  251/538 - Train Accuracy:  0.943, Validation Accuracy:  0.924, Loss:  0.115
    Epoch   5 Batch  252/538 - Train Accuracy:  0.936, Validation Accuracy:  0.927, Loss:  0.118
    Epoch   5 Batch  253/538 - Train Accuracy:  0.922, Validation Accuracy:  0.927, Loss:  0.112
    Epoch   5 Batch  254/538 - Train Accuracy:  0.924, Validation Accuracy:  0.931, Loss:  0.130
    Epoch   5 Batch  255/538 - Train Accuracy:  0.939, Validation Accuracy:  0.934, Loss:  0.123
    Epoch   5 Batch  256/538 - Train Accuracy:  0.924, Validation Accuracy:  0.932, Loss:  0.137
    Epoch   5 Batch  257/538 - Train Accuracy:  0.941, Validation Accuracy:  0.932, Loss:  0.139
    Epoch   5 Batch  258/538 - Train Accuracy:  0.938, Validation Accuracy:  0.930, Loss:  0.138
    Epoch   5 Batch  259/538 - Train Accuracy:  0.953, Validation Accuracy:  0.931, Loss:  0.121
    Epoch   5 Batch  260/538 - Train Accuracy:  0.907, Validation Accuracy:  0.932, Loss:  0.153
    Epoch   5 Batch  261/538 - Train Accuracy:  0.941, Validation Accuracy:  0.933, Loss:  0.145
    Epoch   5 Batch  262/538 - Train Accuracy:  0.928, Validation Accuracy:  0.933, Loss:  0.123
    Epoch   5 Batch  263/538 - Train Accuracy:  0.933, Validation Accuracy:  0.933, Loss:  0.127
    Epoch   5 Batch  264/538 - Train Accuracy:  0.922, Validation Accuracy:  0.938, Loss:  0.142
    Epoch   5 Batch  265/538 - Train Accuracy:  0.918, Validation Accuracy:  0.939, Loss:  0.150
    Epoch   5 Batch  266/538 - Train Accuracy:  0.926, Validation Accuracy:  0.944, Loss:  0.135
    Epoch   5 Batch  267/538 - Train Accuracy:  0.934, Validation Accuracy:  0.940, Loss:  0.138
    Epoch   5 Batch  268/538 - Train Accuracy:  0.951, Validation Accuracy:  0.938, Loss:  0.116
    Epoch   5 Batch  269/538 - Train Accuracy:  0.930, Validation Accuracy:  0.940, Loss:  0.149
    Epoch   5 Batch  270/538 - Train Accuracy:  0.942, Validation Accuracy:  0.940, Loss:  0.124
    Epoch   5 Batch  271/538 - Train Accuracy:  0.939, Validation Accuracy:  0.938, Loss:  0.123
    Epoch   5 Batch  272/538 - Train Accuracy:  0.925, Validation Accuracy:  0.939, Loss:  0.144
    Epoch   5 Batch  273/538 - Train Accuracy:  0.937, Validation Accuracy:  0.938, Loss:  0.148
    Epoch   5 Batch  274/538 - Train Accuracy:  0.911, Validation Accuracy:  0.940, Loss:  0.156
    Epoch   5 Batch  275/538 - Train Accuracy:  0.940, Validation Accuracy:  0.939, Loss:  0.145
    Epoch   5 Batch  276/538 - Train Accuracy:  0.928, Validation Accuracy:  0.938, Loss:  0.147
    Epoch   5 Batch  277/538 - Train Accuracy:  0.927, Validation Accuracy:  0.936, Loss:  0.117
    Epoch   5 Batch  278/538 - Train Accuracy:  0.941, Validation Accuracy:  0.934, Loss:  0.128
    Epoch   5 Batch  279/538 - Train Accuracy:  0.930, Validation Accuracy:  0.935, Loss:  0.132
    Epoch   5 Batch  280/538 - Train Accuracy:  0.938, Validation Accuracy:  0.937, Loss:  0.117
    Epoch   5 Batch  281/538 - Train Accuracy:  0.935, Validation Accuracy:  0.936, Loss:  0.142
    Epoch   5 Batch  282/538 - Train Accuracy:  0.934, Validation Accuracy:  0.935, Loss:  0.143
    Epoch   5 Batch  283/538 - Train Accuracy:  0.935, Validation Accuracy:  0.935, Loss:  0.127
    Epoch   5 Batch  284/538 - Train Accuracy:  0.929, Validation Accuracy:  0.933, Loss:  0.148
    Epoch   5 Batch  285/538 - Train Accuracy:  0.942, Validation Accuracy:  0.933, Loss:  0.120
    Epoch   5 Batch  286/538 - Train Accuracy:  0.923, Validation Accuracy:  0.933, Loss:  0.146
    Epoch   5 Batch  287/538 - Train Accuracy:  0.946, Validation Accuracy:  0.930, Loss:  0.116
    Epoch   5 Batch  288/538 - Train Accuracy:  0.934, Validation Accuracy:  0.939, Loss:  0.146
    Epoch   5 Batch  289/538 - Train Accuracy:  0.934, Validation Accuracy:  0.934, Loss:  0.107
    Epoch   5 Batch  290/538 - Train Accuracy:  0.946, Validation Accuracy:  0.940, Loss:  0.116
    Epoch   5 Batch  291/538 - Train Accuracy:  0.939, Validation Accuracy:  0.940, Loss:  0.140
    Epoch   5 Batch  292/538 - Train Accuracy:  0.938, Validation Accuracy:  0.938, Loss:  0.112
    Epoch   5 Batch  293/538 - Train Accuracy:  0.926, Validation Accuracy:  0.936, Loss:  0.135
    Epoch   5 Batch  294/538 - Train Accuracy:  0.943, Validation Accuracy:  0.935, Loss:  0.131
    Epoch   5 Batch  295/538 - Train Accuracy:  0.933, Validation Accuracy:  0.933, Loss:  0.124
    Epoch   5 Batch  296/538 - Train Accuracy:  0.927, Validation Accuracy:  0.933, Loss:  0.131
    Epoch   5 Batch  297/538 - Train Accuracy:  0.944, Validation Accuracy:  0.933, Loss:  0.125
    Epoch   5 Batch  298/538 - Train Accuracy:  0.932, Validation Accuracy:  0.931, Loss:  0.122
    Epoch   5 Batch  299/538 - Train Accuracy:  0.931, Validation Accuracy:  0.931, Loss:  0.154
    Epoch   5 Batch  300/538 - Train Accuracy:  0.932, Validation Accuracy:  0.934, Loss:  0.127
    Epoch   5 Batch  301/538 - Train Accuracy:  0.931, Validation Accuracy:  0.932, Loss:  0.146
    Epoch   5 Batch  302/538 - Train Accuracy:  0.950, Validation Accuracy:  0.930, Loss:  0.116
    Epoch   5 Batch  303/538 - Train Accuracy:  0.945, Validation Accuracy:  0.931, Loss:  0.134
    Epoch   5 Batch  304/538 - Train Accuracy:  0.936, Validation Accuracy:  0.929, Loss:  0.143
    Epoch   5 Batch  305/538 - Train Accuracy:  0.944, Validation Accuracy:  0.930, Loss:  0.118
    Epoch   5 Batch  306/538 - Train Accuracy:  0.933, Validation Accuracy:  0.928, Loss:  0.121
    Epoch   5 Batch  307/538 - Train Accuracy:  0.949, Validation Accuracy:  0.929, Loss:  0.131
    Epoch   5 Batch  308/538 - Train Accuracy:  0.943, Validation Accuracy:  0.932, Loss:  0.120
    Epoch   5 Batch  309/538 - Train Accuracy:  0.933, Validation Accuracy:  0.934, Loss:  0.110
    Epoch   5 Batch  310/538 - Train Accuracy:  0.950, Validation Accuracy:  0.935, Loss:  0.126
    Epoch   5 Batch  311/538 - Train Accuracy:  0.934, Validation Accuracy:  0.935, Loss:  0.127
    Epoch   5 Batch  312/538 - Train Accuracy:  0.943, Validation Accuracy:  0.935, Loss:  0.121
    Epoch   5 Batch  313/538 - Train Accuracy:  0.929, Validation Accuracy:  0.935, Loss:  0.160
    Epoch   5 Batch  314/538 - Train Accuracy:  0.942, Validation Accuracy:  0.937, Loss:  0.118
    Epoch   5 Batch  315/538 - Train Accuracy:  0.934, Validation Accuracy:  0.937, Loss:  0.120
    Epoch   5 Batch  316/538 - Train Accuracy:  0.932, Validation Accuracy:  0.936, Loss:  0.116
    Epoch   5 Batch  317/538 - Train Accuracy:  0.937, Validation Accuracy:  0.934, Loss:  0.134
    Epoch   5 Batch  318/538 - Train Accuracy:  0.917, Validation Accuracy:  0.933, Loss:  0.143
    Epoch   5 Batch  319/538 - Train Accuracy:  0.942, Validation Accuracy:  0.930, Loss:  0.132
    Epoch   5 Batch  320/538 - Train Accuracy:  0.926, Validation Accuracy:  0.927, Loss:  0.131
    Epoch   5 Batch  321/538 - Train Accuracy:  0.931, Validation Accuracy:  0.924, Loss:  0.117
    Epoch   5 Batch  322/538 - Train Accuracy:  0.943, Validation Accuracy:  0.928, Loss:  0.130
    Epoch   5 Batch  323/538 - Train Accuracy:  0.932, Validation Accuracy:  0.930, Loss:  0.118
    Epoch   5 Batch  324/538 - Train Accuracy:  0.946, Validation Accuracy:  0.928, Loss:  0.140
    Epoch   5 Batch  325/538 - Train Accuracy:  0.947, Validation Accuracy:  0.934, Loss:  0.128
    Epoch   5 Batch  326/538 - Train Accuracy:  0.946, Validation Accuracy:  0.930, Loss:  0.117
    Epoch   5 Batch  327/538 - Train Accuracy:  0.926, Validation Accuracy:  0.929, Loss:  0.135
    Epoch   5 Batch  328/538 - Train Accuracy:  0.947, Validation Accuracy:  0.926, Loss:  0.119
    Epoch   5 Batch  329/538 - Train Accuracy:  0.936, Validation Accuracy:  0.928, Loss:  0.124
    Epoch   5 Batch  330/538 - Train Accuracy:  0.948, Validation Accuracy:  0.924, Loss:  0.113
    Epoch   5 Batch  331/538 - Train Accuracy:  0.942, Validation Accuracy:  0.933, Loss:  0.122
    Epoch   5 Batch  332/538 - Train Accuracy:  0.938, Validation Accuracy:  0.932, Loss:  0.131
    Epoch   5 Batch  333/538 - Train Accuracy:  0.945, Validation Accuracy:  0.929, Loss:  0.126
    Epoch   5 Batch  334/538 - Train Accuracy:  0.944, Validation Accuracy:  0.927, Loss:  0.114
    Epoch   5 Batch  335/538 - Train Accuracy:  0.945, Validation Accuracy:  0.927, Loss:  0.128
    Epoch   5 Batch  336/538 - Train Accuracy:  0.931, Validation Accuracy:  0.927, Loss:  0.136
    Epoch   5 Batch  337/538 - Train Accuracy:  0.936, Validation Accuracy:  0.928, Loss:  0.133
    Epoch   5 Batch  338/538 - Train Accuracy:  0.934, Validation Accuracy:  0.928, Loss:  0.124
    Epoch   5 Batch  339/538 - Train Accuracy:  0.929, Validation Accuracy:  0.930, Loss:  0.139
    Epoch   5 Batch  340/538 - Train Accuracy:  0.928, Validation Accuracy:  0.932, Loss:  0.137
    Epoch   5 Batch  341/538 - Train Accuracy:  0.931, Validation Accuracy:  0.931, Loss:  0.129
    Epoch   5 Batch  342/538 - Train Accuracy:  0.939, Validation Accuracy:  0.931, Loss:  0.124
    Epoch   5 Batch  343/538 - Train Accuracy:  0.947, Validation Accuracy:  0.931, Loss:  0.133
    Epoch   5 Batch  344/538 - Train Accuracy:  0.942, Validation Accuracy:  0.930, Loss:  0.124
    Epoch   5 Batch  345/538 - Train Accuracy:  0.938, Validation Accuracy:  0.928, Loss:  0.117
    Epoch   5 Batch  346/538 - Train Accuracy:  0.926, Validation Accuracy:  0.928, Loss:  0.141
    Epoch   5 Batch  347/538 - Train Accuracy:  0.946, Validation Accuracy:  0.933, Loss:  0.120
    Epoch   5 Batch  348/538 - Train Accuracy:  0.938, Validation Accuracy:  0.934, Loss:  0.135
    Epoch   5 Batch  349/538 - Train Accuracy:  0.951, Validation Accuracy:  0.930, Loss:  0.109
    Epoch   5 Batch  350/538 - Train Accuracy:  0.935, Validation Accuracy:  0.926, Loss:  0.140
    Epoch   5 Batch  351/538 - Train Accuracy:  0.935, Validation Accuracy:  0.924, Loss:  0.140
    Epoch   5 Batch  352/538 - Train Accuracy:  0.918, Validation Accuracy:  0.924, Loss:  0.153
    Epoch   5 Batch  353/538 - Train Accuracy:  0.923, Validation Accuracy:  0.929, Loss:  0.132
    Epoch   5 Batch  354/538 - Train Accuracy:  0.928, Validation Accuracy:  0.926, Loss:  0.134
    Epoch   5 Batch  355/538 - Train Accuracy:  0.932, Validation Accuracy:  0.929, Loss:  0.136
    Epoch   5 Batch  356/538 - Train Accuracy:  0.929, Validation Accuracy:  0.933, Loss:  0.121
    Epoch   5 Batch  357/538 - Train Accuracy:  0.944, Validation Accuracy:  0.932, Loss:  0.132
    Epoch   5 Batch  358/538 - Train Accuracy:  0.941, Validation Accuracy:  0.930, Loss:  0.110
    Epoch   5 Batch  359/538 - Train Accuracy:  0.916, Validation Accuracy:  0.930, Loss:  0.124
    Epoch   5 Batch  360/538 - Train Accuracy:  0.933, Validation Accuracy:  0.929, Loss:  0.133
    Epoch   5 Batch  361/538 - Train Accuracy:  0.942, Validation Accuracy:  0.934, Loss:  0.121
    Epoch   5 Batch  362/538 - Train Accuracy:  0.952, Validation Accuracy:  0.936, Loss:  0.117
    Epoch   5 Batch  363/538 - Train Accuracy:  0.931, Validation Accuracy:  0.938, Loss:  0.121
    Epoch   5 Batch  364/538 - Train Accuracy:  0.924, Validation Accuracy:  0.936, Loss:  0.142
    Epoch   5 Batch  365/538 - Train Accuracy:  0.925, Validation Accuracy:  0.936, Loss:  0.128
    Epoch   5 Batch  366/538 - Train Accuracy:  0.942, Validation Accuracy:  0.931, Loss:  0.124
    Epoch   5 Batch  367/538 - Train Accuracy:  0.933, Validation Accuracy:  0.934, Loss:  0.105
    Epoch   5 Batch  368/538 - Train Accuracy:  0.948, Validation Accuracy:  0.934, Loss:  0.107
    Epoch   5 Batch  369/538 - Train Accuracy:  0.937, Validation Accuracy:  0.934, Loss:  0.125
    Epoch   5 Batch  370/538 - Train Accuracy:  0.933, Validation Accuracy:  0.938, Loss:  0.136
    Epoch   5 Batch  371/538 - Train Accuracy:  0.946, Validation Accuracy:  0.940, Loss:  0.127
    Epoch   5 Batch  372/538 - Train Accuracy:  0.942, Validation Accuracy:  0.939, Loss:  0.114
    Epoch   5 Batch  373/538 - Train Accuracy:  0.934, Validation Accuracy:  0.942, Loss:  0.108
    Epoch   5 Batch  374/538 - Train Accuracy:  0.940, Validation Accuracy:  0.936, Loss:  0.133
    Epoch   5 Batch  375/538 - Train Accuracy:  0.929, Validation Accuracy:  0.938, Loss:  0.121
    Epoch   5 Batch  376/538 - Train Accuracy:  0.930, Validation Accuracy:  0.936, Loss:  0.128
    Epoch   5 Batch  377/538 - Train Accuracy:  0.955, Validation Accuracy:  0.936, Loss:  0.119
    Epoch   5 Batch  378/538 - Train Accuracy:  0.944, Validation Accuracy:  0.937, Loss:  0.125
    Epoch   5 Batch  379/538 - Train Accuracy:  0.954, Validation Accuracy:  0.934, Loss:  0.127
    Epoch   5 Batch  380/538 - Train Accuracy:  0.932, Validation Accuracy:  0.940, Loss:  0.117
    Epoch   5 Batch  381/538 - Train Accuracy:  0.940, Validation Accuracy:  0.939, Loss:  0.114
    Epoch   5 Batch  382/538 - Train Accuracy:  0.925, Validation Accuracy:  0.940, Loss:  0.135
    Epoch   5 Batch  383/538 - Train Accuracy:  0.934, Validation Accuracy:  0.939, Loss:  0.119
    Epoch   5 Batch  384/538 - Train Accuracy:  0.919, Validation Accuracy:  0.941, Loss:  0.119
    Epoch   5 Batch  385/538 - Train Accuracy:  0.935, Validation Accuracy:  0.941, Loss:  0.134
    Epoch   5 Batch  386/538 - Train Accuracy:  0.944, Validation Accuracy:  0.936, Loss:  0.135
    Epoch   5 Batch  387/538 - Train Accuracy:  0.928, Validation Accuracy:  0.932, Loss:  0.118
    Epoch   5 Batch  388/538 - Train Accuracy:  0.930, Validation Accuracy:  0.933, Loss:  0.127
    Epoch   5 Batch  389/538 - Train Accuracy:  0.920, Validation Accuracy:  0.935, Loss:  0.157
    Epoch   5 Batch  390/538 - Train Accuracy:  0.945, Validation Accuracy:  0.934, Loss:  0.109
    Epoch   5 Batch  391/538 - Train Accuracy:  0.924, Validation Accuracy:  0.935, Loss:  0.131
    Epoch   5 Batch  392/538 - Train Accuracy:  0.934, Validation Accuracy:  0.938, Loss:  0.103
    Epoch   5 Batch  393/538 - Train Accuracy:  0.946, Validation Accuracy:  0.935, Loss:  0.116
    Epoch   5 Batch  394/538 - Train Accuracy:  0.913, Validation Accuracy:  0.938, Loss:  0.131
    Epoch   5 Batch  395/538 - Train Accuracy:  0.939, Validation Accuracy:  0.936, Loss:  0.131
    Epoch   5 Batch  396/538 - Train Accuracy:  0.935, Validation Accuracy:  0.935, Loss:  0.124
    Epoch   5 Batch  397/538 - Train Accuracy:  0.935, Validation Accuracy:  0.935, Loss:  0.133
    Epoch   5 Batch  398/538 - Train Accuracy:  0.933, Validation Accuracy:  0.940, Loss:  0.118
    Epoch   5 Batch  399/538 - Train Accuracy:  0.925, Validation Accuracy:  0.940, Loss:  0.138
    Epoch   5 Batch  400/538 - Train Accuracy:  0.939, Validation Accuracy:  0.940, Loss:  0.128
    Epoch   5 Batch  401/538 - Train Accuracy:  0.953, Validation Accuracy:  0.939, Loss:  0.123
    Epoch   5 Batch  402/538 - Train Accuracy:  0.939, Validation Accuracy:  0.939, Loss:  0.117
    Epoch   5 Batch  403/538 - Train Accuracy:  0.943, Validation Accuracy:  0.936, Loss:  0.110
    Epoch   5 Batch  404/538 - Train Accuracy:  0.944, Validation Accuracy:  0.936, Loss:  0.116
    Epoch   5 Batch  405/538 - Train Accuracy:  0.936, Validation Accuracy:  0.935, Loss:  0.122
    Epoch   5 Batch  406/538 - Train Accuracy:  0.930, Validation Accuracy:  0.937, Loss:  0.123
    Epoch   5 Batch  407/538 - Train Accuracy:  0.946, Validation Accuracy:  0.937, Loss:  0.124
    Epoch   5 Batch  408/538 - Train Accuracy:  0.925, Validation Accuracy:  0.938, Loss:  0.141
    Epoch   5 Batch  409/538 - Train Accuracy:  0.919, Validation Accuracy:  0.938, Loss:  0.119
    Epoch   5 Batch  410/538 - Train Accuracy:  0.947, Validation Accuracy:  0.937, Loss:  0.113
    Epoch   5 Batch  411/538 - Train Accuracy:  0.946, Validation Accuracy:  0.936, Loss:  0.111
    Epoch   5 Batch  412/538 - Train Accuracy:  0.944, Validation Accuracy:  0.930, Loss:  0.111
    Epoch   5 Batch  413/538 - Train Accuracy:  0.948, Validation Accuracy:  0.930, Loss:  0.111
    Epoch   5 Batch  414/538 - Train Accuracy:  0.912, Validation Accuracy:  0.929, Loss:  0.159
    Epoch   5 Batch  415/538 - Train Accuracy:  0.917, Validation Accuracy:  0.929, Loss:  0.122
    Epoch   5 Batch  416/538 - Train Accuracy:  0.946, Validation Accuracy:  0.928, Loss:  0.120
    Epoch   5 Batch  417/538 - Train Accuracy:  0.941, Validation Accuracy:  0.927, Loss:  0.119
    Epoch   5 Batch  418/538 - Train Accuracy:  0.945, Validation Accuracy:  0.923, Loss:  0.142
    Epoch   5 Batch  419/538 - Train Accuracy:  0.948, Validation Accuracy:  0.923, Loss:  0.116
    Epoch   5 Batch  420/538 - Train Accuracy:  0.941, Validation Accuracy:  0.928, Loss:  0.119
    Epoch   5 Batch  421/538 - Train Accuracy:  0.947, Validation Accuracy:  0.934, Loss:  0.112
    Epoch   5 Batch  422/538 - Train Accuracy:  0.936, Validation Accuracy:  0.933, Loss:  0.128
    Epoch   5 Batch  423/538 - Train Accuracy:  0.934, Validation Accuracy:  0.932, Loss:  0.136
    Epoch   5 Batch  424/538 - Train Accuracy:  0.933, Validation Accuracy:  0.935, Loss:  0.132
    Epoch   5 Batch  425/538 - Train Accuracy:  0.926, Validation Accuracy:  0.935, Loss:  0.132
    Epoch   5 Batch  426/538 - Train Accuracy:  0.942, Validation Accuracy:  0.936, Loss:  0.113
    Epoch   5 Batch  427/538 - Train Accuracy:  0.920, Validation Accuracy:  0.935, Loss:  0.134
    Epoch   5 Batch  428/538 - Train Accuracy:  0.949, Validation Accuracy:  0.937, Loss:  0.110
    Epoch   5 Batch  429/538 - Train Accuracy:  0.938, Validation Accuracy:  0.935, Loss:  0.123
    Epoch   5 Batch  430/538 - Train Accuracy:  0.930, Validation Accuracy:  0.933, Loss:  0.122
    Epoch   5 Batch  431/538 - Train Accuracy:  0.927, Validation Accuracy:  0.935, Loss:  0.115
    Epoch   5 Batch  432/538 - Train Accuracy:  0.941, Validation Accuracy:  0.941, Loss:  0.129
    Epoch   5 Batch  433/538 - Train Accuracy:  0.937, Validation Accuracy:  0.944, Loss:  0.164
    Epoch   5 Batch  434/538 - Train Accuracy:  0.935, Validation Accuracy:  0.946, Loss:  0.125
    Epoch   5 Batch  435/538 - Train Accuracy:  0.926, Validation Accuracy:  0.945, Loss:  0.125
    Epoch   5 Batch  436/538 - Train Accuracy:  0.935, Validation Accuracy:  0.945, Loss:  0.129
    Epoch   5 Batch  437/538 - Train Accuracy:  0.940, Validation Accuracy:  0.947, Loss:  0.128
    Epoch   5 Batch  438/538 - Train Accuracy:  0.936, Validation Accuracy:  0.945, Loss:  0.123
    Epoch   5 Batch  439/538 - Train Accuracy:  0.955, Validation Accuracy:  0.939, Loss:  0.114
    Epoch   5 Batch  440/538 - Train Accuracy:  0.940, Validation Accuracy:  0.938, Loss:  0.152
    Epoch   5 Batch  441/538 - Train Accuracy:  0.928, Validation Accuracy:  0.932, Loss:  0.139
    Epoch   5 Batch  442/538 - Train Accuracy:  0.936, Validation Accuracy:  0.934, Loss:  0.104
    Epoch   5 Batch  443/538 - Train Accuracy:  0.931, Validation Accuracy:  0.930, Loss:  0.127
    Epoch   5 Batch  444/538 - Train Accuracy:  0.945, Validation Accuracy:  0.929, Loss:  0.112
    Epoch   5 Batch  445/538 - Train Accuracy:  0.945, Validation Accuracy:  0.928, Loss:  0.107
    Epoch   5 Batch  446/538 - Train Accuracy:  0.952, Validation Accuracy:  0.929, Loss:  0.117
    Epoch   5 Batch  447/538 - Train Accuracy:  0.938, Validation Accuracy:  0.928, Loss:  0.120
    Epoch   5 Batch  448/538 - Train Accuracy:  0.939, Validation Accuracy:  0.934, Loss:  0.103
    Epoch   5 Batch  449/538 - Train Accuracy:  0.941, Validation Accuracy:  0.937, Loss:  0.132
    Epoch   5 Batch  450/538 - Train Accuracy:  0.926, Validation Accuracy:  0.941, Loss:  0.141
    Epoch   5 Batch  451/538 - Train Accuracy:  0.926, Validation Accuracy:  0.940, Loss:  0.124
    Epoch   5 Batch  452/538 - Train Accuracy:  0.932, Validation Accuracy:  0.937, Loss:  0.109
    Epoch   5 Batch  453/538 - Train Accuracy:  0.938, Validation Accuracy:  0.939, Loss:  0.123
    Epoch   5 Batch  454/538 - Train Accuracy:  0.931, Validation Accuracy:  0.936, Loss:  0.118
    Epoch   5 Batch  455/538 - Train Accuracy:  0.941, Validation Accuracy:  0.933, Loss:  0.116
    Epoch   5 Batch  456/538 - Train Accuracy:  0.947, Validation Accuracy:  0.930, Loss:  0.146
    Epoch   5 Batch  457/538 - Train Accuracy:  0.938, Validation Accuracy:  0.933, Loss:  0.128
    Epoch   5 Batch  458/538 - Train Accuracy:  0.943, Validation Accuracy:  0.935, Loss:  0.104
    Epoch   5 Batch  459/538 - Train Accuracy:  0.939, Validation Accuracy:  0.934, Loss:  0.115
    Epoch   5 Batch  460/538 - Train Accuracy:  0.923, Validation Accuracy:  0.933, Loss:  0.137
    Epoch   5 Batch  461/538 - Train Accuracy:  0.952, Validation Accuracy:  0.937, Loss:  0.135
    Epoch   5 Batch  462/538 - Train Accuracy:  0.931, Validation Accuracy:  0.937, Loss:  0.129
    Epoch   5 Batch  463/538 - Train Accuracy:  0.916, Validation Accuracy:  0.935, Loss:  0.128
    Epoch   5 Batch  464/538 - Train Accuracy:  0.944, Validation Accuracy:  0.935, Loss:  0.123
    Epoch   5 Batch  465/538 - Train Accuracy:  0.938, Validation Accuracy:  0.939, Loss:  0.111
    Epoch   5 Batch  466/538 - Train Accuracy:  0.929, Validation Accuracy:  0.936, Loss:  0.122
    Epoch   5 Batch  467/538 - Train Accuracy:  0.943, Validation Accuracy:  0.936, Loss:  0.123
    Epoch   5 Batch  468/538 - Train Accuracy:  0.954, Validation Accuracy:  0.936, Loss:  0.131
    Epoch   5 Batch  469/538 - Train Accuracy:  0.925, Validation Accuracy:  0.933, Loss:  0.126
    Epoch   5 Batch  470/538 - Train Accuracy:  0.940, Validation Accuracy:  0.932, Loss:  0.115
    Epoch   5 Batch  471/538 - Train Accuracy:  0.938, Validation Accuracy:  0.930, Loss:  0.107
    Epoch   5 Batch  472/538 - Train Accuracy:  0.969, Validation Accuracy:  0.926, Loss:  0.094
    Epoch   5 Batch  473/538 - Train Accuracy:  0.935, Validation Accuracy:  0.927, Loss:  0.116
    Epoch   5 Batch  474/538 - Train Accuracy:  0.951, Validation Accuracy:  0.926, Loss:  0.111
    Epoch   5 Batch  475/538 - Train Accuracy:  0.930, Validation Accuracy:  0.930, Loss:  0.116
    Epoch   5 Batch  476/538 - Train Accuracy:  0.946, Validation Accuracy:  0.930, Loss:  0.105
    Epoch   5 Batch  477/538 - Train Accuracy:  0.945, Validation Accuracy:  0.926, Loss:  0.134
    Epoch   5 Batch  478/538 - Train Accuracy:  0.946, Validation Accuracy:  0.933, Loss:  0.101
    Epoch   5 Batch  479/538 - Train Accuracy:  0.943, Validation Accuracy:  0.928, Loss:  0.112
    Epoch   5 Batch  480/538 - Train Accuracy:  0.940, Validation Accuracy:  0.927, Loss:  0.117
    Epoch   5 Batch  481/538 - Train Accuracy:  0.935, Validation Accuracy:  0.930, Loss:  0.118
    Epoch   5 Batch  482/538 - Train Accuracy:  0.923, Validation Accuracy:  0.927, Loss:  0.112
    Epoch   5 Batch  483/538 - Train Accuracy:  0.919, Validation Accuracy:  0.929, Loss:  0.145
    Epoch   5 Batch  484/538 - Train Accuracy:  0.928, Validation Accuracy:  0.931, Loss:  0.148
    Epoch   5 Batch  485/538 - Train Accuracy:  0.929, Validation Accuracy:  0.930, Loss:  0.123
    Epoch   5 Batch  486/538 - Train Accuracy:  0.949, Validation Accuracy:  0.930, Loss:  0.101
    Epoch   5 Batch  487/538 - Train Accuracy:  0.948, Validation Accuracy:  0.926, Loss:  0.101
    Epoch   5 Batch  488/538 - Train Accuracy:  0.938, Validation Accuracy:  0.932, Loss:  0.113
    Epoch   5 Batch  489/538 - Train Accuracy:  0.931, Validation Accuracy:  0.935, Loss:  0.115
    Epoch   5 Batch  490/538 - Train Accuracy:  0.930, Validation Accuracy:  0.937, Loss:  0.123
    Epoch   5 Batch  491/538 - Train Accuracy:  0.917, Validation Accuracy:  0.934, Loss:  0.122
    Epoch   5 Batch  492/538 - Train Accuracy:  0.929, Validation Accuracy:  0.931, Loss:  0.115
    Epoch   5 Batch  493/538 - Train Accuracy:  0.937, Validation Accuracy:  0.927, Loss:  0.107
    Epoch   5 Batch  494/538 - Train Accuracy:  0.946, Validation Accuracy:  0.927, Loss:  0.123
    Epoch   5 Batch  495/538 - Train Accuracy:  0.942, Validation Accuracy:  0.931, Loss:  0.123
    Epoch   5 Batch  496/538 - Train Accuracy:  0.943, Validation Accuracy:  0.929, Loss:  0.105
    Epoch   5 Batch  497/538 - Train Accuracy:  0.949, Validation Accuracy:  0.932, Loss:  0.113
    Epoch   5 Batch  498/538 - Train Accuracy:  0.951, Validation Accuracy:  0.928, Loss:  0.109
    Epoch   5 Batch  499/538 - Train Accuracy:  0.932, Validation Accuracy:  0.930, Loss:  0.122
    Epoch   5 Batch  500/538 - Train Accuracy:  0.950, Validation Accuracy:  0.937, Loss:  0.097
    Epoch   5 Batch  501/538 - Train Accuracy:  0.949, Validation Accuracy:  0.934, Loss:  0.132
    Epoch   5 Batch  502/538 - Train Accuracy:  0.938, Validation Accuracy:  0.937, Loss:  0.111
    Epoch   5 Batch  503/538 - Train Accuracy:  0.948, Validation Accuracy:  0.936, Loss:  0.126
    Epoch   5 Batch  504/538 - Train Accuracy:  0.957, Validation Accuracy:  0.933, Loss:  0.104
    Epoch   5 Batch  505/538 - Train Accuracy:  0.960, Validation Accuracy:  0.935, Loss:  0.108
    Epoch   5 Batch  506/538 - Train Accuracy:  0.952, Validation Accuracy:  0.929, Loss:  0.111
    Epoch   5 Batch  507/538 - Train Accuracy:  0.928, Validation Accuracy:  0.928, Loss:  0.124
    Epoch   5 Batch  508/538 - Train Accuracy:  0.928, Validation Accuracy:  0.933, Loss:  0.113
    Epoch   5 Batch  509/538 - Train Accuracy:  0.948, Validation Accuracy:  0.931, Loss:  0.117
    Epoch   5 Batch  510/538 - Train Accuracy:  0.941, Validation Accuracy:  0.932, Loss:  0.100
    Epoch   5 Batch  511/538 - Train Accuracy:  0.934, Validation Accuracy:  0.929, Loss:  0.134
    Epoch   5 Batch  512/538 - Train Accuracy:  0.946, Validation Accuracy:  0.929, Loss:  0.113
    Epoch   5 Batch  513/538 - Train Accuracy:  0.927, Validation Accuracy:  0.931, Loss:  0.118
    Epoch   5 Batch  514/538 - Train Accuracy:  0.944, Validation Accuracy:  0.931, Loss:  0.121
    Epoch   5 Batch  515/538 - Train Accuracy:  0.932, Validation Accuracy:  0.929, Loss:  0.124
    Epoch   5 Batch  516/538 - Train Accuracy:  0.927, Validation Accuracy:  0.923, Loss:  0.117
    Epoch   5 Batch  517/538 - Train Accuracy:  0.942, Validation Accuracy:  0.926, Loss:  0.117
    Epoch   5 Batch  518/538 - Train Accuracy:  0.927, Validation Accuracy:  0.932, Loss:  0.125
    Epoch   5 Batch  519/538 - Train Accuracy:  0.942, Validation Accuracy:  0.929, Loss:  0.120
    Epoch   5 Batch  520/538 - Train Accuracy:  0.927, Validation Accuracy:  0.927, Loss:  0.142
    Epoch   5 Batch  521/538 - Train Accuracy:  0.933, Validation Accuracy:  0.929, Loss:  0.133
    Epoch   5 Batch  522/538 - Train Accuracy:  0.943, Validation Accuracy:  0.928, Loss:  0.105
    Epoch   5 Batch  523/538 - Train Accuracy:  0.934, Validation Accuracy:  0.925, Loss:  0.104
    Epoch   5 Batch  524/538 - Train Accuracy:  0.933, Validation Accuracy:  0.929, Loss:  0.119
    Epoch   5 Batch  525/538 - Train Accuracy:  0.942, Validation Accuracy:  0.928, Loss:  0.115
    Epoch   5 Batch  526/538 - Train Accuracy:  0.938, Validation Accuracy:  0.926, Loss:  0.114
    Epoch   5 Batch  527/538 - Train Accuracy:  0.938, Validation Accuracy:  0.927, Loss:  0.113
    Epoch   5 Batch  528/538 - Train Accuracy:  0.949, Validation Accuracy:  0.929, Loss:  0.118
    Epoch   5 Batch  529/538 - Train Accuracy:  0.921, Validation Accuracy:  0.930, Loss:  0.132
    Epoch   5 Batch  530/538 - Train Accuracy:  0.932, Validation Accuracy:  0.930, Loss:  0.128
    Epoch   5 Batch  531/538 - Train Accuracy:  0.939, Validation Accuracy:  0.930, Loss:  0.115
    Epoch   5 Batch  532/538 - Train Accuracy:  0.928, Validation Accuracy:  0.928, Loss:  0.109
    Epoch   5 Batch  533/538 - Train Accuracy:  0.939, Validation Accuracy:  0.930, Loss:  0.106
    Epoch   5 Batch  534/538 - Train Accuracy:  0.942, Validation Accuracy:  0.936, Loss:  0.107
    Epoch   5 Batch  535/538 - Train Accuracy:  0.945, Validation Accuracy:  0.936, Loss:  0.105
    Epoch   5 Batch  536/538 - Train Accuracy:  0.948, Validation Accuracy:  0.938, Loss:  0.131
    Epoch   6 Batch    0/538 - Train Accuracy:  0.955, Validation Accuracy:  0.939, Loss:  0.102
    Epoch   6 Batch    1/538 - Train Accuracy:  0.954, Validation Accuracy:  0.937, Loss:  0.114
    Epoch   6 Batch    2/538 - Train Accuracy:  0.946, Validation Accuracy:  0.934, Loss:  0.129
    Epoch   6 Batch    3/538 - Train Accuracy:  0.947, Validation Accuracy:  0.938, Loss:  0.108
    Epoch   6 Batch    4/538 - Train Accuracy:  0.937, Validation Accuracy:  0.936, Loss:  0.116
    Epoch   6 Batch    5/538 - Train Accuracy:  0.932, Validation Accuracy:  0.933, Loss:  0.119
    Epoch   6 Batch    6/538 - Train Accuracy:  0.931, Validation Accuracy:  0.932, Loss:  0.112
    Epoch   6 Batch    7/538 - Train Accuracy:  0.955, Validation Accuracy:  0.935, Loss:  0.121
    Epoch   6 Batch    8/538 - Train Accuracy:  0.938, Validation Accuracy:  0.934, Loss:  0.111
    Epoch   6 Batch    9/538 - Train Accuracy:  0.940, Validation Accuracy:  0.936, Loss:  0.110
    Epoch   6 Batch   10/538 - Train Accuracy:  0.922, Validation Accuracy:  0.937, Loss:  0.132
    Epoch   6 Batch   11/538 - Train Accuracy:  0.951, Validation Accuracy:  0.939, Loss:  0.106
    Epoch   6 Batch   12/538 - Train Accuracy:  0.951, Validation Accuracy:  0.938, Loss:  0.104
    Epoch   6 Batch   13/538 - Train Accuracy:  0.946, Validation Accuracy:  0.939, Loss:  0.097
    Epoch   6 Batch   14/538 - Train Accuracy:  0.949, Validation Accuracy:  0.939, Loss:  0.104
    Epoch   6 Batch   15/538 - Train Accuracy:  0.947, Validation Accuracy:  0.938, Loss:  0.119
    Epoch   6 Batch   16/538 - Train Accuracy:  0.944, Validation Accuracy:  0.934, Loss:  0.102
    Epoch   6 Batch   17/538 - Train Accuracy:  0.949, Validation Accuracy:  0.933, Loss:  0.112
    Epoch   6 Batch   18/538 - Train Accuracy:  0.951, Validation Accuracy:  0.933, Loss:  0.122
    Epoch   6 Batch   19/538 - Train Accuracy:  0.946, Validation Accuracy:  0.932, Loss:  0.111
    Epoch   6 Batch   20/538 - Train Accuracy:  0.938, Validation Accuracy:  0.931, Loss:  0.110
    Epoch   6 Batch   21/538 - Train Accuracy:  0.956, Validation Accuracy:  0.932, Loss:  0.094
    Epoch   6 Batch   22/538 - Train Accuracy:  0.930, Validation Accuracy:  0.929, Loss:  0.118
    Epoch   6 Batch   23/538 - Train Accuracy:  0.923, Validation Accuracy:  0.937, Loss:  0.135
    Epoch   6 Batch   24/538 - Train Accuracy:  0.943, Validation Accuracy:  0.938, Loss:  0.112
    Epoch   6 Batch   25/538 - Train Accuracy:  0.934, Validation Accuracy:  0.937, Loss:  0.109
    Epoch   6 Batch   26/538 - Train Accuracy:  0.931, Validation Accuracy:  0.937, Loss:  0.129
    Epoch   6 Batch   27/538 - Train Accuracy:  0.943, Validation Accuracy:  0.935, Loss:  0.108
    Epoch   6 Batch   28/538 - Train Accuracy:  0.940, Validation Accuracy:  0.933, Loss:  0.099
    Epoch   6 Batch   29/538 - Train Accuracy:  0.941, Validation Accuracy:  0.930, Loss:  0.098
    Epoch   6 Batch   30/538 - Train Accuracy:  0.930, Validation Accuracy:  0.926, Loss:  0.128
    Epoch   6 Batch   31/538 - Train Accuracy:  0.950, Validation Accuracy:  0.929, Loss:  0.101
    Epoch   6 Batch   32/538 - Train Accuracy:  0.939, Validation Accuracy:  0.929, Loss:  0.093
    Epoch   6 Batch   33/538 - Train Accuracy:  0.940, Validation Accuracy:  0.931, Loss:  0.104
    Epoch   6 Batch   34/538 - Train Accuracy:  0.935, Validation Accuracy:  0.925, Loss:  0.121
    Epoch   6 Batch   35/538 - Train Accuracy:  0.954, Validation Accuracy:  0.932, Loss:  0.102
    Epoch   6 Batch   36/538 - Train Accuracy:  0.932, Validation Accuracy:  0.928, Loss:  0.091
    Epoch   6 Batch   37/538 - Train Accuracy:  0.954, Validation Accuracy:  0.933, Loss:  0.119
    Epoch   6 Batch   38/538 - Train Accuracy:  0.928, Validation Accuracy:  0.936, Loss:  0.119
    Epoch   6 Batch   39/538 - Train Accuracy:  0.950, Validation Accuracy:  0.933, Loss:  0.105
    Epoch   6 Batch   40/538 - Train Accuracy:  0.950, Validation Accuracy:  0.931, Loss:  0.097
    Epoch   6 Batch   41/538 - Train Accuracy:  0.945, Validation Accuracy:  0.934, Loss:  0.111
    Epoch   6 Batch   42/538 - Train Accuracy:  0.935, Validation Accuracy:  0.932, Loss:  0.109
    Epoch   6 Batch   43/538 - Train Accuracy:  0.918, Validation Accuracy:  0.933, Loss:  0.138
    Epoch   6 Batch   44/538 - Train Accuracy:  0.934, Validation Accuracy:  0.935, Loss:  0.113
    Epoch   6 Batch   45/538 - Train Accuracy:  0.940, Validation Accuracy:  0.933, Loss:  0.105
    Epoch   6 Batch   46/538 - Train Accuracy:  0.949, Validation Accuracy:  0.932, Loss:  0.103
    Epoch   6 Batch   47/538 - Train Accuracy:  0.946, Validation Accuracy:  0.931, Loss:  0.123
    Epoch   6 Batch   48/538 - Train Accuracy:  0.939, Validation Accuracy:  0.929, Loss:  0.128
    Epoch   6 Batch   49/538 - Train Accuracy:  0.940, Validation Accuracy:  0.929, Loss:  0.112
    Epoch   6 Batch   50/538 - Train Accuracy:  0.936, Validation Accuracy:  0.928, Loss:  0.112
    Epoch   6 Batch   51/538 - Train Accuracy:  0.931, Validation Accuracy:  0.925, Loss:  0.129
    Epoch   6 Batch   52/538 - Train Accuracy:  0.938, Validation Accuracy:  0.925, Loss:  0.116
    Epoch   6 Batch   53/538 - Train Accuracy:  0.929, Validation Accuracy:  0.930, Loss:  0.110
    Epoch   6 Batch   54/538 - Train Accuracy:  0.947, Validation Accuracy:  0.931, Loss:  0.102
    Epoch   6 Batch   55/538 - Train Accuracy:  0.929, Validation Accuracy:  0.933, Loss:  0.101
    Epoch   6 Batch   56/538 - Train Accuracy:  0.929, Validation Accuracy:  0.930, Loss:  0.125
    Epoch   6 Batch   57/538 - Train Accuracy:  0.914, Validation Accuracy:  0.930, Loss:  0.124
    Epoch   6 Batch   58/538 - Train Accuracy:  0.939, Validation Accuracy:  0.924, Loss:  0.110
    Epoch   6 Batch   59/538 - Train Accuracy:  0.933, Validation Accuracy:  0.928, Loss:  0.122
    Epoch   6 Batch   60/538 - Train Accuracy:  0.943, Validation Accuracy:  0.928, Loss:  0.106
    Epoch   6 Batch   61/538 - Train Accuracy:  0.929, Validation Accuracy:  0.927, Loss:  0.112
    Epoch   6 Batch   62/538 - Train Accuracy:  0.934, Validation Accuracy:  0.931, Loss:  0.106
    Epoch   6 Batch   63/538 - Train Accuracy:  0.946, Validation Accuracy:  0.933, Loss:  0.099
    Epoch   6 Batch   64/538 - Train Accuracy:  0.938, Validation Accuracy:  0.936, Loss:  0.113
    Epoch   6 Batch   65/538 - Train Accuracy:  0.930, Validation Accuracy:  0.941, Loss:  0.116
    Epoch   6 Batch   66/538 - Train Accuracy:  0.954, Validation Accuracy:  0.936, Loss:  0.097
    Epoch   6 Batch   67/538 - Train Accuracy:  0.950, Validation Accuracy:  0.937, Loss:  0.105
    Epoch   6 Batch   68/538 - Train Accuracy:  0.935, Validation Accuracy:  0.939, Loss:  0.101
    Epoch   6 Batch   69/538 - Train Accuracy:  0.946, Validation Accuracy:  0.942, Loss:  0.110
    Epoch   6 Batch   70/538 - Train Accuracy:  0.928, Validation Accuracy:  0.938, Loss:  0.110
    Epoch   6 Batch   71/538 - Train Accuracy:  0.935, Validation Accuracy:  0.934, Loss:  0.126
    Epoch   6 Batch   72/538 - Train Accuracy:  0.945, Validation Accuracy:  0.936, Loss:  0.143
    Epoch   6 Batch   73/538 - Train Accuracy:  0.928, Validation Accuracy:  0.937, Loss:  0.112
    Epoch   6 Batch   74/538 - Train Accuracy:  0.944, Validation Accuracy:  0.939, Loss:  0.098
    Epoch   6 Batch   75/538 - Train Accuracy:  0.941, Validation Accuracy:  0.940, Loss:  0.107
    Epoch   6 Batch   76/538 - Train Accuracy:  0.934, Validation Accuracy:  0.939, Loss:  0.132
    Epoch   6 Batch   77/538 - Train Accuracy:  0.937, Validation Accuracy:  0.938, Loss:  0.107
    Epoch   6 Batch   78/538 - Train Accuracy:  0.928, Validation Accuracy:  0.936, Loss:  0.123
    Epoch   6 Batch   79/538 - Train Accuracy:  0.942, Validation Accuracy:  0.942, Loss:  0.098
    Epoch   6 Batch   80/538 - Train Accuracy:  0.938, Validation Accuracy:  0.939, Loss:  0.124
    Epoch   6 Batch   81/538 - Train Accuracy:  0.920, Validation Accuracy:  0.942, Loss:  0.113
    Epoch   6 Batch   82/538 - Train Accuracy:  0.933, Validation Accuracy:  0.942, Loss:  0.112
    Epoch   6 Batch   83/538 - Train Accuracy:  0.937, Validation Accuracy:  0.938, Loss:  0.115
    Epoch   6 Batch   84/538 - Train Accuracy:  0.933, Validation Accuracy:  0.938, Loss:  0.121
    Epoch   6 Batch   85/538 - Train Accuracy:  0.943, Validation Accuracy:  0.936, Loss:  0.100
    Epoch   6 Batch   86/538 - Train Accuracy:  0.942, Validation Accuracy:  0.931, Loss:  0.101
    Epoch   6 Batch   87/538 - Train Accuracy:  0.930, Validation Accuracy:  0.934, Loss:  0.113
    Epoch   6 Batch   88/538 - Train Accuracy:  0.940, Validation Accuracy:  0.934, Loss:  0.109
    Epoch   6 Batch   89/538 - Train Accuracy:  0.936, Validation Accuracy:  0.937, Loss:  0.103
    Epoch   6 Batch   90/538 - Train Accuracy:  0.939, Validation Accuracy:  0.935, Loss:  0.121
    Epoch   6 Batch   91/538 - Train Accuracy:  0.944, Validation Accuracy:  0.934, Loss:  0.105
    Epoch   6 Batch   92/538 - Train Accuracy:  0.927, Validation Accuracy:  0.942, Loss:  0.119
    Epoch   6 Batch   93/538 - Train Accuracy:  0.945, Validation Accuracy:  0.941, Loss:  0.096
    Epoch   6 Batch   94/538 - Train Accuracy:  0.942, Validation Accuracy:  0.943, Loss:  0.100
    Epoch   6 Batch   95/538 - Train Accuracy:  0.937, Validation Accuracy:  0.938, Loss:  0.100
    Epoch   6 Batch   96/538 - Train Accuracy:  0.945, Validation Accuracy:  0.938, Loss:  0.099
    Epoch   6 Batch   97/538 - Train Accuracy:  0.938, Validation Accuracy:  0.941, Loss:  0.094
    Epoch   6 Batch   98/538 - Train Accuracy:  0.940, Validation Accuracy:  0.942, Loss:  0.108
    Epoch   6 Batch   99/538 - Train Accuracy:  0.938, Validation Accuracy:  0.942, Loss:  0.116
    Epoch   6 Batch  100/538 - Train Accuracy:  0.945, Validation Accuracy:  0.937, Loss:  0.100
    Epoch   6 Batch  101/538 - Train Accuracy:  0.931, Validation Accuracy:  0.938, Loss:  0.127
    Epoch   6 Batch  102/538 - Train Accuracy:  0.929, Validation Accuracy:  0.938, Loss:  0.116
    Epoch   6 Batch  103/538 - Train Accuracy:  0.954, Validation Accuracy:  0.937, Loss:  0.102
    Epoch   6 Batch  104/538 - Train Accuracy:  0.950, Validation Accuracy:  0.936, Loss:  0.096
    Epoch   6 Batch  105/538 - Train Accuracy:  0.942, Validation Accuracy:  0.938, Loss:  0.091
    Epoch   6 Batch  106/538 - Train Accuracy:  0.934, Validation Accuracy:  0.935, Loss:  0.093
    Epoch   6 Batch  107/538 - Train Accuracy:  0.935, Validation Accuracy:  0.935, Loss:  0.114
    Epoch   6 Batch  108/538 - Train Accuracy:  0.950, Validation Accuracy:  0.936, Loss:  0.107
    Epoch   6 Batch  109/538 - Train Accuracy:  0.948, Validation Accuracy:  0.933, Loss:  0.088
    Epoch   6 Batch  110/538 - Train Accuracy:  0.932, Validation Accuracy:  0.935, Loss:  0.113
    Epoch   6 Batch  111/538 - Train Accuracy:  0.948, Validation Accuracy:  0.936, Loss:  0.126
    Epoch   6 Batch  112/538 - Train Accuracy:  0.943, Validation Accuracy:  0.938, Loss:  0.116
    Epoch   6 Batch  113/538 - Train Accuracy:  0.925, Validation Accuracy:  0.934, Loss:  0.125
    Epoch   6 Batch  114/538 - Train Accuracy:  0.944, Validation Accuracy:  0.933, Loss:  0.100
    Epoch   6 Batch  115/538 - Train Accuracy:  0.943, Validation Accuracy:  0.935, Loss:  0.101
    Epoch   6 Batch  116/538 - Train Accuracy:  0.933, Validation Accuracy:  0.936, Loss:  0.132
    Epoch   6 Batch  117/538 - Train Accuracy:  0.947, Validation Accuracy:  0.939, Loss:  0.109
    Epoch   6 Batch  118/538 - Train Accuracy:  0.950, Validation Accuracy:  0.938, Loss:  0.098
    Epoch   6 Batch  119/538 - Train Accuracy:  0.947, Validation Accuracy:  0.936, Loss:  0.097
    Epoch   6 Batch  120/538 - Train Accuracy:  0.956, Validation Accuracy:  0.934, Loss:  0.095
    Epoch   6 Batch  121/538 - Train Accuracy:  0.948, Validation Accuracy:  0.934, Loss:  0.097
    Epoch   6 Batch  122/538 - Train Accuracy:  0.938, Validation Accuracy:  0.934, Loss:  0.103
    Epoch   6 Batch  123/538 - Train Accuracy:  0.940, Validation Accuracy:  0.936, Loss:  0.103
    Epoch   6 Batch  124/538 - Train Accuracy:  0.950, Validation Accuracy:  0.938, Loss:  0.100
    Epoch   6 Batch  125/538 - Train Accuracy:  0.939, Validation Accuracy:  0.937, Loss:  0.121
    Epoch   6 Batch  126/538 - Train Accuracy:  0.915, Validation Accuracy:  0.939, Loss:  0.117
    Epoch   6 Batch  127/538 - Train Accuracy:  0.920, Validation Accuracy:  0.939, Loss:  0.135
    Epoch   6 Batch  128/538 - Train Accuracy:  0.942, Validation Accuracy:  0.941, Loss:  0.124
    Epoch   6 Batch  129/538 - Train Accuracy:  0.953, Validation Accuracy:  0.943, Loss:  0.087
    Epoch   6 Batch  130/538 - Train Accuracy:  0.951, Validation Accuracy:  0.943, Loss:  0.098
    Epoch   6 Batch  131/538 - Train Accuracy:  0.945, Validation Accuracy:  0.940, Loss:  0.098
    Epoch   6 Batch  132/538 - Train Accuracy:  0.944, Validation Accuracy:  0.940, Loss:  0.117
    Epoch   6 Batch  133/538 - Train Accuracy:  0.937, Validation Accuracy:  0.937, Loss:  0.103
    Epoch   6 Batch  134/538 - Train Accuracy:  0.924, Validation Accuracy:  0.939, Loss:  0.130
    Epoch   6 Batch  135/538 - Train Accuracy:  0.941, Validation Accuracy:  0.939, Loss:  0.137
    Epoch   6 Batch  136/538 - Train Accuracy:  0.935, Validation Accuracy:  0.936, Loss:  0.122
    Epoch   6 Batch  137/538 - Train Accuracy:  0.936, Validation Accuracy:  0.937, Loss:  0.132
    Epoch   6 Batch  138/538 - Train Accuracy:  0.936, Validation Accuracy:  0.938, Loss:  0.113
    Epoch   6 Batch  139/538 - Train Accuracy:  0.929, Validation Accuracy:  0.939, Loss:  0.134
    Epoch   6 Batch  140/538 - Train Accuracy:  0.934, Validation Accuracy:  0.937, Loss:  0.139
    Epoch   6 Batch  141/538 - Train Accuracy:  0.938, Validation Accuracy:  0.940, Loss:  0.140
    Epoch   6 Batch  142/538 - Train Accuracy:  0.938, Validation Accuracy:  0.937, Loss:  0.113
    Epoch   6 Batch  143/538 - Train Accuracy:  0.937, Validation Accuracy:  0.936, Loss:  0.109
    Epoch   6 Batch  144/538 - Train Accuracy:  0.942, Validation Accuracy:  0.940, Loss:  0.128
    Epoch   6 Batch  145/538 - Train Accuracy:  0.920, Validation Accuracy:  0.938, Loss:  0.137
    Epoch   6 Batch  146/538 - Train Accuracy:  0.940, Validation Accuracy:  0.938, Loss:  0.117
    Epoch   6 Batch  147/538 - Train Accuracy:  0.942, Validation Accuracy:  0.937, Loss:  0.124
    Epoch   6 Batch  148/538 - Train Accuracy:  0.928, Validation Accuracy:  0.937, Loss:  0.141
    Epoch   6 Batch  149/538 - Train Accuracy:  0.955, Validation Accuracy:  0.941, Loss:  0.103
    Epoch   6 Batch  150/538 - Train Accuracy:  0.946, Validation Accuracy:  0.939, Loss:  0.099
    Epoch   6 Batch  151/538 - Train Accuracy:  0.939, Validation Accuracy:  0.940, Loss:  0.109
    Epoch   6 Batch  152/538 - Train Accuracy:  0.954, Validation Accuracy:  0.940, Loss:  0.116
    Epoch   6 Batch  153/538 - Train Accuracy:  0.941, Validation Accuracy:  0.943, Loss:  0.117
    Epoch   6 Batch  154/538 - Train Accuracy:  0.941, Validation Accuracy:  0.941, Loss:  0.107
    Epoch   6 Batch  155/538 - Train Accuracy:  0.935, Validation Accuracy:  0.942, Loss:  0.122
    Epoch   6 Batch  156/538 - Train Accuracy:  0.946, Validation Accuracy:  0.940, Loss:  0.107
    Epoch   6 Batch  157/538 - Train Accuracy:  0.945, Validation Accuracy:  0.939, Loss:  0.105
    Epoch   6 Batch  158/538 - Train Accuracy:  0.944, Validation Accuracy:  0.939, Loss:  0.100
    Epoch   6 Batch  159/538 - Train Accuracy:  0.944, Validation Accuracy:  0.941, Loss:  0.111
    Epoch   6 Batch  160/538 - Train Accuracy:  0.937, Validation Accuracy:  0.938, Loss:  0.104
    Epoch   6 Batch  161/538 - Train Accuracy:  0.945, Validation Accuracy:  0.933, Loss:  0.111
    Epoch   6 Batch  162/538 - Train Accuracy:  0.940, Validation Accuracy:  0.937, Loss:  0.109
    Epoch   6 Batch  163/538 - Train Accuracy:  0.942, Validation Accuracy:  0.937, Loss:  0.118
    Epoch   6 Batch  164/538 - Train Accuracy:  0.938, Validation Accuracy:  0.937, Loss:  0.125
    Epoch   6 Batch  165/538 - Train Accuracy:  0.943, Validation Accuracy:  0.935, Loss:  0.096
    Epoch   6 Batch  166/538 - Train Accuracy:  0.949, Validation Accuracy:  0.936, Loss:  0.104
    Epoch   6 Batch  167/538 - Train Accuracy:  0.938, Validation Accuracy:  0.937, Loss:  0.125
    Epoch   6 Batch  168/538 - Train Accuracy:  0.917, Validation Accuracy:  0.937, Loss:  0.140
    Epoch   6 Batch  169/538 - Train Accuracy:  0.963, Validation Accuracy:  0.933, Loss:  0.084
    Epoch   6 Batch  170/538 - Train Accuracy:  0.938, Validation Accuracy:  0.934, Loss:  0.111
    Epoch   6 Batch  171/538 - Train Accuracy:  0.934, Validation Accuracy:  0.934, Loss:  0.107
    Epoch   6 Batch  172/538 - Train Accuracy:  0.937, Validation Accuracy:  0.932, Loss:  0.112
    Epoch   6 Batch  173/538 - Train Accuracy:  0.946, Validation Accuracy:  0.930, Loss:  0.093
    Epoch   6 Batch  174/538 - Train Accuracy:  0.943, Validation Accuracy:  0.930, Loss:  0.107
    Epoch   6 Batch  175/538 - Train Accuracy:  0.945, Validation Accuracy:  0.928, Loss:  0.103
    Epoch   6 Batch  176/538 - Train Accuracy:  0.941, Validation Accuracy:  0.930, Loss:  0.123
    Epoch   6 Batch  177/538 - Train Accuracy:  0.947, Validation Accuracy:  0.930, Loss:  0.108
    Epoch   6 Batch  178/538 - Train Accuracy:  0.931, Validation Accuracy:  0.934, Loss:  0.115
    Epoch   6 Batch  179/538 - Train Accuracy:  0.946, Validation Accuracy:  0.936, Loss:  0.106
    Epoch   6 Batch  180/538 - Train Accuracy:  0.945, Validation Accuracy:  0.935, Loss:  0.116
    Epoch   6 Batch  181/538 - Train Accuracy:  0.935, Validation Accuracy:  0.936, Loss:  0.114
    Epoch   6 Batch  182/538 - Train Accuracy:  0.951, Validation Accuracy:  0.938, Loss:  0.096
    Epoch   6 Batch  183/538 - Train Accuracy:  0.950, Validation Accuracy:  0.938, Loss:  0.089
    Epoch   6 Batch  184/538 - Train Accuracy:  0.939, Validation Accuracy:  0.938, Loss:  0.112
    Epoch   6 Batch  185/538 - Train Accuracy:  0.958, Validation Accuracy:  0.939, Loss:  0.094
    Epoch   6 Batch  186/538 - Train Accuracy:  0.945, Validation Accuracy:  0.939, Loss:  0.112
    Epoch   6 Batch  187/538 - Train Accuracy:  0.957, Validation Accuracy:  0.939, Loss:  0.111
    Epoch   6 Batch  188/538 - Train Accuracy:  0.938, Validation Accuracy:  0.941, Loss:  0.100
    Epoch   6 Batch  189/538 - Train Accuracy:  0.947, Validation Accuracy:  0.941, Loss:  0.104
    Epoch   6 Batch  190/538 - Train Accuracy:  0.931, Validation Accuracy:  0.938, Loss:  0.136
    Epoch   6 Batch  191/538 - Train Accuracy:  0.941, Validation Accuracy:  0.940, Loss:  0.100
    Epoch   6 Batch  192/538 - Train Accuracy:  0.947, Validation Accuracy:  0.932, Loss:  0.112
    Epoch   6 Batch  193/538 - Train Accuracy:  0.935, Validation Accuracy:  0.933, Loss:  0.103
    Epoch   6 Batch  194/538 - Train Accuracy:  0.935, Validation Accuracy:  0.931, Loss:  0.127
    Epoch   6 Batch  195/538 - Train Accuracy:  0.940, Validation Accuracy:  0.930, Loss:  0.107
    Epoch   6 Batch  196/538 - Train Accuracy:  0.934, Validation Accuracy:  0.933, Loss:  0.101
    Epoch   6 Batch  197/538 - Train Accuracy:  0.941, Validation Accuracy:  0.937, Loss:  0.116
    Epoch   6 Batch  198/538 - Train Accuracy:  0.940, Validation Accuracy:  0.939, Loss:  0.110
    Epoch   6 Batch  199/538 - Train Accuracy:  0.944, Validation Accuracy:  0.939, Loss:  0.128
    Epoch   6 Batch  200/538 - Train Accuracy:  0.957, Validation Accuracy:  0.939, Loss:  0.095
    Epoch   6 Batch  201/538 - Train Accuracy:  0.945, Validation Accuracy:  0.941, Loss:  0.106
    Epoch   6 Batch  202/538 - Train Accuracy:  0.942, Validation Accuracy:  0.940, Loss:  0.101
    Epoch   6 Batch  203/538 - Train Accuracy:  0.934, Validation Accuracy:  0.937, Loss:  0.120
    Epoch   6 Batch  204/538 - Train Accuracy:  0.929, Validation Accuracy:  0.937, Loss:  0.126
    Epoch   6 Batch  205/538 - Train Accuracy:  0.942, Validation Accuracy:  0.938, Loss:  0.097
    Epoch   6 Batch  206/538 - Train Accuracy:  0.941, Validation Accuracy:  0.940, Loss:  0.106
    Epoch   6 Batch  207/538 - Train Accuracy:  0.950, Validation Accuracy:  0.939, Loss:  0.114
    Epoch   6 Batch  208/538 - Train Accuracy:  0.936, Validation Accuracy:  0.937, Loss:  0.130
    Epoch   6 Batch  209/538 - Train Accuracy:  0.947, Validation Accuracy:  0.937, Loss:  0.101
    Epoch   6 Batch  210/538 - Train Accuracy:  0.946, Validation Accuracy:  0.940, Loss:  0.115
    Epoch   6 Batch  211/538 - Train Accuracy:  0.948, Validation Accuracy:  0.937, Loss:  0.114
    Epoch   6 Batch  212/538 - Train Accuracy:  0.922, Validation Accuracy:  0.933, Loss:  0.112
    Epoch   6 Batch  213/538 - Train Accuracy:  0.955, Validation Accuracy:  0.935, Loss:  0.094
    Epoch   6 Batch  214/538 - Train Accuracy:  0.951, Validation Accuracy:  0.937, Loss:  0.103
    Epoch   6 Batch  215/538 - Train Accuracy:  0.945, Validation Accuracy:  0.939, Loss:  0.095
    Epoch   6 Batch  216/538 - Train Accuracy:  0.951, Validation Accuracy:  0.939, Loss:  0.105
    Epoch   6 Batch  217/538 - Train Accuracy:  0.947, Validation Accuracy:  0.941, Loss:  0.104
    Epoch   6 Batch  218/538 - Train Accuracy:  0.947, Validation Accuracy:  0.936, Loss:  0.105
    Epoch   6 Batch  219/538 - Train Accuracy:  0.931, Validation Accuracy:  0.936, Loss:  0.123
    Epoch   6 Batch  220/538 - Train Accuracy:  0.928, Validation Accuracy:  0.934, Loss:  0.112
    Epoch   6 Batch  221/538 - Train Accuracy:  0.947, Validation Accuracy:  0.936, Loss:  0.098
    Epoch   6 Batch  222/538 - Train Accuracy:  0.925, Validation Accuracy:  0.934, Loss:  0.104
    Epoch   6 Batch  223/538 - Train Accuracy:  0.928, Validation Accuracy:  0.933, Loss:  0.117
    Epoch   6 Batch  224/538 - Train Accuracy:  0.939, Validation Accuracy:  0.931, Loss:  0.119
    Epoch   6 Batch  225/538 - Train Accuracy:  0.940, Validation Accuracy:  0.938, Loss:  0.109
    Epoch   6 Batch  226/538 - Train Accuracy:  0.927, Validation Accuracy:  0.935, Loss:  0.110
    Epoch   6 Batch  227/538 - Train Accuracy:  0.940, Validation Accuracy:  0.935, Loss:  0.108
    Epoch   6 Batch  228/538 - Train Accuracy:  0.919, Validation Accuracy:  0.936, Loss:  0.105
    Epoch   6 Batch  229/538 - Train Accuracy:  0.938, Validation Accuracy:  0.937, Loss:  0.106
    Epoch   6 Batch  230/538 - Train Accuracy:  0.943, Validation Accuracy:  0.937, Loss:  0.108
    Epoch   6 Batch  231/538 - Train Accuracy:  0.953, Validation Accuracy:  0.939, Loss:  0.104
    Epoch   6 Batch  232/538 - Train Accuracy:  0.940, Validation Accuracy:  0.939, Loss:  0.092
    Epoch   6 Batch  233/538 - Train Accuracy:  0.946, Validation Accuracy:  0.939, Loss:  0.117
    Epoch   6 Batch  234/538 - Train Accuracy:  0.944, Validation Accuracy:  0.939, Loss:  0.101
    Epoch   6 Batch  235/538 - Train Accuracy:  0.950, Validation Accuracy:  0.939, Loss:  0.095
    Epoch   6 Batch  236/538 - Train Accuracy:  0.940, Validation Accuracy:  0.941, Loss:  0.105
    Epoch   6 Batch  237/538 - Train Accuracy:  0.949, Validation Accuracy:  0.941, Loss:  0.097
    Epoch   6 Batch  238/538 - Train Accuracy:  0.950, Validation Accuracy:  0.936, Loss:  0.108
    Epoch   6 Batch  239/538 - Train Accuracy:  0.938, Validation Accuracy:  0.936, Loss:  0.128
    Epoch   6 Batch  240/538 - Train Accuracy:  0.928, Validation Accuracy:  0.937, Loss:  0.118
    Epoch   6 Batch  241/538 - Train Accuracy:  0.922, Validation Accuracy:  0.937, Loss:  0.121
    Epoch   6 Batch  242/538 - Train Accuracy:  0.951, Validation Accuracy:  0.937, Loss:  0.101
    Epoch   6 Batch  243/538 - Train Accuracy:  0.955, Validation Accuracy:  0.940, Loss:  0.092
    Epoch   6 Batch  244/538 - Train Accuracy:  0.936, Validation Accuracy:  0.942, Loss:  0.102
    Epoch   6 Batch  245/538 - Train Accuracy:  0.926, Validation Accuracy:  0.936, Loss:  0.135
    Epoch   6 Batch  246/538 - Train Accuracy:  0.953, Validation Accuracy:  0.933, Loss:  0.087
    Epoch   6 Batch  247/538 - Train Accuracy:  0.934, Validation Accuracy:  0.934, Loss:  0.102
    Epoch   6 Batch  248/538 - Train Accuracy:  0.950, Validation Accuracy:  0.934, Loss:  0.110
    Epoch   6 Batch  249/538 - Train Accuracy:  0.951, Validation Accuracy:  0.937, Loss:  0.086
    Epoch   6 Batch  250/538 - Train Accuracy:  0.958, Validation Accuracy:  0.937, Loss:  0.096
    Epoch   6 Batch  251/538 - Train Accuracy:  0.945, Validation Accuracy:  0.933, Loss:  0.099
    Epoch   6 Batch  252/538 - Train Accuracy:  0.941, Validation Accuracy:  0.934, Loss:  0.092
    Epoch   6 Batch  253/538 - Train Accuracy:  0.927, Validation Accuracy:  0.934, Loss:  0.096
    Epoch   6 Batch  254/538 - Train Accuracy:  0.934, Validation Accuracy:  0.937, Loss:  0.102
    Epoch   6 Batch  255/538 - Train Accuracy:  0.955, Validation Accuracy:  0.937, Loss:  0.093
    Epoch   6 Batch  256/538 - Train Accuracy:  0.939, Validation Accuracy:  0.939, Loss:  0.117
    Epoch   6 Batch  257/538 - Train Accuracy:  0.945, Validation Accuracy:  0.941, Loss:  0.103
    Epoch   6 Batch  258/538 - Train Accuracy:  0.953, Validation Accuracy:  0.940, Loss:  0.104
    Epoch   6 Batch  259/538 - Train Accuracy:  0.948, Validation Accuracy:  0.939, Loss:  0.102
    Epoch   6 Batch  260/538 - Train Accuracy:  0.914, Validation Accuracy:  0.934, Loss:  0.113
    Epoch   6 Batch  261/538 - Train Accuracy:  0.939, Validation Accuracy:  0.933, Loss:  0.113
    Epoch   6 Batch  262/538 - Train Accuracy:  0.946, Validation Accuracy:  0.936, Loss:  0.109
    Epoch   6 Batch  263/538 - Train Accuracy:  0.927, Validation Accuracy:  0.939, Loss:  0.102
    Epoch   6 Batch  264/538 - Train Accuracy:  0.933, Validation Accuracy:  0.939, Loss:  0.103
    Epoch   6 Batch  265/538 - Train Accuracy:  0.938, Validation Accuracy:  0.939, Loss:  0.130
    Epoch   6 Batch  266/538 - Train Accuracy:  0.939, Validation Accuracy:  0.940, Loss:  0.110
    Epoch   6 Batch  267/538 - Train Accuracy:  0.935, Validation Accuracy:  0.938, Loss:  0.107
    Epoch   6 Batch  268/538 - Train Accuracy:  0.954, Validation Accuracy:  0.939, Loss:  0.093
    Epoch   6 Batch  269/538 - Train Accuracy:  0.934, Validation Accuracy:  0.939, Loss:  0.126
    Epoch   6 Batch  270/538 - Train Accuracy:  0.935, Validation Accuracy:  0.942, Loss:  0.105
    Epoch   6 Batch  271/538 - Train Accuracy:  0.943, Validation Accuracy:  0.945, Loss:  0.097
    Epoch   6 Batch  272/538 - Train Accuracy:  0.942, Validation Accuracy:  0.946, Loss:  0.110
    Epoch   6 Batch  273/538 - Train Accuracy:  0.928, Validation Accuracy:  0.942, Loss:  0.122
    Epoch   6 Batch  274/538 - Train Accuracy:  0.914, Validation Accuracy:  0.941, Loss:  0.110
    Epoch   6 Batch  275/538 - Train Accuracy:  0.944, Validation Accuracy:  0.938, Loss:  0.125
    Epoch   6 Batch  276/538 - Train Accuracy:  0.935, Validation Accuracy:  0.941, Loss:  0.110
    Epoch   6 Batch  277/538 - Train Accuracy:  0.932, Validation Accuracy:  0.944, Loss:  0.100
    Epoch   6 Batch  278/538 - Train Accuracy:  0.952, Validation Accuracy:  0.943, Loss:  0.098
    Epoch   6 Batch  279/538 - Train Accuracy:  0.935, Validation Accuracy:  0.943, Loss:  0.106
    Epoch   6 Batch  280/538 - Train Accuracy:  0.944, Validation Accuracy:  0.944, Loss:  0.105
    Epoch   6 Batch  281/538 - Train Accuracy:  0.939, Validation Accuracy:  0.946, Loss:  0.135
    Epoch   6 Batch  282/538 - Train Accuracy:  0.942, Validation Accuracy:  0.941, Loss:  0.117
    Epoch   6 Batch  283/538 - Train Accuracy:  0.947, Validation Accuracy:  0.939, Loss:  0.103
    Epoch   6 Batch  284/538 - Train Accuracy:  0.939, Validation Accuracy:  0.939, Loss:  0.117
    Epoch   6 Batch  285/538 - Train Accuracy:  0.948, Validation Accuracy:  0.931, Loss:  0.093
    Epoch   6 Batch  286/538 - Train Accuracy:  0.935, Validation Accuracy:  0.929, Loss:  0.131
    Epoch   6 Batch  287/538 - Train Accuracy:  0.947, Validation Accuracy:  0.931, Loss:  0.102
    Epoch   6 Batch  288/538 - Train Accuracy:  0.936, Validation Accuracy:  0.931, Loss:  0.109
    Epoch   6 Batch  289/538 - Train Accuracy:  0.942, Validation Accuracy:  0.935, Loss:  0.091
    Epoch   6 Batch  290/538 - Train Accuracy:  0.962, Validation Accuracy:  0.936, Loss:  0.090
    Epoch   6 Batch  291/538 - Train Accuracy:  0.945, Validation Accuracy:  0.937, Loss:  0.112
    Epoch   6 Batch  292/538 - Train Accuracy:  0.951, Validation Accuracy:  0.937, Loss:  0.087
    Epoch   6 Batch  293/538 - Train Accuracy:  0.933, Validation Accuracy:  0.934, Loss:  0.105
    Epoch   6 Batch  294/538 - Train Accuracy:  0.946, Validation Accuracy:  0.931, Loss:  0.101
    Epoch   6 Batch  295/538 - Train Accuracy:  0.943, Validation Accuracy:  0.928, Loss:  0.100
    Epoch   6 Batch  296/538 - Train Accuracy:  0.935, Validation Accuracy:  0.926, Loss:  0.113
    Epoch   6 Batch  297/538 - Train Accuracy:  0.952, Validation Accuracy:  0.929, Loss:  0.098
    Epoch   6 Batch  298/538 - Train Accuracy:  0.944, Validation Accuracy:  0.931, Loss:  0.106
    Epoch   6 Batch  299/538 - Train Accuracy:  0.943, Validation Accuracy:  0.936, Loss:  0.123
    Epoch   6 Batch  300/538 - Train Accuracy:  0.939, Validation Accuracy:  0.937, Loss:  0.109
    Epoch   6 Batch  301/538 - Train Accuracy:  0.932, Validation Accuracy:  0.938, Loss:  0.126
    Epoch   6 Batch  302/538 - Train Accuracy:  0.951, Validation Accuracy:  0.941, Loss:  0.100
    Epoch   6 Batch  303/538 - Train Accuracy:  0.946, Validation Accuracy:  0.939, Loss:  0.105
    Epoch   6 Batch  304/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.114
    Epoch   6 Batch  305/538 - Train Accuracy:  0.952, Validation Accuracy:  0.942, Loss:  0.091
    Epoch   6 Batch  306/538 - Train Accuracy:  0.941, Validation Accuracy:  0.942, Loss:  0.096
    Epoch   6 Batch  307/538 - Train Accuracy:  0.959, Validation Accuracy:  0.942, Loss:  0.099
    Epoch   6 Batch  308/538 - Train Accuracy:  0.941, Validation Accuracy:  0.939, Loss:  0.102
    Epoch   6 Batch  309/538 - Train Accuracy:  0.940, Validation Accuracy:  0.941, Loss:  0.085
    Epoch   6 Batch  310/538 - Train Accuracy:  0.961, Validation Accuracy:  0.943, Loss:  0.106
    Epoch   6 Batch  311/538 - Train Accuracy:  0.935, Validation Accuracy:  0.943, Loss:  0.103
    Epoch   6 Batch  312/538 - Train Accuracy:  0.948, Validation Accuracy:  0.941, Loss:  0.097
    Epoch   6 Batch  313/538 - Train Accuracy:  0.937, Validation Accuracy:  0.944, Loss:  0.105
    Epoch   6 Batch  314/538 - Train Accuracy:  0.946, Validation Accuracy:  0.944, Loss:  0.103
    Epoch   6 Batch  315/538 - Train Accuracy:  0.926, Validation Accuracy:  0.941, Loss:  0.098
    Epoch   6 Batch  316/538 - Train Accuracy:  0.940, Validation Accuracy:  0.941, Loss:  0.090
    Epoch   6 Batch  317/538 - Train Accuracy:  0.942, Validation Accuracy:  0.938, Loss:  0.105
    Epoch   6 Batch  318/538 - Train Accuracy:  0.930, Validation Accuracy:  0.943, Loss:  0.112
    Epoch   6 Batch  319/538 - Train Accuracy:  0.957, Validation Accuracy:  0.943, Loss:  0.103
    Epoch   6 Batch  320/538 - Train Accuracy:  0.948, Validation Accuracy:  0.942, Loss:  0.104
    Epoch   6 Batch  321/538 - Train Accuracy:  0.945, Validation Accuracy:  0.941, Loss:  0.090
    Epoch   6 Batch  322/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.105
    Epoch   6 Batch  323/538 - Train Accuracy:  0.945, Validation Accuracy:  0.943, Loss:  0.098
    Epoch   6 Batch  324/538 - Train Accuracy:  0.957, Validation Accuracy:  0.941, Loss:  0.105
    Epoch   6 Batch  325/538 - Train Accuracy:  0.944, Validation Accuracy:  0.941, Loss:  0.110
    Epoch   6 Batch  326/538 - Train Accuracy:  0.946, Validation Accuracy:  0.945, Loss:  0.090
    Epoch   6 Batch  327/538 - Train Accuracy:  0.934, Validation Accuracy:  0.942, Loss:  0.117
    Epoch   6 Batch  328/538 - Train Accuracy:  0.959, Validation Accuracy:  0.943, Loss:  0.084
    Epoch   6 Batch  329/538 - Train Accuracy:  0.950, Validation Accuracy:  0.943, Loss:  0.090
    Epoch   6 Batch  330/538 - Train Accuracy:  0.951, Validation Accuracy:  0.943, Loss:  0.098
    Epoch   6 Batch  331/538 - Train Accuracy:  0.937, Validation Accuracy:  0.941, Loss:  0.097
    Epoch   6 Batch  332/538 - Train Accuracy:  0.953, Validation Accuracy:  0.941, Loss:  0.101
    Epoch   6 Batch  333/538 - Train Accuracy:  0.947, Validation Accuracy:  0.946, Loss:  0.108
    Epoch   6 Batch  334/538 - Train Accuracy:  0.954, Validation Accuracy:  0.942, Loss:  0.097
    Epoch   6 Batch  335/538 - Train Accuracy:  0.951, Validation Accuracy:  0.934, Loss:  0.103
    Epoch   6 Batch  336/538 - Train Accuracy:  0.936, Validation Accuracy:  0.934, Loss:  0.103
    Epoch   6 Batch  337/538 - Train Accuracy:  0.940, Validation Accuracy:  0.928, Loss:  0.108
    Epoch   6 Batch  338/538 - Train Accuracy:  0.944, Validation Accuracy:  0.928, Loss:  0.101
    Epoch   6 Batch  339/538 - Train Accuracy:  0.937, Validation Accuracy:  0.933, Loss:  0.103
    Epoch   6 Batch  340/538 - Train Accuracy:  0.931, Validation Accuracy:  0.936, Loss:  0.101
    Epoch   6 Batch  341/538 - Train Accuracy:  0.940, Validation Accuracy:  0.941, Loss:  0.102
    Epoch   6 Batch  342/538 - Train Accuracy:  0.945, Validation Accuracy:  0.940, Loss:  0.102
    Epoch   6 Batch  343/538 - Train Accuracy:  0.956, Validation Accuracy:  0.938, Loss:  0.097
    Epoch   6 Batch  344/538 - Train Accuracy:  0.946, Validation Accuracy:  0.942, Loss:  0.100
    Epoch   6 Batch  345/538 - Train Accuracy:  0.940, Validation Accuracy:  0.942, Loss:  0.101
    Epoch   6 Batch  346/538 - Train Accuracy:  0.927, Validation Accuracy:  0.939, Loss:  0.120
    Epoch   6 Batch  347/538 - Train Accuracy:  0.951, Validation Accuracy:  0.936, Loss:  0.096
    Epoch   6 Batch  348/538 - Train Accuracy:  0.935, Validation Accuracy:  0.941, Loss:  0.102
    Epoch   6 Batch  349/538 - Train Accuracy:  0.951, Validation Accuracy:  0.943, Loss:  0.091
    Epoch   6 Batch  350/538 - Train Accuracy:  0.935, Validation Accuracy:  0.943, Loss:  0.124
    Epoch   6 Batch  351/538 - Train Accuracy:  0.939, Validation Accuracy:  0.943, Loss:  0.112
    Epoch   6 Batch  352/538 - Train Accuracy:  0.931, Validation Accuracy:  0.945, Loss:  0.125
    Epoch   6 Batch  353/538 - Train Accuracy:  0.932, Validation Accuracy:  0.949, Loss:  0.107
    Epoch   6 Batch  354/538 - Train Accuracy:  0.941, Validation Accuracy:  0.945, Loss:  0.099
    Epoch   6 Batch  355/538 - Train Accuracy:  0.946, Validation Accuracy:  0.948, Loss:  0.106
    Epoch   6 Batch  356/538 - Train Accuracy:  0.943, Validation Accuracy:  0.942, Loss:  0.097
    Epoch   6 Batch  357/538 - Train Accuracy:  0.949, Validation Accuracy:  0.943, Loss:  0.105
    Epoch   6 Batch  358/538 - Train Accuracy:  0.950, Validation Accuracy:  0.942, Loss:  0.090
    Epoch   6 Batch  359/538 - Train Accuracy:  0.930, Validation Accuracy:  0.943, Loss:  0.103
    Epoch   6 Batch  360/538 - Train Accuracy:  0.935, Validation Accuracy:  0.944, Loss:  0.114
    Epoch   6 Batch  361/538 - Train Accuracy:  0.945, Validation Accuracy:  0.942, Loss:  0.103
    Epoch   6 Batch  362/538 - Train Accuracy:  0.955, Validation Accuracy:  0.944, Loss:  0.098
    Epoch   6 Batch  363/538 - Train Accuracy:  0.934, Validation Accuracy:  0.940, Loss:  0.101
    Epoch   6 Batch  364/538 - Train Accuracy:  0.930, Validation Accuracy:  0.942, Loss:  0.128
    Epoch   6 Batch  365/538 - Train Accuracy:  0.925, Validation Accuracy:  0.947, Loss:  0.098
    Epoch   6 Batch  366/538 - Train Accuracy:  0.945, Validation Accuracy:  0.946, Loss:  0.109
    Epoch   6 Batch  367/538 - Train Accuracy:  0.949, Validation Accuracy:  0.944, Loss:  0.080
    Epoch   6 Batch  368/538 - Train Accuracy:  0.953, Validation Accuracy:  0.941, Loss:  0.080
    Epoch   6 Batch  369/538 - Train Accuracy:  0.931, Validation Accuracy:  0.941, Loss:  0.094
    Epoch   6 Batch  370/538 - Train Accuracy:  0.939, Validation Accuracy:  0.938, Loss:  0.104
    Epoch   6 Batch  371/538 - Train Accuracy:  0.958, Validation Accuracy:  0.940, Loss:  0.101
    Epoch   6 Batch  372/538 - Train Accuracy:  0.956, Validation Accuracy:  0.942, Loss:  0.092
    Epoch   6 Batch  373/538 - Train Accuracy:  0.946, Validation Accuracy:  0.941, Loss:  0.088
    Epoch   6 Batch  374/538 - Train Accuracy:  0.947, Validation Accuracy:  0.942, Loss:  0.099
    Epoch   6 Batch  375/538 - Train Accuracy:  0.948, Validation Accuracy:  0.943, Loss:  0.097
    Epoch   6 Batch  376/538 - Train Accuracy:  0.948, Validation Accuracy:  0.943, Loss:  0.113
    Epoch   6 Batch  377/538 - Train Accuracy:  0.957, Validation Accuracy:  0.944, Loss:  0.090
    Epoch   6 Batch  378/538 - Train Accuracy:  0.955, Validation Accuracy:  0.942, Loss:  0.092
    Epoch   6 Batch  379/538 - Train Accuracy:  0.960, Validation Accuracy:  0.940, Loss:  0.104
    Epoch   6 Batch  380/538 - Train Accuracy:  0.943, Validation Accuracy:  0.937, Loss:  0.097
    Epoch   6 Batch  381/538 - Train Accuracy:  0.945, Validation Accuracy:  0.940, Loss:  0.090
    Epoch   6 Batch  382/538 - Train Accuracy:  0.944, Validation Accuracy:  0.940, Loss:  0.109
    Epoch   6 Batch  383/538 - Train Accuracy:  0.938, Validation Accuracy:  0.938, Loss:  0.091
    Epoch   6 Batch  384/538 - Train Accuracy:  0.935, Validation Accuracy:  0.936, Loss:  0.109
    Epoch   6 Batch  385/538 - Train Accuracy:  0.939, Validation Accuracy:  0.938, Loss:  0.101
    Epoch   6 Batch  386/538 - Train Accuracy:  0.944, Validation Accuracy:  0.938, Loss:  0.104
    Epoch   6 Batch  387/538 - Train Accuracy:  0.938, Validation Accuracy:  0.940, Loss:  0.094
    Epoch   6 Batch  388/538 - Train Accuracy:  0.937, Validation Accuracy:  0.942, Loss:  0.108
    Epoch   6 Batch  389/538 - Train Accuracy:  0.931, Validation Accuracy:  0.943, Loss:  0.133
    Epoch   6 Batch  390/538 - Train Accuracy:  0.949, Validation Accuracy:  0.944, Loss:  0.085
    Epoch   6 Batch  391/538 - Train Accuracy:  0.944, Validation Accuracy:  0.939, Loss:  0.098
    Epoch   6 Batch  392/538 - Train Accuracy:  0.942, Validation Accuracy:  0.939, Loss:  0.084
    Epoch   6 Batch  393/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.087
    Epoch   6 Batch  394/538 - Train Accuracy:  0.926, Validation Accuracy:  0.944, Loss:  0.110
    Epoch   6 Batch  395/538 - Train Accuracy:  0.936, Validation Accuracy:  0.941, Loss:  0.100
    Epoch   6 Batch  396/538 - Train Accuracy:  0.943, Validation Accuracy:  0.942, Loss:  0.100
    Epoch   6 Batch  397/538 - Train Accuracy:  0.944, Validation Accuracy:  0.942, Loss:  0.104
    Epoch   6 Batch  398/538 - Train Accuracy:  0.938, Validation Accuracy:  0.945, Loss:  0.105
    Epoch   6 Batch  399/538 - Train Accuracy:  0.939, Validation Accuracy:  0.949, Loss:  0.107
    Epoch   6 Batch  400/538 - Train Accuracy:  0.954, Validation Accuracy:  0.943, Loss:  0.107
    Epoch   6 Batch  401/538 - Train Accuracy:  0.955, Validation Accuracy:  0.942, Loss:  0.100
    Epoch   6 Batch  402/538 - Train Accuracy:  0.944, Validation Accuracy:  0.942, Loss:  0.095
    Epoch   6 Batch  403/538 - Train Accuracy:  0.953, Validation Accuracy:  0.942, Loss:  0.104
    Epoch   6 Batch  404/538 - Train Accuracy:  0.945, Validation Accuracy:  0.942, Loss:  0.095
    Epoch   6 Batch  405/538 - Train Accuracy:  0.944, Validation Accuracy:  0.940, Loss:  0.102
    Epoch   6 Batch  406/538 - Train Accuracy:  0.934, Validation Accuracy:  0.937, Loss:  0.105
    Epoch   6 Batch  407/538 - Train Accuracy:  0.956, Validation Accuracy:  0.943, Loss:  0.111
    Epoch   6 Batch  408/538 - Train Accuracy:  0.932, Validation Accuracy:  0.942, Loss:  0.125
    Epoch   6 Batch  409/538 - Train Accuracy:  0.934, Validation Accuracy:  0.941, Loss:  0.100
    Epoch   6 Batch  410/538 - Train Accuracy:  0.949, Validation Accuracy:  0.943, Loss:  0.101
    Epoch   6 Batch  411/538 - Train Accuracy:  0.956, Validation Accuracy:  0.943, Loss:  0.093
    Epoch   6 Batch  412/538 - Train Accuracy:  0.953, Validation Accuracy:  0.946, Loss:  0.091
    Epoch   6 Batch  413/538 - Train Accuracy:  0.951, Validation Accuracy:  0.943, Loss:  0.102
    Epoch   6 Batch  414/538 - Train Accuracy:  0.927, Validation Accuracy:  0.940, Loss:  0.134
    Epoch   6 Batch  415/538 - Train Accuracy:  0.926, Validation Accuracy:  0.938, Loss:  0.111
    Epoch   6 Batch  416/538 - Train Accuracy:  0.947, Validation Accuracy:  0.936, Loss:  0.095
    Epoch   6 Batch  417/538 - Train Accuracy:  0.941, Validation Accuracy:  0.934, Loss:  0.097
    Epoch   6 Batch  418/538 - Train Accuracy:  0.946, Validation Accuracy:  0.937, Loss:  0.120
    Epoch   6 Batch  419/538 - Train Accuracy:  0.957, Validation Accuracy:  0.932, Loss:  0.089
    Epoch   6 Batch  420/538 - Train Accuracy:  0.949, Validation Accuracy:  0.930, Loss:  0.093
    Epoch   6 Batch  421/538 - Train Accuracy:  0.957, Validation Accuracy:  0.931, Loss:  0.092
    Epoch   6 Batch  422/538 - Train Accuracy:  0.939, Validation Accuracy:  0.934, Loss:  0.104
    Epoch   6 Batch  423/538 - Train Accuracy:  0.936, Validation Accuracy:  0.943, Loss:  0.106
    Epoch   6 Batch  424/538 - Train Accuracy:  0.946, Validation Accuracy:  0.944, Loss:  0.099
    Epoch   6 Batch  425/538 - Train Accuracy:  0.938, Validation Accuracy:  0.946, Loss:  0.114
    Epoch   6 Batch  426/538 - Train Accuracy:  0.946, Validation Accuracy:  0.948, Loss:  0.104
    Epoch   6 Batch  427/538 - Train Accuracy:  0.939, Validation Accuracy:  0.946, Loss:  0.108
    Epoch   6 Batch  428/538 - Train Accuracy:  0.950, Validation Accuracy:  0.949, Loss:  0.086
    Epoch   6 Batch  429/538 - Train Accuracy:  0.953, Validation Accuracy:  0.947, Loss:  0.098
    Epoch   6 Batch  430/538 - Train Accuracy:  0.928, Validation Accuracy:  0.951, Loss:  0.098
    Epoch   6 Batch  431/538 - Train Accuracy:  0.937, Validation Accuracy:  0.948, Loss:  0.093
    Epoch   6 Batch  432/538 - Train Accuracy:  0.955, Validation Accuracy:  0.952, Loss:  0.109
    Epoch   6 Batch  433/538 - Train Accuracy:  0.936, Validation Accuracy:  0.955, Loss:  0.122
    Epoch   6 Batch  434/538 - Train Accuracy:  0.946, Validation Accuracy:  0.954, Loss:  0.097
    Epoch   6 Batch  435/538 - Train Accuracy:  0.939, Validation Accuracy:  0.956, Loss:  0.105
    Epoch   6 Batch  436/538 - Train Accuracy:  0.932, Validation Accuracy:  0.962, Loss:  0.111
    Epoch   6 Batch  437/538 - Train Accuracy:  0.957, Validation Accuracy:  0.957, Loss:  0.097
    Epoch   6 Batch  438/538 - Train Accuracy:  0.937, Validation Accuracy:  0.952, Loss:  0.090
    Epoch   6 Batch  439/538 - Train Accuracy:  0.957, Validation Accuracy:  0.948, Loss:  0.086
    Epoch   6 Batch  440/538 - Train Accuracy:  0.945, Validation Accuracy:  0.945, Loss:  0.114
    Epoch   6 Batch  441/538 - Train Accuracy:  0.936, Validation Accuracy:  0.944, Loss:  0.114
    Epoch   6 Batch  442/538 - Train Accuracy:  0.945, Validation Accuracy:  0.944, Loss:  0.093
    Epoch   6 Batch  443/538 - Train Accuracy:  0.943, Validation Accuracy:  0.945, Loss:  0.105
    Epoch   6 Batch  444/538 - Train Accuracy:  0.958, Validation Accuracy:  0.945, Loss:  0.089
    Epoch   6 Batch  445/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.083
    Epoch   6 Batch  446/538 - Train Accuracy:  0.954, Validation Accuracy:  0.944, Loss:  0.092
    Epoch   6 Batch  447/538 - Train Accuracy:  0.930, Validation Accuracy:  0.946, Loss:  0.105
    Epoch   6 Batch  448/538 - Train Accuracy:  0.950, Validation Accuracy:  0.948, Loss:  0.085
    Epoch   6 Batch  449/538 - Train Accuracy:  0.955, Validation Accuracy:  0.948, Loss:  0.102
    Epoch   6 Batch  450/538 - Train Accuracy:  0.922, Validation Accuracy:  0.944, Loss:  0.133
    Epoch   6 Batch  451/538 - Train Accuracy:  0.940, Validation Accuracy:  0.944, Loss:  0.094
    Epoch   6 Batch  452/538 - Train Accuracy:  0.953, Validation Accuracy:  0.942, Loss:  0.088
    Epoch   6 Batch  453/538 - Train Accuracy:  0.939, Validation Accuracy:  0.939, Loss:  0.101
    Epoch   6 Batch  454/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.098
    Epoch   6 Batch  455/538 - Train Accuracy:  0.945, Validation Accuracy:  0.942, Loss:  0.108
    Epoch   6 Batch  456/538 - Train Accuracy:  0.956, Validation Accuracy:  0.947, Loss:  0.121
    Epoch   6 Batch  457/538 - Train Accuracy:  0.951, Validation Accuracy:  0.946, Loss:  0.102
    Epoch   6 Batch  458/538 - Train Accuracy:  0.956, Validation Accuracy:  0.946, Loss:  0.093
    Epoch   6 Batch  459/538 - Train Accuracy:  0.946, Validation Accuracy:  0.947, Loss:  0.091
    Epoch   6 Batch  460/538 - Train Accuracy:  0.946, Validation Accuracy:  0.949, Loss:  0.108
    Epoch   6 Batch  461/538 - Train Accuracy:  0.956, Validation Accuracy:  0.945, Loss:  0.097
    Epoch   6 Batch  462/538 - Train Accuracy:  0.940, Validation Accuracy:  0.941, Loss:  0.099
    Epoch   6 Batch  463/538 - Train Accuracy:  0.918, Validation Accuracy:  0.937, Loss:  0.107
    Epoch   6 Batch  464/538 - Train Accuracy:  0.951, Validation Accuracy:  0.940, Loss:  0.098
    Epoch   6 Batch  465/538 - Train Accuracy:  0.938, Validation Accuracy:  0.942, Loss:  0.097
    Epoch   6 Batch  466/538 - Train Accuracy:  0.933, Validation Accuracy:  0.944, Loss:  0.100
    Epoch   6 Batch  467/538 - Train Accuracy:  0.949, Validation Accuracy:  0.943, Loss:  0.104
    Epoch   6 Batch  468/538 - Train Accuracy:  0.951, Validation Accuracy:  0.942, Loss:  0.103
    Epoch   6 Batch  469/538 - Train Accuracy:  0.940, Validation Accuracy:  0.945, Loss:  0.103
    Epoch   6 Batch  470/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.091
    Epoch   6 Batch  471/538 - Train Accuracy:  0.957, Validation Accuracy:  0.944, Loss:  0.085
    Epoch   6 Batch  472/538 - Train Accuracy:  0.974, Validation Accuracy:  0.944, Loss:  0.080
    Epoch   6 Batch  473/538 - Train Accuracy:  0.933, Validation Accuracy:  0.944, Loss:  0.103
    Epoch   6 Batch  474/538 - Train Accuracy:  0.950, Validation Accuracy:  0.943, Loss:  0.102
    Epoch   6 Batch  475/538 - Train Accuracy:  0.937, Validation Accuracy:  0.942, Loss:  0.098
    Epoch   6 Batch  476/538 - Train Accuracy:  0.937, Validation Accuracy:  0.940, Loss:  0.095
    Epoch   6 Batch  477/538 - Train Accuracy:  0.928, Validation Accuracy:  0.937, Loss:  0.108
    Epoch   6 Batch  478/538 - Train Accuracy:  0.937, Validation Accuracy:  0.937, Loss:  0.098
    Epoch   6 Batch  479/538 - Train Accuracy:  0.944, Validation Accuracy:  0.942, Loss:  0.096
    Epoch   6 Batch  480/538 - Train Accuracy:  0.950, Validation Accuracy:  0.950, Loss:  0.097
    Epoch   6 Batch  481/538 - Train Accuracy:  0.946, Validation Accuracy:  0.952, Loss:  0.099
    Epoch   6 Batch  482/538 - Train Accuracy:  0.941, Validation Accuracy:  0.952, Loss:  0.093
    Epoch   6 Batch  483/538 - Train Accuracy:  0.924, Validation Accuracy:  0.951, Loss:  0.118
    Epoch   6 Batch  484/538 - Train Accuracy:  0.941, Validation Accuracy:  0.953, Loss:  0.127
    Epoch   6 Batch  485/538 - Train Accuracy:  0.941, Validation Accuracy:  0.953, Loss:  0.102
    Epoch   6 Batch  486/538 - Train Accuracy:  0.960, Validation Accuracy:  0.948, Loss:  0.081
    Epoch   6 Batch  487/538 - Train Accuracy:  0.961, Validation Accuracy:  0.938, Loss:  0.080
    Epoch   6 Batch  488/538 - Train Accuracy:  0.946, Validation Accuracy:  0.937, Loss:  0.093
    Epoch   6 Batch  489/538 - Train Accuracy:  0.935, Validation Accuracy:  0.934, Loss:  0.102
    Epoch   6 Batch  490/538 - Train Accuracy:  0.935, Validation Accuracy:  0.936, Loss:  0.108
    Epoch   6 Batch  491/538 - Train Accuracy:  0.920, Validation Accuracy:  0.943, Loss:  0.104
    Epoch   6 Batch  492/538 - Train Accuracy:  0.954, Validation Accuracy:  0.937, Loss:  0.099
    Epoch   6 Batch  493/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.091
    Epoch   6 Batch  494/538 - Train Accuracy:  0.944, Validation Accuracy:  0.944, Loss:  0.112
    Epoch   6 Batch  495/538 - Train Accuracy:  0.948, Validation Accuracy:  0.942, Loss:  0.105
    Epoch   6 Batch  496/538 - Train Accuracy:  0.950, Validation Accuracy:  0.943, Loss:  0.090
    Epoch   6 Batch  497/538 - Train Accuracy:  0.948, Validation Accuracy:  0.945, Loss:  0.085
    Epoch   6 Batch  498/538 - Train Accuracy:  0.955, Validation Accuracy:  0.946, Loss:  0.096
    Epoch   6 Batch  499/538 - Train Accuracy:  0.946, Validation Accuracy:  0.945, Loss:  0.097
    Epoch   6 Batch  500/538 - Train Accuracy:  0.948, Validation Accuracy:  0.947, Loss:  0.084
    Epoch   6 Batch  501/538 - Train Accuracy:  0.953, Validation Accuracy:  0.947, Loss:  0.107
    Epoch   6 Batch  502/538 - Train Accuracy:  0.937, Validation Accuracy:  0.937, Loss:  0.099
    Epoch   6 Batch  503/538 - Train Accuracy:  0.956, Validation Accuracy:  0.933, Loss:  0.101
    Epoch   6 Batch  504/538 - Train Accuracy:  0.955, Validation Accuracy:  0.930, Loss:  0.095
    Epoch   6 Batch  505/538 - Train Accuracy:  0.952, Validation Accuracy:  0.931, Loss:  0.084
    Epoch   6 Batch  506/538 - Train Accuracy:  0.952, Validation Accuracy:  0.931, Loss:  0.096
    Epoch   6 Batch  507/538 - Train Accuracy:  0.923, Validation Accuracy:  0.934, Loss:  0.122
    Epoch   6 Batch  508/538 - Train Accuracy:  0.936, Validation Accuracy:  0.934, Loss:  0.093
    Epoch   6 Batch  509/538 - Train Accuracy:  0.941, Validation Accuracy:  0.935, Loss:  0.092
    Epoch   6 Batch  510/538 - Train Accuracy:  0.948, Validation Accuracy:  0.936, Loss:  0.088
    Epoch   6 Batch  511/538 - Train Accuracy:  0.933, Validation Accuracy:  0.936, Loss:  0.105
    Epoch   6 Batch  512/538 - Train Accuracy:  0.958, Validation Accuracy:  0.933, Loss:  0.099
    Epoch   6 Batch  513/538 - Train Accuracy:  0.930, Validation Accuracy:  0.931, Loss:  0.098
    Epoch   6 Batch  514/538 - Train Accuracy:  0.942, Validation Accuracy:  0.929, Loss:  0.102
    Epoch   6 Batch  515/538 - Train Accuracy:  0.942, Validation Accuracy:  0.926, Loss:  0.107
    Epoch   6 Batch  516/538 - Train Accuracy:  0.931, Validation Accuracy:  0.927, Loss:  0.099
    Epoch   6 Batch  517/538 - Train Accuracy:  0.948, Validation Accuracy:  0.927, Loss:  0.113
    Epoch   6 Batch  518/538 - Train Accuracy:  0.931, Validation Accuracy:  0.929, Loss:  0.108
    Epoch   6 Batch  519/538 - Train Accuracy:  0.945, Validation Accuracy:  0.926, Loss:  0.099
    Epoch   6 Batch  520/538 - Train Accuracy:  0.932, Validation Accuracy:  0.924, Loss:  0.109
    Epoch   6 Batch  521/538 - Train Accuracy:  0.952, Validation Accuracy:  0.924, Loss:  0.103
    Epoch   6 Batch  522/538 - Train Accuracy:  0.943, Validation Accuracy:  0.926, Loss:  0.079
    Epoch   6 Batch  523/538 - Train Accuracy:  0.946, Validation Accuracy:  0.930, Loss:  0.086
    Epoch   6 Batch  524/538 - Train Accuracy:  0.942, Validation Accuracy:  0.933, Loss:  0.113
    Epoch   6 Batch  525/538 - Train Accuracy:  0.946, Validation Accuracy:  0.935, Loss:  0.099
    Epoch   6 Batch  526/538 - Train Accuracy:  0.951, Validation Accuracy:  0.932, Loss:  0.099
    Epoch   6 Batch  527/538 - Train Accuracy:  0.949, Validation Accuracy:  0.936, Loss:  0.090
    Epoch   6 Batch  528/538 - Train Accuracy:  0.939, Validation Accuracy:  0.935, Loss:  0.102
    Epoch   6 Batch  529/538 - Train Accuracy:  0.919, Validation Accuracy:  0.944, Loss:  0.105
    Epoch   6 Batch  530/538 - Train Accuracy:  0.930, Validation Accuracy:  0.943, Loss:  0.111
    Epoch   6 Batch  531/538 - Train Accuracy:  0.939, Validation Accuracy:  0.941, Loss:  0.096
    Epoch   6 Batch  532/538 - Train Accuracy:  0.949, Validation Accuracy:  0.944, Loss:  0.094
    Epoch   6 Batch  533/538 - Train Accuracy:  0.948, Validation Accuracy:  0.944, Loss:  0.093
    Epoch   6 Batch  534/538 - Train Accuracy:  0.945, Validation Accuracy:  0.947, Loss:  0.086
    Epoch   6 Batch  535/538 - Train Accuracy:  0.948, Validation Accuracy:  0.947, Loss:  0.090
    Epoch   6 Batch  536/538 - Train Accuracy:  0.952, Validation Accuracy:  0.945, Loss:  0.116
    Epoch   7 Batch    0/538 - Train Accuracy:  0.964, Validation Accuracy:  0.943, Loss:  0.080
    Epoch   7 Batch    1/538 - Train Accuracy:  0.954, Validation Accuracy:  0.939, Loss:  0.097
    Epoch   7 Batch    2/538 - Train Accuracy:  0.942, Validation Accuracy:  0.940, Loss:  0.111
    Epoch   7 Batch    3/538 - Train Accuracy:  0.946, Validation Accuracy:  0.940, Loss:  0.093
    Epoch   7 Batch    4/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.085
    Epoch   7 Batch    5/538 - Train Accuracy:  0.945, Validation Accuracy:  0.939, Loss:  0.092
    Epoch   7 Batch    6/538 - Train Accuracy:  0.946, Validation Accuracy:  0.939, Loss:  0.092
    Epoch   7 Batch    7/538 - Train Accuracy:  0.951, Validation Accuracy:  0.939, Loss:  0.092
    Epoch   7 Batch    8/538 - Train Accuracy:  0.940, Validation Accuracy:  0.938, Loss:  0.094
    Epoch   7 Batch    9/538 - Train Accuracy:  0.940, Validation Accuracy:  0.936, Loss:  0.091
    Epoch   7 Batch   10/538 - Train Accuracy:  0.932, Validation Accuracy:  0.936, Loss:  0.100
    Epoch   7 Batch   11/538 - Train Accuracy:  0.948, Validation Accuracy:  0.940, Loss:  0.092
    Epoch   7 Batch   12/538 - Train Accuracy:  0.945, Validation Accuracy:  0.943, Loss:  0.086
    Epoch   7 Batch   13/538 - Train Accuracy:  0.951, Validation Accuracy:  0.942, Loss:  0.081
    Epoch   7 Batch   14/538 - Train Accuracy:  0.950, Validation Accuracy:  0.940, Loss:  0.090
    Epoch   7 Batch   15/538 - Train Accuracy:  0.948, Validation Accuracy:  0.941, Loss:  0.099
    Epoch   7 Batch   16/538 - Train Accuracy:  0.952, Validation Accuracy:  0.942, Loss:  0.089
    Epoch   7 Batch   17/538 - Train Accuracy:  0.946, Validation Accuracy:  0.944, Loss:  0.101
    Epoch   7 Batch   18/538 - Train Accuracy:  0.956, Validation Accuracy:  0.944, Loss:  0.101
    Epoch   7 Batch   19/538 - Train Accuracy:  0.946, Validation Accuracy:  0.942, Loss:  0.111
    Epoch   7 Batch   20/538 - Train Accuracy:  0.942, Validation Accuracy:  0.940, Loss:  0.096
    Epoch   7 Batch   21/538 - Train Accuracy:  0.959, Validation Accuracy:  0.942, Loss:  0.082
    Epoch   7 Batch   22/538 - Train Accuracy:  0.934, Validation Accuracy:  0.942, Loss:  0.093
    Epoch   7 Batch   23/538 - Train Accuracy:  0.939, Validation Accuracy:  0.939, Loss:  0.121
    Epoch   7 Batch   24/538 - Train Accuracy:  0.954, Validation Accuracy:  0.938, Loss:  0.095
    Epoch   7 Batch   25/538 - Train Accuracy:  0.934, Validation Accuracy:  0.940, Loss:  0.092
    Epoch   7 Batch   26/538 - Train Accuracy:  0.937, Validation Accuracy:  0.940, Loss:  0.102
    Epoch   7 Batch   27/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.078
    Epoch   7 Batch   28/538 - Train Accuracy:  0.940, Validation Accuracy:  0.939, Loss:  0.092
    Epoch   7 Batch   29/538 - Train Accuracy:  0.953, Validation Accuracy:  0.939, Loss:  0.083
    Epoch   7 Batch   30/538 - Train Accuracy:  0.937, Validation Accuracy:  0.945, Loss:  0.119
    Epoch   7 Batch   31/538 - Train Accuracy:  0.947, Validation Accuracy:  0.940, Loss:  0.082
    Epoch   7 Batch   32/538 - Train Accuracy:  0.935, Validation Accuracy:  0.944, Loss:  0.074
    Epoch   7 Batch   33/538 - Train Accuracy:  0.937, Validation Accuracy:  0.943, Loss:  0.094
    Epoch   7 Batch   34/538 - Train Accuracy:  0.942, Validation Accuracy:  0.948, Loss:  0.116
    Epoch   7 Batch   35/538 - Train Accuracy:  0.952, Validation Accuracy:  0.952, Loss:  0.082
    Epoch   7 Batch   36/538 - Train Accuracy:  0.942, Validation Accuracy:  0.951, Loss:  0.080
    Epoch   7 Batch   37/538 - Train Accuracy:  0.949, Validation Accuracy:  0.951, Loss:  0.098
    Epoch   7 Batch   38/538 - Train Accuracy:  0.929, Validation Accuracy:  0.954, Loss:  0.099
    Epoch   7 Batch   39/538 - Train Accuracy:  0.951, Validation Accuracy:  0.952, Loss:  0.095
    Epoch   7 Batch   40/538 - Train Accuracy:  0.950, Validation Accuracy:  0.953, Loss:  0.082
    Epoch   7 Batch   41/538 - Train Accuracy:  0.951, Validation Accuracy:  0.951, Loss:  0.098
    Epoch   7 Batch   42/538 - Train Accuracy:  0.947, Validation Accuracy:  0.946, Loss:  0.096
    Epoch   7 Batch   43/538 - Train Accuracy:  0.928, Validation Accuracy:  0.944, Loss:  0.124
    Epoch   7 Batch   44/538 - Train Accuracy:  0.939, Validation Accuracy:  0.944, Loss:  0.095
    Epoch   7 Batch   45/538 - Train Accuracy:  0.949, Validation Accuracy:  0.938, Loss:  0.085
    Epoch   7 Batch   46/538 - Train Accuracy:  0.949, Validation Accuracy:  0.938, Loss:  0.080
    Epoch   7 Batch   47/538 - Train Accuracy:  0.943, Validation Accuracy:  0.937, Loss:  0.105
    Epoch   7 Batch   48/538 - Train Accuracy:  0.937, Validation Accuracy:  0.938, Loss:  0.112
    Epoch   7 Batch   49/538 - Train Accuracy:  0.942, Validation Accuracy:  0.939, Loss:  0.089
    Epoch   7 Batch   50/538 - Train Accuracy:  0.942, Validation Accuracy:  0.942, Loss:  0.097
    Epoch   7 Batch   51/538 - Train Accuracy:  0.938, Validation Accuracy:  0.942, Loss:  0.107
    Epoch   7 Batch   52/538 - Train Accuracy:  0.944, Validation Accuracy:  0.944, Loss:  0.095
    Epoch   7 Batch   53/538 - Train Accuracy:  0.936, Validation Accuracy:  0.946, Loss:  0.090
    Epoch   7 Batch   54/538 - Train Accuracy:  0.954, Validation Accuracy:  0.948, Loss:  0.078
    Epoch   7 Batch   55/538 - Train Accuracy:  0.942, Validation Accuracy:  0.948, Loss:  0.091
    Epoch   7 Batch   56/538 - Train Accuracy:  0.942, Validation Accuracy:  0.948, Loss:  0.094
    Epoch   7 Batch   57/538 - Train Accuracy:  0.929, Validation Accuracy:  0.948, Loss:  0.097
    Epoch   7 Batch   58/538 - Train Accuracy:  0.952, Validation Accuracy:  0.947, Loss:  0.096
    Epoch   7 Batch   59/538 - Train Accuracy:  0.939, Validation Accuracy:  0.944, Loss:  0.098
    Epoch   7 Batch   60/538 - Train Accuracy:  0.932, Validation Accuracy:  0.943, Loss:  0.090
    Epoch   7 Batch   61/538 - Train Accuracy:  0.944, Validation Accuracy:  0.944, Loss:  0.093
    Epoch   7 Batch   62/538 - Train Accuracy:  0.954, Validation Accuracy:  0.947, Loss:  0.094
    Epoch   7 Batch   63/538 - Train Accuracy:  0.955, Validation Accuracy:  0.947, Loss:  0.087
    Epoch   7 Batch   64/538 - Train Accuracy:  0.951, Validation Accuracy:  0.944, Loss:  0.091
    Epoch   7 Batch   65/538 - Train Accuracy:  0.938, Validation Accuracy:  0.944, Loss:  0.097
    Epoch   7 Batch   66/538 - Train Accuracy:  0.955, Validation Accuracy:  0.943, Loss:  0.078
    Epoch   7 Batch   67/538 - Train Accuracy:  0.955, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   7 Batch   68/538 - Train Accuracy:  0.938, Validation Accuracy:  0.944, Loss:  0.081
    Epoch   7 Batch   69/538 - Train Accuracy:  0.949, Validation Accuracy:  0.948, Loss:  0.094
    Epoch   7 Batch   70/538 - Train Accuracy:  0.946, Validation Accuracy:  0.947, Loss:  0.083
    Epoch   7 Batch   71/538 - Train Accuracy:  0.937, Validation Accuracy:  0.948, Loss:  0.102
    Epoch   7 Batch   72/538 - Train Accuracy:  0.955, Validation Accuracy:  0.943, Loss:  0.119
    Epoch   7 Batch   73/538 - Train Accuracy:  0.938, Validation Accuracy:  0.946, Loss:  0.099
    Epoch   7 Batch   74/538 - Train Accuracy:  0.949, Validation Accuracy:  0.942, Loss:  0.080
    Epoch   7 Batch   75/538 - Train Accuracy:  0.945, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   7 Batch   76/538 - Train Accuracy:  0.950, Validation Accuracy:  0.944, Loss:  0.104
    Epoch   7 Batch   77/538 - Train Accuracy:  0.943, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   7 Batch   78/538 - Train Accuracy:  0.944, Validation Accuracy:  0.947, Loss:  0.094
    Epoch   7 Batch   79/538 - Train Accuracy:  0.945, Validation Accuracy:  0.945, Loss:  0.071
    Epoch   7 Batch   80/538 - Train Accuracy:  0.938, Validation Accuracy:  0.948, Loss:  0.096
    Epoch   7 Batch   81/538 - Train Accuracy:  0.938, Validation Accuracy:  0.948, Loss:  0.102
    Epoch   7 Batch   82/538 - Train Accuracy:  0.941, Validation Accuracy:  0.947, Loss:  0.091
    Epoch   7 Batch   83/538 - Train Accuracy:  0.945, Validation Accuracy:  0.946, Loss:  0.089
    Epoch   7 Batch   84/538 - Train Accuracy:  0.924, Validation Accuracy:  0.940, Loss:  0.107
    Epoch   7 Batch   85/538 - Train Accuracy:  0.953, Validation Accuracy:  0.934, Loss:  0.088
    Epoch   7 Batch   86/538 - Train Accuracy:  0.950, Validation Accuracy:  0.936, Loss:  0.092
    Epoch   7 Batch   87/538 - Train Accuracy:  0.938, Validation Accuracy:  0.933, Loss:  0.104
    Epoch   7 Batch   88/538 - Train Accuracy:  0.935, Validation Accuracy:  0.933, Loss:  0.094
    Epoch   7 Batch   89/538 - Train Accuracy:  0.942, Validation Accuracy:  0.933, Loss:  0.090
    Epoch   7 Batch   90/538 - Train Accuracy:  0.945, Validation Accuracy:  0.935, Loss:  0.096
    Epoch   7 Batch   91/538 - Train Accuracy:  0.941, Validation Accuracy:  0.934, Loss:  0.083
    Epoch   7 Batch   92/538 - Train Accuracy:  0.939, Validation Accuracy:  0.936, Loss:  0.099
    Epoch   7 Batch   93/538 - Train Accuracy:  0.953, Validation Accuracy:  0.941, Loss:  0.082
    Epoch   7 Batch   94/538 - Train Accuracy:  0.935, Validation Accuracy:  0.943, Loss:  0.089
    Epoch   7 Batch   95/538 - Train Accuracy:  0.949, Validation Accuracy:  0.946, Loss:  0.083
    Epoch   7 Batch   96/538 - Train Accuracy:  0.962, Validation Accuracy:  0.946, Loss:  0.085
    Epoch   7 Batch   97/538 - Train Accuracy:  0.943, Validation Accuracy:  0.948, Loss:  0.089
    Epoch   7 Batch   98/538 - Train Accuracy:  0.948, Validation Accuracy:  0.943, Loss:  0.084
    Epoch   7 Batch   99/538 - Train Accuracy:  0.948, Validation Accuracy:  0.945, Loss:  0.095
    Epoch   7 Batch  100/538 - Train Accuracy:  0.954, Validation Accuracy:  0.942, Loss:  0.088
    Epoch   7 Batch  101/538 - Train Accuracy:  0.938, Validation Accuracy:  0.939, Loss:  0.116
    Epoch   7 Batch  102/538 - Train Accuracy:  0.931, Validation Accuracy:  0.937, Loss:  0.096
    Epoch   7 Batch  103/538 - Train Accuracy:  0.947, Validation Accuracy:  0.937, Loss:  0.097
    Epoch   7 Batch  104/538 - Train Accuracy:  0.954, Validation Accuracy:  0.935, Loss:  0.079
    Epoch   7 Batch  105/538 - Train Accuracy:  0.938, Validation Accuracy:  0.934, Loss:  0.085
    Epoch   7 Batch  106/538 - Train Accuracy:  0.948, Validation Accuracy:  0.934, Loss:  0.079
    Epoch   7 Batch  107/538 - Train Accuracy:  0.937, Validation Accuracy:  0.933, Loss:  0.108
    Epoch   7 Batch  108/538 - Train Accuracy:  0.957, Validation Accuracy:  0.933, Loss:  0.090
    Epoch   7 Batch  109/538 - Train Accuracy:  0.959, Validation Accuracy:  0.933, Loss:  0.084
    Epoch   7 Batch  110/538 - Train Accuracy:  0.946, Validation Accuracy:  0.933, Loss:  0.108
    Epoch   7 Batch  111/538 - Train Accuracy:  0.942, Validation Accuracy:  0.933, Loss:  0.089
    Epoch   7 Batch  112/538 - Train Accuracy:  0.945, Validation Accuracy:  0.932, Loss:  0.099
    Epoch   7 Batch  113/538 - Train Accuracy:  0.922, Validation Accuracy:  0.932, Loss:  0.110
    Epoch   7 Batch  114/538 - Train Accuracy:  0.954, Validation Accuracy:  0.936, Loss:  0.091
    Epoch   7 Batch  115/538 - Train Accuracy:  0.956, Validation Accuracy:  0.937, Loss:  0.091
    Epoch   7 Batch  116/538 - Train Accuracy:  0.944, Validation Accuracy:  0.936, Loss:  0.114
    Epoch   7 Batch  117/538 - Train Accuracy:  0.948, Validation Accuracy:  0.938, Loss:  0.091
    Epoch   7 Batch  118/538 - Train Accuracy:  0.956, Validation Accuracy:  0.940, Loss:  0.078
    Epoch   7 Batch  119/538 - Train Accuracy:  0.958, Validation Accuracy:  0.941, Loss:  0.073
    Epoch   7 Batch  120/538 - Train Accuracy:  0.947, Validation Accuracy:  0.942, Loss:  0.079
    Epoch   7 Batch  121/538 - Train Accuracy:  0.953, Validation Accuracy:  0.943, Loss:  0.086
    Epoch   7 Batch  122/538 - Train Accuracy:  0.941, Validation Accuracy:  0.943, Loss:  0.085
    Epoch   7 Batch  123/538 - Train Accuracy:  0.942, Validation Accuracy:  0.939, Loss:  0.097
    Epoch   7 Batch  124/538 - Train Accuracy:  0.945, Validation Accuracy:  0.941, Loss:  0.086
    Epoch   7 Batch  125/538 - Train Accuracy:  0.936, Validation Accuracy:  0.940, Loss:  0.096
    Epoch   7 Batch  126/538 - Train Accuracy:  0.929, Validation Accuracy:  0.939, Loss:  0.102
    Epoch   7 Batch  127/538 - Train Accuracy:  0.930, Validation Accuracy:  0.939, Loss:  0.109
    Epoch   7 Batch  128/538 - Train Accuracy:  0.946, Validation Accuracy:  0.939, Loss:  0.101
    Epoch   7 Batch  129/538 - Train Accuracy:  0.953, Validation Accuracy:  0.942, Loss:  0.080
    Epoch   7 Batch  130/538 - Train Accuracy:  0.942, Validation Accuracy:  0.943, Loss:  0.088
    Epoch   7 Batch  131/538 - Train Accuracy:  0.946, Validation Accuracy:  0.943, Loss:  0.078
    Epoch   7 Batch  132/538 - Train Accuracy:  0.941, Validation Accuracy:  0.943, Loss:  0.092
    Epoch   7 Batch  133/538 - Train Accuracy:  0.937, Validation Accuracy:  0.942, Loss:  0.084
    Epoch   7 Batch  134/538 - Train Accuracy:  0.949, Validation Accuracy:  0.942, Loss:  0.107
    Epoch   7 Batch  135/538 - Train Accuracy:  0.946, Validation Accuracy:  0.947, Loss:  0.120
    Epoch   7 Batch  136/538 - Train Accuracy:  0.953, Validation Accuracy:  0.945, Loss:  0.102
    Epoch   7 Batch  137/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.112
    Epoch   7 Batch  138/538 - Train Accuracy:  0.942, Validation Accuracy:  0.936, Loss:  0.092
    Epoch   7 Batch  139/538 - Train Accuracy:  0.931, Validation Accuracy:  0.933, Loss:  0.111
    Epoch   7 Batch  140/538 - Train Accuracy:  0.929, Validation Accuracy:  0.935, Loss:  0.112
    Epoch   7 Batch  141/538 - Train Accuracy:  0.948, Validation Accuracy:  0.934, Loss:  0.113
    Epoch   7 Batch  142/538 - Train Accuracy:  0.938, Validation Accuracy:  0.934, Loss:  0.088
    Epoch   7 Batch  143/538 - Train Accuracy:  0.936, Validation Accuracy:  0.941, Loss:  0.088
    Epoch   7 Batch  144/538 - Train Accuracy:  0.956, Validation Accuracy:  0.942, Loss:  0.104
    Epoch   7 Batch  145/538 - Train Accuracy:  0.929, Validation Accuracy:  0.944, Loss:  0.116
    Epoch   7 Batch  146/538 - Train Accuracy:  0.946, Validation Accuracy:  0.945, Loss:  0.103
    Epoch   7 Batch  147/538 - Train Accuracy:  0.949, Validation Accuracy:  0.948, Loss:  0.090
    Epoch   7 Batch  148/538 - Train Accuracy:  0.922, Validation Accuracy:  0.950, Loss:  0.132
    Epoch   7 Batch  149/538 - Train Accuracy:  0.959, Validation Accuracy:  0.946, Loss:  0.082
    Epoch   7 Batch  150/538 - Train Accuracy:  0.955, Validation Accuracy:  0.942, Loss:  0.084
    Epoch   7 Batch  151/538 - Train Accuracy:  0.938, Validation Accuracy:  0.946, Loss:  0.090
    Epoch   7 Batch  152/538 - Train Accuracy:  0.949, Validation Accuracy:  0.944, Loss:  0.103
    Epoch   7 Batch  153/538 - Train Accuracy:  0.940, Validation Accuracy:  0.939, Loss:  0.092
    Epoch   7 Batch  154/538 - Train Accuracy:  0.947, Validation Accuracy:  0.939, Loss:  0.088
    Epoch   7 Batch  155/538 - Train Accuracy:  0.936, Validation Accuracy:  0.932, Loss:  0.092
    Epoch   7 Batch  156/538 - Train Accuracy:  0.946, Validation Accuracy:  0.937, Loss:  0.093
    Epoch   7 Batch  157/538 - Train Accuracy:  0.948, Validation Accuracy:  0.941, Loss:  0.084
    Epoch   7 Batch  158/538 - Train Accuracy:  0.955, Validation Accuracy:  0.946, Loss:  0.092
    Epoch   7 Batch  159/538 - Train Accuracy:  0.931, Validation Accuracy:  0.949, Loss:  0.102
    Epoch   7 Batch  160/538 - Train Accuracy:  0.934, Validation Accuracy:  0.945, Loss:  0.087
    Epoch   7 Batch  161/538 - Train Accuracy:  0.949, Validation Accuracy:  0.942, Loss:  0.084
    Epoch   7 Batch  162/538 - Train Accuracy:  0.941, Validation Accuracy:  0.946, Loss:  0.091
    Epoch   7 Batch  163/538 - Train Accuracy:  0.942, Validation Accuracy:  0.940, Loss:  0.107
    Epoch   7 Batch  164/538 - Train Accuracy:  0.938, Validation Accuracy:  0.941, Loss:  0.111
    Epoch   7 Batch  165/538 - Train Accuracy:  0.953, Validation Accuracy:  0.936, Loss:  0.081
    Epoch   7 Batch  166/538 - Train Accuracy:  0.956, Validation Accuracy:  0.940, Loss:  0.087
    Epoch   7 Batch  167/538 - Train Accuracy:  0.951, Validation Accuracy:  0.941, Loss:  0.112
    Epoch   7 Batch  168/538 - Train Accuracy:  0.921, Validation Accuracy:  0.938, Loss:  0.123
    Epoch   7 Batch  169/538 - Train Accuracy:  0.966, Validation Accuracy:  0.943, Loss:  0.079
    Epoch   7 Batch  170/538 - Train Accuracy:  0.942, Validation Accuracy:  0.944, Loss:  0.098
    Epoch   7 Batch  171/538 - Train Accuracy:  0.940, Validation Accuracy:  0.940, Loss:  0.094
    Epoch   7 Batch  172/538 - Train Accuracy:  0.934, Validation Accuracy:  0.937, Loss:  0.096
    Epoch   7 Batch  173/538 - Train Accuracy:  0.956, Validation Accuracy:  0.933, Loss:  0.075
    Epoch   7 Batch  174/538 - Train Accuracy:  0.946, Validation Accuracy:  0.934, Loss:  0.083
    Epoch   7 Batch  175/538 - Train Accuracy:  0.952, Validation Accuracy:  0.935, Loss:  0.090
    Epoch   7 Batch  176/538 - Train Accuracy:  0.946, Validation Accuracy:  0.934, Loss:  0.105
    Epoch   7 Batch  177/538 - Train Accuracy:  0.953, Validation Accuracy:  0.936, Loss:  0.100
    Epoch   7 Batch  178/538 - Train Accuracy:  0.930, Validation Accuracy:  0.936, Loss:  0.094
    Epoch   7 Batch  179/538 - Train Accuracy:  0.957, Validation Accuracy:  0.938, Loss:  0.085
    Epoch   7 Batch  180/538 - Train Accuracy:  0.942, Validation Accuracy:  0.943, Loss:  0.091
    Epoch   7 Batch  181/538 - Train Accuracy:  0.938, Validation Accuracy:  0.943, Loss:  0.114
    Epoch   7 Batch  182/538 - Train Accuracy:  0.957, Validation Accuracy:  0.941, Loss:  0.077
    Epoch   7 Batch  183/538 - Train Accuracy:  0.955, Validation Accuracy:  0.941, Loss:  0.073
    Epoch   7 Batch  184/538 - Train Accuracy:  0.945, Validation Accuracy:  0.942, Loss:  0.098
    Epoch   7 Batch  185/538 - Train Accuracy:  0.966, Validation Accuracy:  0.942, Loss:  0.071
    Epoch   7 Batch  186/538 - Train Accuracy:  0.942, Validation Accuracy:  0.935, Loss:  0.082
    Epoch   7 Batch  187/538 - Train Accuracy:  0.955, Validation Accuracy:  0.935, Loss:  0.088
    Epoch   7 Batch  188/538 - Train Accuracy:  0.941, Validation Accuracy:  0.937, Loss:  0.085
    Epoch   7 Batch  189/538 - Train Accuracy:  0.955, Validation Accuracy:  0.936, Loss:  0.100
    Epoch   7 Batch  190/538 - Train Accuracy:  0.927, Validation Accuracy:  0.937, Loss:  0.116
    Epoch   7 Batch  191/538 - Train Accuracy:  0.950, Validation Accuracy:  0.939, Loss:  0.082
    Epoch   7 Batch  192/538 - Train Accuracy:  0.948, Validation Accuracy:  0.938, Loss:  0.092
    Epoch   7 Batch  193/538 - Train Accuracy:  0.951, Validation Accuracy:  0.943, Loss:  0.085
    Epoch   7 Batch  194/538 - Train Accuracy:  0.932, Validation Accuracy:  0.941, Loss:  0.104
    Epoch   7 Batch  195/538 - Train Accuracy:  0.952, Validation Accuracy:  0.940, Loss:  0.101
    Epoch   7 Batch  196/538 - Train Accuracy:  0.925, Validation Accuracy:  0.939, Loss:  0.086
    Epoch   7 Batch  197/538 - Train Accuracy:  0.944, Validation Accuracy:  0.936, Loss:  0.092
    Epoch   7 Batch  198/538 - Train Accuracy:  0.945, Validation Accuracy:  0.936, Loss:  0.095
    Epoch   7 Batch  199/538 - Train Accuracy:  0.941, Validation Accuracy:  0.938, Loss:  0.104
    Epoch   7 Batch  200/538 - Train Accuracy:  0.958, Validation Accuracy:  0.935, Loss:  0.082
    Epoch   7 Batch  201/538 - Train Accuracy:  0.957, Validation Accuracy:  0.933, Loss:  0.099
    Epoch   7 Batch  202/538 - Train Accuracy:  0.953, Validation Accuracy:  0.934, Loss:  0.098
    Epoch   7 Batch  203/538 - Train Accuracy:  0.947, Validation Accuracy:  0.936, Loss:  0.106
    Epoch   7 Batch  204/538 - Train Accuracy:  0.939, Validation Accuracy:  0.940, Loss:  0.103
    Epoch   7 Batch  205/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.085
    Epoch   7 Batch  206/538 - Train Accuracy:  0.944, Validation Accuracy:  0.941, Loss:  0.093
    Epoch   7 Batch  207/538 - Train Accuracy:  0.957, Validation Accuracy:  0.944, Loss:  0.096
    Epoch   7 Batch  208/538 - Train Accuracy:  0.939, Validation Accuracy:  0.941, Loss:  0.105
    Epoch   7 Batch  209/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   7 Batch  210/538 - Train Accuracy:  0.940, Validation Accuracy:  0.945, Loss:  0.108
    Epoch   7 Batch  211/538 - Train Accuracy:  0.945, Validation Accuracy:  0.945, Loss:  0.101
    Epoch   7 Batch  212/538 - Train Accuracy:  0.921, Validation Accuracy:  0.945, Loss:  0.091
    Epoch   7 Batch  213/538 - Train Accuracy:  0.944, Validation Accuracy:  0.947, Loss:  0.085
    Epoch   7 Batch  214/538 - Train Accuracy:  0.953, Validation Accuracy:  0.946, Loss:  0.082
    Epoch   7 Batch  215/538 - Train Accuracy:  0.956, Validation Accuracy:  0.944, Loss:  0.077
    Epoch   7 Batch  216/538 - Train Accuracy:  0.953, Validation Accuracy:  0.948, Loss:  0.086
    Epoch   7 Batch  217/538 - Train Accuracy:  0.960, Validation Accuracy:  0.944, Loss:  0.083
    Epoch   7 Batch  218/538 - Train Accuracy:  0.945, Validation Accuracy:  0.939, Loss:  0.089
    Epoch   7 Batch  219/538 - Train Accuracy:  0.939, Validation Accuracy:  0.936, Loss:  0.102
    Epoch   7 Batch  220/538 - Train Accuracy:  0.928, Validation Accuracy:  0.934, Loss:  0.112
    Epoch   7 Batch  221/538 - Train Accuracy:  0.944, Validation Accuracy:  0.933, Loss:  0.090
    Epoch   7 Batch  222/538 - Train Accuracy:  0.939, Validation Accuracy:  0.935, Loss:  0.088
    Epoch   7 Batch  223/538 - Train Accuracy:  0.934, Validation Accuracy:  0.934, Loss:  0.122
    Epoch   7 Batch  224/538 - Train Accuracy:  0.939, Validation Accuracy:  0.933, Loss:  0.097
    Epoch   7 Batch  225/538 - Train Accuracy:  0.950, Validation Accuracy:  0.935, Loss:  0.104
    Epoch   7 Batch  226/538 - Train Accuracy:  0.930, Validation Accuracy:  0.935, Loss:  0.097
    Epoch   7 Batch  227/538 - Train Accuracy:  0.943, Validation Accuracy:  0.937, Loss:  0.089
    Epoch   7 Batch  228/538 - Train Accuracy:  0.930, Validation Accuracy:  0.937, Loss:  0.095
    Epoch   7 Batch  229/538 - Train Accuracy:  0.946, Validation Accuracy:  0.936, Loss:  0.101
    Epoch   7 Batch  230/538 - Train Accuracy:  0.946, Validation Accuracy:  0.937, Loss:  0.095
    Epoch   7 Batch  231/538 - Train Accuracy:  0.938, Validation Accuracy:  0.939, Loss:  0.095
    Epoch   7 Batch  232/538 - Train Accuracy:  0.933, Validation Accuracy:  0.937, Loss:  0.085
    Epoch   7 Batch  233/538 - Train Accuracy:  0.956, Validation Accuracy:  0.940, Loss:  0.106
    Epoch   7 Batch  234/538 - Train Accuracy:  0.949, Validation Accuracy:  0.941, Loss:  0.089
    Epoch   7 Batch  235/538 - Train Accuracy:  0.954, Validation Accuracy:  0.940, Loss:  0.081
    Epoch   7 Batch  236/538 - Train Accuracy:  0.940, Validation Accuracy:  0.942, Loss:  0.110
    Epoch   7 Batch  237/538 - Train Accuracy:  0.950, Validation Accuracy:  0.941, Loss:  0.075
    Epoch   7 Batch  238/538 - Train Accuracy:  0.952, Validation Accuracy:  0.942, Loss:  0.087
    Epoch   7 Batch  239/538 - Train Accuracy:  0.940, Validation Accuracy:  0.944, Loss:  0.107
    Epoch   7 Batch  240/538 - Train Accuracy:  0.942, Validation Accuracy:  0.944, Loss:  0.107
    Epoch   7 Batch  241/538 - Train Accuracy:  0.928, Validation Accuracy:  0.945, Loss:  0.106
    Epoch   7 Batch  242/538 - Train Accuracy:  0.950, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   7 Batch  243/538 - Train Accuracy:  0.954, Validation Accuracy:  0.939, Loss:  0.096
    Epoch   7 Batch  244/538 - Train Accuracy:  0.940, Validation Accuracy:  0.939, Loss:  0.087
    Epoch   7 Batch  245/538 - Train Accuracy:  0.938, Validation Accuracy:  0.940, Loss:  0.113
    Epoch   7 Batch  246/538 - Train Accuracy:  0.950, Validation Accuracy:  0.942, Loss:  0.080
    Epoch   7 Batch  247/538 - Train Accuracy:  0.932, Validation Accuracy:  0.941, Loss:  0.094
    Epoch   7 Batch  248/538 - Train Accuracy:  0.941, Validation Accuracy:  0.942, Loss:  0.100
    Epoch   7 Batch  249/538 - Train Accuracy:  0.956, Validation Accuracy:  0.947, Loss:  0.072
    Epoch   7 Batch  250/538 - Train Accuracy:  0.953, Validation Accuracy:  0.944, Loss:  0.079
    Epoch   7 Batch  251/538 - Train Accuracy:  0.948, Validation Accuracy:  0.945, Loss:  0.083
    Epoch   7 Batch  252/538 - Train Accuracy:  0.946, Validation Accuracy:  0.944, Loss:  0.080
    Epoch   7 Batch  253/538 - Train Accuracy:  0.941, Validation Accuracy:  0.942, Loss:  0.075
    Epoch   7 Batch  254/538 - Train Accuracy:  0.933, Validation Accuracy:  0.938, Loss:  0.101
    Epoch   7 Batch  255/538 - Train Accuracy:  0.954, Validation Accuracy:  0.937, Loss:  0.076
    Epoch   7 Batch  256/538 - Train Accuracy:  0.942, Validation Accuracy:  0.938, Loss:  0.094
    Epoch   7 Batch  257/538 - Train Accuracy:  0.961, Validation Accuracy:  0.939, Loss:  0.080
    Epoch   7 Batch  258/538 - Train Accuracy:  0.948, Validation Accuracy:  0.941, Loss:  0.087
    Epoch   7 Batch  259/538 - Train Accuracy:  0.957, Validation Accuracy:  0.944, Loss:  0.087
    Epoch   7 Batch  260/538 - Train Accuracy:  0.920, Validation Accuracy:  0.947, Loss:  0.102
    Epoch   7 Batch  261/538 - Train Accuracy:  0.946, Validation Accuracy:  0.947, Loss:  0.099
    Epoch   7 Batch  262/538 - Train Accuracy:  0.955, Validation Accuracy:  0.945, Loss:  0.084
    Epoch   7 Batch  263/538 - Train Accuracy:  0.927, Validation Accuracy:  0.943, Loss:  0.091
    Epoch   7 Batch  264/538 - Train Accuracy:  0.934, Validation Accuracy:  0.941, Loss:  0.092
    Epoch   7 Batch  265/538 - Train Accuracy:  0.937, Validation Accuracy:  0.941, Loss:  0.100
    Epoch   7 Batch  266/538 - Train Accuracy:  0.944, Validation Accuracy:  0.941, Loss:  0.093
    Epoch   7 Batch  267/538 - Train Accuracy:  0.936, Validation Accuracy:  0.939, Loss:  0.093
    Epoch   7 Batch  268/538 - Train Accuracy:  0.953, Validation Accuracy:  0.936, Loss:  0.074
    Epoch   7 Batch  269/538 - Train Accuracy:  0.942, Validation Accuracy:  0.938, Loss:  0.101
    Epoch   7 Batch  270/538 - Train Accuracy:  0.947, Validation Accuracy:  0.942, Loss:  0.088
    Epoch   7 Batch  271/538 - Train Accuracy:  0.948, Validation Accuracy:  0.948, Loss:  0.072
    Epoch   7 Batch  272/538 - Train Accuracy:  0.949, Validation Accuracy:  0.947, Loss:  0.088
    Epoch   7 Batch  273/538 - Train Accuracy:  0.937, Validation Accuracy:  0.950, Loss:  0.100
    Epoch   7 Batch  274/538 - Train Accuracy:  0.928, Validation Accuracy:  0.949, Loss:  0.105
    Epoch   7 Batch  275/538 - Train Accuracy:  0.949, Validation Accuracy:  0.949, Loss:  0.115
    Epoch   7 Batch  276/538 - Train Accuracy:  0.941, Validation Accuracy:  0.949, Loss:  0.095
    Epoch   7 Batch  277/538 - Train Accuracy:  0.945, Validation Accuracy:  0.945, Loss:  0.078
    Epoch   7 Batch  278/538 - Train Accuracy:  0.942, Validation Accuracy:  0.944, Loss:  0.075
    Epoch   7 Batch  279/538 - Train Accuracy:  0.944, Validation Accuracy:  0.945, Loss:  0.090
    Epoch   7 Batch  280/538 - Train Accuracy:  0.951, Validation Accuracy:  0.942, Loss:  0.074
    Epoch   7 Batch  281/538 - Train Accuracy:  0.958, Validation Accuracy:  0.944, Loss:  0.109
    Epoch   7 Batch  282/538 - Train Accuracy:  0.955, Validation Accuracy:  0.942, Loss:  0.095
    Epoch   7 Batch  283/538 - Train Accuracy:  0.945, Validation Accuracy:  0.944, Loss:  0.097
    Epoch   7 Batch  284/538 - Train Accuracy:  0.944, Validation Accuracy:  0.946, Loss:  0.104
    Epoch   7 Batch  285/538 - Train Accuracy:  0.943, Validation Accuracy:  0.946, Loss:  0.079
    Epoch   7 Batch  286/538 - Train Accuracy:  0.938, Validation Accuracy:  0.946, Loss:  0.110
    Epoch   7 Batch  287/538 - Train Accuracy:  0.957, Validation Accuracy:  0.947, Loss:  0.070
    Epoch   7 Batch  288/538 - Train Accuracy:  0.950, Validation Accuracy:  0.945, Loss:  0.096
    Epoch   7 Batch  289/538 - Train Accuracy:  0.957, Validation Accuracy:  0.945, Loss:  0.074
    Epoch   7 Batch  290/538 - Train Accuracy:  0.960, Validation Accuracy:  0.945, Loss:  0.069
    Epoch   7 Batch  291/538 - Train Accuracy:  0.942, Validation Accuracy:  0.953, Loss:  0.081
    Epoch   7 Batch  292/538 - Train Accuracy:  0.962, Validation Accuracy:  0.948, Loss:  0.072
    Epoch   7 Batch  293/538 - Train Accuracy:  0.944, Validation Accuracy:  0.948, Loss:  0.083
    Epoch   7 Batch  294/538 - Train Accuracy:  0.939, Validation Accuracy:  0.946, Loss:  0.091
    Epoch   7 Batch  295/538 - Train Accuracy:  0.953, Validation Accuracy:  0.947, Loss:  0.083
    Epoch   7 Batch  296/538 - Train Accuracy:  0.941, Validation Accuracy:  0.949, Loss:  0.089
    Epoch   7 Batch  297/538 - Train Accuracy:  0.958, Validation Accuracy:  0.947, Loss:  0.085
    Epoch   7 Batch  298/538 - Train Accuracy:  0.950, Validation Accuracy:  0.945, Loss:  0.078
    Epoch   7 Batch  299/538 - Train Accuracy:  0.943, Validation Accuracy:  0.945, Loss:  0.103
    Epoch   7 Batch  300/538 - Train Accuracy:  0.947, Validation Accuracy:  0.942, Loss:  0.089
    Epoch   7 Batch  301/538 - Train Accuracy:  0.934, Validation Accuracy:  0.939, Loss:  0.099
    Epoch   7 Batch  302/538 - Train Accuracy:  0.963, Validation Accuracy:  0.942, Loss:  0.087
    Epoch   7 Batch  303/538 - Train Accuracy:  0.955, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   7 Batch  304/538 - Train Accuracy:  0.949, Validation Accuracy:  0.943, Loss:  0.091
    Epoch   7 Batch  305/538 - Train Accuracy:  0.962, Validation Accuracy:  0.938, Loss:  0.077
    Epoch   7 Batch  306/538 - Train Accuracy:  0.945, Validation Accuracy:  0.940, Loss:  0.078
    Epoch   7 Batch  307/538 - Train Accuracy:  0.957, Validation Accuracy:  0.937, Loss:  0.086
    Epoch   7 Batch  308/538 - Train Accuracy:  0.942, Validation Accuracy:  0.936, Loss:  0.087
    Epoch   7 Batch  309/538 - Train Accuracy:  0.949, Validation Accuracy:  0.934, Loss:  0.079
    Epoch   7 Batch  310/538 - Train Accuracy:  0.965, Validation Accuracy:  0.936, Loss:  0.095
    Epoch   7 Batch  311/538 - Train Accuracy:  0.943, Validation Accuracy:  0.939, Loss:  0.093
    Epoch   7 Batch  312/538 - Train Accuracy:  0.953, Validation Accuracy:  0.942, Loss:  0.089
    Epoch   7 Batch  313/538 - Train Accuracy:  0.946, Validation Accuracy:  0.943, Loss:  0.088
    Epoch   7 Batch  314/538 - Train Accuracy:  0.951, Validation Accuracy:  0.941, Loss:  0.084
    Epoch   7 Batch  315/538 - Train Accuracy:  0.944, Validation Accuracy:  0.940, Loss:  0.071
    Epoch   7 Batch  316/538 - Train Accuracy:  0.939, Validation Accuracy:  0.947, Loss:  0.081
    Epoch   7 Batch  317/538 - Train Accuracy:  0.946, Validation Accuracy:  0.947, Loss:  0.097
    Epoch   7 Batch  318/538 - Train Accuracy:  0.940, Validation Accuracy:  0.950, Loss:  0.101
    Epoch   7 Batch  319/538 - Train Accuracy:  0.948, Validation Accuracy:  0.943, Loss:  0.092
    Epoch   7 Batch  320/538 - Train Accuracy:  0.949, Validation Accuracy:  0.943, Loss:  0.085
    Epoch   7 Batch  321/538 - Train Accuracy:  0.941, Validation Accuracy:  0.942, Loss:  0.080
    Epoch   7 Batch  322/538 - Train Accuracy:  0.950, Validation Accuracy:  0.945, Loss:  0.103
    Epoch   7 Batch  323/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.081
    Epoch   7 Batch  324/538 - Train Accuracy:  0.961, Validation Accuracy:  0.945, Loss:  0.089
    Epoch   7 Batch  325/538 - Train Accuracy:  0.949, Validation Accuracy:  0.942, Loss:  0.095
    Epoch   7 Batch  326/538 - Train Accuracy:  0.950, Validation Accuracy:  0.942, Loss:  0.088
    Epoch   7 Batch  327/538 - Train Accuracy:  0.945, Validation Accuracy:  0.943, Loss:  0.103
    Epoch   7 Batch  328/538 - Train Accuracy:  0.965, Validation Accuracy:  0.945, Loss:  0.077
    Epoch   7 Batch  329/538 - Train Accuracy:  0.959, Validation Accuracy:  0.945, Loss:  0.089
    Epoch   7 Batch  330/538 - Train Accuracy:  0.953, Validation Accuracy:  0.942, Loss:  0.093
    Epoch   7 Batch  331/538 - Train Accuracy:  0.952, Validation Accuracy:  0.946, Loss:  0.085
    Epoch   7 Batch  332/538 - Train Accuracy:  0.950, Validation Accuracy:  0.946, Loss:  0.092
    Epoch   7 Batch  333/538 - Train Accuracy:  0.944, Validation Accuracy:  0.945, Loss:  0.090
    Epoch   7 Batch  334/538 - Train Accuracy:  0.950, Validation Accuracy:  0.947, Loss:  0.084
    Epoch   7 Batch  335/538 - Train Accuracy:  0.962, Validation Accuracy:  0.943, Loss:  0.078
    Epoch   7 Batch  336/538 - Train Accuracy:  0.947, Validation Accuracy:  0.945, Loss:  0.090
    Epoch   7 Batch  337/538 - Train Accuracy:  0.946, Validation Accuracy:  0.944, Loss:  0.094
    Epoch   7 Batch  338/538 - Train Accuracy:  0.955, Validation Accuracy:  0.943, Loss:  0.099
    Epoch   7 Batch  339/538 - Train Accuracy:  0.942, Validation Accuracy:  0.945, Loss:  0.090
    Epoch   7 Batch  340/538 - Train Accuracy:  0.935, Validation Accuracy:  0.946, Loss:  0.092
    Epoch   7 Batch  341/538 - Train Accuracy:  0.943, Validation Accuracy:  0.945, Loss:  0.087
    Epoch   7 Batch  342/538 - Train Accuracy:  0.949, Validation Accuracy:  0.948, Loss:  0.096
    Epoch   7 Batch  343/538 - Train Accuracy:  0.959, Validation Accuracy:  0.947, Loss:  0.090
    Epoch   7 Batch  344/538 - Train Accuracy:  0.944, Validation Accuracy:  0.947, Loss:  0.081
    Epoch   7 Batch  345/538 - Train Accuracy:  0.958, Validation Accuracy:  0.947, Loss:  0.086
    Epoch   7 Batch  346/538 - Train Accuracy:  0.929, Validation Accuracy:  0.947, Loss:  0.101
    Epoch   7 Batch  347/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.075
    Epoch   7 Batch  348/538 - Train Accuracy:  0.936, Validation Accuracy:  0.947, Loss:  0.090
    Epoch   7 Batch  349/538 - Train Accuracy:  0.958, Validation Accuracy:  0.947, Loss:  0.072
    Epoch   7 Batch  350/538 - Train Accuracy:  0.945, Validation Accuracy:  0.947, Loss:  0.100
    Epoch   7 Batch  351/538 - Train Accuracy:  0.946, Validation Accuracy:  0.948, Loss:  0.093
    Epoch   7 Batch  352/538 - Train Accuracy:  0.933, Validation Accuracy:  0.946, Loss:  0.111
    Epoch   7 Batch  353/538 - Train Accuracy:  0.941, Validation Accuracy:  0.946, Loss:  0.099
    Epoch   7 Batch  354/538 - Train Accuracy:  0.949, Validation Accuracy:  0.943, Loss:  0.086
    Epoch   7 Batch  355/538 - Train Accuracy:  0.946, Validation Accuracy:  0.946, Loss:  0.094
    Epoch   7 Batch  356/538 - Train Accuracy:  0.944, Validation Accuracy:  0.942, Loss:  0.087
    Epoch   7 Batch  357/538 - Train Accuracy:  0.953, Validation Accuracy:  0.944, Loss:  0.090
    Epoch   7 Batch  358/538 - Train Accuracy:  0.955, Validation Accuracy:  0.947, Loss:  0.071
    Epoch   7 Batch  359/538 - Train Accuracy:  0.944, Validation Accuracy:  0.945, Loss:  0.089
    Epoch   7 Batch  360/538 - Train Accuracy:  0.950, Validation Accuracy:  0.945, Loss:  0.096
    Epoch   7 Batch  361/538 - Train Accuracy:  0.956, Validation Accuracy:  0.946, Loss:  0.087
    Epoch   7 Batch  362/538 - Train Accuracy:  0.954, Validation Accuracy:  0.947, Loss:  0.078
    Epoch   7 Batch  363/538 - Train Accuracy:  0.938, Validation Accuracy:  0.948, Loss:  0.086
    Epoch   7 Batch  364/538 - Train Accuracy:  0.944, Validation Accuracy:  0.946, Loss:  0.109
    Epoch   7 Batch  365/538 - Train Accuracy:  0.937, Validation Accuracy:  0.942, Loss:  0.084
    Epoch   7 Batch  366/538 - Train Accuracy:  0.952, Validation Accuracy:  0.942, Loss:  0.083
    Epoch   7 Batch  367/538 - Train Accuracy:  0.955, Validation Accuracy:  0.936, Loss:  0.069
    Epoch   7 Batch  368/538 - Train Accuracy:  0.960, Validation Accuracy:  0.936, Loss:  0.070
    Epoch   7 Batch  369/538 - Train Accuracy:  0.938, Validation Accuracy:  0.940, Loss:  0.088
    Epoch   7 Batch  370/538 - Train Accuracy:  0.948, Validation Accuracy:  0.938, Loss:  0.100
    Epoch   7 Batch  371/538 - Train Accuracy:  0.957, Validation Accuracy:  0.939, Loss:  0.091
    Epoch   7 Batch  372/538 - Train Accuracy:  0.950, Validation Accuracy:  0.943, Loss:  0.080
    Epoch   7 Batch  373/538 - Train Accuracy:  0.953, Validation Accuracy:  0.943, Loss:  0.076
    Epoch   7 Batch  374/538 - Train Accuracy:  0.953, Validation Accuracy:  0.941, Loss:  0.087
    Epoch   7 Batch  375/538 - Train Accuracy:  0.951, Validation Accuracy:  0.941, Loss:  0.076
    Epoch   7 Batch  376/538 - Train Accuracy:  0.944, Validation Accuracy:  0.941, Loss:  0.086
    Epoch   7 Batch  377/538 - Train Accuracy:  0.961, Validation Accuracy:  0.945, Loss:  0.078
    Epoch   7 Batch  378/538 - Train Accuracy:  0.950, Validation Accuracy:  0.943, Loss:  0.079
    Epoch   7 Batch  379/538 - Train Accuracy:  0.959, Validation Accuracy:  0.943, Loss:  0.085
    Epoch   7 Batch  380/538 - Train Accuracy:  0.950, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   7 Batch  381/538 - Train Accuracy:  0.956, Validation Accuracy:  0.944, Loss:  0.102
    Epoch   7 Batch  382/538 - Train Accuracy:  0.941, Validation Accuracy:  0.936, Loss:  0.089
    Epoch   7 Batch  383/538 - Train Accuracy:  0.946, Validation Accuracy:  0.938, Loss:  0.080
    Epoch   7 Batch  384/538 - Train Accuracy:  0.935, Validation Accuracy:  0.938, Loss:  0.089
    Epoch   7 Batch  385/538 - Train Accuracy:  0.945, Validation Accuracy:  0.939, Loss:  0.088
    Epoch   7 Batch  386/538 - Train Accuracy:  0.945, Validation Accuracy:  0.938, Loss:  0.097
    Epoch   7 Batch  387/538 - Train Accuracy:  0.949, Validation Accuracy:  0.936, Loss:  0.084
    Epoch   7 Batch  388/538 - Train Accuracy:  0.947, Validation Accuracy:  0.938, Loss:  0.090
    Epoch   7 Batch  389/538 - Train Accuracy:  0.941, Validation Accuracy:  0.940, Loss:  0.111
    Epoch   7 Batch  390/538 - Train Accuracy:  0.963, Validation Accuracy:  0.942, Loss:  0.077
    Epoch   7 Batch  391/538 - Train Accuracy:  0.949, Validation Accuracy:  0.940, Loss:  0.081
    Epoch   7 Batch  392/538 - Train Accuracy:  0.944, Validation Accuracy:  0.941, Loss:  0.082
    Epoch   7 Batch  393/538 - Train Accuracy:  0.959, Validation Accuracy:  0.941, Loss:  0.074
    Epoch   7 Batch  394/538 - Train Accuracy:  0.931, Validation Accuracy:  0.936, Loss:  0.089
    Epoch   7 Batch  395/538 - Train Accuracy:  0.942, Validation Accuracy:  0.940, Loss:  0.096
    Epoch   7 Batch  396/538 - Train Accuracy:  0.945, Validation Accuracy:  0.939, Loss:  0.093
    Epoch   7 Batch  397/538 - Train Accuracy:  0.950, Validation Accuracy:  0.941, Loss:  0.092
    Epoch   7 Batch  398/538 - Train Accuracy:  0.940, Validation Accuracy:  0.944, Loss:  0.091
    Epoch   7 Batch  399/538 - Train Accuracy:  0.934, Validation Accuracy:  0.947, Loss:  0.092
    Epoch   7 Batch  400/538 - Train Accuracy:  0.960, Validation Accuracy:  0.949, Loss:  0.090
    Epoch   7 Batch  401/538 - Train Accuracy:  0.962, Validation Accuracy:  0.952, Loss:  0.085
    Epoch   7 Batch  402/538 - Train Accuracy:  0.942, Validation Accuracy:  0.948, Loss:  0.093
    Epoch   7 Batch  403/538 - Train Accuracy:  0.952, Validation Accuracy:  0.943, Loss:  0.091
    Epoch   7 Batch  404/538 - Train Accuracy:  0.962, Validation Accuracy:  0.943, Loss:  0.082
    Epoch   7 Batch  405/538 - Train Accuracy:  0.942, Validation Accuracy:  0.943, Loss:  0.077
    Epoch   7 Batch  406/538 - Train Accuracy:  0.937, Validation Accuracy:  0.943, Loss:  0.089
    Epoch   7 Batch  407/538 - Train Accuracy:  0.961, Validation Accuracy:  0.942, Loss:  0.082
    Epoch   7 Batch  408/538 - Train Accuracy:  0.925, Validation Accuracy:  0.947, Loss:  0.120
    Epoch   7 Batch  409/538 - Train Accuracy:  0.938, Validation Accuracy:  0.947, Loss:  0.093
    Epoch   7 Batch  410/538 - Train Accuracy:  0.955, Validation Accuracy:  0.942, Loss:  0.084
    Epoch   7 Batch  411/538 - Train Accuracy:  0.958, Validation Accuracy:  0.944, Loss:  0.089
    Epoch   7 Batch  412/538 - Train Accuracy:  0.954, Validation Accuracy:  0.938, Loss:  0.076
    Epoch   7 Batch  413/538 - Train Accuracy:  0.960, Validation Accuracy:  0.929, Loss:  0.081
    Epoch   7 Batch  414/538 - Train Accuracy:  0.923, Validation Accuracy:  0.929, Loss:  0.112
    Epoch   7 Batch  415/538 - Train Accuracy:  0.938, Validation Accuracy:  0.928, Loss:  0.085
    Epoch   7 Batch  416/538 - Train Accuracy:  0.954, Validation Accuracy:  0.930, Loss:  0.085
    Epoch   7 Batch  417/538 - Train Accuracy:  0.954, Validation Accuracy:  0.932, Loss:  0.085
    Epoch   7 Batch  418/538 - Train Accuracy:  0.946, Validation Accuracy:  0.936, Loss:  0.107
    Epoch   7 Batch  419/538 - Train Accuracy:  0.962, Validation Accuracy:  0.936, Loss:  0.073
    Epoch   7 Batch  420/538 - Train Accuracy:  0.952, Validation Accuracy:  0.936, Loss:  0.078
    Epoch   7 Batch  421/538 - Train Accuracy:  0.954, Validation Accuracy:  0.938, Loss:  0.076
    Epoch   7 Batch  422/538 - Train Accuracy:  0.929, Validation Accuracy:  0.938, Loss:  0.095
    Epoch   7 Batch  423/538 - Train Accuracy:  0.941, Validation Accuracy:  0.935, Loss:  0.101
    Epoch   7 Batch  424/538 - Train Accuracy:  0.944, Validation Accuracy:  0.936, Loss:  0.085
    Epoch   7 Batch  425/538 - Train Accuracy:  0.934, Validation Accuracy:  0.939, Loss:  0.099
    Epoch   7 Batch  426/538 - Train Accuracy:  0.951, Validation Accuracy:  0.941, Loss:  0.085
    Epoch   7 Batch  427/538 - Train Accuracy:  0.937, Validation Accuracy:  0.940, Loss:  0.089
    Epoch   7 Batch  428/538 - Train Accuracy:  0.953, Validation Accuracy:  0.939, Loss:  0.070
    Epoch   7 Batch  429/538 - Train Accuracy:  0.952, Validation Accuracy:  0.939, Loss:  0.085
    Epoch   7 Batch  430/538 - Train Accuracy:  0.950, Validation Accuracy:  0.939, Loss:  0.088
    Epoch   7 Batch  431/538 - Train Accuracy:  0.939, Validation Accuracy:  0.936, Loss:  0.093
    Epoch   7 Batch  432/538 - Train Accuracy:  0.951, Validation Accuracy:  0.936, Loss:  0.092
    Epoch   7 Batch  433/538 - Train Accuracy:  0.938, Validation Accuracy:  0.938, Loss:  0.107
    Epoch   7 Batch  434/538 - Train Accuracy:  0.948, Validation Accuracy:  0.934, Loss:  0.080
    Epoch   7 Batch  435/538 - Train Accuracy:  0.936, Validation Accuracy:  0.944, Loss:  0.092
    Epoch   7 Batch  436/538 - Train Accuracy:  0.929, Validation Accuracy:  0.947, Loss:  0.100
    Epoch   7 Batch  437/538 - Train Accuracy:  0.955, Validation Accuracy:  0.948, Loss:  0.083
    Epoch   7 Batch  438/538 - Train Accuracy:  0.950, Validation Accuracy:  0.948, Loss:  0.073
    Epoch   7 Batch  439/538 - Train Accuracy:  0.950, Validation Accuracy:  0.946, Loss:  0.080
    Epoch   7 Batch  440/538 - Train Accuracy:  0.949, Validation Accuracy:  0.948, Loss:  0.100
    Epoch   7 Batch  441/538 - Train Accuracy:  0.949, Validation Accuracy:  0.946, Loss:  0.096
    Epoch   7 Batch  442/538 - Train Accuracy:  0.951, Validation Accuracy:  0.944, Loss:  0.074
    Epoch   7 Batch  443/538 - Train Accuracy:  0.945, Validation Accuracy:  0.944, Loss:  0.083
    Epoch   7 Batch  444/538 - Train Accuracy:  0.953, Validation Accuracy:  0.945, Loss:  0.077
    Epoch   7 Batch  445/538 - Train Accuracy:  0.957, Validation Accuracy:  0.944, Loss:  0.073
    Epoch   7 Batch  446/538 - Train Accuracy:  0.959, Validation Accuracy:  0.944, Loss:  0.078
    Epoch   7 Batch  447/538 - Train Accuracy:  0.941, Validation Accuracy:  0.947, Loss:  0.087
    Epoch   7 Batch  448/538 - Train Accuracy:  0.948, Validation Accuracy:  0.946, Loss:  0.075
    Epoch   7 Batch  449/538 - Train Accuracy:  0.954, Validation Accuracy:  0.950, Loss:  0.090
    Epoch   7 Batch  450/538 - Train Accuracy:  0.935, Validation Accuracy:  0.948, Loss:  0.115
    Epoch   7 Batch  451/538 - Train Accuracy:  0.932, Validation Accuracy:  0.950, Loss:  0.083
    Epoch   7 Batch  452/538 - Train Accuracy:  0.954, Validation Accuracy:  0.944, Loss:  0.081
    Epoch   7 Batch  453/538 - Train Accuracy:  0.942, Validation Accuracy:  0.942, Loss:  0.093
    Epoch   7 Batch  454/538 - Train Accuracy:  0.943, Validation Accuracy:  0.943, Loss:  0.082
    Epoch   7 Batch  455/538 - Train Accuracy:  0.947, Validation Accuracy:  0.942, Loss:  0.105
    Epoch   7 Batch  456/538 - Train Accuracy:  0.956, Validation Accuracy:  0.941, Loss:  0.108
    Epoch   7 Batch  457/538 - Train Accuracy:  0.959, Validation Accuracy:  0.945, Loss:  0.080
    Epoch   7 Batch  458/538 - Train Accuracy:  0.955, Validation Accuracy:  0.951, Loss:  0.075
    Epoch   7 Batch  459/538 - Train Accuracy:  0.950, Validation Accuracy:  0.953, Loss:  0.076
    Epoch   7 Batch  460/538 - Train Accuracy:  0.948, Validation Accuracy:  0.949, Loss:  0.089
    Epoch   7 Batch  461/538 - Train Accuracy:  0.955, Validation Accuracy:  0.949, Loss:  0.087
    Epoch   7 Batch  462/538 - Train Accuracy:  0.952, Validation Accuracy:  0.949, Loss:  0.090
    Epoch   7 Batch  463/538 - Train Accuracy:  0.928, Validation Accuracy:  0.949, Loss:  0.090
    Epoch   7 Batch  464/538 - Train Accuracy:  0.950, Validation Accuracy:  0.947, Loss:  0.083
    Epoch   7 Batch  465/538 - Train Accuracy:  0.953, Validation Accuracy:  0.943, Loss:  0.076
    Epoch   7 Batch  466/538 - Train Accuracy:  0.941, Validation Accuracy:  0.944, Loss:  0.083
    Epoch   7 Batch  467/538 - Train Accuracy:  0.951, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   7 Batch  468/538 - Train Accuracy:  0.966, Validation Accuracy:  0.937, Loss:  0.091
    Epoch   7 Batch  469/538 - Train Accuracy:  0.937, Validation Accuracy:  0.938, Loss:  0.092
    Epoch   7 Batch  470/538 - Train Accuracy:  0.949, Validation Accuracy:  0.941, Loss:  0.085
    Epoch   7 Batch  471/538 - Train Accuracy:  0.968, Validation Accuracy:  0.941, Loss:  0.075
    Epoch   7 Batch  472/538 - Train Accuracy:  0.975, Validation Accuracy:  0.942, Loss:  0.067
    Epoch   7 Batch  473/538 - Train Accuracy:  0.942, Validation Accuracy:  0.942, Loss:  0.075
    Epoch   7 Batch  474/538 - Train Accuracy:  0.959, Validation Accuracy:  0.940, Loss:  0.076
    Epoch   7 Batch  475/538 - Train Accuracy:  0.954, Validation Accuracy:  0.943, Loss:  0.072
    Epoch   7 Batch  476/538 - Train Accuracy:  0.956, Validation Accuracy:  0.943, Loss:  0.079
    Epoch   7 Batch  477/538 - Train Accuracy:  0.942, Validation Accuracy:  0.946, Loss:  0.094
    Epoch   7 Batch  478/538 - Train Accuracy:  0.950, Validation Accuracy:  0.943, Loss:  0.075
    Epoch   7 Batch  479/538 - Train Accuracy:  0.957, Validation Accuracy:  0.943, Loss:  0.078
    Epoch   7 Batch  480/538 - Train Accuracy:  0.956, Validation Accuracy:  0.944, Loss:  0.081
    Epoch   7 Batch  481/538 - Train Accuracy:  0.950, Validation Accuracy:  0.946, Loss:  0.082
    Epoch   7 Batch  482/538 - Train Accuracy:  0.942, Validation Accuracy:  0.945, Loss:  0.087
    Epoch   7 Batch  483/538 - Train Accuracy:  0.933, Validation Accuracy:  0.940, Loss:  0.105
    Epoch   7 Batch  484/538 - Train Accuracy:  0.948, Validation Accuracy:  0.938, Loss:  0.106
    Epoch   7 Batch  485/538 - Train Accuracy:  0.949, Validation Accuracy:  0.936, Loss:  0.082
    Epoch   7 Batch  486/538 - Train Accuracy:  0.958, Validation Accuracy:  0.933, Loss:  0.069
    Epoch   7 Batch  487/538 - Train Accuracy:  0.953, Validation Accuracy:  0.930, Loss:  0.070
    Epoch   7 Batch  488/538 - Train Accuracy:  0.952, Validation Accuracy:  0.933, Loss:  0.084
    Epoch   7 Batch  489/538 - Train Accuracy:  0.956, Validation Accuracy:  0.939, Loss:  0.090
    Epoch   7 Batch  490/538 - Train Accuracy:  0.943, Validation Accuracy:  0.938, Loss:  0.083
    Epoch   7 Batch  491/538 - Train Accuracy:  0.924, Validation Accuracy:  0.937, Loss:  0.089
    Epoch   7 Batch  492/538 - Train Accuracy:  0.958, Validation Accuracy:  0.939, Loss:  0.086
    Epoch   7 Batch  493/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.079
    Epoch   7 Batch  494/538 - Train Accuracy:  0.959, Validation Accuracy:  0.943, Loss:  0.089
    Epoch   7 Batch  495/538 - Train Accuracy:  0.949, Validation Accuracy:  0.939, Loss:  0.089
    Epoch   7 Batch  496/538 - Train Accuracy:  0.955, Validation Accuracy:  0.938, Loss:  0.074
    Epoch   7 Batch  497/538 - Train Accuracy:  0.961, Validation Accuracy:  0.940, Loss:  0.076
    Epoch   7 Batch  498/538 - Train Accuracy:  0.955, Validation Accuracy:  0.944, Loss:  0.076
    Epoch   7 Batch  499/538 - Train Accuracy:  0.944, Validation Accuracy:  0.943, Loss:  0.089
    Epoch   7 Batch  500/538 - Train Accuracy:  0.960, Validation Accuracy:  0.944, Loss:  0.067
    Epoch   7 Batch  501/538 - Train Accuracy:  0.956, Validation Accuracy:  0.944, Loss:  0.094
    Epoch   7 Batch  502/538 - Train Accuracy:  0.941, Validation Accuracy:  0.941, Loss:  0.083
    Epoch   7 Batch  503/538 - Train Accuracy:  0.956, Validation Accuracy:  0.946, Loss:  0.085
    Epoch   7 Batch  504/538 - Train Accuracy:  0.958, Validation Accuracy:  0.946, Loss:  0.080
    Epoch   7 Batch  505/538 - Train Accuracy:  0.963, Validation Accuracy:  0.946, Loss:  0.072
    Epoch   7 Batch  506/538 - Train Accuracy:  0.955, Validation Accuracy:  0.942, Loss:  0.072
    Epoch   7 Batch  507/538 - Train Accuracy:  0.931, Validation Accuracy:  0.941, Loss:  0.110
    Epoch   7 Batch  508/538 - Train Accuracy:  0.940, Validation Accuracy:  0.939, Loss:  0.079
    Epoch   7 Batch  509/538 - Train Accuracy:  0.948, Validation Accuracy:  0.939, Loss:  0.087
    Epoch   7 Batch  510/538 - Train Accuracy:  0.960, Validation Accuracy:  0.939, Loss:  0.077
    Epoch   7 Batch  511/538 - Train Accuracy:  0.943, Validation Accuracy:  0.940, Loss:  0.092
    Epoch   7 Batch  512/538 - Train Accuracy:  0.958, Validation Accuracy:  0.939, Loss:  0.082
    Epoch   7 Batch  513/538 - Train Accuracy:  0.934, Validation Accuracy:  0.940, Loss:  0.084
    Epoch   7 Batch  514/538 - Train Accuracy:  0.952, Validation Accuracy:  0.938, Loss:  0.081
    Epoch   7 Batch  515/538 - Train Accuracy:  0.936, Validation Accuracy:  0.936, Loss:  0.097
    Epoch   7 Batch  516/538 - Train Accuracy:  0.945, Validation Accuracy:  0.936, Loss:  0.094
    Epoch   7 Batch  517/538 - Train Accuracy:  0.955, Validation Accuracy:  0.936, Loss:  0.082
    Epoch   7 Batch  518/538 - Train Accuracy:  0.943, Validation Accuracy:  0.937, Loss:  0.083
    Epoch   7 Batch  519/538 - Train Accuracy:  0.949, Validation Accuracy:  0.933, Loss:  0.092
    Epoch   7 Batch  520/538 - Train Accuracy:  0.934, Validation Accuracy:  0.933, Loss:  0.088
    Epoch   7 Batch  521/538 - Train Accuracy:  0.947, Validation Accuracy:  0.940, Loss:  0.091
    Epoch   7 Batch  522/538 - Train Accuracy:  0.956, Validation Accuracy:  0.944, Loss:  0.070
    Epoch   7 Batch  523/538 - Train Accuracy:  0.949, Validation Accuracy:  0.946, Loss:  0.084
    Epoch   7 Batch  524/538 - Train Accuracy:  0.950, Validation Accuracy:  0.949, Loss:  0.101
    Epoch   7 Batch  525/538 - Train Accuracy:  0.945, Validation Accuracy:  0.946, Loss:  0.085
    Epoch   7 Batch  526/538 - Train Accuracy:  0.953, Validation Accuracy:  0.946, Loss:  0.101
    Epoch   7 Batch  527/538 - Train Accuracy:  0.948, Validation Accuracy:  0.945, Loss:  0.078
    Epoch   7 Batch  528/538 - Train Accuracy:  0.946, Validation Accuracy:  0.946, Loss:  0.087
    Epoch   7 Batch  529/538 - Train Accuracy:  0.929, Validation Accuracy:  0.944, Loss:  0.098
    Epoch   7 Batch  530/538 - Train Accuracy:  0.944, Validation Accuracy:  0.940, Loss:  0.093
    Epoch   7 Batch  531/538 - Train Accuracy:  0.944, Validation Accuracy:  0.941, Loss:  0.079
    Epoch   7 Batch  532/538 - Train Accuracy:  0.943, Validation Accuracy:  0.943, Loss:  0.083
    Epoch   7 Batch  533/538 - Train Accuracy:  0.954, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   7 Batch  534/538 - Train Accuracy:  0.953, Validation Accuracy:  0.946, Loss:  0.078
    Epoch   7 Batch  535/538 - Train Accuracy:  0.952, Validation Accuracy:  0.950, Loss:  0.083
    Epoch   7 Batch  536/538 - Train Accuracy:  0.963, Validation Accuracy:  0.950, Loss:  0.094
    Epoch   8 Batch    0/538 - Train Accuracy:  0.972, Validation Accuracy:  0.952, Loss:  0.064
    Epoch   8 Batch    1/538 - Train Accuracy:  0.966, Validation Accuracy:  0.950, Loss:  0.086
    Epoch   8 Batch    2/538 - Train Accuracy:  0.965, Validation Accuracy:  0.944, Loss:  0.097
    Epoch   8 Batch    3/538 - Train Accuracy:  0.954, Validation Accuracy:  0.942, Loss:  0.071
    Epoch   8 Batch    4/538 - Train Accuracy:  0.942, Validation Accuracy:  0.948, Loss:  0.087
    Epoch   8 Batch    5/538 - Train Accuracy:  0.942, Validation Accuracy:  0.946, Loss:  0.086
    Epoch   8 Batch    6/538 - Train Accuracy:  0.950, Validation Accuracy:  0.948, Loss:  0.080
    Epoch   8 Batch    7/538 - Train Accuracy:  0.960, Validation Accuracy:  0.948, Loss:  0.087
    Epoch   8 Batch    8/538 - Train Accuracy:  0.946, Validation Accuracy:  0.950, Loss:  0.086
    Epoch   8 Batch    9/538 - Train Accuracy:  0.950, Validation Accuracy:  0.951, Loss:  0.075
    Epoch   8 Batch   10/538 - Train Accuracy:  0.943, Validation Accuracy:  0.950, Loss:  0.084
    Epoch   8 Batch   11/538 - Train Accuracy:  0.957, Validation Accuracy:  0.949, Loss:  0.079
    Epoch   8 Batch   12/538 - Train Accuracy:  0.954, Validation Accuracy:  0.945, Loss:  0.083
    Epoch   8 Batch   13/538 - Train Accuracy:  0.951, Validation Accuracy:  0.942, Loss:  0.069
    Epoch   8 Batch   14/538 - Train Accuracy:  0.952, Validation Accuracy:  0.941, Loss:  0.074
    Epoch   8 Batch   15/538 - Train Accuracy:  0.949, Validation Accuracy:  0.941, Loss:  0.091
    Epoch   8 Batch   16/538 - Train Accuracy:  0.956, Validation Accuracy:  0.949, Loss:  0.070
    Epoch   8 Batch   17/538 - Train Accuracy:  0.954, Validation Accuracy:  0.946, Loss:  0.076
    Epoch   8 Batch   18/538 - Train Accuracy:  0.954, Validation Accuracy:  0.942, Loss:  0.091
    Epoch   8 Batch   19/538 - Train Accuracy:  0.948, Validation Accuracy:  0.940, Loss:  0.091
    Epoch   8 Batch   20/538 - Train Accuracy:  0.942, Validation Accuracy:  0.940, Loss:  0.089
    Epoch   8 Batch   21/538 - Train Accuracy:  0.961, Validation Accuracy:  0.945, Loss:  0.065
    Epoch   8 Batch   22/538 - Train Accuracy:  0.939, Validation Accuracy:  0.940, Loss:  0.082
    Epoch   8 Batch   23/538 - Train Accuracy:  0.931, Validation Accuracy:  0.944, Loss:  0.098
    Epoch   8 Batch   24/538 - Train Accuracy:  0.953, Validation Accuracy:  0.949, Loss:  0.087
    Epoch   8 Batch   25/538 - Train Accuracy:  0.942, Validation Accuracy:  0.946, Loss:  0.078
    Epoch   8 Batch   26/538 - Train Accuracy:  0.940, Validation Accuracy:  0.947, Loss:  0.088
    Epoch   8 Batch   27/538 - Train Accuracy:  0.958, Validation Accuracy:  0.949, Loss:  0.066
    Epoch   8 Batch   28/538 - Train Accuracy:  0.953, Validation Accuracy:  0.947, Loss:  0.078
    Epoch   8 Batch   29/538 - Train Accuracy:  0.952, Validation Accuracy:  0.946, Loss:  0.076
    Epoch   8 Batch   30/538 - Train Accuracy:  0.946, Validation Accuracy:  0.946, Loss:  0.094
    Epoch   8 Batch   31/538 - Train Accuracy:  0.953, Validation Accuracy:  0.946, Loss:  0.066
    Epoch   8 Batch   32/538 - Train Accuracy:  0.953, Validation Accuracy:  0.947, Loss:  0.073
    Epoch   8 Batch   33/538 - Train Accuracy:  0.938, Validation Accuracy:  0.944, Loss:  0.083
    Epoch   8 Batch   34/538 - Train Accuracy:  0.942, Validation Accuracy:  0.944, Loss:  0.092
    Epoch   8 Batch   35/538 - Train Accuracy:  0.953, Validation Accuracy:  0.945, Loss:  0.067
    Epoch   8 Batch   36/538 - Train Accuracy:  0.948, Validation Accuracy:  0.945, Loss:  0.072
    Epoch   8 Batch   37/538 - Train Accuracy:  0.946, Validation Accuracy:  0.946, Loss:  0.085
    Epoch   8 Batch   38/538 - Train Accuracy:  0.941, Validation Accuracy:  0.948, Loss:  0.090
    Epoch   8 Batch   39/538 - Train Accuracy:  0.956, Validation Accuracy:  0.947, Loss:  0.073
    Epoch   8 Batch   40/538 - Train Accuracy:  0.953, Validation Accuracy:  0.949, Loss:  0.075
    Epoch   8 Batch   41/538 - Train Accuracy:  0.951, Validation Accuracy:  0.947, Loss:  0.077
    Epoch   8 Batch   42/538 - Train Accuracy:  0.955, Validation Accuracy:  0.944, Loss:  0.078
    Epoch   8 Batch   43/538 - Train Accuracy:  0.933, Validation Accuracy:  0.947, Loss:  0.100
    Epoch   8 Batch   44/538 - Train Accuracy:  0.952, Validation Accuracy:  0.951, Loss:  0.081
    Epoch   8 Batch   45/538 - Train Accuracy:  0.959, Validation Accuracy:  0.950, Loss:  0.082
    Epoch   8 Batch   46/538 - Train Accuracy:  0.962, Validation Accuracy:  0.948, Loss:  0.074
    Epoch   8 Batch   47/538 - Train Accuracy:  0.941, Validation Accuracy:  0.944, Loss:  0.088
    Epoch   8 Batch   48/538 - Train Accuracy:  0.952, Validation Accuracy:  0.947, Loss:  0.093
    Epoch   8 Batch   49/538 - Train Accuracy:  0.952, Validation Accuracy:  0.946, Loss:  0.077
    Epoch   8 Batch   50/538 - Train Accuracy:  0.954, Validation Accuracy:  0.949, Loss:  0.081
    Epoch   8 Batch   51/538 - Train Accuracy:  0.939, Validation Accuracy:  0.947, Loss:  0.090
    Epoch   8 Batch   52/538 - Train Accuracy:  0.947, Validation Accuracy:  0.944, Loss:  0.082
    Epoch   8 Batch   53/538 - Train Accuracy:  0.931, Validation Accuracy:  0.941, Loss:  0.084
    Epoch   8 Batch   54/538 - Train Accuracy:  0.960, Validation Accuracy:  0.940, Loss:  0.069
    Epoch   8 Batch   55/538 - Train Accuracy:  0.953, Validation Accuracy:  0.938, Loss:  0.077
    Epoch   8 Batch   56/538 - Train Accuracy:  0.944, Validation Accuracy:  0.942, Loss:  0.082
    Epoch   8 Batch   57/538 - Train Accuracy:  0.931, Validation Accuracy:  0.942, Loss:  0.099
    Epoch   8 Batch   58/538 - Train Accuracy:  0.953, Validation Accuracy:  0.942, Loss:  0.074
    Epoch   8 Batch   59/538 - Train Accuracy:  0.946, Validation Accuracy:  0.941, Loss:  0.081
    Epoch   8 Batch   60/538 - Train Accuracy:  0.943, Validation Accuracy:  0.943, Loss:  0.078
    Epoch   8 Batch   61/538 - Train Accuracy:  0.952, Validation Accuracy:  0.948, Loss:  0.082
    Epoch   8 Batch   62/538 - Train Accuracy:  0.941, Validation Accuracy:  0.948, Loss:  0.078
    Epoch   8 Batch   63/538 - Train Accuracy:  0.959, Validation Accuracy:  0.949, Loss:  0.081
    Epoch   8 Batch   64/538 - Train Accuracy:  0.951, Validation Accuracy:  0.949, Loss:  0.096
    Epoch   8 Batch   65/538 - Train Accuracy:  0.942, Validation Accuracy:  0.949, Loss:  0.085
    Epoch   8 Batch   66/538 - Train Accuracy:  0.961, Validation Accuracy:  0.950, Loss:  0.067
    Epoch   8 Batch   67/538 - Train Accuracy:  0.959, Validation Accuracy:  0.949, Loss:  0.066
    Epoch   8 Batch   68/538 - Train Accuracy:  0.938, Validation Accuracy:  0.951, Loss:  0.067
    Epoch   8 Batch   69/538 - Train Accuracy:  0.959, Validation Accuracy:  0.953, Loss:  0.085
    Epoch   8 Batch   70/538 - Train Accuracy:  0.943, Validation Accuracy:  0.953, Loss:  0.069
    Epoch   8 Batch   71/538 - Train Accuracy:  0.949, Validation Accuracy:  0.952, Loss:  0.093
    Epoch   8 Batch   72/538 - Train Accuracy:  0.954, Validation Accuracy:  0.952, Loss:  0.114
    Epoch   8 Batch   73/538 - Train Accuracy:  0.937, Validation Accuracy:  0.951, Loss:  0.077
    Epoch   8 Batch   74/538 - Train Accuracy:  0.951, Validation Accuracy:  0.949, Loss:  0.066
    Epoch   8 Batch   75/538 - Train Accuracy:  0.933, Validation Accuracy:  0.945, Loss:  0.077
    Epoch   8 Batch   76/538 - Train Accuracy:  0.954, Validation Accuracy:  0.945, Loss:  0.086
    Epoch   8 Batch   77/538 - Train Accuracy:  0.947, Validation Accuracy:  0.945, Loss:  0.076
    Epoch   8 Batch   78/538 - Train Accuracy:  0.942, Validation Accuracy:  0.944, Loss:  0.081
    Epoch   8 Batch   79/538 - Train Accuracy:  0.949, Validation Accuracy:  0.947, Loss:  0.061
    Epoch   8 Batch   80/538 - Train Accuracy:  0.948, Validation Accuracy:  0.947, Loss:  0.085
    Epoch   8 Batch   81/538 - Train Accuracy:  0.938, Validation Accuracy:  0.946, Loss:  0.090
    Epoch   8 Batch   82/538 - Train Accuracy:  0.947, Validation Accuracy:  0.945, Loss:  0.089
    Epoch   8 Batch   83/538 - Train Accuracy:  0.946, Validation Accuracy:  0.949, Loss:  0.076
    Epoch   8 Batch   84/538 - Train Accuracy:  0.937, Validation Accuracy:  0.947, Loss:  0.089
    Epoch   8 Batch   85/538 - Train Accuracy:  0.958, Validation Accuracy:  0.944, Loss:  0.081
    Epoch   8 Batch   86/538 - Train Accuracy:  0.949, Validation Accuracy:  0.937, Loss:  0.074
    Epoch   8 Batch   87/538 - Train Accuracy:  0.944, Validation Accuracy:  0.939, Loss:  0.079
    Epoch   8 Batch   88/538 - Train Accuracy:  0.945, Validation Accuracy:  0.940, Loss:  0.089
    Epoch   8 Batch   89/538 - Train Accuracy:  0.951, Validation Accuracy:  0.940, Loss:  0.077
    Epoch   8 Batch   90/538 - Train Accuracy:  0.952, Validation Accuracy:  0.945, Loss:  0.094
    Epoch   8 Batch   91/538 - Train Accuracy:  0.949, Validation Accuracy:  0.946, Loss:  0.076
    Epoch   8 Batch   92/538 - Train Accuracy:  0.944, Validation Accuracy:  0.951, Loss:  0.078
    Epoch   8 Batch   93/538 - Train Accuracy:  0.956, Validation Accuracy:  0.951, Loss:  0.071
    Epoch   8 Batch   94/538 - Train Accuracy:  0.941, Validation Accuracy:  0.950, Loss:  0.071
    Epoch   8 Batch   95/538 - Train Accuracy:  0.951, Validation Accuracy:  0.950, Loss:  0.070
    Epoch   8 Batch   96/538 - Train Accuracy:  0.958, Validation Accuracy:  0.951, Loss:  0.073
    Epoch   8 Batch   97/538 - Train Accuracy:  0.951, Validation Accuracy:  0.953, Loss:  0.069
    Epoch   8 Batch   98/538 - Train Accuracy:  0.952, Validation Accuracy:  0.955, Loss:  0.077
    Epoch   8 Batch   99/538 - Train Accuracy:  0.954, Validation Accuracy:  0.954, Loss:  0.080
    Epoch   8 Batch  100/538 - Train Accuracy:  0.956, Validation Accuracy:  0.947, Loss:  0.078
    Epoch   8 Batch  101/538 - Train Accuracy:  0.943, Validation Accuracy:  0.947, Loss:  0.091
    Epoch   8 Batch  102/538 - Train Accuracy:  0.933, Validation Accuracy:  0.945, Loss:  0.084
    Epoch   8 Batch  103/538 - Train Accuracy:  0.951, Validation Accuracy:  0.943, Loss:  0.083
    Epoch   8 Batch  104/538 - Train Accuracy:  0.959, Validation Accuracy:  0.950, Loss:  0.064
    Epoch   8 Batch  105/538 - Train Accuracy:  0.947, Validation Accuracy:  0.955, Loss:  0.066
    Epoch   8 Batch  106/538 - Train Accuracy:  0.950, Validation Accuracy:  0.959, Loss:  0.067
    Epoch   8 Batch  107/538 - Train Accuracy:  0.940, Validation Accuracy:  0.959, Loss:  0.085
    Epoch   8 Batch  108/538 - Train Accuracy:  0.962, Validation Accuracy:  0.960, Loss:  0.074
    Epoch   8 Batch  109/538 - Train Accuracy:  0.957, Validation Accuracy:  0.960, Loss:  0.066
    Epoch   8 Batch  110/538 - Train Accuracy:  0.956, Validation Accuracy:  0.955, Loss:  0.091
    Epoch   8 Batch  111/538 - Train Accuracy:  0.943, Validation Accuracy:  0.947, Loss:  0.065
    Epoch   8 Batch  112/538 - Train Accuracy:  0.946, Validation Accuracy:  0.947, Loss:  0.080
    Epoch   8 Batch  113/538 - Train Accuracy:  0.935, Validation Accuracy:  0.947, Loss:  0.088
    Epoch   8 Batch  114/538 - Train Accuracy:  0.954, Validation Accuracy:  0.945, Loss:  0.079
    Epoch   8 Batch  115/538 - Train Accuracy:  0.948, Validation Accuracy:  0.947, Loss:  0.079
    Epoch   8 Batch  116/538 - Train Accuracy:  0.948, Validation Accuracy:  0.950, Loss:  0.096
    Epoch   8 Batch  117/538 - Train Accuracy:  0.948, Validation Accuracy:  0.954, Loss:  0.077
    Epoch   8 Batch  118/538 - Train Accuracy:  0.952, Validation Accuracy:  0.955, Loss:  0.071
    Epoch   8 Batch  119/538 - Train Accuracy:  0.957, Validation Accuracy:  0.950, Loss:  0.064
    Epoch   8 Batch  120/538 - Train Accuracy:  0.951, Validation Accuracy:  0.952, Loss:  0.058
    Epoch   8 Batch  121/538 - Train Accuracy:  0.950, Validation Accuracy:  0.953, Loss:  0.068
    Epoch   8 Batch  122/538 - Train Accuracy:  0.948, Validation Accuracy:  0.954, Loss:  0.075
    Epoch   8 Batch  123/538 - Train Accuracy:  0.932, Validation Accuracy:  0.951, Loss:  0.079
    Epoch   8 Batch  124/538 - Train Accuracy:  0.950, Validation Accuracy:  0.949, Loss:  0.069
    Epoch   8 Batch  125/538 - Train Accuracy:  0.945, Validation Accuracy:  0.949, Loss:  0.082
    Epoch   8 Batch  126/538 - Train Accuracy:  0.935, Validation Accuracy:  0.953, Loss:  0.087
    Epoch   8 Batch  127/538 - Train Accuracy:  0.939, Validation Accuracy:  0.953, Loss:  0.102
    Epoch   8 Batch  128/538 - Train Accuracy:  0.946, Validation Accuracy:  0.952, Loss:  0.079
    Epoch   8 Batch  129/538 - Train Accuracy:  0.957, Validation Accuracy:  0.952, Loss:  0.071
    Epoch   8 Batch  130/538 - Train Accuracy:  0.955, Validation Accuracy:  0.952, Loss:  0.078
    Epoch   8 Batch  131/538 - Train Accuracy:  0.965, Validation Accuracy:  0.950, Loss:  0.072
    Epoch   8 Batch  132/538 - Train Accuracy:  0.959, Validation Accuracy:  0.945, Loss:  0.080
    Epoch   8 Batch  133/538 - Train Accuracy:  0.936, Validation Accuracy:  0.943, Loss:  0.077
    Epoch   8 Batch  134/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.095
    Epoch   8 Batch  135/538 - Train Accuracy:  0.950, Validation Accuracy:  0.944, Loss:  0.098
    Epoch   8 Batch  136/538 - Train Accuracy:  0.951, Validation Accuracy:  0.941, Loss:  0.090
    Epoch   8 Batch  137/538 - Train Accuracy:  0.938, Validation Accuracy:  0.941, Loss:  0.096
    Epoch   8 Batch  138/538 - Train Accuracy:  0.952, Validation Accuracy:  0.940, Loss:  0.080
    Epoch   8 Batch  139/538 - Train Accuracy:  0.938, Validation Accuracy:  0.938, Loss:  0.094
    Epoch   8 Batch  140/538 - Train Accuracy:  0.932, Validation Accuracy:  0.938, Loss:  0.112
    Epoch   8 Batch  141/538 - Train Accuracy:  0.957, Validation Accuracy:  0.939, Loss:  0.101
    Epoch   8 Batch  142/538 - Train Accuracy:  0.945, Validation Accuracy:  0.939, Loss:  0.081
    Epoch   8 Batch  143/538 - Train Accuracy:  0.948, Validation Accuracy:  0.943, Loss:  0.092
    Epoch   8 Batch  144/538 - Train Accuracy:  0.949, Validation Accuracy:  0.945, Loss:  0.090
    Epoch   8 Batch  145/538 - Train Accuracy:  0.936, Validation Accuracy:  0.945, Loss:  0.106
    Epoch   8 Batch  146/538 - Train Accuracy:  0.948, Validation Accuracy:  0.943, Loss:  0.077
    Epoch   8 Batch  147/538 - Train Accuracy:  0.956, Validation Accuracy:  0.943, Loss:  0.081
    Epoch   8 Batch  148/538 - Train Accuracy:  0.936, Validation Accuracy:  0.943, Loss:  0.098
    Epoch   8 Batch  149/538 - Train Accuracy:  0.963, Validation Accuracy:  0.942, Loss:  0.072
    Epoch   8 Batch  150/538 - Train Accuracy:  0.962, Validation Accuracy:  0.939, Loss:  0.062
    Epoch   8 Batch  151/538 - Train Accuracy:  0.948, Validation Accuracy:  0.941, Loss:  0.087
    Epoch   8 Batch  152/538 - Train Accuracy:  0.952, Validation Accuracy:  0.943, Loss:  0.083
    Epoch   8 Batch  153/538 - Train Accuracy:  0.943, Validation Accuracy:  0.944, Loss:  0.081
    Epoch   8 Batch  154/538 - Train Accuracy:  0.955, Validation Accuracy:  0.944, Loss:  0.068
    Epoch   8 Batch  155/538 - Train Accuracy:  0.943, Validation Accuracy:  0.943, Loss:  0.088
    Epoch   8 Batch  156/538 - Train Accuracy:  0.952, Validation Accuracy:  0.946, Loss:  0.078
    Epoch   8 Batch  157/538 - Train Accuracy:  0.953, Validation Accuracy:  0.946, Loss:  0.077
    Epoch   8 Batch  158/538 - Train Accuracy:  0.967, Validation Accuracy:  0.945, Loss:  0.076
    Epoch   8 Batch  159/538 - Train Accuracy:  0.948, Validation Accuracy:  0.945, Loss:  0.075
    Epoch   8 Batch  160/538 - Train Accuracy:  0.947, Validation Accuracy:  0.944, Loss:  0.074
    Epoch   8 Batch  161/538 - Train Accuracy:  0.955, Validation Accuracy:  0.944, Loss:  0.071
    Epoch   8 Batch  162/538 - Train Accuracy:  0.952, Validation Accuracy:  0.945, Loss:  0.078
    Epoch   8 Batch  163/538 - Train Accuracy:  0.949, Validation Accuracy:  0.945, Loss:  0.094
    Epoch   8 Batch  164/538 - Train Accuracy:  0.951, Validation Accuracy:  0.945, Loss:  0.094
    Epoch   8 Batch  165/538 - Train Accuracy:  0.961, Validation Accuracy:  0.945, Loss:  0.070
    Epoch   8 Batch  166/538 - Train Accuracy:  0.959, Validation Accuracy:  0.945, Loss:  0.077
    Epoch   8 Batch  167/538 - Train Accuracy:  0.945, Validation Accuracy:  0.948, Loss:  0.107
    Epoch   8 Batch  168/538 - Train Accuracy:  0.930, Validation Accuracy:  0.948, Loss:  0.097
    Epoch   8 Batch  169/538 - Train Accuracy:  0.966, Validation Accuracy:  0.947, Loss:  0.058
    Epoch   8 Batch  170/538 - Train Accuracy:  0.946, Validation Accuracy:  0.946, Loss:  0.087
    Epoch   8 Batch  171/538 - Train Accuracy:  0.952, Validation Accuracy:  0.945, Loss:  0.070
    Epoch   8 Batch  172/538 - Train Accuracy:  0.938, Validation Accuracy:  0.946, Loss:  0.075
    Epoch   8 Batch  173/538 - Train Accuracy:  0.963, Validation Accuracy:  0.945, Loss:  0.062
    Epoch   8 Batch  174/538 - Train Accuracy:  0.957, Validation Accuracy:  0.946, Loss:  0.072
    Epoch   8 Batch  175/538 - Train Accuracy:  0.956, Validation Accuracy:  0.943, Loss:  0.070
    Epoch   8 Batch  176/538 - Train Accuracy:  0.950, Validation Accuracy:  0.943, Loss:  0.088
    Epoch   8 Batch  177/538 - Train Accuracy:  0.962, Validation Accuracy:  0.942, Loss:  0.079
    Epoch   8 Batch  178/538 - Train Accuracy:  0.936, Validation Accuracy:  0.944, Loss:  0.099
    Epoch   8 Batch  179/538 - Train Accuracy:  0.957, Validation Accuracy:  0.942, Loss:  0.067
    Epoch   8 Batch  180/538 - Train Accuracy:  0.946, Validation Accuracy:  0.939, Loss:  0.069
    Epoch   8 Batch  181/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.088
    Epoch   8 Batch  182/538 - Train Accuracy:  0.961, Validation Accuracy:  0.945, Loss:  0.075
    Epoch   8 Batch  183/538 - Train Accuracy:  0.954, Validation Accuracy:  0.945, Loss:  0.062
    Epoch   8 Batch  184/538 - Train Accuracy:  0.954, Validation Accuracy:  0.950, Loss:  0.076
    Epoch   8 Batch  185/538 - Train Accuracy:  0.963, Validation Accuracy:  0.952, Loss:  0.061
    Epoch   8 Batch  186/538 - Train Accuracy:  0.956, Validation Accuracy:  0.951, Loss:  0.086
    Epoch   8 Batch  187/538 - Train Accuracy:  0.956, Validation Accuracy:  0.950, Loss:  0.069
    Epoch   8 Batch  188/538 - Train Accuracy:  0.946, Validation Accuracy:  0.950, Loss:  0.069
    Epoch   8 Batch  189/538 - Train Accuracy:  0.957, Validation Accuracy:  0.950, Loss:  0.086
    Epoch   8 Batch  190/538 - Train Accuracy:  0.923, Validation Accuracy:  0.948, Loss:  0.100
    Epoch   8 Batch  191/538 - Train Accuracy:  0.961, Validation Accuracy:  0.949, Loss:  0.072
    Epoch   8 Batch  192/538 - Train Accuracy:  0.951, Validation Accuracy:  0.951, Loss:  0.087
    Epoch   8 Batch  193/538 - Train Accuracy:  0.951, Validation Accuracy:  0.949, Loss:  0.069
    Epoch   8 Batch  194/538 - Train Accuracy:  0.938, Validation Accuracy:  0.947, Loss:  0.088
    Epoch   8 Batch  195/538 - Train Accuracy:  0.956, Validation Accuracy:  0.945, Loss:  0.082
    Epoch   8 Batch  196/538 - Train Accuracy:  0.943, Validation Accuracy:  0.946, Loss:  0.074
    Epoch   8 Batch  197/538 - Train Accuracy:  0.955, Validation Accuracy:  0.945, Loss:  0.081
    Epoch   8 Batch  198/538 - Train Accuracy:  0.949, Validation Accuracy:  0.945, Loss:  0.081
    Epoch   8 Batch  199/538 - Train Accuracy:  0.945, Validation Accuracy:  0.952, Loss:  0.093
    Epoch   8 Batch  200/538 - Train Accuracy:  0.963, Validation Accuracy:  0.953, Loss:  0.063
    Epoch   8 Batch  201/538 - Train Accuracy:  0.957, Validation Accuracy:  0.947, Loss:  0.092
    Epoch   8 Batch  202/538 - Train Accuracy:  0.966, Validation Accuracy:  0.950, Loss:  0.076
    Epoch   8 Batch  203/538 - Train Accuracy:  0.956, Validation Accuracy:  0.952, Loss:  0.091
    Epoch   8 Batch  204/538 - Train Accuracy:  0.935, Validation Accuracy:  0.954, Loss:  0.090
    Epoch   8 Batch  205/538 - Train Accuracy:  0.962, Validation Accuracy:  0.952, Loss:  0.077
    Epoch   8 Batch  206/538 - Train Accuracy:  0.952, Validation Accuracy:  0.951, Loss:  0.082
    Epoch   8 Batch  207/538 - Train Accuracy:  0.962, Validation Accuracy:  0.951, Loss:  0.072
    Epoch   8 Batch  208/538 - Train Accuracy:  0.939, Validation Accuracy:  0.954, Loss:  0.093
    Epoch   8 Batch  209/538 - Train Accuracy:  0.951, Validation Accuracy:  0.955, Loss:  0.078
    Epoch   8 Batch  210/538 - Train Accuracy:  0.957, Validation Accuracy:  0.953, Loss:  0.091
    Epoch   8 Batch  211/538 - Train Accuracy:  0.947, Validation Accuracy:  0.948, Loss:  0.086
    Epoch   8 Batch  212/538 - Train Accuracy:  0.931, Validation Accuracy:  0.950, Loss:  0.086
    Epoch   8 Batch  213/538 - Train Accuracy:  0.962, Validation Accuracy:  0.952, Loss:  0.068
    Epoch   8 Batch  214/538 - Train Accuracy:  0.950, Validation Accuracy:  0.953, Loss:  0.067
    Epoch   8 Batch  215/538 - Train Accuracy:  0.959, Validation Accuracy:  0.949, Loss:  0.072
    Epoch   8 Batch  216/538 - Train Accuracy:  0.951, Validation Accuracy:  0.947, Loss:  0.082
    Epoch   8 Batch  217/538 - Train Accuracy:  0.948, Validation Accuracy:  0.947, Loss:  0.084
    Epoch   8 Batch  218/538 - Train Accuracy:  0.951, Validation Accuracy:  0.949, Loss:  0.071
    Epoch   8 Batch  219/538 - Train Accuracy:  0.944, Validation Accuracy:  0.945, Loss:  0.094
    Epoch   8 Batch  220/538 - Train Accuracy:  0.941, Validation Accuracy:  0.945, Loss:  0.082
    Epoch   8 Batch  221/538 - Train Accuracy:  0.964, Validation Accuracy:  0.945, Loss:  0.070
    Epoch   8 Batch  222/538 - Train Accuracy:  0.941, Validation Accuracy:  0.944, Loss:  0.075
    Epoch   8 Batch  223/538 - Train Accuracy:  0.946, Validation Accuracy:  0.941, Loss:  0.085
    Epoch   8 Batch  224/538 - Train Accuracy:  0.950, Validation Accuracy:  0.942, Loss:  0.100
    Epoch   8 Batch  225/538 - Train Accuracy:  0.954, Validation Accuracy:  0.944, Loss:  0.082
    Epoch   8 Batch  226/538 - Train Accuracy:  0.947, Validation Accuracy:  0.945, Loss:  0.080
    Epoch   8 Batch  227/538 - Train Accuracy:  0.939, Validation Accuracy:  0.944, Loss:  0.080
    Epoch   8 Batch  228/538 - Train Accuracy:  0.937, Validation Accuracy:  0.944, Loss:  0.081
    Epoch   8 Batch  229/538 - Train Accuracy:  0.945, Validation Accuracy:  0.945, Loss:  0.082
    Epoch   8 Batch  230/538 - Train Accuracy:  0.946, Validation Accuracy:  0.945, Loss:  0.075
    Epoch   8 Batch  231/538 - Train Accuracy:  0.948, Validation Accuracy:  0.945, Loss:  0.084
    Epoch   8 Batch  232/538 - Train Accuracy:  0.938, Validation Accuracy:  0.947, Loss:  0.073
    Epoch   8 Batch  233/538 - Train Accuracy:  0.953, Validation Accuracy:  0.947, Loss:  0.083
    Epoch   8 Batch  234/538 - Train Accuracy:  0.949, Validation Accuracy:  0.943, Loss:  0.077
    Epoch   8 Batch  235/538 - Train Accuracy:  0.953, Validation Accuracy:  0.943, Loss:  0.064
    Epoch   8 Batch  236/538 - Train Accuracy:  0.950, Validation Accuracy:  0.945, Loss:  0.096
    Epoch   8 Batch  237/538 - Train Accuracy:  0.953, Validation Accuracy:  0.947, Loss:  0.075
    Epoch   8 Batch  238/538 - Train Accuracy:  0.961, Validation Accuracy:  0.943, Loss:  0.073
    Epoch   8 Batch  239/538 - Train Accuracy:  0.948, Validation Accuracy:  0.949, Loss:  0.081
    Epoch   8 Batch  240/538 - Train Accuracy:  0.945, Validation Accuracy:  0.945, Loss:  0.092
    Epoch   8 Batch  241/538 - Train Accuracy:  0.932, Validation Accuracy:  0.946, Loss:  0.088
    Epoch   8 Batch  242/538 - Train Accuracy:  0.954, Validation Accuracy:  0.944, Loss:  0.083
    Epoch   8 Batch  243/538 - Train Accuracy:  0.966, Validation Accuracy:  0.941, Loss:  0.071
    Epoch   8 Batch  244/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.079
    Epoch   8 Batch  245/538 - Train Accuracy:  0.932, Validation Accuracy:  0.941, Loss:  0.110
    Epoch   8 Batch  246/538 - Train Accuracy:  0.954, Validation Accuracy:  0.942, Loss:  0.066
    Epoch   8 Batch  247/538 - Train Accuracy:  0.939, Validation Accuracy:  0.943, Loss:  0.077
    Epoch   8 Batch  248/538 - Train Accuracy:  0.952, Validation Accuracy:  0.945, Loss:  0.082
    Epoch   8 Batch  249/538 - Train Accuracy:  0.960, Validation Accuracy:  0.948, Loss:  0.066
    Epoch   8 Batch  250/538 - Train Accuracy:  0.960, Validation Accuracy:  0.948, Loss:  0.065
    Epoch   8 Batch  251/538 - Train Accuracy:  0.950, Validation Accuracy:  0.948, Loss:  0.066
    Epoch   8 Batch  252/538 - Train Accuracy:  0.949, Validation Accuracy:  0.948, Loss:  0.070
    Epoch   8 Batch  253/538 - Train Accuracy:  0.947, Validation Accuracy:  0.947, Loss:  0.065
    Epoch   8 Batch  254/538 - Train Accuracy:  0.940, Validation Accuracy:  0.947, Loss:  0.075
    Epoch   8 Batch  255/538 - Train Accuracy:  0.969, Validation Accuracy:  0.947, Loss:  0.069
    Epoch   8 Batch  256/538 - Train Accuracy:  0.943, Validation Accuracy:  0.947, Loss:  0.089
    Epoch   8 Batch  257/538 - Train Accuracy:  0.958, Validation Accuracy:  0.947, Loss:  0.073
    Epoch   8 Batch  258/538 - Train Accuracy:  0.958, Validation Accuracy:  0.947, Loss:  0.093
    Epoch   8 Batch  259/538 - Train Accuracy:  0.956, Validation Accuracy:  0.948, Loss:  0.075
    Epoch   8 Batch  260/538 - Train Accuracy:  0.936, Validation Accuracy:  0.948, Loss:  0.088
    Epoch   8 Batch  261/538 - Train Accuracy:  0.957, Validation Accuracy:  0.949, Loss:  0.090
    Epoch   8 Batch  262/538 - Train Accuracy:  0.950, Validation Accuracy:  0.947, Loss:  0.074
    Epoch   8 Batch  263/538 - Train Accuracy:  0.942, Validation Accuracy:  0.947, Loss:  0.085
    Epoch   8 Batch  264/538 - Train Accuracy:  0.953, Validation Accuracy:  0.949, Loss:  0.084
    Epoch   8 Batch  265/538 - Train Accuracy:  0.934, Validation Accuracy:  0.945, Loss:  0.094
    Epoch   8 Batch  266/538 - Train Accuracy:  0.943, Validation Accuracy:  0.946, Loss:  0.091
    Epoch   8 Batch  267/538 - Train Accuracy:  0.942, Validation Accuracy:  0.946, Loss:  0.071
    Epoch   8 Batch  268/538 - Train Accuracy:  0.962, Validation Accuracy:  0.947, Loss:  0.055
    Epoch   8 Batch  269/538 - Train Accuracy:  0.952, Validation Accuracy:  0.948, Loss:  0.092
    Epoch   8 Batch  270/538 - Train Accuracy:  0.950, Validation Accuracy:  0.948, Loss:  0.069
    Epoch   8 Batch  271/538 - Train Accuracy:  0.948, Validation Accuracy:  0.949, Loss:  0.076
    Epoch   8 Batch  272/538 - Train Accuracy:  0.950, Validation Accuracy:  0.948, Loss:  0.082
    Epoch   8 Batch  273/538 - Train Accuracy:  0.947, Validation Accuracy:  0.945, Loss:  0.094
    Epoch   8 Batch  274/538 - Train Accuracy:  0.927, Validation Accuracy:  0.945, Loss:  0.101
    Epoch   8 Batch  275/538 - Train Accuracy:  0.943, Validation Accuracy:  0.949, Loss:  0.099
    Epoch   8 Batch  276/538 - Train Accuracy:  0.938, Validation Accuracy:  0.949, Loss:  0.091
    Epoch   8 Batch  277/538 - Train Accuracy:  0.951, Validation Accuracy:  0.950, Loss:  0.073
    Epoch   8 Batch  278/538 - Train Accuracy:  0.953, Validation Accuracy:  0.948, Loss:  0.067
    Epoch   8 Batch  279/538 - Train Accuracy:  0.938, Validation Accuracy:  0.946, Loss:  0.081
    Epoch   8 Batch  280/538 - Train Accuracy:  0.947, Validation Accuracy:  0.944, Loss:  0.065
    Epoch   8 Batch  281/538 - Train Accuracy:  0.950, Validation Accuracy:  0.941, Loss:  0.082
    Epoch   8 Batch  282/538 - Train Accuracy:  0.954, Validation Accuracy:  0.942, Loss:  0.086
    Epoch   8 Batch  283/538 - Train Accuracy:  0.954, Validation Accuracy:  0.944, Loss:  0.075
    Epoch   8 Batch  284/538 - Train Accuracy:  0.950, Validation Accuracy:  0.949, Loss:  0.088
    Epoch   8 Batch  285/538 - Train Accuracy:  0.956, Validation Accuracy:  0.951, Loss:  0.069
    Epoch   8 Batch  286/538 - Train Accuracy:  0.947, Validation Accuracy:  0.950, Loss:  0.086
    Epoch   8 Batch  287/538 - Train Accuracy:  0.961, Validation Accuracy:  0.951, Loss:  0.066
    Epoch   8 Batch  288/538 - Train Accuracy:  0.950, Validation Accuracy:  0.949, Loss:  0.076
    Epoch   8 Batch  289/538 - Train Accuracy:  0.948, Validation Accuracy:  0.949, Loss:  0.062
    Epoch   8 Batch  290/538 - Train Accuracy:  0.956, Validation Accuracy:  0.948, Loss:  0.061
    Epoch   8 Batch  291/538 - Train Accuracy:  0.953, Validation Accuracy:  0.950, Loss:  0.078
    Epoch   8 Batch  292/538 - Train Accuracy:  0.958, Validation Accuracy:  0.952, Loss:  0.062
    Epoch   8 Batch  293/538 - Train Accuracy:  0.952, Validation Accuracy:  0.952, Loss:  0.072
    Epoch   8 Batch  294/538 - Train Accuracy:  0.950, Validation Accuracy:  0.956, Loss:  0.082
    Epoch   8 Batch  295/538 - Train Accuracy:  0.958, Validation Accuracy:  0.954, Loss:  0.079
    Epoch   8 Batch  296/538 - Train Accuracy:  0.946, Validation Accuracy:  0.952, Loss:  0.085
    Epoch   8 Batch  297/538 - Train Accuracy:  0.968, Validation Accuracy:  0.951, Loss:  0.072
    Epoch   8 Batch  298/538 - Train Accuracy:  0.950, Validation Accuracy:  0.950, Loss:  0.075
    Epoch   8 Batch  299/538 - Train Accuracy:  0.948, Validation Accuracy:  0.948, Loss:  0.087
    Epoch   8 Batch  300/538 - Train Accuracy:  0.951, Validation Accuracy:  0.949, Loss:  0.080
    Epoch   8 Batch  301/538 - Train Accuracy:  0.942, Validation Accuracy:  0.948, Loss:  0.092
    Epoch   8 Batch  302/538 - Train Accuracy:  0.960, Validation Accuracy:  0.949, Loss:  0.071
    Epoch   8 Batch  303/538 - Train Accuracy:  0.959, Validation Accuracy:  0.949, Loss:  0.075
    Epoch   8 Batch  304/538 - Train Accuracy:  0.948, Validation Accuracy:  0.950, Loss:  0.080
    Epoch   8 Batch  305/538 - Train Accuracy:  0.963, Validation Accuracy:  0.950, Loss:  0.065
    Epoch   8 Batch  306/538 - Train Accuracy:  0.947, Validation Accuracy:  0.950, Loss:  0.082
    Epoch   8 Batch  307/538 - Train Accuracy:  0.966, Validation Accuracy:  0.952, Loss:  0.073
    Epoch   8 Batch  308/538 - Train Accuracy:  0.955, Validation Accuracy:  0.952, Loss:  0.083
    Epoch   8 Batch  309/538 - Train Accuracy:  0.951, Validation Accuracy:  0.953, Loss:  0.062
    Epoch   8 Batch  310/538 - Train Accuracy:  0.967, Validation Accuracy:  0.953, Loss:  0.077
    Epoch   8 Batch  311/538 - Train Accuracy:  0.942, Validation Accuracy:  0.948, Loss:  0.079
    Epoch   8 Batch  312/538 - Train Accuracy:  0.952, Validation Accuracy:  0.948, Loss:  0.073
    Epoch   8 Batch  313/538 - Train Accuracy:  0.947, Validation Accuracy:  0.951, Loss:  0.081
    Epoch   8 Batch  314/538 - Train Accuracy:  0.953, Validation Accuracy:  0.953, Loss:  0.075
    Epoch   8 Batch  315/538 - Train Accuracy:  0.949, Validation Accuracy:  0.954, Loss:  0.067
    Epoch   8 Batch  316/538 - Train Accuracy:  0.951, Validation Accuracy:  0.951, Loss:  0.065
    Epoch   8 Batch  317/538 - Train Accuracy:  0.949, Validation Accuracy:  0.951, Loss:  0.080
    Epoch   8 Batch  318/538 - Train Accuracy:  0.944, Validation Accuracy:  0.951, Loss:  0.074
    Epoch   8 Batch  319/538 - Train Accuracy:  0.951, Validation Accuracy:  0.952, Loss:  0.074
    Epoch   8 Batch  320/538 - Train Accuracy:  0.960, Validation Accuracy:  0.951, Loss:  0.073
    Epoch   8 Batch  321/538 - Train Accuracy:  0.950, Validation Accuracy:  0.952, Loss:  0.071
    Epoch   8 Batch  322/538 - Train Accuracy:  0.957, Validation Accuracy:  0.952, Loss:  0.087
    Epoch   8 Batch  323/538 - Train Accuracy:  0.945, Validation Accuracy:  0.953, Loss:  0.072
    Epoch   8 Batch  324/538 - Train Accuracy:  0.965, Validation Accuracy:  0.956, Loss:  0.078
    Epoch   8 Batch  325/538 - Train Accuracy:  0.956, Validation Accuracy:  0.957, Loss:  0.073
    Epoch   8 Batch  326/538 - Train Accuracy:  0.953, Validation Accuracy:  0.957, Loss:  0.074
    Epoch   8 Batch  327/538 - Train Accuracy:  0.939, Validation Accuracy:  0.956, Loss:  0.086
    Epoch   8 Batch  328/538 - Train Accuracy:  0.963, Validation Accuracy:  0.954, Loss:  0.063
    Epoch   8 Batch  329/538 - Train Accuracy:  0.963, Validation Accuracy:  0.951, Loss:  0.075
    Epoch   8 Batch  330/538 - Train Accuracy:  0.954, Validation Accuracy:  0.953, Loss:  0.068
    Epoch   8 Batch  331/538 - Train Accuracy:  0.955, Validation Accuracy:  0.953, Loss:  0.074
    Epoch   8 Batch  332/538 - Train Accuracy:  0.953, Validation Accuracy:  0.952, Loss:  0.078
    Epoch   8 Batch  333/538 - Train Accuracy:  0.953, Validation Accuracy:  0.950, Loss:  0.075
    Epoch   8 Batch  334/538 - Train Accuracy:  0.953, Validation Accuracy:  0.951, Loss:  0.072
    Epoch   8 Batch  335/538 - Train Accuracy:  0.954, Validation Accuracy:  0.936, Loss:  0.078
    Epoch   8 Batch  336/538 - Train Accuracy:  0.952, Validation Accuracy:  0.933, Loss:  0.077
    Epoch   8 Batch  337/538 - Train Accuracy:  0.950, Validation Accuracy:  0.938, Loss:  0.085
    Epoch   8 Batch  338/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.079
    Epoch   8 Batch  339/538 - Train Accuracy:  0.951, Validation Accuracy:  0.941, Loss:  0.072
    Epoch   8 Batch  340/538 - Train Accuracy:  0.930, Validation Accuracy:  0.942, Loss:  0.083
    Epoch   8 Batch  341/538 - Train Accuracy:  0.943, Validation Accuracy:  0.944, Loss:  0.072
    Epoch   8 Batch  342/538 - Train Accuracy:  0.944, Validation Accuracy:  0.943, Loss:  0.072
    Epoch   8 Batch  343/538 - Train Accuracy:  0.958, Validation Accuracy:  0.944, Loss:  0.079
    Epoch   8 Batch  344/538 - Train Accuracy:  0.951, Validation Accuracy:  0.942, Loss:  0.073
    Epoch   8 Batch  345/538 - Train Accuracy:  0.955, Validation Accuracy:  0.945, Loss:  0.077
    Epoch   8 Batch  346/538 - Train Accuracy:  0.942, Validation Accuracy:  0.949, Loss:  0.092
    Epoch   8 Batch  347/538 - Train Accuracy:  0.952, Validation Accuracy:  0.949, Loss:  0.065
    Epoch   8 Batch  348/538 - Train Accuracy:  0.946, Validation Accuracy:  0.949, Loss:  0.072
    Epoch   8 Batch  349/538 - Train Accuracy:  0.968, Validation Accuracy:  0.951, Loss:  0.061
    Epoch   8 Batch  350/538 - Train Accuracy:  0.953, Validation Accuracy:  0.952, Loss:  0.093
    Epoch   8 Batch  351/538 - Train Accuracy:  0.950, Validation Accuracy:  0.955, Loss:  0.076
    Epoch   8 Batch  352/538 - Train Accuracy:  0.937, Validation Accuracy:  0.948, Loss:  0.109
    Epoch   8 Batch  353/538 - Train Accuracy:  0.940, Validation Accuracy:  0.941, Loss:  0.088
    Epoch   8 Batch  354/538 - Train Accuracy:  0.938, Validation Accuracy:  0.941, Loss:  0.078
    Epoch   8 Batch  355/538 - Train Accuracy:  0.950, Validation Accuracy:  0.943, Loss:  0.095
    Epoch   8 Batch  356/538 - Train Accuracy:  0.946, Validation Accuracy:  0.944, Loss:  0.064
    Epoch   8 Batch  357/538 - Train Accuracy:  0.962, Validation Accuracy:  0.945, Loss:  0.079
    Epoch   8 Batch  358/538 - Train Accuracy:  0.958, Validation Accuracy:  0.949, Loss:  0.065
    Epoch   8 Batch  359/538 - Train Accuracy:  0.949, Validation Accuracy:  0.949, Loss:  0.082
    Epoch   8 Batch  360/538 - Train Accuracy:  0.945, Validation Accuracy:  0.951, Loss:  0.089
    Epoch   8 Batch  361/538 - Train Accuracy:  0.956, Validation Accuracy:  0.949, Loss:  0.076
    Epoch   8 Batch  362/538 - Train Accuracy:  0.955, Validation Accuracy:  0.947, Loss:  0.073
    Epoch   8 Batch  363/538 - Train Accuracy:  0.942, Validation Accuracy:  0.948, Loss:  0.085
    Epoch   8 Batch  364/538 - Train Accuracy:  0.953, Validation Accuracy:  0.949, Loss:  0.088
    Epoch   8 Batch  365/538 - Train Accuracy:  0.936, Validation Accuracy:  0.944, Loss:  0.083
    Epoch   8 Batch  366/538 - Train Accuracy:  0.955, Validation Accuracy:  0.947, Loss:  0.073
    Epoch   8 Batch  367/538 - Train Accuracy:  0.947, Validation Accuracy:  0.948, Loss:  0.062
    Epoch   8 Batch  368/538 - Train Accuracy:  0.958, Validation Accuracy:  0.949, Loss:  0.061
    Epoch   8 Batch  369/538 - Train Accuracy:  0.958, Validation Accuracy:  0.950, Loss:  0.066
    Epoch   8 Batch  370/538 - Train Accuracy:  0.945, Validation Accuracy:  0.952, Loss:  0.087
    Epoch   8 Batch  371/538 - Train Accuracy:  0.961, Validation Accuracy:  0.950, Loss:  0.085
    Epoch   8 Batch  372/538 - Train Accuracy:  0.955, Validation Accuracy:  0.948, Loss:  0.068
    Epoch   8 Batch  373/538 - Train Accuracy:  0.952, Validation Accuracy:  0.948, Loss:  0.068
    Epoch   8 Batch  374/538 - Train Accuracy:  0.951, Validation Accuracy:  0.948, Loss:  0.077
    Epoch   8 Batch  375/538 - Train Accuracy:  0.955, Validation Accuracy:  0.950, Loss:  0.072
    Epoch   8 Batch  376/538 - Train Accuracy:  0.955, Validation Accuracy:  0.946, Loss:  0.082
    Epoch   8 Batch  377/538 - Train Accuracy:  0.961, Validation Accuracy:  0.947, Loss:  0.072
    Epoch   8 Batch  378/538 - Train Accuracy:  0.938, Validation Accuracy:  0.946, Loss:  0.069
    Epoch   8 Batch  379/538 - Train Accuracy:  0.961, Validation Accuracy:  0.949, Loss:  0.075
    Epoch   8 Batch  380/538 - Train Accuracy:  0.959, Validation Accuracy:  0.948, Loss:  0.071
    Epoch   8 Batch  381/538 - Train Accuracy:  0.968, Validation Accuracy:  0.947, Loss:  0.070
    Epoch   8 Batch  382/538 - Train Accuracy:  0.944, Validation Accuracy:  0.944, Loss:  0.080
    Epoch   8 Batch  383/538 - Train Accuracy:  0.949, Validation Accuracy:  0.946, Loss:  0.068
    Epoch   8 Batch  384/538 - Train Accuracy:  0.943, Validation Accuracy:  0.944, Loss:  0.080
    Epoch   8 Batch  385/538 - Train Accuracy:  0.946, Validation Accuracy:  0.945, Loss:  0.077
    Epoch   8 Batch  386/538 - Train Accuracy:  0.955, Validation Accuracy:  0.947, Loss:  0.081
    Epoch   8 Batch  387/538 - Train Accuracy:  0.956, Validation Accuracy:  0.949, Loss:  0.075
    Epoch   8 Batch  388/538 - Train Accuracy:  0.960, Validation Accuracy:  0.947, Loss:  0.081
    Epoch   8 Batch  389/538 - Train Accuracy:  0.943, Validation Accuracy:  0.948, Loss:  0.092
    Epoch   8 Batch  390/538 - Train Accuracy:  0.961, Validation Accuracy:  0.941, Loss:  0.065
    Epoch   8 Batch  391/538 - Train Accuracy:  0.953, Validation Accuracy:  0.940, Loss:  0.077
    Epoch   8 Batch  392/538 - Train Accuracy:  0.947, Validation Accuracy:  0.940, Loss:  0.070
    Epoch   8 Batch  393/538 - Train Accuracy:  0.957, Validation Accuracy:  0.940, Loss:  0.074
    Epoch   8 Batch  394/538 - Train Accuracy:  0.944, Validation Accuracy:  0.942, Loss:  0.090
    Epoch   8 Batch  395/538 - Train Accuracy:  0.949, Validation Accuracy:  0.943, Loss:  0.086
    Epoch   8 Batch  396/538 - Train Accuracy:  0.942, Validation Accuracy:  0.941, Loss:  0.081
    Epoch   8 Batch  397/538 - Train Accuracy:  0.944, Validation Accuracy:  0.938, Loss:  0.096
    Epoch   8 Batch  398/538 - Train Accuracy:  0.932, Validation Accuracy:  0.937, Loss:  0.075
    Epoch   8 Batch  399/538 - Train Accuracy:  0.946, Validation Accuracy:  0.937, Loss:  0.082
    Epoch   8 Batch  400/538 - Train Accuracy:  0.948, Validation Accuracy:  0.939, Loss:  0.089
    Epoch   8 Batch  401/538 - Train Accuracy:  0.960, Validation Accuracy:  0.941, Loss:  0.073
    Epoch   8 Batch  402/538 - Train Accuracy:  0.950, Validation Accuracy:  0.944, Loss:  0.077
    Epoch   8 Batch  403/538 - Train Accuracy:  0.955, Validation Accuracy:  0.947, Loss:  0.082
    Epoch   8 Batch  404/538 - Train Accuracy:  0.953, Validation Accuracy:  0.945, Loss:  0.079
    Epoch   8 Batch  405/538 - Train Accuracy:  0.952, Validation Accuracy:  0.943, Loss:  0.065
    Epoch   8 Batch  406/538 - Train Accuracy:  0.940, Validation Accuracy:  0.943, Loss:  0.087
    Epoch   8 Batch  407/538 - Train Accuracy:  0.957, Validation Accuracy:  0.943, Loss:  0.082
    Epoch   8 Batch  408/538 - Train Accuracy:  0.935, Validation Accuracy:  0.936, Loss:  0.100
    Epoch   8 Batch  409/538 - Train Accuracy:  0.946, Validation Accuracy:  0.932, Loss:  0.080
    Epoch   8 Batch  410/538 - Train Accuracy:  0.954, Validation Accuracy:  0.936, Loss:  0.069
    Epoch   8 Batch  411/538 - Train Accuracy:  0.950, Validation Accuracy:  0.936, Loss:  0.077
    Epoch   8 Batch  412/538 - Train Accuracy:  0.965, Validation Accuracy:  0.934, Loss:  0.067
    Epoch   8 Batch  413/538 - Train Accuracy:  0.957, Validation Accuracy:  0.938, Loss:  0.079
    Epoch   8 Batch  414/538 - Train Accuracy:  0.937, Validation Accuracy:  0.937, Loss:  0.101
    Epoch   8 Batch  415/538 - Train Accuracy:  0.936, Validation Accuracy:  0.940, Loss:  0.077
    Epoch   8 Batch  416/538 - Train Accuracy:  0.954, Validation Accuracy:  0.939, Loss:  0.076
    Epoch   8 Batch  417/538 - Train Accuracy:  0.956, Validation Accuracy:  0.939, Loss:  0.078
    Epoch   8 Batch  418/538 - Train Accuracy:  0.955, Validation Accuracy:  0.938, Loss:  0.097
    Epoch   8 Batch  419/538 - Train Accuracy:  0.961, Validation Accuracy:  0.939, Loss:  0.061
    Epoch   8 Batch  420/538 - Train Accuracy:  0.954, Validation Accuracy:  0.940, Loss:  0.077
    Epoch   8 Batch  421/538 - Train Accuracy:  0.949, Validation Accuracy:  0.937, Loss:  0.071
    Epoch   8 Batch  422/538 - Train Accuracy:  0.953, Validation Accuracy:  0.940, Loss:  0.084
    Epoch   8 Batch  423/538 - Train Accuracy:  0.955, Validation Accuracy:  0.938, Loss:  0.082
    Epoch   8 Batch  424/538 - Train Accuracy:  0.947, Validation Accuracy:  0.939, Loss:  0.079
    Epoch   8 Batch  425/538 - Train Accuracy:  0.932, Validation Accuracy:  0.939, Loss:  0.088
    Epoch   8 Batch  426/538 - Train Accuracy:  0.953, Validation Accuracy:  0.936, Loss:  0.073
    Epoch   8 Batch  427/538 - Train Accuracy:  0.939, Validation Accuracy:  0.936, Loss:  0.081
    Epoch   8 Batch  428/538 - Train Accuracy:  0.957, Validation Accuracy:  0.936, Loss:  0.067
    Epoch   8 Batch  429/538 - Train Accuracy:  0.954, Validation Accuracy:  0.938, Loss:  0.078
    Epoch   8 Batch  430/538 - Train Accuracy:  0.954, Validation Accuracy:  0.938, Loss:  0.068
    Epoch   8 Batch  431/538 - Train Accuracy:  0.937, Validation Accuracy:  0.939, Loss:  0.069
    Epoch   8 Batch  432/538 - Train Accuracy:  0.945, Validation Accuracy:  0.941, Loss:  0.083
    Epoch   8 Batch  433/538 - Train Accuracy:  0.931, Validation Accuracy:  0.944, Loss:  0.105
    Epoch   8 Batch  434/538 - Train Accuracy:  0.951, Validation Accuracy:  0.944, Loss:  0.071
    Epoch   8 Batch  435/538 - Train Accuracy:  0.939, Validation Accuracy:  0.944, Loss:  0.079
    Epoch   8 Batch  436/538 - Train Accuracy:  0.940, Validation Accuracy:  0.946, Loss:  0.094
    Epoch   8 Batch  437/538 - Train Accuracy:  0.956, Validation Accuracy:  0.948, Loss:  0.072
    Epoch   8 Batch  438/538 - Train Accuracy:  0.959, Validation Accuracy:  0.950, Loss:  0.077
    Epoch   8 Batch  439/538 - Train Accuracy:  0.963, Validation Accuracy:  0.950, Loss:  0.066
    Epoch   8 Batch  440/538 - Train Accuracy:  0.953, Validation Accuracy:  0.952, Loss:  0.080
    Epoch   8 Batch  441/538 - Train Accuracy:  0.947, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   8 Batch  442/538 - Train Accuracy:  0.954, Validation Accuracy:  0.944, Loss:  0.067
    Epoch   8 Batch  443/538 - Train Accuracy:  0.949, Validation Accuracy:  0.947, Loss:  0.076
    Epoch   8 Batch  444/538 - Train Accuracy:  0.962, Validation Accuracy:  0.948, Loss:  0.071
    Epoch   8 Batch  445/538 - Train Accuracy:  0.958, Validation Accuracy:  0.941, Loss:  0.068
    Epoch   8 Batch  446/538 - Train Accuracy:  0.960, Validation Accuracy:  0.942, Loss:  0.071
    Epoch   8 Batch  447/538 - Train Accuracy:  0.947, Validation Accuracy:  0.941, Loss:  0.078
    Epoch   8 Batch  448/538 - Train Accuracy:  0.949, Validation Accuracy:  0.942, Loss:  0.067
    Epoch   8 Batch  449/538 - Train Accuracy:  0.960, Validation Accuracy:  0.941, Loss:  0.078
    Epoch   8 Batch  450/538 - Train Accuracy:  0.929, Validation Accuracy:  0.941, Loss:  0.102
    Epoch   8 Batch  451/538 - Train Accuracy:  0.937, Validation Accuracy:  0.943, Loss:  0.077
    Epoch   8 Batch  452/538 - Train Accuracy:  0.960, Validation Accuracy:  0.941, Loss:  0.066
    Epoch   8 Batch  453/538 - Train Accuracy:  0.951, Validation Accuracy:  0.950, Loss:  0.079
    Epoch   8 Batch  454/538 - Train Accuracy:  0.954, Validation Accuracy:  0.948, Loss:  0.072
    Epoch   8 Batch  455/538 - Train Accuracy:  0.957, Validation Accuracy:  0.949, Loss:  0.082
    Epoch   8 Batch  456/538 - Train Accuracy:  0.958, Validation Accuracy:  0.950, Loss:  0.097
    Epoch   8 Batch  457/538 - Train Accuracy:  0.961, Validation Accuracy:  0.950, Loss:  0.066
    Epoch   8 Batch  458/538 - Train Accuracy:  0.954, Validation Accuracy:  0.952, Loss:  0.074
    Epoch   8 Batch  459/538 - Train Accuracy:  0.959, Validation Accuracy:  0.949, Loss:  0.059
    Epoch   8 Batch  460/538 - Train Accuracy:  0.938, Validation Accuracy:  0.952, Loss:  0.082
    Epoch   8 Batch  461/538 - Train Accuracy:  0.961, Validation Accuracy:  0.952, Loss:  0.074
    Epoch   8 Batch  462/538 - Train Accuracy:  0.949, Validation Accuracy:  0.953, Loss:  0.076
    Epoch   8 Batch  463/538 - Train Accuracy:  0.934, Validation Accuracy:  0.953, Loss:  0.073
    Epoch   8 Batch  464/538 - Train Accuracy:  0.959, Validation Accuracy:  0.949, Loss:  0.070
    Epoch   8 Batch  465/538 - Train Accuracy:  0.949, Validation Accuracy:  0.948, Loss:  0.065
    Epoch   8 Batch  466/538 - Train Accuracy:  0.941, Validation Accuracy:  0.947, Loss:  0.077
    Epoch   8 Batch  467/538 - Train Accuracy:  0.958, Validation Accuracy:  0.951, Loss:  0.086
    Epoch   8 Batch  468/538 - Train Accuracy:  0.966, Validation Accuracy:  0.951, Loss:  0.074
    Epoch   8 Batch  469/538 - Train Accuracy:  0.947, Validation Accuracy:  0.951, Loss:  0.068
    Epoch   8 Batch  470/538 - Train Accuracy:  0.949, Validation Accuracy:  0.952, Loss:  0.075
    Epoch   8 Batch  471/538 - Train Accuracy:  0.971, Validation Accuracy:  0.953, Loss:  0.051
    Epoch   8 Batch  472/538 - Train Accuracy:  0.978, Validation Accuracy:  0.950, Loss:  0.055
    Epoch   8 Batch  473/538 - Train Accuracy:  0.951, Validation Accuracy:  0.950, Loss:  0.074
    Epoch   8 Batch  474/538 - Train Accuracy:  0.960, Validation Accuracy:  0.945, Loss:  0.066
    Epoch   8 Batch  475/538 - Train Accuracy:  0.941, Validation Accuracy:  0.943, Loss:  0.065
    Epoch   8 Batch  476/538 - Train Accuracy:  0.953, Validation Accuracy:  0.939, Loss:  0.071
    Epoch   8 Batch  477/538 - Train Accuracy:  0.941, Validation Accuracy:  0.940, Loss:  0.082
    Epoch   8 Batch  478/538 - Train Accuracy:  0.957, Validation Accuracy:  0.938, Loss:  0.065
    Epoch   8 Batch  479/538 - Train Accuracy:  0.949, Validation Accuracy:  0.937, Loss:  0.067
    Epoch   8 Batch  480/538 - Train Accuracy:  0.953, Validation Accuracy:  0.937, Loss:  0.063
    Epoch   8 Batch  481/538 - Train Accuracy:  0.960, Validation Accuracy:  0.941, Loss:  0.070
    Epoch   8 Batch  482/538 - Train Accuracy:  0.950, Validation Accuracy:  0.939, Loss:  0.070
    Epoch   8 Batch  483/538 - Train Accuracy:  0.932, Validation Accuracy:  0.939, Loss:  0.082
    Epoch   8 Batch  484/538 - Train Accuracy:  0.945, Validation Accuracy:  0.939, Loss:  0.092
    Epoch   8 Batch  485/538 - Train Accuracy:  0.958, Validation Accuracy:  0.943, Loss:  0.075
    Epoch   8 Batch  486/538 - Train Accuracy:  0.961, Validation Accuracy:  0.944, Loss:  0.062
    Epoch   8 Batch  487/538 - Train Accuracy:  0.959, Validation Accuracy:  0.943, Loss:  0.057
    Epoch   8 Batch  488/538 - Train Accuracy:  0.949, Validation Accuracy:  0.945, Loss:  0.065
    Epoch   8 Batch  489/538 - Train Accuracy:  0.944, Validation Accuracy:  0.945, Loss:  0.068
    Epoch   8 Batch  490/538 - Train Accuracy:  0.942, Validation Accuracy:  0.941, Loss:  0.069
    Epoch   8 Batch  491/538 - Train Accuracy:  0.940, Validation Accuracy:  0.941, Loss:  0.082
    Epoch   8 Batch  492/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.067
    Epoch   8 Batch  493/538 - Train Accuracy:  0.948, Validation Accuracy:  0.937, Loss:  0.077
    Epoch   8 Batch  494/538 - Train Accuracy:  0.957, Validation Accuracy:  0.938, Loss:  0.082
    Epoch   8 Batch  495/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.079
    Epoch   8 Batch  496/538 - Train Accuracy:  0.960, Validation Accuracy:  0.941, Loss:  0.064
    Epoch   8 Batch  497/538 - Train Accuracy:  0.967, Validation Accuracy:  0.940, Loss:  0.064
    Epoch   8 Batch  498/538 - Train Accuracy:  0.956, Validation Accuracy:  0.940, Loss:  0.074
    Epoch   8 Batch  499/538 - Train Accuracy:  0.948, Validation Accuracy:  0.940, Loss:  0.065
    Epoch   8 Batch  500/538 - Train Accuracy:  0.965, Validation Accuracy:  0.939, Loss:  0.057
    Epoch   8 Batch  501/538 - Train Accuracy:  0.956, Validation Accuracy:  0.940, Loss:  0.084
    Epoch   8 Batch  502/538 - Train Accuracy:  0.953, Validation Accuracy:  0.941, Loss:  0.073
    Epoch   8 Batch  503/538 - Train Accuracy:  0.952, Validation Accuracy:  0.938, Loss:  0.076
    Epoch   8 Batch  504/538 - Train Accuracy:  0.964, Validation Accuracy:  0.939, Loss:  0.077
    Epoch   8 Batch  505/538 - Train Accuracy:  0.960, Validation Accuracy:  0.942, Loss:  0.061
    Epoch   8 Batch  506/538 - Train Accuracy:  0.956, Validation Accuracy:  0.942, Loss:  0.065
    Epoch   8 Batch  507/538 - Train Accuracy:  0.943, Validation Accuracy:  0.947, Loss:  0.085
    Epoch   8 Batch  508/538 - Train Accuracy:  0.950, Validation Accuracy:  0.947, Loss:  0.081
    Epoch   8 Batch  509/538 - Train Accuracy:  0.955, Validation Accuracy:  0.943, Loss:  0.073
    Epoch   8 Batch  510/538 - Train Accuracy:  0.965, Validation Accuracy:  0.944, Loss:  0.067
    Epoch   8 Batch  511/538 - Train Accuracy:  0.950, Validation Accuracy:  0.944, Loss:  0.082
    Epoch   8 Batch  512/538 - Train Accuracy:  0.960, Validation Accuracy:  0.948, Loss:  0.081
    Epoch   8 Batch  513/538 - Train Accuracy:  0.942, Validation Accuracy:  0.948, Loss:  0.073
    Epoch   8 Batch  514/538 - Train Accuracy:  0.955, Validation Accuracy:  0.950, Loss:  0.075
    Epoch   8 Batch  515/538 - Train Accuracy:  0.938, Validation Accuracy:  0.945, Loss:  0.081
    Epoch   8 Batch  516/538 - Train Accuracy:  0.949, Validation Accuracy:  0.944, Loss:  0.082
    Epoch   8 Batch  517/538 - Train Accuracy:  0.955, Validation Accuracy:  0.942, Loss:  0.078
    Epoch   8 Batch  518/538 - Train Accuracy:  0.948, Validation Accuracy:  0.943, Loss:  0.086
    Epoch   8 Batch  519/538 - Train Accuracy:  0.962, Validation Accuracy:  0.943, Loss:  0.068
    Epoch   8 Batch  520/538 - Train Accuracy:  0.942, Validation Accuracy:  0.941, Loss:  0.076
    Epoch   8 Batch  521/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.077
    Epoch   8 Batch  522/538 - Train Accuracy:  0.953, Validation Accuracy:  0.940, Loss:  0.054
    Epoch   8 Batch  523/538 - Train Accuracy:  0.952, Validation Accuracy:  0.943, Loss:  0.069
    Epoch   8 Batch  524/538 - Train Accuracy:  0.962, Validation Accuracy:  0.946, Loss:  0.075
    Epoch   8 Batch  525/538 - Train Accuracy:  0.951, Validation Accuracy:  0.948, Loss:  0.088
    Epoch   8 Batch  526/538 - Train Accuracy:  0.951, Validation Accuracy:  0.952, Loss:  0.083
    Epoch   8 Batch  527/538 - Train Accuracy:  0.952, Validation Accuracy:  0.950, Loss:  0.067
    Epoch   8 Batch  528/538 - Train Accuracy:  0.954, Validation Accuracy:  0.948, Loss:  0.071
    Epoch   8 Batch  529/538 - Train Accuracy:  0.928, Validation Accuracy:  0.950, Loss:  0.089
    Epoch   8 Batch  530/538 - Train Accuracy:  0.951, Validation Accuracy:  0.952, Loss:  0.080
    Epoch   8 Batch  531/538 - Train Accuracy:  0.950, Validation Accuracy:  0.946, Loss:  0.077
    Epoch   8 Batch  532/538 - Train Accuracy:  0.953, Validation Accuracy:  0.948, Loss:  0.068
    Epoch   8 Batch  533/538 - Train Accuracy:  0.952, Validation Accuracy:  0.943, Loss:  0.079
    Epoch   8 Batch  534/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.059
    Epoch   8 Batch  535/538 - Train Accuracy:  0.951, Validation Accuracy:  0.945, Loss:  0.074
    Epoch   8 Batch  536/538 - Train Accuracy:  0.958, Validation Accuracy:  0.942, Loss:  0.081
    Epoch   9 Batch    0/538 - Train Accuracy:  0.963, Validation Accuracy:  0.945, Loss:  0.058
    Epoch   9 Batch    1/538 - Train Accuracy:  0.963, Validation Accuracy:  0.946, Loss:  0.069
    Epoch   9 Batch    2/538 - Train Accuracy:  0.960, Validation Accuracy:  0.946, Loss:  0.081
    Epoch   9 Batch    3/538 - Train Accuracy:  0.956, Validation Accuracy:  0.949, Loss:  0.068
    Epoch   9 Batch    4/538 - Train Accuracy:  0.957, Validation Accuracy:  0.949, Loss:  0.069
    Epoch   9 Batch    5/538 - Train Accuracy:  0.949, Validation Accuracy:  0.947, Loss:  0.074
    Epoch   9 Batch    6/538 - Train Accuracy:  0.952, Validation Accuracy:  0.947, Loss:  0.058
    Epoch   9 Batch    7/538 - Train Accuracy:  0.961, Validation Accuracy:  0.947, Loss:  0.074
    Epoch   9 Batch    8/538 - Train Accuracy:  0.952, Validation Accuracy:  0.946, Loss:  0.070
    Epoch   9 Batch    9/538 - Train Accuracy:  0.959, Validation Accuracy:  0.946, Loss:  0.064
    Epoch   9 Batch   10/538 - Train Accuracy:  0.941, Validation Accuracy:  0.946, Loss:  0.068
    Epoch   9 Batch   11/538 - Train Accuracy:  0.961, Validation Accuracy:  0.947, Loss:  0.076
    Epoch   9 Batch   12/538 - Train Accuracy:  0.955, Validation Accuracy:  0.946, Loss:  0.067
    Epoch   9 Batch   13/538 - Train Accuracy:  0.955, Validation Accuracy:  0.951, Loss:  0.066
    Epoch   9 Batch   14/538 - Train Accuracy:  0.965, Validation Accuracy:  0.951, Loss:  0.060
    Epoch   9 Batch   15/538 - Train Accuracy:  0.961, Validation Accuracy:  0.949, Loss:  0.078
    Epoch   9 Batch   16/538 - Train Accuracy:  0.955, Validation Accuracy:  0.949, Loss:  0.071
    Epoch   9 Batch   17/538 - Train Accuracy:  0.955, Validation Accuracy:  0.945, Loss:  0.068
    Epoch   9 Batch   18/538 - Train Accuracy:  0.957, Validation Accuracy:  0.943, Loss:  0.093
    Epoch   9 Batch   19/538 - Train Accuracy:  0.954, Validation Accuracy:  0.940, Loss:  0.077
    Epoch   9 Batch   20/538 - Train Accuracy:  0.951, Validation Accuracy:  0.943, Loss:  0.083
    Epoch   9 Batch   21/538 - Train Accuracy:  0.963, Validation Accuracy:  0.944, Loss:  0.057
    Epoch   9 Batch   22/538 - Train Accuracy:  0.934, Validation Accuracy:  0.941, Loss:  0.073
    Epoch   9 Batch   23/538 - Train Accuracy:  0.939, Validation Accuracy:  0.944, Loss:  0.093
    Epoch   9 Batch   24/538 - Train Accuracy:  0.950, Validation Accuracy:  0.949, Loss:  0.078
    Epoch   9 Batch   25/538 - Train Accuracy:  0.946, Validation Accuracy:  0.953, Loss:  0.073
    Epoch   9 Batch   26/538 - Train Accuracy:  0.940, Validation Accuracy:  0.951, Loss:  0.075
    Epoch   9 Batch   27/538 - Train Accuracy:  0.959, Validation Accuracy:  0.956, Loss:  0.062
    Epoch   9 Batch   28/538 - Train Accuracy:  0.949, Validation Accuracy:  0.952, Loss:  0.075
    Epoch   9 Batch   29/538 - Train Accuracy:  0.955, Validation Accuracy:  0.952, Loss:  0.061
    Epoch   9 Batch   30/538 - Train Accuracy:  0.944, Validation Accuracy:  0.952, Loss:  0.085
    Epoch   9 Batch   31/538 - Train Accuracy:  0.957, Validation Accuracy:  0.952, Loss:  0.059
    Epoch   9 Batch   32/538 - Train Accuracy:  0.960, Validation Accuracy:  0.954, Loss:  0.063
    Epoch   9 Batch   33/538 - Train Accuracy:  0.948, Validation Accuracy:  0.956, Loss:  0.071
    Epoch   9 Batch   34/538 - Train Accuracy:  0.943, Validation Accuracy:  0.950, Loss:  0.111
    Epoch   9 Batch   35/538 - Train Accuracy:  0.950, Validation Accuracy:  0.949, Loss:  0.068
    Epoch   9 Batch   36/538 - Train Accuracy:  0.947, Validation Accuracy:  0.942, Loss:  0.066
    Epoch   9 Batch   37/538 - Train Accuracy:  0.958, Validation Accuracy:  0.940, Loss:  0.074
    Epoch   9 Batch   38/538 - Train Accuracy:  0.940, Validation Accuracy:  0.941, Loss:  0.078
    Epoch   9 Batch   39/538 - Train Accuracy:  0.959, Validation Accuracy:  0.941, Loss:  0.074
    Epoch   9 Batch   40/538 - Train Accuracy:  0.955, Validation Accuracy:  0.945, Loss:  0.062
    Epoch   9 Batch   41/538 - Train Accuracy:  0.959, Validation Accuracy:  0.941, Loss:  0.068
    Epoch   9 Batch   42/538 - Train Accuracy:  0.952, Validation Accuracy:  0.946, Loss:  0.073
    Epoch   9 Batch   43/538 - Train Accuracy:  0.939, Validation Accuracy:  0.945, Loss:  0.092
    Epoch   9 Batch   44/538 - Train Accuracy:  0.940, Validation Accuracy:  0.945, Loss:  0.074
    Epoch   9 Batch   45/538 - Train Accuracy:  0.961, Validation Accuracy:  0.946, Loss:  0.073
    Epoch   9 Batch   46/538 - Train Accuracy:  0.953, Validation Accuracy:  0.944, Loss:  0.059
    Epoch   9 Batch   47/538 - Train Accuracy:  0.948, Validation Accuracy:  0.943, Loss:  0.078
    Epoch   9 Batch   48/538 - Train Accuracy:  0.942, Validation Accuracy:  0.945, Loss:  0.085
    Epoch   9 Batch   49/538 - Train Accuracy:  0.958, Validation Accuracy:  0.950, Loss:  0.073
    Epoch   9 Batch   50/538 - Train Accuracy:  0.947, Validation Accuracy:  0.952, Loss:  0.063
    Epoch   9 Batch   51/538 - Train Accuracy:  0.947, Validation Accuracy:  0.950, Loss:  0.085
    Epoch   9 Batch   52/538 - Train Accuracy:  0.949, Validation Accuracy:  0.944, Loss:  0.075
    Epoch   9 Batch   53/538 - Train Accuracy:  0.933, Validation Accuracy:  0.944, Loss:  0.075
    Epoch   9 Batch   54/538 - Train Accuracy:  0.963, Validation Accuracy:  0.945, Loss:  0.062
    Epoch   9 Batch   55/538 - Train Accuracy:  0.946, Validation Accuracy:  0.944, Loss:  0.061
    Epoch   9 Batch   56/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.086
    Epoch   9 Batch   57/538 - Train Accuracy:  0.931, Validation Accuracy:  0.941, Loss:  0.085
    Epoch   9 Batch   58/538 - Train Accuracy:  0.953, Validation Accuracy:  0.948, Loss:  0.064
    Epoch   9 Batch   59/538 - Train Accuracy:  0.949, Validation Accuracy:  0.946, Loss:  0.078
    Epoch   9 Batch   60/538 - Train Accuracy:  0.957, Validation Accuracy:  0.946, Loss:  0.074
    Epoch   9 Batch   61/538 - Train Accuracy:  0.952, Validation Accuracy:  0.946, Loss:  0.071
    Epoch   9 Batch   62/538 - Train Accuracy:  0.949, Validation Accuracy:  0.946, Loss:  0.076
    Epoch   9 Batch   63/538 - Train Accuracy:  0.956, Validation Accuracy:  0.948, Loss:  0.070
    Epoch   9 Batch   64/538 - Train Accuracy:  0.951, Validation Accuracy:  0.947, Loss:  0.072
    Epoch   9 Batch   65/538 - Train Accuracy:  0.946, Validation Accuracy:  0.946, Loss:  0.074
    Epoch   9 Batch   66/538 - Train Accuracy:  0.962, Validation Accuracy:  0.949, Loss:  0.063
    Epoch   9 Batch   67/538 - Train Accuracy:  0.963, Validation Accuracy:  0.948, Loss:  0.058
    Epoch   9 Batch   68/538 - Train Accuracy:  0.946, Validation Accuracy:  0.951, Loss:  0.062
    Epoch   9 Batch   69/538 - Train Accuracy:  0.956, Validation Accuracy:  0.951, Loss:  0.072
    Epoch   9 Batch   70/538 - Train Accuracy:  0.944, Validation Accuracy:  0.942, Loss:  0.065
    Epoch   9 Batch   71/538 - Train Accuracy:  0.944, Validation Accuracy:  0.943, Loss:  0.074
    Epoch   9 Batch   72/538 - Train Accuracy:  0.945, Validation Accuracy:  0.944, Loss:  0.114
    Epoch   9 Batch   73/538 - Train Accuracy:  0.938, Validation Accuracy:  0.944, Loss:  0.069
    Epoch   9 Batch   74/538 - Train Accuracy:  0.964, Validation Accuracy:  0.945, Loss:  0.061
    Epoch   9 Batch   75/538 - Train Accuracy:  0.943, Validation Accuracy:  0.946, Loss:  0.071
    Epoch   9 Batch   76/538 - Train Accuracy:  0.951, Validation Accuracy:  0.953, Loss:  0.085
    Epoch   9 Batch   77/538 - Train Accuracy:  0.951, Validation Accuracy:  0.953, Loss:  0.067
    Epoch   9 Batch   78/538 - Train Accuracy:  0.946, Validation Accuracy:  0.956, Loss:  0.075
    Epoch   9 Batch   79/538 - Train Accuracy:  0.952, Validation Accuracy:  0.951, Loss:  0.056
    Epoch   9 Batch   80/538 - Train Accuracy:  0.942, Validation Accuracy:  0.950, Loss:  0.090
    Epoch   9 Batch   81/538 - Train Accuracy:  0.941, Validation Accuracy:  0.950, Loss:  0.082
    Epoch   9 Batch   82/538 - Train Accuracy:  0.944, Validation Accuracy:  0.949, Loss:  0.068
    Epoch   9 Batch   83/538 - Train Accuracy:  0.947, Validation Accuracy:  0.944, Loss:  0.074
    Epoch   9 Batch   84/538 - Train Accuracy:  0.942, Validation Accuracy:  0.945, Loss:  0.076
    Epoch   9 Batch   85/538 - Train Accuracy:  0.963, Validation Accuracy:  0.946, Loss:  0.066
    Epoch   9 Batch   86/538 - Train Accuracy:  0.959, Validation Accuracy:  0.947, Loss:  0.071
    Epoch   9 Batch   87/538 - Train Accuracy:  0.937, Validation Accuracy:  0.944, Loss:  0.073
    Epoch   9 Batch   88/538 - Train Accuracy:  0.951, Validation Accuracy:  0.944, Loss:  0.068
    Epoch   9 Batch   89/538 - Train Accuracy:  0.949, Validation Accuracy:  0.941, Loss:  0.067
    Epoch   9 Batch   90/538 - Train Accuracy:  0.952, Validation Accuracy:  0.942, Loss:  0.078
    Epoch   9 Batch   91/538 - Train Accuracy:  0.944, Validation Accuracy:  0.944, Loss:  0.066
    Epoch   9 Batch   92/538 - Train Accuracy:  0.936, Validation Accuracy:  0.944, Loss:  0.069
    Epoch   9 Batch   93/538 - Train Accuracy:  0.964, Validation Accuracy:  0.950, Loss:  0.066
    Epoch   9 Batch   94/538 - Train Accuracy:  0.945, Validation Accuracy:  0.950, Loss:  0.064
    Epoch   9 Batch   95/538 - Train Accuracy:  0.955, Validation Accuracy:  0.953, Loss:  0.068
    Epoch   9 Batch   96/538 - Train Accuracy:  0.964, Validation Accuracy:  0.952, Loss:  0.068
    Epoch   9 Batch   97/538 - Train Accuracy:  0.954, Validation Accuracy:  0.954, Loss:  0.061
    Epoch   9 Batch   98/538 - Train Accuracy:  0.953, Validation Accuracy:  0.952, Loss:  0.075
    Epoch   9 Batch   99/538 - Train Accuracy:  0.948, Validation Accuracy:  0.952, Loss:  0.065
    Epoch   9 Batch  100/538 - Train Accuracy:  0.958, Validation Accuracy:  0.950, Loss:  0.063
    Epoch   9 Batch  101/538 - Train Accuracy:  0.941, Validation Accuracy:  0.952, Loss:  0.093
    Epoch   9 Batch  102/538 - Train Accuracy:  0.932, Validation Accuracy:  0.950, Loss:  0.077
    Epoch   9 Batch  103/538 - Train Accuracy:  0.958, Validation Accuracy:  0.950, Loss:  0.060
    Epoch   9 Batch  104/538 - Train Accuracy:  0.959, Validation Accuracy:  0.947, Loss:  0.059
    Epoch   9 Batch  105/538 - Train Accuracy:  0.958, Validation Accuracy:  0.947, Loss:  0.058
    Epoch   9 Batch  106/538 - Train Accuracy:  0.958, Validation Accuracy:  0.947, Loss:  0.058
    Epoch   9 Batch  107/538 - Train Accuracy:  0.951, Validation Accuracy:  0.944, Loss:  0.080
    Epoch   9 Batch  108/538 - Train Accuracy:  0.952, Validation Accuracy:  0.947, Loss:  0.076
    Epoch   9 Batch  109/538 - Train Accuracy:  0.963, Validation Accuracy:  0.950, Loss:  0.058
    Epoch   9 Batch  110/538 - Train Accuracy:  0.957, Validation Accuracy:  0.950, Loss:  0.061
    Epoch   9 Batch  111/538 - Train Accuracy:  0.953, Validation Accuracy:  0.948, Loss:  0.061
    Epoch   9 Batch  112/538 - Train Accuracy:  0.947, Validation Accuracy:  0.947, Loss:  0.067
    Epoch   9 Batch  113/538 - Train Accuracy:  0.933, Validation Accuracy:  0.946, Loss:  0.073
    Epoch   9 Batch  114/538 - Train Accuracy:  0.964, Validation Accuracy:  0.946, Loss:  0.071
    Epoch   9 Batch  115/538 - Train Accuracy:  0.956, Validation Accuracy:  0.949, Loss:  0.081
    Epoch   9 Batch  116/538 - Train Accuracy:  0.951, Validation Accuracy:  0.949, Loss:  0.087
    Epoch   9 Batch  117/538 - Train Accuracy:  0.950, Validation Accuracy:  0.950, Loss:  0.072
    Epoch   9 Batch  118/538 - Train Accuracy:  0.964, Validation Accuracy:  0.948, Loss:  0.063
    Epoch   9 Batch  119/538 - Train Accuracy:  0.965, Validation Accuracy:  0.947, Loss:  0.053
    Epoch   9 Batch  120/538 - Train Accuracy:  0.948, Validation Accuracy:  0.947, Loss:  0.054
    Epoch   9 Batch  121/538 - Train Accuracy:  0.951, Validation Accuracy:  0.947, Loss:  0.061
    Epoch   9 Batch  122/538 - Train Accuracy:  0.958, Validation Accuracy:  0.946, Loss:  0.061
    Epoch   9 Batch  123/538 - Train Accuracy:  0.940, Validation Accuracy:  0.946, Loss:  0.062
    Epoch   9 Batch  124/538 - Train Accuracy:  0.959, Validation Accuracy:  0.946, Loss:  0.063
    Epoch   9 Batch  125/538 - Train Accuracy:  0.953, Validation Accuracy:  0.946, Loss:  0.075
    Epoch   9 Batch  126/538 - Train Accuracy:  0.946, Validation Accuracy:  0.951, Loss:  0.068
    Epoch   9 Batch  127/538 - Train Accuracy:  0.939, Validation Accuracy:  0.951, Loss:  0.089
    Epoch   9 Batch  128/538 - Train Accuracy:  0.959, Validation Accuracy:  0.954, Loss:  0.067
    Epoch   9 Batch  129/538 - Train Accuracy:  0.964, Validation Accuracy:  0.954, Loss:  0.063
    Epoch   9 Batch  130/538 - Train Accuracy:  0.959, Validation Accuracy:  0.951, Loss:  0.062
    Epoch   9 Batch  131/538 - Train Accuracy:  0.959, Validation Accuracy:  0.950, Loss:  0.055
    Epoch   9 Batch  132/538 - Train Accuracy:  0.957, Validation Accuracy:  0.951, Loss:  0.077
    Epoch   9 Batch  133/538 - Train Accuracy:  0.938, Validation Accuracy:  0.948, Loss:  0.059
    Epoch   9 Batch  134/538 - Train Accuracy:  0.945, Validation Accuracy:  0.948, Loss:  0.087
    Epoch   9 Batch  135/538 - Train Accuracy:  0.949, Validation Accuracy:  0.947, Loss:  0.091
    Epoch   9 Batch  136/538 - Train Accuracy:  0.958, Validation Accuracy:  0.945, Loss:  0.086
    Epoch   9 Batch  137/538 - Train Accuracy:  0.949, Validation Accuracy:  0.945, Loss:  0.079
    Epoch   9 Batch  138/538 - Train Accuracy:  0.952, Validation Accuracy:  0.950, Loss:  0.068
    Epoch   9 Batch  139/538 - Train Accuracy:  0.951, Validation Accuracy:  0.951, Loss:  0.084
    Epoch   9 Batch  140/538 - Train Accuracy:  0.950, Validation Accuracy:  0.948, Loss:  0.094
    Epoch   9 Batch  141/538 - Train Accuracy:  0.957, Validation Accuracy:  0.945, Loss:  0.086
    Epoch   9 Batch  142/538 - Train Accuracy:  0.957, Validation Accuracy:  0.946, Loss:  0.064
    Epoch   9 Batch  143/538 - Train Accuracy:  0.950, Validation Accuracy:  0.947, Loss:  0.075
    Epoch   9 Batch  144/538 - Train Accuracy:  0.947, Validation Accuracy:  0.949, Loss:  0.079
    Epoch   9 Batch  145/538 - Train Accuracy:  0.935, Validation Accuracy:  0.948, Loss:  0.090
    Epoch   9 Batch  146/538 - Train Accuracy:  0.952, Validation Accuracy:  0.949, Loss:  0.074
    Epoch   9 Batch  147/538 - Train Accuracy:  0.951, Validation Accuracy:  0.949, Loss:  0.068
    Epoch   9 Batch  148/538 - Train Accuracy:  0.943, Validation Accuracy:  0.949, Loss:  0.117
    Epoch   9 Batch  149/538 - Train Accuracy:  0.963, Validation Accuracy:  0.946, Loss:  0.067
    Epoch   9 Batch  150/538 - Train Accuracy:  0.961, Validation Accuracy:  0.944, Loss:  0.060
    Epoch   9 Batch  151/538 - Train Accuracy:  0.939, Validation Accuracy:  0.942, Loss:  0.070
    Epoch   9 Batch  152/538 - Train Accuracy:  0.956, Validation Accuracy:  0.941, Loss:  0.075
    Epoch   9 Batch  153/538 - Train Accuracy:  0.943, Validation Accuracy:  0.944, Loss:  0.070
    Epoch   9 Batch  154/538 - Train Accuracy:  0.961, Validation Accuracy:  0.949, Loss:  0.061
    Epoch   9 Batch  155/538 - Train Accuracy:  0.956, Validation Accuracy:  0.949, Loss:  0.068
    Epoch   9 Batch  156/538 - Train Accuracy:  0.966, Validation Accuracy:  0.951, Loss:  0.064
    Epoch   9 Batch  157/538 - Train Accuracy:  0.958, Validation Accuracy:  0.951, Loss:  0.063
    Epoch   9 Batch  158/538 - Train Accuracy:  0.971, Validation Accuracy:  0.947, Loss:  0.075
    Epoch   9 Batch  159/538 - Train Accuracy:  0.952, Validation Accuracy:  0.946, Loss:  0.077
    Epoch   9 Batch  160/538 - Train Accuracy:  0.948, Validation Accuracy:  0.947, Loss:  0.064
    Epoch   9 Batch  161/538 - Train Accuracy:  0.947, Validation Accuracy:  0.949, Loss:  0.075
    Epoch   9 Batch  162/538 - Train Accuracy:  0.963, Validation Accuracy:  0.946, Loss:  0.071
    Epoch   9 Batch  163/538 - Train Accuracy:  0.947, Validation Accuracy:  0.950, Loss:  0.093
    Epoch   9 Batch  164/538 - Train Accuracy:  0.956, Validation Accuracy:  0.952, Loss:  0.080
    Epoch   9 Batch  165/538 - Train Accuracy:  0.963, Validation Accuracy:  0.952, Loss:  0.062
    Epoch   9 Batch  166/538 - Train Accuracy:  0.963, Validation Accuracy:  0.947, Loss:  0.068
    Epoch   9 Batch  167/538 - Train Accuracy:  0.944, Validation Accuracy:  0.945, Loss:  0.109
    Epoch   9 Batch  168/538 - Train Accuracy:  0.934, Validation Accuracy:  0.944, Loss:  0.090
    Epoch   9 Batch  169/538 - Train Accuracy:  0.965, Validation Accuracy:  0.944, Loss:  0.055
    Epoch   9 Batch  170/538 - Train Accuracy:  0.948, Validation Accuracy:  0.946, Loss:  0.082
    Epoch   9 Batch  171/538 - Train Accuracy:  0.949, Validation Accuracy:  0.946, Loss:  0.072
    Epoch   9 Batch  172/538 - Train Accuracy:  0.951, Validation Accuracy:  0.946, Loss:  0.069
    Epoch   9 Batch  173/538 - Train Accuracy:  0.965, Validation Accuracy:  0.947, Loss:  0.062
    Epoch   9 Batch  174/538 - Train Accuracy:  0.955, Validation Accuracy:  0.945, Loss:  0.063
    Epoch   9 Batch  175/538 - Train Accuracy:  0.959, Validation Accuracy:  0.948, Loss:  0.079
    Epoch   9 Batch  176/538 - Train Accuracy:  0.950, Validation Accuracy:  0.951, Loss:  0.080
    Epoch   9 Batch  177/538 - Train Accuracy:  0.962, Validation Accuracy:  0.948, Loss:  0.078
    Epoch   9 Batch  178/538 - Train Accuracy:  0.939, Validation Accuracy:  0.945, Loss:  0.076
    Epoch   9 Batch  179/538 - Train Accuracy:  0.957, Validation Accuracy:  0.944, Loss:  0.066
    Epoch   9 Batch  180/538 - Train Accuracy:  0.954, Validation Accuracy:  0.945, Loss:  0.066
    Epoch   9 Batch  181/538 - Train Accuracy:  0.948, Validation Accuracy:  0.945, Loss:  0.081
    Epoch   9 Batch  182/538 - Train Accuracy:  0.962, Validation Accuracy:  0.943, Loss:  0.061
    Epoch   9 Batch  183/538 - Train Accuracy:  0.964, Validation Accuracy:  0.945, Loss:  0.054
    Epoch   9 Batch  184/538 - Train Accuracy:  0.957, Validation Accuracy:  0.944, Loss:  0.071
    Epoch   9 Batch  185/538 - Train Accuracy:  0.969, Validation Accuracy:  0.944, Loss:  0.059
    Epoch   9 Batch  186/538 - Train Accuracy:  0.953, Validation Accuracy:  0.942, Loss:  0.067
    Epoch   9 Batch  187/538 - Train Accuracy:  0.968, Validation Accuracy:  0.942, Loss:  0.066
    Epoch   9 Batch  188/538 - Train Accuracy:  0.946, Validation Accuracy:  0.942, Loss:  0.064
    Epoch   9 Batch  189/538 - Train Accuracy:  0.956, Validation Accuracy:  0.941, Loss:  0.068
    Epoch   9 Batch  190/538 - Train Accuracy:  0.939, Validation Accuracy:  0.943, Loss:  0.095
    Epoch   9 Batch  191/538 - Train Accuracy:  0.967, Validation Accuracy:  0.943, Loss:  0.057
    Epoch   9 Batch  192/538 - Train Accuracy:  0.958, Validation Accuracy:  0.949, Loss:  0.074
    Epoch   9 Batch  193/538 - Train Accuracy:  0.957, Validation Accuracy:  0.951, Loss:  0.067
    Epoch   9 Batch  194/538 - Train Accuracy:  0.943, Validation Accuracy:  0.951, Loss:  0.072
    Epoch   9 Batch  195/538 - Train Accuracy:  0.961, Validation Accuracy:  0.955, Loss:  0.073
    Epoch   9 Batch  196/538 - Train Accuracy:  0.946, Validation Accuracy:  0.957, Loss:  0.066
    Epoch   9 Batch  197/538 - Train Accuracy:  0.952, Validation Accuracy:  0.957, Loss:  0.078
    Epoch   9 Batch  198/538 - Train Accuracy:  0.959, Validation Accuracy:  0.956, Loss:  0.064
    Epoch   9 Batch  199/538 - Train Accuracy:  0.952, Validation Accuracy:  0.957, Loss:  0.087
    Epoch   9 Batch  200/538 - Train Accuracy:  0.966, Validation Accuracy:  0.957, Loss:  0.059
    Epoch   9 Batch  201/538 - Train Accuracy:  0.953, Validation Accuracy:  0.955, Loss:  0.077
    Epoch   9 Batch  202/538 - Train Accuracy:  0.957, Validation Accuracy:  0.955, Loss:  0.070
    Epoch   9 Batch  203/538 - Train Accuracy:  0.953, Validation Accuracy:  0.948, Loss:  0.067
    Epoch   9 Batch  204/538 - Train Accuracy:  0.936, Validation Accuracy:  0.945, Loss:  0.078
    Epoch   9 Batch  205/538 - Train Accuracy:  0.953, Validation Accuracy:  0.944, Loss:  0.068
    Epoch   9 Batch  206/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.071
    Epoch   9 Batch  207/538 - Train Accuracy:  0.966, Validation Accuracy:  0.941, Loss:  0.072
    Epoch   9 Batch  208/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.089
    Epoch   9 Batch  209/538 - Train Accuracy:  0.960, Validation Accuracy:  0.941, Loss:  0.062
    Epoch   9 Batch  210/538 - Train Accuracy:  0.956, Validation Accuracy:  0.939, Loss:  0.079
    Epoch   9 Batch  211/538 - Train Accuracy:  0.949, Validation Accuracy:  0.942, Loss:  0.067
    Epoch   9 Batch  212/538 - Train Accuracy:  0.949, Validation Accuracy:  0.944, Loss:  0.070
    Epoch   9 Batch  213/538 - Train Accuracy:  0.957, Validation Accuracy:  0.946, Loss:  0.056
    Epoch   9 Batch  214/538 - Train Accuracy:  0.960, Validation Accuracy:  0.941, Loss:  0.058
    Epoch   9 Batch  215/538 - Train Accuracy:  0.956, Validation Accuracy:  0.946, Loss:  0.058
    Epoch   9 Batch  216/538 - Train Accuracy:  0.951, Validation Accuracy:  0.946, Loss:  0.071
    Epoch   9 Batch  217/538 - Train Accuracy:  0.959, Validation Accuracy:  0.948, Loss:  0.065
    Epoch   9 Batch  218/538 - Train Accuracy:  0.952, Validation Accuracy:  0.949, Loss:  0.060
    Epoch   9 Batch  219/538 - Train Accuracy:  0.948, Validation Accuracy:  0.948, Loss:  0.080
    Epoch   9 Batch  220/538 - Train Accuracy:  0.949, Validation Accuracy:  0.948, Loss:  0.070
    Epoch   9 Batch  221/538 - Train Accuracy:  0.964, Validation Accuracy:  0.947, Loss:  0.060
    Epoch   9 Batch  222/538 - Train Accuracy:  0.952, Validation Accuracy:  0.947, Loss:  0.065
    Epoch   9 Batch  223/538 - Train Accuracy:  0.949, Validation Accuracy:  0.947, Loss:  0.079
    Epoch   9 Batch  224/538 - Train Accuracy:  0.950, Validation Accuracy:  0.947, Loss:  0.072
    Epoch   9 Batch  225/538 - Train Accuracy:  0.957, Validation Accuracy:  0.950, Loss:  0.076
    Epoch   9 Batch  226/538 - Train Accuracy:  0.943, Validation Accuracy:  0.950, Loss:  0.073
    Epoch   9 Batch  227/538 - Train Accuracy:  0.949, Validation Accuracy:  0.950, Loss:  0.068
    Epoch   9 Batch  228/538 - Train Accuracy:  0.930, Validation Accuracy:  0.950, Loss:  0.074
    Epoch   9 Batch  229/538 - Train Accuracy:  0.952, Validation Accuracy:  0.950, Loss:  0.078
    Epoch   9 Batch  230/538 - Train Accuracy:  0.948, Validation Accuracy:  0.946, Loss:  0.078
    Epoch   9 Batch  231/538 - Train Accuracy:  0.944, Validation Accuracy:  0.946, Loss:  0.096
    Epoch   9 Batch  232/538 - Train Accuracy:  0.950, Validation Accuracy:  0.938, Loss:  0.075
    Epoch   9 Batch  233/538 - Train Accuracy:  0.963, Validation Accuracy:  0.940, Loss:  0.079
    Epoch   9 Batch  234/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.073
    Epoch   9 Batch  235/538 - Train Accuracy:  0.954, Validation Accuracy:  0.943, Loss:  0.060
    Epoch   9 Batch  236/538 - Train Accuracy:  0.947, Validation Accuracy:  0.942, Loss:  0.075
    Epoch   9 Batch  237/538 - Train Accuracy:  0.957, Validation Accuracy:  0.940, Loss:  0.062
    Epoch   9 Batch  238/538 - Train Accuracy:  0.956, Validation Accuracy:  0.942, Loss:  0.072
    Epoch   9 Batch  239/538 - Train Accuracy:  0.946, Validation Accuracy:  0.942, Loss:  0.079
    Epoch   9 Batch  240/538 - Train Accuracy:  0.951, Validation Accuracy:  0.943, Loss:  0.083
    Epoch   9 Batch  241/538 - Train Accuracy:  0.943, Validation Accuracy:  0.945, Loss:  0.080
    Epoch   9 Batch  242/538 - Train Accuracy:  0.956, Validation Accuracy:  0.945, Loss:  0.064
    Epoch   9 Batch  243/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.072
    Epoch   9 Batch  244/538 - Train Accuracy:  0.948, Validation Accuracy:  0.947, Loss:  0.065
    Epoch   9 Batch  245/538 - Train Accuracy:  0.948, Validation Accuracy:  0.946, Loss:  0.109
    Epoch   9 Batch  246/538 - Train Accuracy:  0.953, Validation Accuracy:  0.945, Loss:  0.056
    Epoch   9 Batch  247/538 - Train Accuracy:  0.935, Validation Accuracy:  0.945, Loss:  0.067
    Epoch   9 Batch  248/538 - Train Accuracy:  0.946, Validation Accuracy:  0.943, Loss:  0.075
    Epoch   9 Batch  249/538 - Train Accuracy:  0.967, Validation Accuracy:  0.945, Loss:  0.050
    Epoch   9 Batch  250/538 - Train Accuracy:  0.959, Validation Accuracy:  0.945, Loss:  0.059
    Epoch   9 Batch  251/538 - Train Accuracy:  0.951, Validation Accuracy:  0.945, Loss:  0.061
    Epoch   9 Batch  252/538 - Train Accuracy:  0.954, Validation Accuracy:  0.946, Loss:  0.065
    Epoch   9 Batch  253/538 - Train Accuracy:  0.947, Validation Accuracy:  0.945, Loss:  0.060
    Epoch   9 Batch  254/538 - Train Accuracy:  0.945, Validation Accuracy:  0.943, Loss:  0.075
    Epoch   9 Batch  255/538 - Train Accuracy:  0.960, Validation Accuracy:  0.943, Loss:  0.062
    Epoch   9 Batch  256/538 - Train Accuracy:  0.952, Validation Accuracy:  0.940, Loss:  0.066
    Epoch   9 Batch  257/538 - Train Accuracy:  0.961, Validation Accuracy:  0.939, Loss:  0.058
    Epoch   9 Batch  258/538 - Train Accuracy:  0.952, Validation Accuracy:  0.940, Loss:  0.075
    Epoch   9 Batch  259/538 - Train Accuracy:  0.958, Validation Accuracy:  0.945, Loss:  0.072
    Epoch   9 Batch  260/538 - Train Accuracy:  0.936, Validation Accuracy:  0.945, Loss:  0.074
    Epoch   9 Batch  261/538 - Train Accuracy:  0.963, Validation Accuracy:  0.943, Loss:  0.074
    Epoch   9 Batch  262/538 - Train Accuracy:  0.948, Validation Accuracy:  0.941, Loss:  0.063
    Epoch   9 Batch  263/538 - Train Accuracy:  0.950, Validation Accuracy:  0.939, Loss:  0.060
    Epoch   9 Batch  264/538 - Train Accuracy:  0.953, Validation Accuracy:  0.937, Loss:  0.065
    Epoch   9 Batch  265/538 - Train Accuracy:  0.945, Validation Accuracy:  0.939, Loss:  0.086
    Epoch   9 Batch  266/538 - Train Accuracy:  0.952, Validation Accuracy:  0.939, Loss:  0.072
    Epoch   9 Batch  267/538 - Train Accuracy:  0.943, Validation Accuracy:  0.941, Loss:  0.066
    Epoch   9 Batch  268/538 - Train Accuracy:  0.959, Validation Accuracy:  0.940, Loss:  0.055
    Epoch   9 Batch  269/538 - Train Accuracy:  0.944, Validation Accuracy:  0.937, Loss:  0.084
    Epoch   9 Batch  270/538 - Train Accuracy:  0.945, Validation Accuracy:  0.935, Loss:  0.072
    Epoch   9 Batch  271/538 - Train Accuracy:  0.955, Validation Accuracy:  0.935, Loss:  0.059
    Epoch   9 Batch  272/538 - Train Accuracy:  0.950, Validation Accuracy:  0.938, Loss:  0.072
    Epoch   9 Batch  273/538 - Train Accuracy:  0.952, Validation Accuracy:  0.940, Loss:  0.082
    Epoch   9 Batch  274/538 - Train Accuracy:  0.928, Validation Accuracy:  0.947, Loss:  0.103
    Epoch   9 Batch  275/538 - Train Accuracy:  0.954, Validation Accuracy:  0.950, Loss:  0.084
    Epoch   9 Batch  276/538 - Train Accuracy:  0.948, Validation Accuracy:  0.951, Loss:  0.072
    Epoch   9 Batch  277/538 - Train Accuracy:  0.952, Validation Accuracy:  0.952, Loss:  0.067
    Epoch   9 Batch  278/538 - Train Accuracy:  0.957, Validation Accuracy:  0.953, Loss:  0.082
    Epoch   9 Batch  279/538 - Train Accuracy:  0.951, Validation Accuracy:  0.950, Loss:  0.070
    Epoch   9 Batch  280/538 - Train Accuracy:  0.956, Validation Accuracy:  0.949, Loss:  0.061
    Epoch   9 Batch  281/538 - Train Accuracy:  0.953, Validation Accuracy:  0.954, Loss:  0.079
    Epoch   9 Batch  282/538 - Train Accuracy:  0.952, Validation Accuracy:  0.953, Loss:  0.080
    Epoch   9 Batch  283/538 - Train Accuracy:  0.957, Validation Accuracy:  0.953, Loss:  0.075
    Epoch   9 Batch  284/538 - Train Accuracy:  0.960, Validation Accuracy:  0.950, Loss:  0.082
    Epoch   9 Batch  285/538 - Train Accuracy:  0.953, Validation Accuracy:  0.948, Loss:  0.063
    Epoch   9 Batch  286/538 - Train Accuracy:  0.952, Validation Accuracy:  0.950, Loss:  0.088
    Epoch   9 Batch  287/538 - Train Accuracy:  0.964, Validation Accuracy:  0.955, Loss:  0.064
    Epoch   9 Batch  288/538 - Train Accuracy:  0.956, Validation Accuracy:  0.955, Loss:  0.068
    Epoch   9 Batch  289/538 - Train Accuracy:  0.954, Validation Accuracy:  0.955, Loss:  0.058
    Epoch   9 Batch  290/538 - Train Accuracy:  0.966, Validation Accuracy:  0.955, Loss:  0.054
    Epoch   9 Batch  291/538 - Train Accuracy:  0.961, Validation Accuracy:  0.953, Loss:  0.066
    Epoch   9 Batch  292/538 - Train Accuracy:  0.966, Validation Accuracy:  0.956, Loss:  0.054
    Epoch   9 Batch  293/538 - Train Accuracy:  0.949, Validation Accuracy:  0.957, Loss:  0.077
    Epoch   9 Batch  294/538 - Train Accuracy:  0.954, Validation Accuracy:  0.958, Loss:  0.071
    Epoch   9 Batch  295/538 - Train Accuracy:  0.947, Validation Accuracy:  0.957, Loss:  0.078
    Epoch   9 Batch  296/538 - Train Accuracy:  0.953, Validation Accuracy:  0.956, Loss:  0.076
    Epoch   9 Batch  297/538 - Train Accuracy:  0.965, Validation Accuracy:  0.952, Loss:  0.067
    Epoch   9 Batch  298/538 - Train Accuracy:  0.949, Validation Accuracy:  0.952, Loss:  0.062
    Epoch   9 Batch  299/538 - Train Accuracy:  0.952, Validation Accuracy:  0.954, Loss:  0.088
    Epoch   9 Batch  300/538 - Train Accuracy:  0.953, Validation Accuracy:  0.953, Loss:  0.071
    Epoch   9 Batch  301/538 - Train Accuracy:  0.944, Validation Accuracy:  0.953, Loss:  0.080
    Epoch   9 Batch  302/538 - Train Accuracy:  0.965, Validation Accuracy:  0.953, Loss:  0.062
    Epoch   9 Batch  303/538 - Train Accuracy:  0.953, Validation Accuracy:  0.955, Loss:  0.074
    Epoch   9 Batch  304/538 - Train Accuracy:  0.952, Validation Accuracy:  0.955, Loss:  0.078
    Epoch   9 Batch  305/538 - Train Accuracy:  0.960, Validation Accuracy:  0.955, Loss:  0.064
    Epoch   9 Batch  306/538 - Train Accuracy:  0.958, Validation Accuracy:  0.955, Loss:  0.066
    Epoch   9 Batch  307/538 - Train Accuracy:  0.964, Validation Accuracy:  0.953, Loss:  0.069
    Epoch   9 Batch  308/538 - Train Accuracy:  0.953, Validation Accuracy:  0.954, Loss:  0.056
    Epoch   9 Batch  309/538 - Train Accuracy:  0.948, Validation Accuracy:  0.955, Loss:  0.052
    Epoch   9 Batch  310/538 - Train Accuracy:  0.962, Validation Accuracy:  0.951, Loss:  0.075
    Epoch   9 Batch  311/538 - Train Accuracy:  0.947, Validation Accuracy:  0.951, Loss:  0.084
    Epoch   9 Batch  312/538 - Train Accuracy:  0.957, Validation Accuracy:  0.951, Loss:  0.064
    Epoch   9 Batch  313/538 - Train Accuracy:  0.956, Validation Accuracy:  0.953, Loss:  0.068
    Epoch   9 Batch  314/538 - Train Accuracy:  0.959, Validation Accuracy:  0.953, Loss:  0.072
    Epoch   9 Batch  315/538 - Train Accuracy:  0.955, Validation Accuracy:  0.953, Loss:  0.063
    Epoch   9 Batch  316/538 - Train Accuracy:  0.947, Validation Accuracy:  0.959, Loss:  0.062
    Epoch   9 Batch  317/538 - Train Accuracy:  0.955, Validation Accuracy:  0.955, Loss:  0.075
    Epoch   9 Batch  318/538 - Train Accuracy:  0.950, Validation Accuracy:  0.957, Loss:  0.071
    Epoch   9 Batch  319/538 - Train Accuracy:  0.961, Validation Accuracy:  0.957, Loss:  0.060
    Epoch   9 Batch  320/538 - Train Accuracy:  0.954, Validation Accuracy:  0.957, Loss:  0.070
    Epoch   9 Batch  321/538 - Train Accuracy:  0.945, Validation Accuracy:  0.955, Loss:  0.060
    Epoch   9 Batch  322/538 - Train Accuracy:  0.959, Validation Accuracy:  0.952, Loss:  0.080
    Epoch   9 Batch  323/538 - Train Accuracy:  0.955, Validation Accuracy:  0.954, Loss:  0.068
    Epoch   9 Batch  324/538 - Train Accuracy:  0.968, Validation Accuracy:  0.952, Loss:  0.068
    Epoch   9 Batch  325/538 - Train Accuracy:  0.951, Validation Accuracy:  0.950, Loss:  0.067
    Epoch   9 Batch  326/538 - Train Accuracy:  0.953, Validation Accuracy:  0.952, Loss:  0.062
    Epoch   9 Batch  327/538 - Train Accuracy:  0.953, Validation Accuracy:  0.957, Loss:  0.077
    Epoch   9 Batch  328/538 - Train Accuracy:  0.968, Validation Accuracy:  0.952, Loss:  0.063
    Epoch   9 Batch  329/538 - Train Accuracy:  0.961, Validation Accuracy:  0.952, Loss:  0.066
    Epoch   9 Batch  330/538 - Train Accuracy:  0.948, Validation Accuracy:  0.951, Loss:  0.056
    Epoch   9 Batch  331/538 - Train Accuracy:  0.953, Validation Accuracy:  0.950, Loss:  0.068
    Epoch   9 Batch  332/538 - Train Accuracy:  0.947, Validation Accuracy:  0.953, Loss:  0.073
    Epoch   9 Batch  333/538 - Train Accuracy:  0.960, Validation Accuracy:  0.953, Loss:  0.071
    Epoch   9 Batch  334/538 - Train Accuracy:  0.953, Validation Accuracy:  0.952, Loss:  0.067
    Epoch   9 Batch  335/538 - Train Accuracy:  0.951, Validation Accuracy:  0.948, Loss:  0.068
    Epoch   9 Batch  336/538 - Train Accuracy:  0.953, Validation Accuracy:  0.948, Loss:  0.071
    Epoch   9 Batch  337/538 - Train Accuracy:  0.947, Validation Accuracy:  0.952, Loss:  0.067
    Epoch   9 Batch  338/538 - Train Accuracy:  0.959, Validation Accuracy:  0.952, Loss:  0.064
    Epoch   9 Batch  339/538 - Train Accuracy:  0.951, Validation Accuracy:  0.952, Loss:  0.069
    Epoch   9 Batch  340/538 - Train Accuracy:  0.938, Validation Accuracy:  0.952, Loss:  0.070
    Epoch   9 Batch  341/538 - Train Accuracy:  0.943, Validation Accuracy:  0.952, Loss:  0.069
    Epoch   9 Batch  342/538 - Train Accuracy:  0.958, Validation Accuracy:  0.949, Loss:  0.073
    Epoch   9 Batch  343/538 - Train Accuracy:  0.961, Validation Accuracy:  0.950, Loss:  0.070
    Epoch   9 Batch  344/538 - Train Accuracy:  0.958, Validation Accuracy:  0.940, Loss:  0.062
    Epoch   9 Batch  345/538 - Train Accuracy:  0.959, Validation Accuracy:  0.938, Loss:  0.068
    Epoch   9 Batch  346/538 - Train Accuracy:  0.953, Validation Accuracy:  0.938, Loss:  0.073
    Epoch   9 Batch  347/538 - Train Accuracy:  0.963, Validation Accuracy:  0.941, Loss:  0.058
    Epoch   9 Batch  348/538 - Train Accuracy:  0.950, Validation Accuracy:  0.944, Loss:  0.076
    Epoch   9 Batch  349/538 - Train Accuracy:  0.969, Validation Accuracy:  0.944, Loss:  0.058
    Epoch   9 Batch  350/538 - Train Accuracy:  0.951, Validation Accuracy:  0.944, Loss:  0.083
    Epoch   9 Batch  351/538 - Train Accuracy:  0.950, Validation Accuracy:  0.948, Loss:  0.079
    Epoch   9 Batch  352/538 - Train Accuracy:  0.942, Validation Accuracy:  0.951, Loss:  0.090
    Epoch   9 Batch  353/538 - Train Accuracy:  0.941, Validation Accuracy:  0.951, Loss:  0.081
    Epoch   9 Batch  354/538 - Train Accuracy:  0.951, Validation Accuracy:  0.951, Loss:  0.076
    Epoch   9 Batch  355/538 - Train Accuracy:  0.953, Validation Accuracy:  0.949, Loss:  0.070
    Epoch   9 Batch  356/538 - Train Accuracy:  0.956, Validation Accuracy:  0.952, Loss:  0.069
    Epoch   9 Batch  357/538 - Train Accuracy:  0.964, Validation Accuracy:  0.951, Loss:  0.070
    Epoch   9 Batch  358/538 - Train Accuracy:  0.966, Validation Accuracy:  0.951, Loss:  0.053
    Epoch   9 Batch  359/538 - Train Accuracy:  0.942, Validation Accuracy:  0.950, Loss:  0.072
    Epoch   9 Batch  360/538 - Train Accuracy:  0.952, Validation Accuracy:  0.949, Loss:  0.081
    Epoch   9 Batch  361/538 - Train Accuracy:  0.954, Validation Accuracy:  0.950, Loss:  0.070
    Epoch   9 Batch  362/538 - Train Accuracy:  0.959, Validation Accuracy:  0.952, Loss:  0.053
    Epoch   9 Batch  363/538 - Train Accuracy:  0.943, Validation Accuracy:  0.952, Loss:  0.064
    Epoch   9 Batch  364/538 - Train Accuracy:  0.949, Validation Accuracy:  0.955, Loss:  0.086
    Epoch   9 Batch  365/538 - Train Accuracy:  0.945, Validation Accuracy:  0.953, Loss:  0.063
    Epoch   9 Batch  366/538 - Train Accuracy:  0.959, Validation Accuracy:  0.952, Loss:  0.070
    Epoch   9 Batch  367/538 - Train Accuracy:  0.955, Validation Accuracy:  0.957, Loss:  0.054
    Epoch   9 Batch  368/538 - Train Accuracy:  0.971, Validation Accuracy:  0.957, Loss:  0.059
    Epoch   9 Batch  369/538 - Train Accuracy:  0.948, Validation Accuracy:  0.955, Loss:  0.068
    Epoch   9 Batch  370/538 - Train Accuracy:  0.944, Validation Accuracy:  0.955, Loss:  0.069
    Epoch   9 Batch  371/538 - Train Accuracy:  0.961, Validation Accuracy:  0.953, Loss:  0.071
    Epoch   9 Batch  372/538 - Train Accuracy:  0.958, Validation Accuracy:  0.949, Loss:  0.061
    Epoch   9 Batch  373/538 - Train Accuracy:  0.951, Validation Accuracy:  0.945, Loss:  0.055
    Epoch   9 Batch  374/538 - Train Accuracy:  0.961, Validation Accuracy:  0.942, Loss:  0.066
    Epoch   9 Batch  375/538 - Train Accuracy:  0.951, Validation Accuracy:  0.942, Loss:  0.065
    Epoch   9 Batch  376/538 - Train Accuracy:  0.950, Validation Accuracy:  0.941, Loss:  0.068
    Epoch   9 Batch  377/538 - Train Accuracy:  0.965, Validation Accuracy:  0.944, Loss:  0.064
    Epoch   9 Batch  378/538 - Train Accuracy:  0.958, Validation Accuracy:  0.945, Loss:  0.061
    Epoch   9 Batch  379/538 - Train Accuracy:  0.966, Validation Accuracy:  0.950, Loss:  0.078
    Epoch   9 Batch  380/538 - Train Accuracy:  0.953, Validation Accuracy:  0.950, Loss:  0.055
    Epoch   9 Batch  381/538 - Train Accuracy:  0.966, Validation Accuracy:  0.952, Loss:  0.061
    Epoch   9 Batch  382/538 - Train Accuracy:  0.950, Validation Accuracy:  0.952, Loss:  0.077
    Epoch   9 Batch  383/538 - Train Accuracy:  0.956, Validation Accuracy:  0.949, Loss:  0.053
    Epoch   9 Batch  384/538 - Train Accuracy:  0.956, Validation Accuracy:  0.949, Loss:  0.068
    Epoch   9 Batch  385/538 - Train Accuracy:  0.956, Validation Accuracy:  0.947, Loss:  0.076
    Epoch   9 Batch  386/538 - Train Accuracy:  0.955, Validation Accuracy:  0.946, Loss:  0.066
    Epoch   9 Batch  387/538 - Train Accuracy:  0.952, Validation Accuracy:  0.950, Loss:  0.067
    Epoch   9 Batch  388/538 - Train Accuracy:  0.956, Validation Accuracy:  0.949, Loss:  0.072
    Epoch   9 Batch  389/538 - Train Accuracy:  0.946, Validation Accuracy:  0.949, Loss:  0.086
    Epoch   9 Batch  390/538 - Train Accuracy:  0.965, Validation Accuracy:  0.950, Loss:  0.064
    Epoch   9 Batch  391/538 - Train Accuracy:  0.955, Validation Accuracy:  0.952, Loss:  0.069
    Epoch   9 Batch  392/538 - Train Accuracy:  0.957, Validation Accuracy:  0.952, Loss:  0.065
    Epoch   9 Batch  393/538 - Train Accuracy:  0.953, Validation Accuracy:  0.954, Loss:  0.062
    Epoch   9 Batch  394/538 - Train Accuracy:  0.947, Validation Accuracy:  0.950, Loss:  0.078
    Epoch   9 Batch  395/538 - Train Accuracy:  0.950, Validation Accuracy:  0.944, Loss:  0.072
    Epoch   9 Batch  396/538 - Train Accuracy:  0.953, Validation Accuracy:  0.948, Loss:  0.074
    Epoch   9 Batch  397/538 - Train Accuracy:  0.953, Validation Accuracy:  0.945, Loss:  0.068
    Epoch   9 Batch  398/538 - Train Accuracy:  0.944, Validation Accuracy:  0.946, Loss:  0.066
    Epoch   9 Batch  399/538 - Train Accuracy:  0.946, Validation Accuracy:  0.946, Loss:  0.074
    Epoch   9 Batch  400/538 - Train Accuracy:  0.961, Validation Accuracy:  0.948, Loss:  0.071
    Epoch   9 Batch  401/538 - Train Accuracy:  0.962, Validation Accuracy:  0.950, Loss:  0.074
    Epoch   9 Batch  402/538 - Train Accuracy:  0.957, Validation Accuracy:  0.949, Loss:  0.062
    Epoch   9 Batch  403/538 - Train Accuracy:  0.956, Validation Accuracy:  0.953, Loss:  0.071
    Epoch   9 Batch  404/538 - Train Accuracy:  0.954, Validation Accuracy:  0.949, Loss:  0.076
    Epoch   9 Batch  405/538 - Train Accuracy:  0.958, Validation Accuracy:  0.945, Loss:  0.067
    Epoch   9 Batch  406/538 - Train Accuracy:  0.951, Validation Accuracy:  0.944, Loss:  0.068
    Epoch   9 Batch  407/538 - Train Accuracy:  0.949, Validation Accuracy:  0.945, Loss:  0.070
    Epoch   9 Batch  408/538 - Train Accuracy:  0.946, Validation Accuracy:  0.944, Loss:  0.085
    Epoch   9 Batch  409/538 - Train Accuracy:  0.952, Validation Accuracy:  0.942, Loss:  0.066
    Epoch   9 Batch  410/538 - Train Accuracy:  0.958, Validation Accuracy:  0.944, Loss:  0.066
    Epoch   9 Batch  411/538 - Train Accuracy:  0.961, Validation Accuracy:  0.941, Loss:  0.064
    Epoch   9 Batch  412/538 - Train Accuracy:  0.964, Validation Accuracy:  0.939, Loss:  0.057
    Epoch   9 Batch  413/538 - Train Accuracy:  0.956, Validation Accuracy:  0.942, Loss:  0.067
    Epoch   9 Batch  414/538 - Train Accuracy:  0.930, Validation Accuracy:  0.943, Loss:  0.082
    Epoch   9 Batch  415/538 - Train Accuracy:  0.935, Validation Accuracy:  0.944, Loss:  0.075
    Epoch   9 Batch  416/538 - Train Accuracy:  0.960, Validation Accuracy:  0.938, Loss:  0.070
    Epoch   9 Batch  417/538 - Train Accuracy:  0.959, Validation Accuracy:  0.935, Loss:  0.071
    Epoch   9 Batch  418/538 - Train Accuracy:  0.961, Validation Accuracy:  0.938, Loss:  0.085
    Epoch   9 Batch  419/538 - Train Accuracy:  0.963, Validation Accuracy:  0.943, Loss:  0.051
    Epoch   9 Batch  420/538 - Train Accuracy:  0.957, Validation Accuracy:  0.944, Loss:  0.065
    Epoch   9 Batch  421/538 - Train Accuracy:  0.954, Validation Accuracy:  0.943, Loss:  0.054
    Epoch   9 Batch  422/538 - Train Accuracy:  0.956, Validation Accuracy:  0.945, Loss:  0.063
    Epoch   9 Batch  423/538 - Train Accuracy:  0.944, Validation Accuracy:  0.944, Loss:  0.073
    Epoch   9 Batch  424/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.074
    Epoch   9 Batch  425/538 - Train Accuracy:  0.938, Validation Accuracy:  0.944, Loss:  0.088
    Epoch   9 Batch  426/538 - Train Accuracy:  0.958, Validation Accuracy:  0.941, Loss:  0.061
    Epoch   9 Batch  427/538 - Train Accuracy:  0.939, Validation Accuracy:  0.941, Loss:  0.082
    Epoch   9 Batch  428/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.055
    Epoch   9 Batch  429/538 - Train Accuracy:  0.958, Validation Accuracy:  0.941, Loss:  0.069
    Epoch   9 Batch  430/538 - Train Accuracy:  0.955, Validation Accuracy:  0.947, Loss:  0.068
    Epoch   9 Batch  431/538 - Train Accuracy:  0.941, Validation Accuracy:  0.947, Loss:  0.071
    Epoch   9 Batch  432/538 - Train Accuracy:  0.960, Validation Accuracy:  0.947, Loss:  0.077
    Epoch   9 Batch  433/538 - Train Accuracy:  0.939, Validation Accuracy:  0.949, Loss:  0.102
    Epoch   9 Batch  434/538 - Train Accuracy:  0.952, Validation Accuracy:  0.947, Loss:  0.064
    Epoch   9 Batch  435/538 - Train Accuracy:  0.949, Validation Accuracy:  0.946, Loss:  0.070
    Epoch   9 Batch  436/538 - Train Accuracy:  0.946, Validation Accuracy:  0.946, Loss:  0.077
    Epoch   9 Batch  437/538 - Train Accuracy:  0.956, Validation Accuracy:  0.947, Loss:  0.071
    Epoch   9 Batch  438/538 - Train Accuracy:  0.959, Validation Accuracy:  0.946, Loss:  0.068
    Epoch   9 Batch  439/538 - Train Accuracy:  0.962, Validation Accuracy:  0.948, Loss:  0.054
    Epoch   9 Batch  440/538 - Train Accuracy:  0.953, Validation Accuracy:  0.950, Loss:  0.077
    Epoch   9 Batch  441/538 - Train Accuracy:  0.945, Validation Accuracy:  0.952, Loss:  0.094
    Epoch   9 Batch  442/538 - Train Accuracy:  0.948, Validation Accuracy:  0.947, Loss:  0.054
    Epoch   9 Batch  443/538 - Train Accuracy:  0.955, Validation Accuracy:  0.944, Loss:  0.061
    Epoch   9 Batch  444/538 - Train Accuracy:  0.964, Validation Accuracy:  0.942, Loss:  0.062
    Epoch   9 Batch  445/538 - Train Accuracy:  0.957, Validation Accuracy:  0.942, Loss:  0.059
    Epoch   9 Batch  446/538 - Train Accuracy:  0.965, Validation Accuracy:  0.942, Loss:  0.067
    Epoch   9 Batch  447/538 - Train Accuracy:  0.945, Validation Accuracy:  0.942, Loss:  0.068
    Epoch   9 Batch  448/538 - Train Accuracy:  0.953, Validation Accuracy:  0.940, Loss:  0.059
    Epoch   9 Batch  449/538 - Train Accuracy:  0.955, Validation Accuracy:  0.942, Loss:  0.079
    Epoch   9 Batch  450/538 - Train Accuracy:  0.939, Validation Accuracy:  0.944, Loss:  0.094
    Epoch   9 Batch  451/538 - Train Accuracy:  0.945, Validation Accuracy:  0.947, Loss:  0.061
    Epoch   9 Batch  452/538 - Train Accuracy:  0.965, Validation Accuracy:  0.947, Loss:  0.057
    Epoch   9 Batch  453/538 - Train Accuracy:  0.956, Validation Accuracy:  0.947, Loss:  0.081
    Epoch   9 Batch  454/538 - Train Accuracy:  0.951, Validation Accuracy:  0.947, Loss:  0.063
    Epoch   9 Batch  455/538 - Train Accuracy:  0.955, Validation Accuracy:  0.943, Loss:  0.074
    Epoch   9 Batch  456/538 - Train Accuracy:  0.961, Validation Accuracy:  0.943, Loss:  0.095
    Epoch   9 Batch  457/538 - Train Accuracy:  0.960, Validation Accuracy:  0.947, Loss:  0.070
    Epoch   9 Batch  458/538 - Train Accuracy:  0.961, Validation Accuracy:  0.947, Loss:  0.060
    Epoch   9 Batch  459/538 - Train Accuracy:  0.963, Validation Accuracy:  0.947, Loss:  0.054
    Epoch   9 Batch  460/538 - Train Accuracy:  0.950, Validation Accuracy:  0.944, Loss:  0.071
    Epoch   9 Batch  461/538 - Train Accuracy:  0.963, Validation Accuracy:  0.944, Loss:  0.064
    Epoch   9 Batch  462/538 - Train Accuracy:  0.952, Validation Accuracy:  0.948, Loss:  0.070
    Epoch   9 Batch  463/538 - Train Accuracy:  0.937, Validation Accuracy:  0.950, Loss:  0.067
    Epoch   9 Batch  464/538 - Train Accuracy:  0.963, Validation Accuracy:  0.949, Loss:  0.066
    Epoch   9 Batch  465/538 - Train Accuracy:  0.952, Validation Accuracy:  0.949, Loss:  0.061
    Epoch   9 Batch  466/538 - Train Accuracy:  0.938, Validation Accuracy:  0.949, Loss:  0.070
    Epoch   9 Batch  467/538 - Train Accuracy:  0.967, Validation Accuracy:  0.947, Loss:  0.073
    Epoch   9 Batch  468/538 - Train Accuracy:  0.970, Validation Accuracy:  0.948, Loss:  0.072
    Epoch   9 Batch  469/538 - Train Accuracy:  0.946, Validation Accuracy:  0.945, Loss:  0.071
    Epoch   9 Batch  470/538 - Train Accuracy:  0.952, Validation Accuracy:  0.944, Loss:  0.069
    Epoch   9 Batch  471/538 - Train Accuracy:  0.965, Validation Accuracy:  0.947, Loss:  0.051
    Epoch   9 Batch  472/538 - Train Accuracy:  0.977, Validation Accuracy:  0.943, Loss:  0.051
    Epoch   9 Batch  473/538 - Train Accuracy:  0.949, Validation Accuracy:  0.940, Loss:  0.082
    Epoch   9 Batch  474/538 - Train Accuracy:  0.951, Validation Accuracy:  0.940, Loss:  0.064
    Epoch   9 Batch  475/538 - Train Accuracy:  0.952, Validation Accuracy:  0.935, Loss:  0.067
    Epoch   9 Batch  476/538 - Train Accuracy:  0.961, Validation Accuracy:  0.930, Loss:  0.058
    Epoch   9 Batch  477/538 - Train Accuracy:  0.941, Validation Accuracy:  0.933, Loss:  0.079
    Epoch   9 Batch  478/538 - Train Accuracy:  0.959, Validation Accuracy:  0.935, Loss:  0.065
    Epoch   9 Batch  479/538 - Train Accuracy:  0.945, Validation Accuracy:  0.938, Loss:  0.065
    Epoch   9 Batch  480/538 - Train Accuracy:  0.952, Validation Accuracy:  0.946, Loss:  0.063
    Epoch   9 Batch  481/538 - Train Accuracy:  0.961, Validation Accuracy:  0.945, Loss:  0.072
    Epoch   9 Batch  482/538 - Train Accuracy:  0.949, Validation Accuracy:  0.951, Loss:  0.061
    Epoch   9 Batch  483/538 - Train Accuracy:  0.938, Validation Accuracy:  0.948, Loss:  0.073
    Epoch   9 Batch  484/538 - Train Accuracy:  0.950, Validation Accuracy:  0.950, Loss:  0.090
    Epoch   9 Batch  485/538 - Train Accuracy:  0.956, Validation Accuracy:  0.946, Loss:  0.064
    Epoch   9 Batch  486/538 - Train Accuracy:  0.960, Validation Accuracy:  0.946, Loss:  0.059
    Epoch   9 Batch  487/538 - Train Accuracy:  0.964, Validation Accuracy:  0.944, Loss:  0.050
    Epoch   9 Batch  488/538 - Train Accuracy:  0.961, Validation Accuracy:  0.945, Loss:  0.063
    Epoch   9 Batch  489/538 - Train Accuracy:  0.952, Validation Accuracy:  0.949, Loss:  0.065
    Epoch   9 Batch  490/538 - Train Accuracy:  0.953, Validation Accuracy:  0.950, Loss:  0.070
    Epoch   9 Batch  491/538 - Train Accuracy:  0.935, Validation Accuracy:  0.947, Loss:  0.078
    Epoch   9 Batch  492/538 - Train Accuracy:  0.957, Validation Accuracy:  0.947, Loss:  0.062
    Epoch   9 Batch  493/538 - Train Accuracy:  0.950, Validation Accuracy:  0.946, Loss:  0.063
    Epoch   9 Batch  494/538 - Train Accuracy:  0.954, Validation Accuracy:  0.945, Loss:  0.071
    Epoch   9 Batch  495/538 - Train Accuracy:  0.959, Validation Accuracy:  0.947, Loss:  0.073
    Epoch   9 Batch  496/538 - Train Accuracy:  0.963, Validation Accuracy:  0.948, Loss:  0.063
    Epoch   9 Batch  497/538 - Train Accuracy:  0.961, Validation Accuracy:  0.953, Loss:  0.060
    Epoch   9 Batch  498/538 - Train Accuracy:  0.964, Validation Accuracy:  0.954, Loss:  0.064
    Epoch   9 Batch  499/538 - Train Accuracy:  0.950, Validation Accuracy:  0.951, Loss:  0.067
    Epoch   9 Batch  500/538 - Train Accuracy:  0.963, Validation Accuracy:  0.949, Loss:  0.053
    Epoch   9 Batch  501/538 - Train Accuracy:  0.959, Validation Accuracy:  0.948, Loss:  0.077
    Epoch   9 Batch  502/538 - Train Accuracy:  0.951, Validation Accuracy:  0.945, Loss:  0.072
    Epoch   9 Batch  503/538 - Train Accuracy:  0.961, Validation Accuracy:  0.945, Loss:  0.074
    Epoch   9 Batch  504/538 - Train Accuracy:  0.962, Validation Accuracy:  0.944, Loss:  0.061
    Epoch   9 Batch  505/538 - Train Accuracy:  0.953, Validation Accuracy:  0.944, Loss:  0.058
    Epoch   9 Batch  506/538 - Train Accuracy:  0.953, Validation Accuracy:  0.946, Loss:  0.056
    Epoch   9 Batch  507/538 - Train Accuracy:  0.942, Validation Accuracy:  0.947, Loss:  0.069
    Epoch   9 Batch  508/538 - Train Accuracy:  0.936, Validation Accuracy:  0.946, Loss:  0.071
    Epoch   9 Batch  509/538 - Train Accuracy:  0.954, Validation Accuracy:  0.947, Loss:  0.066
    Epoch   9 Batch  510/538 - Train Accuracy:  0.962, Validation Accuracy:  0.947, Loss:  0.062
    Epoch   9 Batch  511/538 - Train Accuracy:  0.954, Validation Accuracy:  0.946, Loss:  0.063
    Epoch   9 Batch  512/538 - Train Accuracy:  0.958, Validation Accuracy:  0.946, Loss:  0.065
    Epoch   9 Batch  513/538 - Train Accuracy:  0.954, Validation Accuracy:  0.948, Loss:  0.062
    Epoch   9 Batch  514/538 - Train Accuracy:  0.962, Validation Accuracy:  0.949, Loss:  0.064
    Epoch   9 Batch  515/538 - Train Accuracy:  0.951, Validation Accuracy:  0.944, Loss:  0.072
    Epoch   9 Batch  516/538 - Train Accuracy:  0.954, Validation Accuracy:  0.945, Loss:  0.059
    Epoch   9 Batch  517/538 - Train Accuracy:  0.958, Validation Accuracy:  0.943, Loss:  0.071
    Epoch   9 Batch  518/538 - Train Accuracy:  0.946, Validation Accuracy:  0.941, Loss:  0.077
    Epoch   9 Batch  519/538 - Train Accuracy:  0.954, Validation Accuracy:  0.941, Loss:  0.063
    Epoch   9 Batch  520/538 - Train Accuracy:  0.931, Validation Accuracy:  0.939, Loss:  0.077
    Epoch   9 Batch  521/538 - Train Accuracy:  0.950, Validation Accuracy:  0.941, Loss:  0.067
    Epoch   9 Batch  522/538 - Train Accuracy:  0.956, Validation Accuracy:  0.941, Loss:  0.054
    Epoch   9 Batch  523/538 - Train Accuracy:  0.956, Validation Accuracy:  0.941, Loss:  0.060
    Epoch   9 Batch  524/538 - Train Accuracy:  0.956, Validation Accuracy:  0.944, Loss:  0.064
    Epoch   9 Batch  525/538 - Train Accuracy:  0.960, Validation Accuracy:  0.944, Loss:  0.070
    Epoch   9 Batch  526/538 - Train Accuracy:  0.952, Validation Accuracy:  0.943, Loss:  0.071
    Epoch   9 Batch  527/538 - Train Accuracy:  0.953, Validation Accuracy:  0.946, Loss:  0.064
    Epoch   9 Batch  528/538 - Train Accuracy:  0.949, Validation Accuracy:  0.952, Loss:  0.064
    Epoch   9 Batch  529/538 - Train Accuracy:  0.938, Validation Accuracy:  0.950, Loss:  0.086
    Epoch   9 Batch  530/538 - Train Accuracy:  0.949, Validation Accuracy:  0.955, Loss:  0.077
    Epoch   9 Batch  531/538 - Train Accuracy:  0.959, Validation Accuracy:  0.950, Loss:  0.066
    Epoch   9 Batch  532/538 - Train Accuracy:  0.946, Validation Accuracy:  0.947, Loss:  0.058
    Epoch   9 Batch  533/538 - Train Accuracy:  0.954, Validation Accuracy:  0.944, Loss:  0.064
    Epoch   9 Batch  534/538 - Train Accuracy:  0.957, Validation Accuracy:  0.944, Loss:  0.052
    Epoch   9 Batch  535/538 - Train Accuracy:  0.964, Validation Accuracy:  0.944, Loss:  0.070
    Epoch   9 Batch  536/538 - Train Accuracy:  0.964, Validation Accuracy:  0.941, Loss:  0.071
    Model Trained and Saved


### Save Parameters
Save the `batch_size` and `save_path` parameters for inference.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params(save_path)
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()
```

## Sentence to Sequence
To feed a sentence into the model for translation, you first need to preprocess it.  Implement the function `sentence_to_seq()` to preprocess new sentences.

- Convert the sentence to lowercase
- Convert words into ids using `vocab_to_int`
 - Convert words not in the vocabulary, to the `<UNK>` word id.


```python
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    
    sentence_id = [vocab_to_int.get(word.lower(), vocab_to_int['<UNK>']) for word in sentence.split()]
    return sentence_id


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)
```

    Tests Passed


## Translate
This will translate `translate_sentence` from English to French.


```python
translate_sentence = 'he saw a old yellow truck .'
#translate_sentence = 'this is a big car .'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))
```

    Input
      Word Ids:      [23, 207, 51, 228, 96, 19, 133]
      English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']
    
    Prediction
      Word Ids:      [153, 99, 170, 123, 35, 314, 51, 1]
      French Words: ['il', 'a', 'vu', 'une', 'vieille', 'voiture', '.', '<EOS>']


## Imperfect Translation
You might notice that some sentences translate better than others.  Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words.  For this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.

You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided.  Just make sure you play with the WMT10 corpus after you've submitted this project.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_language_translation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
