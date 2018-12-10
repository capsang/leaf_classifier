# Based on this article:

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from keras.utils.vis_utils import plot_model

import csv

# For tensorboard:
import numpy as np
from IPython.display import clear_output, Image, display, HTML
import tensorflow as tf


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>" % size
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
# end for tensorboard

class AdamOptimizer:
    def __init__(self, params):
        self.params = params
    
class SGDOptimizer:
    def __init__(self, params):
        self.method = keras.optimizers.SGD
        self.params = params

    def instance(self):
        return self.method(lr=self.params.learning_rate, momentum=self.params.momentum,
                           decay=self.params.decay, nesterov=False)


class Optimizer:
    def __init__(self, params):
        self.optimizers = {
            "sgd": SGDOptimizer
        }
        self.optimizer_id = params.optimizer
        self.params = params

    def instance(self):
        return self.optimizers[self.optimizer_id](self.params).instance()


TRAIN_DATA_FILE_PATH = './train.csv'
TEST_DATA_FILE_PATH = './test.csv'


def load_data_from_csv(path_to_csv, x_field_ranges, y_field_range):
    '''
        Note: each range in x_field_ranges corresponds to a given filter e.g. R or G or B.
        Hence, all range values within x_field_ranges must be same.
        This is typically used for conv NN.
    '''
    # x_result shape is (len(x_field_range), x_field_range, num_rows_input)
    x_result = np.empty(shape=(len(x_field_ranges), (x_field_ranges[0].stop - x_field_ranges[0].start), 0))
    y_result = np.empty(shape=(0, (y_field_range.stop - y_field_range.start)))
    with open(path_to_csv, 'r') as csv_file:
        data_iterator = csv.reader(csv_file, delimiter=',')
        # Skip the header
        next(data_iterator)

        for data_list in data_iterator:
            #print(row)
            x_data_all_channels = np.empty(shape=(0, (x_field_ranges[0].stop - x_field_ranges[0].start)))

            for x_field_range in x_field_ranges:
                x_data = data_list[x_field_range.start:x_field_range.stop]
                x_data_all_channels = np.append(x_data_all_channels, np.atleast_2d([x_data]), axis=0)
            x_result = np.append(x_result, np.atleast_3d(x_data_all_channels) , axis=2)
            y_data = data_list[y_field_range.start:y_field_range.stop]
            y_result = np.append(y_result, np.array([y_data]), axis=0)

    return (x_result, y_result)


class ModelFactory:
    def __init__(self, params):
        self.params = params

        # Training hyperparameters
        self.decay = params.decay
        self.learning_rate = params.learning_rate
        self.momentum = params.momentum
        self.optimizer = params.optimizer
        self.loss_function = params.loss_function
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        # Activation function to be applied to hidden layers.
        self.activation_function = params.activation_function

        # Network structure related hyperparameters
        # Number of hidden layers
        self.depth = params.depth
        # Number of neurons in each hidden layer
        self.widths = np.copy(params.widths)
        self.weight_init_strategy = params.weight_init_strategy
        # A regularization method used with larger networks
        #self.dropouts

        self.load_data()

    def extract_image_ids(self, path_to_csv):
        self.x_test_imgids = np.empty(0)
        # Get the test input image_ids (1st col) from the data.
        with open(path_to_csv, 'r') as csv_file:
            data_iterator = csv.reader(csv_file, delimiter=',')
            # Skip the header
            next(data_iterator)

            for data_list in data_iterator:
                self.x_test_imgids = np.append(self.x_test_imgids, data_list[0])

    def load_data(self):
        # Training data:
        #   File: 991 rows, 194 columns
        #   Required:
        #   Input shape: 990 rows, 192 columns (64 margins, 64 shapes, 64 textures)
        #   Output shape: 990 rows, 1 column (species name)
        # Test data:
        #   File: 594 rows, 193 columns
        #   Required:
        #   Input shape: 594 rows, 193 columns (ID, 64 margins, 64 shapes, 64 textures)
        #   Output shape: 594 rows, 100 columns (ID, 99 species probabilities)
        #       Note: Output shape is Model outpupt, Not obtained from file
        (self.x_train, self.y_train_str) = load_data_from_csv(
            TRAIN_DATA_FILE_PATH, [range(2, 66), range(66,130), range(130, 194)], range(1, 2))
        (self.x_test, discard) = load_data_from_csv(
            TEST_DATA_FILE_PATH, [range(1, 65), range(65, 129), range(129, 193)], range(0, 0))

        self.extract_image_ids(TEST_DATA_FILE_PATH)

        # Convert the output labels into integers, for one-hot-encoding
        label_encoder = LabelEncoder()
        self.y_train_num = label_encoder.fit_transform(self.y_train_str)

        print("training input data shape: ", self.x_train.shape)
        print("training output data shape: ", self.y_train_str.shape)
        print("test input data shape: ", self.x_test.shape)

        # Print the extracted data - for validation only
        print("Train data: ")
        for i in range(10):
            print(self.x_train[0, i, 0])

        print("\nTest data: ")
        for i in range(10):
            print(self.x_test[0, i, 0])
 
    def build_conv_model(self):
        #create model
        self.model = Sequential()
        #add model layers
        self.model.add(Conv2D(1, kernel_size=3, activation='relu', input_shape=(192,), kernel_initializer=self.weight_init_strategy))
        self.model.add(Conv2D(1, kernel_size=3, activation='relu',
                              kernel_initializer=self.weight_init_strategy))
        self.model.add(Flatten())
        self.model.add(Dense(99, activation='softmax',
                             kernel_initializer=self.weight_init_strategy))
        self.model.summary()
        #optimizer = Optimizer(self.params)
        self.model.compile(optimizer='adam',
                           loss=self.loss_function, metrics=['accuracy'])

    def build_model(self):
        self.model = Sequential()

        # At every layer's neuron: output = activation(dot(input, weight_input) + bias)
        # Input layer is implicit.
        # Hence, input for this layer becomes 192

        # Hidden layers
        for layer in range(self.depth):
            self.model.add(Dense(units=self.widths[layer], activation=self.activation_function,
                                 input_shape=(192,),
                                 kernel_initializer=self.weight_init_strategy))

        # Output layer: 99 neurons, one neuron for each species of plants.
        # Interesting: Softmax takes the output of all 99 neurons and distributes them into probabilities.
        # If we had applied Sigmoid for instance, then it'd apply sigmoid to each neuron separately.
        # See this (for a picture) https://sebastianraschka.com/faq/docs/softmax_regression.html
        num_classes = 99
        self.model.add(Dense(units=num_classes, activation='softmax',
                             kernel_initializer=self.weight_init_strategy))

        self.model.summary()

        optimizer = Optimizer(self.params)
        self.model.compile(optimizer=optimizer.instance(),
                           loss=self.loss_function, metrics=['accuracy'])

    def train(self):
        # Convert the output labels data to one-hot-vector
        num_classes = 99
        self.y_train_num = keras.utils.to_categorical(
            self.y_train_num, num_classes)
        #self.y_test = keras.utils.to_categorical(self.y_test, num_classes)

        print("training output data shape (one-hot-encoded): ",
              self.y_train_num.shape)

        # Training
        self.history = self.model.fit(self.x_train, self.y_train_num, batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      verbose=False, validation_split=.1)

    def validate(self):
        # Testing
        #self.loss, self.accuracy = self.model.evaluate(
        #    self.x_test, self.y_test, verbose=False) #TODO
        return self.model.predict(self.x_test)

    def plot_stats(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()

        print('Test loss: ', self.loss)
        print('Test accuracy: ', self.accuracy)


class NetworkParams:
    def __init__(self):
        self.weight_init_strategy = "glorot_uniform"
        self.depth = 3
        self.widths = []
        for layer in range(self.depth):
            self.widths.append(512)

        self.activation_function = "relu"
        self.epochs = 10
        self.batch_size = 128

        self.optimizer = "sgd"
        self.loss_function = "categorical_crossentropy"
        self.decay = 0.0
        self.momentum = 0.0
        self.learning_rate = 0.01


params = NetworkParams()
leafClassifier = ModelFactory(params)
#leafClassifier.build_model()
#leafClassifier.train()
#res = leafClassifier.validate()
#overall_res = np.empty((0, 1))
#for i in range(len(res)):
#    species = np.argmax(res[i])
#    print("Img: ", leafClassifier.x_test_imgids[i], " species: ", species)
#    overall_res = np.append(overall_res, species)

#print("Unique: ", len(np.unique(overall_res)))
#mnistFactory.plot_stats()
# Uncomment the below to generate a plot.png from keras utils.
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Uncomment the below line to show the tensorboard.
# show_graph(tf.get_default_graph().as_graph_def())
