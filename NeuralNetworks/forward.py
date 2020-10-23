import tensorflow as tf
import numpy as np
from math import sqrt


class BadLayerSizeException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def WeightInit_StandardDeviation(prev_neurone_count, neurone_count):
    return tf.constant_initializer(value=(np.random.normal(size=(prev_neurone_count, neurone_count)))/sqrt(prev_neurone_count))


class ReinforceTraitNetwork(tf.keras.Model):
    def __init__(self, learning_param, network_layout, dropout=False):
        super().__init__()
        
        self.learning_param = learning_param
        self.network_layout = network_layout
        self.dropout = dropout


        self.spacing = "".join(["-" for _ in range(30)])
        self.sub_spacing = "".join(["~" for _ in range(2)])

        self.step = learning_param[0]
        self.momentum = learning_param[1]

        self.activation_funcs = [layer[0] for layer in network_layout[1:]]


        self.layer_sizes = np.empty(len(network_layout), dtype="int32")
        self.layer_sizes[0] = self.network_layout[0]
        for index, layer in enumerate(network_layout[1:-1]):
            if layer[1] % 8 != 0:
                raise BadLayerSizeException("[ERROR] Please select a network size, which is a multiple of 8 for performance reasons")
            self.layer_sizes[index+1] = layer[1]
        self.layer_sizes[-1] = network_layout[-1][1]

        self.number_of_param = sum([((prev*now)+now) for prev, now in zip(self.layer_sizes, self.layer_sizes[1:])]) 

        self.model = []
        # To take stuff like dropout into account, as they are treated as layers
        for prev, now, func in zip(self.layer_sizes[:-2], self.layer_sizes[1:-1], self.activation_funcs):
            lay = [tf.keras.layers.Dense(now, activation=func, kernel_initializer=WeightInit_StandardDeviation(prev, now), bias_initializer=tf.keras.initializers.Zeros())]

            if dropout:
                lay.append(tf.keras.layers.Dropout(0.5))

            lay[0].build((prev,))
            self.model.append(lay)
        lay = [tf.keras.layers.Dense(self.layer_sizes[-1], activation=self.activation_funcs[-1], kernel_initializer=WeightInit_StandardDeviation(self.layer_sizes[-2], self.layer_sizes[-1]), bias_initializer=tf.keras.initializers.Zeros())]
        lay[0].build((self.layer_sizes[-2],))
        self.model.append(lay)

        self.opti=tf.keras.optimizers.SGD(learning_rate=self.step, momentum=self.momentum)
        self.loss=tf.keras.losses.MeanSquaredError()
        

        self.layers_num = len(self.model)

        # Inforams
        self.reset_info() 
    

    def get_config(self):
        return {"learning_param": self.learning_param, "network_layout": self.network_layout, "dropout": self.dropout}


    def reset_info(self):
        self.network_derivitives = []
        for prev, now in zip(self.layer_sizes, self.layer_sizes[1:]):
            self.network_derivitives.append(tf.constant(np.zeros((now, prev))))
            self.network_derivitives.append(tf.constant(np.zeros(now)))

        ## Cost
        self.network_cost = 0
        
        ## Layer info 
        self.layer_inputs = [np.zeros(size) for size in self.layer_sizes[:-1]]
        self.layer_outputs = [np.zeros(size) for size in self.layer_sizes[1:]]
        

    def print_parameters(self):
        print(self.spacing)
        for index, layer in enumerate(self.model):
            index+=1
            print("Layer " + str(index) + " :")
            print("Weights " + str(index) + " : " + str(layer[0].get_weights())[0])
            print("Bias " + str(index) + " : " + str(layer[0].get_weights()[1]))

            print(self.sub_spacing)
        print("layers parameters: " + str(self.number_of_param))

    def print_derivitives(self):
        print(self.spacing)
        print("layers derivitives:")
        for count, index in enumerate(range(0, len(self.network_derivitives), 2)):
            count+=1
            print("Layer " + str(count) + " :")
            print("Weight derivitive " + str(count) + " : " + str(self.network_derivitives[index]))
            print("Bias derivitive " + str(count) + " : " + str(self.network_derivitives[index+1]))
            print(self.sub_spacing)


    def print_stability(self):
        print(self.spacing)
        print("Stability:")
        print(self.sub_spacing)
        print("Average update: " + str(np.sum(np.array([np.sum(layer) for layer in self.network_derivitives]))/self.number_of_param))
        buffer = np.empty(self.layers_num)
        for count, index in enumerate(range(0, len(self.network_derivitives), 2)):
            buffer[count] = (np.sum(self.network_derivitives[index]) + np.sum(self.network_derivitives[index+1]))/(tf.size(self.network_derivitives[index]) + tf.size(self.network_derivitives[index+1]))
        print("Average update per layer: " + str(buffer))
        print(self.sub_spacing)
        print("Largest absolute update: " + str(np.max(np.array([np.max(np.absolute(layer)) for layer in self.network_derivitives]))))
        
        for count, index in enumerate(range(0, len(self.network_derivitives), 2)):
            buffer[count] = (np.max(np.array([np.max(np.absolute(self.network_derivitives[index])), np.max(np.absolute(self.network_derivitives[index+1]))])))
        print("Largest absolute update per layer: " + str(buffer))

        print(self.sub_spacing)
        print("Largest absolute param: " + str(np.max([np.max([np.max(np.absolute(layer[0].get_weights()[0])), np.max(np.absolute(layer[0].get_weights()[1]))]) for layer in self.model])))

        for count, layer in enumerate(self.model):
            buffer[count] = np.max(np.array([np.max(np.absolute(layer[0].get_weights()[0])), np.max(np.absolute(layer[0].get_weights()[1]))]))
        print("Largest absolute param per layer: " + str(buffer))
        print(self.sub_spacing)
    
    def print_io(self):
        for count, layer_input in enumerate(self.layer_inputs):
            print("Layer INPUT " + str(count+1) + " : " + str(layer_input))
        print("Network output: " + str(self.layer_outputs[-1]))

    def print_cost(self):
        print(self.network_cost)

    def train(self, x, predict):
        truth = np.zeros(self.layer_sizes[-1])
        for position, value in predict:
            truth[position] = value
        
        with tf.GradientTape() as tape:
            y = self(np.array([x]), training=True) 
            cost = self.loss(truth, y)

        grads = tape.gradient(cost, self.trainable_weights)
        self.network_derivitives = [tf.clip_by_value(layer, -5, 5) for layer in grads]
        self.opti.apply_gradients(zip(self.network_derivitives, self.trainable_weights))
        self.network_cost = cost.numpy()
    
    
    #@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=(1,), dtype=tf.bool)])
    def call(self, x, training=False):
        if training:
            for count, ops in enumerate(self.model[:-1]):
                self.layer_inputs[count] = x
                for layer in ops:
                   
                    x = layer(x, training=True)
                self.layer_outputs[count] = x

            self.layer_inputs[-1] = x
            for layer in self.model[-1]:
                x = layer(x, training=True)
            self.layer_outputs[-1] = x

        else:
            for ops in self.model[:-1]:
                for layer in ops:
                    x = layer(x, training=False)
            for layer in self.model[-1]:
                x = layer(x, training=False)
        return x

    
