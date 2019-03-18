import numpy as np

def sigmoid(x): 
  return 1/(1+np.exp(-x))

def dsigmoid(x): 
  return x*(1-x)

def relu(x):
  pass

def drelu(x):
  pass

class Connection(object):
  def __init__(self, neuron):
    self.neuron = neuron
    self.weight = np.random.rand()

class Neuron(object):
  def __init__(self):
    self.activation = 0.0
    self.last_error = 0.0
    self.error = 0.0
    self.delta = 0.0
    self.connections = []
  def connect(self, neuron):
    for c in self.connections:
      if neuron == c.neuron:
        raise RuntimeError("Duplicate connection attempt!")
    self.connections.append(Connection(neuron))
  def set_expected_activation(self, expected):
    self.error = -(expected - self.activation)
  def fire(self):
    self.activation = 0.0
    for c in self.connections:
      self.activation += c.neuron.activation * c.weight
    self.activation = sigmoid(self.activation)
  def back_propagate(self, learning_rate):
    self.delta = dsigmoid(self.activation) * self.error
    for c in self.connections:
      change = self.delta * c.neuron.activation
      c.neuron.error += self.delta * c.weight
      c.weight -= change * learning_rate
    self.last_error = self.error
    self.error = 0.0
      
class Layer(object):
  def __init__(self, n):
   self.neurons = [Neuron() for i in range(n)]
  def connect(self, layer):
    for n in self.neurons:
      for n2 in layer.neurons:
        n.connect(n2)
  def set_activations(self, activations):
    for i in range(len(activations)-1): 
      self.neurons[i].activation = sigmoid(activations[i])
  def get_activations(self):
    activations = []
    for neuron in self.neurons:
      activations.append(neuron.activation)
    return activations
  def set_expected_activations(self, activations):
    for i in range(len(activations)): 
      self.neurons[i].set_expected_activation(activations[i])
  def back_propagate(self, learning_rate):
    for neuron in self.neurons:
      neuron.back_propagate(learning_rate)
  def get_last_error(self):
    last_error = 0.0
    for neuron in self.neurons:
      last_error += 0.5 * (neuron.last_error ** 2)
    return last_error
  def fire(self):
    for neuron in self.neurons:
      neuron.fire()
  def __getitem__(self, i):
    return self.neurons[i]

class BiasNeuron(Neuron):
  @property
  def activation(self):
    return 1.0
  @activation.setter
  def activation(self, x):
    pass

class BiasLayer(Layer):
  def __init__(self):
   self.neurons = [BiasNeuron()] 

class Network(object):
  def __init__(self, topography):
    self.layers = []
    self.bLayer = BiasLayer()
    for i in range(len(topography)):
      self.layers.append(Layer(topography[i])) 
      if i != 0:
        self.layers[i].connect(self.layers[i-1])
        self.layers[i].connect(self.bLayer)
    self.iLayer = self.layers[0]
    self.oLayer = self.layers[-1]
  def train(self, training_data, epochs, learning_rate, report_rate=False):
    for epoch in range(epochs):
      for datum in training_data:
        self.feed_forward(datum[0])
        self.back_propagate(datum[1], learning_rate)
      if report_rate and epoch % report_rate == 0:
        print "E: %-.5f" % self.oLayer.get_last_error()
  def feed_forward(self, vector):
    self.iLayer.set_activations(vector)  
    for i in range(1, len(self.layers)):
      self.layers[i].fire()
  def back_propagate(self, vector, learning_rate):
    self.oLayer.set_expected_activations(vector)
    for i in range(len(self.layers)-1, -1, -1):
      self.layers[i].back_propagate(learning_rate)
  def query(self, vector):
    self.feed_forward(vector)
    return self.oLayer.get_activations()
  def get_last_error(self):
    return self.oLayer.get_last_error()
