import unittest
from ann import *

class NeuronTest(unittest.TestCase):
  def test_connections(self):
    a = Neuron()
    b = Neuron()
    c = Neuron()

    x = Neuron()
    x.connect(a)
    x.connect(b)
    x.connect(c)

    self.assertTrue(len(x.connections) == 3)

    try:
      x.connect(c)
    except RuntimeError:
      self.assertTrue(len(x.connections) == 3)

    d = Neuron()
    x.connect(d)
    self.assertTrue(len(x.connections) == 4)

    
  def test_fire_and_bprop(self):
    a1 = Neuron()
    a2 = Neuron()

    b1 = Neuron()

    b1.connect(a1)
    b1.connect(a2)

    for c in b1.connections:
      c.weight = 2.0

    a1.activation = 0.3
    a2.activation = 0.2

    b1.fire()

    self.assertTrue(b1.activation == sigmoid(1.0))

    b1.set_expected_activation(0.5) 

    self.assertTrue(b1.error == -(0.5 - b1.activation))

    b1.back_propagate(0.2)

  def test_bias_neuron(self):
    b = BiasNeuron()

    self.assertTrue(b.activation == 1.0)
    b.activation = 2.0
    self.assertTrue(b.activation == 1.0)

  def test_bias_layer(self):
    bl = BiasLayer()

    self.assertTrue(bl.neurons[0].activation == 1.0)
    bl.neurons[0].activation = 2.0
    self.assertTrue(bl.neurons[0].activation == 1.0)


class LayerTest(unittest.TestCase):
  def test_connect_layers(self):
    a = Layer(1)
    b = Layer(2)

    b.connect(a)

    self.assertTrue(len(b[0].connections) == 1)
    self.assertTrue(len(b[1].connections) == 1)
    self.assertTrue(b[0].connections[0].neuron == a[0])
    self.assertTrue(b[1].connections[0].neuron == a[0])

    a = Layer(2)
    b = Layer(1)

    b.connect(a)

    self.assertTrue(len(b[0].connections) == 2)
    self.assertTrue(b[0].connections[0].neuron == a[0])
    self.assertTrue(b[0].connections[1].neuron == a[1])


  def test_fire_and_bprop_layers(self):
    iLayer = Layer(2)
    hLayer = Layer(3)
    oLayer = Layer(1)

    oLayer.connect(hLayer)
    hLayer.connect(iLayer)

    iLayer.set_activations([0.5,0.5])
    self.assertTrue(iLayer.neurons[0].activation == 0.5)  
    self.assertTrue(iLayer.neurons[0].activation == 0.5)  

    hLayer.fire()
    oLayer.fire()

    oLayer.set_expected_activations([0])

    oLayer.back_propagate(0.2)
    hLayer.back_propagate(0.2)

  def test_bias_layer_connection(self):
    network = Network([2,3,4,1])
    biasNeuronConnectionPresent = False
    
    for n in network.layers[0].neurons:
      for c in n.connections:
        if type(c.neuron).__name__ == "BiasNeuron":
          biasNeuronConnectionPresent = True
      self.assertFalse(biasNeuronConnectionPresent)
      biasNeuronConnectionPresent = False

    for layer in network.layers[1:]:
      for n in layer.neurons:
        for c in n.connections:
          if type(c.neuron).__name__ == "BiasNeuron":
            biasNeuronConnectionPresent = True
        self.assertTrue(biasNeuronConnectionPresent)
        biasNeuronConnectionPresent = False

if __name__ == "__main__":
 unittest.main() 
