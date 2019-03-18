import unittest
from ann import *

class LearnTest(unittest.TestCase):
  def test_direct_xor_training(self):
    learning_rate = 0.1
    report_rate = 1000

    trainingData = [
      [[0,0], [0]],
      [[0,1], [1]],
      [[1,0], [1]],
      [[1,1], [0]],
    ]

    iLayer = Layer(2)
    hLayer = Layer(5)
    oLayer = Layer(1)

    oLayer.connect(hLayer)
    hLayer.connect(iLayer)

    for epoch in range(5000):
      for datum in trainingData:
        iLayer.set_activations(datum[0])

        hLayer.fire()
        oLayer.fire()

        oLayer.set_expected_activations(datum[1])
        
        oLayer.back_propagate(learning_rate)
        hLayer.back_propagate(learning_rate)

    error1 = oLayer.get_last_error()

    for epoch in range(5000):
      for datum in trainingData:
        iLayer.set_activations(datum[0])

        hLayer.fire()
        oLayer.fire()

        oLayer.set_expected_activations(datum[1])
        
        oLayer.back_propagate(learning_rate)
        hLayer.back_propagate(learning_rate)

    error2 = oLayer.get_last_error()

    self.assertTrue(error2 < error1)


  def test_network_xor_training(self):
    learning_rate = 0.2

    trainingData = [
      [[0,0], [0]],
      [[0,1], [1]],
      [[1,0], [1]],
      [[1,1], [0]],
    ]

    n = Network([2,5,1])

    n.train(trainingData, 5000, learning_rate)
    error1 = n.get_last_error()
    n.train(trainingData, 5000, learning_rate)
    error2 = n.get_last_error()

    self.assertTrue(error1 > error2)

    self.assertLess(n.query([0,0])[0], 0.4)
    self.assertGreater(n.query([0,1])[0], 0.5)
    self.assertGreater(n.query([1,0])[0], 0.5)
    self.assertLess(n.query([1,1])[0], 0.4)



  def test_multi_layer_network_xor_training(self):
    learning_rate = 0.1
    report_rate = 1000

    trainingData = [
      [[0,0], [0]],
      [[0,1], [1]],
      [[1,0], [1]],
      [[1,1], [0]],
    ]

    n = Network([2,5,5,1])

    n.train(trainingData, 10000, learning_rate)
    error1 = n.get_last_error()
    n.train(trainingData, 10000, learning_rate)
    error2 = n.get_last_error()

    self.assertTrue(error2 < error1)

    self.assertLess(n.query([0,0])[0], 0.4)
    self.assertGreater(n.query([0,1])[0], 0.5)
    self.assertGreater(n.query([1,0])[0], 0.5)
    self.assertLess(n.query([1,1])[0], 0.4)

if __name__ == "__main__":
 unittest.main()
