import cProfile
from ann import *

def run():
  learning_rate = 0.1

  trainingData = [
    [[0,0], [0]],
    [[0,1], [1]],
    [[1,0], [1]],
    [[1,1], [0]],
  ]

  n = Network([2,5,1])

  n.train(trainingData, 10000, learning_rate)

if __name__ == "__main__":
  cProfile.run("run()", sort="cumtime")
