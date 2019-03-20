from ann import *

def query(n, x):
  print "[" + str(x) + "]: " + str(n.query([x]))
  

if __name__ == '__main__':
  print "Creating network"

  n = Network([2,3,1])

  training_data = [
    [[0,0],[0]],
    [[0,1],[1]],
    [[1,0],[1]],
    [[1,1],[0]],
  ]

  n.train(training_data, 50000, 0.2, 1000)

  for datum in training_data:
    print "[%i,%i]: %f" % (datum[0][0], datum[0][1], n.query(datum[0])[0])

  print "Done."


  
