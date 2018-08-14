#linear support Vector machine using Spark 
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('spark-lsvm')
sc = SparkContext(conf=conf)

#load and parse the data 
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("data/sample_svm_data.txt")
parsed_data = data.map(parsePoint)

# build the model 
model = SVMWithSGD.train(parsed_data, iterations=100)

#evalluating the model on the training data
labelsAndPreds = parsed_data.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp : lp[0] != lp[1]).count()  / float(parsed_data.count())
print("training error = " + str(trainErr))

#save the model
model.save(sc, 'data/output/tmp/pythonSVMWithSGDModel')
print("================================Model is saved================================")