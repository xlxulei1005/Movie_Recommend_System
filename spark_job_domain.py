from __future__ import print_function
from operator import add
from csv import reader
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


movie_type=['war','western','thriller','scifi','romance','mystery','musical','imax','horror','fantasy','film','drama','crime','document','comedy','children','adventure','animation','action','no_genre']
filename=['train_do_{}.csv'.format(i) for i in movie_type]


data={}
model={}
ratings={}
test_pair={}
userfeature={}
productfeature={}


for i in range(len(filename)):
    data[movie_type[i]]=sc.textFile(filename[i])

header=data['war'].filter(lambda x: 'userId' in x) #But it is for all the movie ratings

for i in data.keys():
    ratings[i]=data[i].subtract(header).mapPartitions(lambda x: reader(x)).map(lambda x:Rating(int(float(x[1])), int(float(x[2])), float(x[3])))
    model[i]= ALS.train(ratings[i], 15, 15)

for i in data.keys():
    userfeature[i]=model[i].userFeatures()
    productfeature[i]=model[i].productFeatures()

def toCSVLine(data):
    return ','.join(str(d) for d in data)

for i in userfeature.keys():
    userfeature[i].map(lambda x: toCSVLine(x)).saveAsTextFile('do_{}_userfeature.csv'.format(i))
    productfeature[i].map(lambda x: toCSVLine(x)).saveAsTextFile('do_{}_productfeature.csv'.format(i))


for i in model:    
    model[i].save(sc, 'model_{}'.format(i))
    
#Save three things, two CSV and one models




