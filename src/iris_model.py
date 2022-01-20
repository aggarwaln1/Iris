import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd

iris = pd.read_csv(r'E:\Nidhi\MLOps training\MLOps_Code\Practice_ML\Iris\data\Iris.csv')
iris = iris.drop(['Id'],axis=1)

print(iris.head())

# our main data split into train and test
# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%
train, test = train_test_split(iris, test_size=0.3) 
print(train.shape)
print(test.shape)

train_X =train.iloc[:,:3]
train_y =train.iloc[:,-1]

test_X =test.iloc[:,:3]
test_y =test.iloc[:,-1]

rfc =RandomForestClassifier(n_estimators=500)
rfc.fit(train_X, train_y)

filename =r'E:\Nidhi\MLOps training\MLOps_Code\Practice_ML\Iris\saved_models\model_rfc.pickle'
pickle.dump(rfc,open(filename,'wb'))

load_model =pickle.load(open(filename,'rb'))
result = load_model.score(test_X,test_y)
print(result)
