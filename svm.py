import math, time 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
#from google.colab import files
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

#uploaded = files.upload()
#MNIST_train_df = pd.read_csv(io.BytesIO(uploaded['mnist_train.csv']))
MNIST_train_df = pd.read_csv('mnist_train.csv')
MNIST_train_small = MNIST_train_df.iloc[0:12000]
MNIST_train_small.to_csv('mnist_train_small.csv')
MNIST_train_df = pd.read_csv('mnist_train_small.csv', sep=',', index_col=0)
print(MNIST_train_df.shape)

#sns.countplot(MNIST_train_df['label'])
plt.show()# looks kinda okay
# or we can just print
print(MNIST_train_df['label'].value_counts())

#Separate label and pixel columns and, label is the 1st column of the data-frame.
X_tr = MNIST_train_df.iloc[:,1:] # iloc ensures X_tr will be a dataframe
y_tr = MNIST_train_df.iloc[:, 0]

kernel = ['poly']
# To decide on the value of C, gamma we will use the GridSearchCV method with 5 folds cross-validation
C_list = [10e5, 100, 0.1, 0.01, 0.001, 0.0001]
gamma_list = [ 0.01,0.1, 1,10,100,1000]

grid = dict(kernel=kernel,C=C_list,gamma= gamma_list )
cross_val = KFold(n_splits=10)
grid_object = GridSearchCV(estimator=SVC(), param_grid=grid, cv=cross_val, scoring='accuracy')
print("Training the model..")
grid_result = grid_object.fit(X_tr, y_tr)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
mean = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(mean, params):
  print("%f with: %r" % (mean, param))

mean = list(grid_result.cv_results_['mean_test_score']*100)

plt.plot(gamma_list,mean[0:6])
plt.plot(gamma_list,mean[6:12])
plt.plot(gamma_list,mean[12:18])
plt.plot(gamma_list,mean[18:24])
plt.plot(gamma_list,mean[24:30])
plt.plot(gamma_list,mean[30:36])
plt.xscale('log')
plt.xlabel("Values of Hyperparameter Gamma")
plt.ylabel("Validation accuracies in %")
plt.legend(['C = 10e5','C = 100','C = 0.1','C = 0.01','C = 0.001','C = 0.0001'])
plt.title("Validation Accuracies for all combinations of C and gamma")
plt.show()

plt.plot(gamma_list,mean[0:6])
plt.xscale('log')
plt.xlabel("Values of Hyperparameter Gamma")
plt.ylabel("Validation accuracies in %")
plt.title("Validation Accuracies for C = %3.2f " % (grid_result.best_params_['C']))
plt.show()

y_pred = grid_result.predict(X_tr)
# use the prediction list and the pixel values from the test list for comparison.
'''
for i in (np.random.randint(0,270,6)):
  two_d = (np.reshape(X_tr.values[i], (28, 28)) * 255).astype(np.uint8)
  plt.title('predicted label: {0}'. format(y_pred[i]))
  plt.imshow(two_d, interpolation='nearest', cmap='gray')
  plt.show()'''

print("confusion matrix: \n ", confusion_matrix(y_tr, y_pred))

# Now we will repeat the process for the test-data set (mnist_test.csv) but instead of going through 
# finding the best parameters for SVM (C, gamma) using GridSearchCV , I have used the same parameters from the training data set.
#uploaded_ = files.upload()
#MNIST_df = pd.read_csv(io.BytesIO(uploaded_['mnist_test.csv']))
MNIST_df = pd.read_csv('mnist_test.csv')
MNIST_test = MNIST_df.iloc[0:5000]
MNIST_test.to_csv('mnist_test.csv')
MNIST_test_df = pd.read_csv('mnist_test.csv', sep=',', index_col=0)

# Choosing features and labels 
X_test = MNIST_test_df.iloc[:,1:]
Y_test = MNIST_test_df.iloc[:,0]

final_model = SVC(kernel = 'poly', C = grid_result.best_params_['C'], gamma= grid_result.best_params_['gamma'] )
print("Training the final model")
final_model.fit(X_tr, y_tr)
y_pred = final_model.predict(X_test)

print("score on the test data set= %3.2f" %(grid_result.score(X_test, Y_test)))
print("confusion matrix: \n ", confusion_matrix(Y_test, y_pred))
