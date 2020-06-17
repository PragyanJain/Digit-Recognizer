
#K-Nearest Neighbor Classification

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
import pandas as pd
from google.colab import files
import io

# load the MNIST digits dataset
#mnist = datasets.load_digits()

uploaded = files.upload()
MNIST_df = pd.read_csv(io.BytesIO(uploaded['mnist_train.csv']))
MNIST_train_small = MNIST_df.iloc[0:17000]
MNIST_train_small.to_csv('mnist_train_small.csv')
MNIST_train_df = pd.read_csv('mnist_train_small.csv', sep=',', index_col=0)

#Separate label and pixel columns and, label is the 1st column of the data-frame.
X_tr = MNIST_train_df.iloc[:,1:] # iloc ensures X_tr will be a dataframe
y_tr = MNIST_train_df.iloc[:, 0]

from sklearn.model_selection import GridSearchCV

# Training and testing split,
# 71% for training and 29% for testing
#(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)
(trainData, testData, trainLabels, testLabels) = train_test_split(X_tr,y_tr,test_size=0.295, random_state=42)

# take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

# Checking sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []
accuracies_train = []

# loop over kVals
for k in range(1, 30, 2):
    # train the classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # evaluate the model and print the accuracies list
    score = model.score(valData, valLabels)
    print("k=%d, accuracy_val=%.2f%%" % (k, score * 100))
    accuracies.append(score)

    # evaluate the model and print the accuracies list on training data
    score_train = model.score(trainData, trainLabels)
    print("k=%d, accuracy_train=%.2f%%" % (k, score_train * 100))
    accuracies_train.append(score_train)


# largest accuracy
# np.argmax returns the indices of the maximum values along an axis
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy_val of %.2f%% on validation data" % (kVals[i],
    accuracies[i] * 100))


# Now that I know the best value of k, re-train the classifier
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)

# Predict labels for the test set
predictions = model.predict(testData)

# Evaluate performance of model for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

plt.plot(range(1, 30, 2),accuracies)
plt.plot(range(1, 30, 2),accuracies_train)
plt.xlabel("Values of k")
plt.ylabel("Accuracies in %")
plt.legend(['Validation','Train'])
plt.title("Accuracies for all values of k")
plt.show()

# some indices are classified correctly 100% of the time (precision = 1)
# high accuracy (98%)

# check predictions against images
# loop over a few random digits
image = testData
print(image)
j = 0
for i in np.random.randint(0, high=len(testLabels), size=(24,)):
        # np.random.randint(low, high=None, size=None, dtype='l')
    i = 3979
    prediction = model.predict(image)[i]
    image0 = image[i].reshape((28, 28)).astype("uint8")
    image0 = exposure.rescale_intensity(image0, out_range=(0, 255))
    plt.subplot(4,6,j+1)
    plt.title(str(prediction))
    plt.imshow(image0,cmap='gray')
    plt.axis('off')
        # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
        # then resize it to 32 x 32 pixels for better visualization

        #image0 = imutils.resize(image[0], width=32, inter=cv2.INTER_CUBIC)

    j = j+1

    # show the prediction
    # print("I think that digit is: {}".format(prediction))
    # print('image0 is ',image0)
    # cv2.imshow("Image", image0)
    # cv2.waitKey(0) # press enter to view each one!
plt.show()
