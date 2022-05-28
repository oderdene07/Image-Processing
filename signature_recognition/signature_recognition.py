import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import keras
from tensorflow.python.framework import ops

real_image_paths = "./real/"
fake_image_paths = "./fake/"

"""
## Preprocessing the image
"""
 
def rgbgrey(img):
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg
 
def greybin(img):
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)  # to remove small components or noise
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)
    return binimg

def preproc(path, img=None, display=True):
    if img is None:
        img = mpimg.imread(path)
    if display:
        plt.imshow(img)
    grey = rgbgrey(img)
    if display:
        plt.imshow(grey, cmap = matplotlib.cm.Greys_r)
    binimg = greybin(grey)
    if display:
        plt.imshow(binimg, cmap = matplotlib.cm.Greys_r)
    r, c = np.where(binimg==1)
    
    signimg = binimg[r.min(): r.max(), c.min(): c.max()]
    if display:
        plt.imshow(signimg, cmap = matplotlib.cm.Greys_r)
    return signimg

"""
## Feature Extraction

"""

def Ratio(img):
    a = 0
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col]==True:
                a = a+1
    total = img.shape[0] * img.shape[1]
    return a/total

def Centroid(img):
    numOfWhites = 0
    a = np.array([0,0])
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col]==True:
                b = np.array([row,col])
                a = np.add(a,b)
                numOfWhites += 1
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a/numOfWhites
    centroid = centroid/rowcols
    return centroid[0], centroid[1]

def EccentricitySolidity(img):
    r = regionprops(img.astype("int8"))
    return r[0].eccentricity, r[0].solidity

def SkewKurtosis(img):
    h,w = img.shape
    x = range(w)  # cols value
    y = range(h)  # rows value

    #calculate projections along the x and y axes
    xp = np.sum(img,axis=0)
    yp = np.sum(img,axis=1)
    
    #centroid
    cx = np.sum(x*xp)/np.sum(xp)
    cy = np.sum(y*yp)/np.sum(yp)
    
    #standard deviation
    x2 = (x-cx)**2
    y2 = (y-cy)**2
    sx = np.sqrt(np.sum(x2*xp)/np.sum(img))
    sy = np.sqrt(np.sum(y2*yp)/np.sum(img))
    
    #skewness
    x3 = (x-cx)**3
    y3 = (y-cy)**3
    skewx = np.sum(xp*x3)/(np.sum(img) * sx**3)
    skewy = np.sum(yp*y3)/(np.sum(img) * sy**3)

    #Kurtosis
    x4 = (x-cx)**4
    y4 = (y-cy)**4
    
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp*x4)/(np.sum(img) * sx**4) - 3
    kurty = np.sum(yp*y4)/(np.sum(img) * sy**4) - 3

    return (skewx , skewy), (kurtx, kurty)

def getFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    img = preproc(path, display=display)
    ratio = Ratio(img)
    centroid = Centroid(img)
    eccentricity, solidity = EccentricitySolidity(img)
    skewness, kurtosis = SkewKurtosis(img)
    retVal = (ratio, centroid, eccentricity, solidity, skewness, kurtosis)
    return retVal

def getCSVFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
    features = (temp[0], temp[1][0], temp[1][1], temp[2], temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1])
    return features

"""
## Saving the features
"""

def makeCSV():
    if not(os.path.exists('./Features')):
        os.mkdir('./Features')
    if not(os.path.exists('./Features/Training')):
        os.mkdir('./Features/Training')
    if not(os.path.exists('./Features/Testing')):
        os.mkdir('./Features/Testing')

    gpath = real_image_paths
    fpath = fake_image_paths

    print('Saving features')
    
    with open('./Features/Training/training_001.csv', 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
        # Training set
        for i in range(0,3):
            source = os.path.join(gpath, '001001_00'+str(i)+'.png')
            features = getCSVFeatures(path=source)
            handle.write(','.join(map(str, features))+',1\n')
        for i in range(0,3):
            source = os.path.join(fpath, '021001_00'+str(i)+'.png')
            features = getCSVFeatures(path=source)
            handle.write(','.join(map(str, features))+',0\n')
    
    with open('./Features/Testing/testing_001.csv', 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
        # Testing set
        for i in range(3, 5):
            source = os.path.join(gpath, '001001_00'+str(i)+'.png')
            features = getCSVFeatures(path=source)
            handle.write(','.join(map(str, features))+',1\n')
        for i in range(3,5):
            source = os.path.join(fpath, '021001_00'+str(i)+'.png')
            features = getCSVFeatures(path=source)
            handle.write(','.join(map(str, features))+',0\n')

"""
# TF Model 
"""
 
def testing(path):
    feature = getCSVFeatures(path)
    if not(os.path.exists('./TestFeatures')):
        os.mkdir('./TestFeatures')
    with open('./TestFeatures/testcsv.csv', 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature))+'\n')

def readCSV(train_path, test_path):
    # Reading train data
    df = pd.read_csv(train_path, usecols=range(n_input))
    train_input = np.array(df.values)
    train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
    df = pd.read_csv(train_path, usecols=(n_input,))
    temp = [elem[0] for elem in df.values]
    correct = np.array(temp)
    corr_train = keras.utils.np_utils.to_categorical(correct,2)
    
    # Reading test data
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)

    return train_input, corr_train, test_input

# Create model
def multilayer_perceptron(x):
    layer_1 = tf.tanh((tf.matmul(x, weights['h1']) + biases['b1']))
    out_layer = tf.tanh(tf.matmul(layer_1, weights['out']) + biases['out'])
    return out_layer

def evaluate(train_path, test_path):   
    train_input, corr_train, test_input = readCSV(train_path, test_path)

    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})
            if cost<0.0001:
                break

        prediction = pred.eval({X: test_input})
        if prediction[0][1]>prediction[0][0]:
            print(prediction[0][1])
            print('Real Signature\nTrue')
        else:
            print(prediction[0][1])
            print('Fake Signature\nFalse')

makeCSV()

n_input = 9
test_image_path = './test_image.png'
train_path = './Features/Training/training_001.csv'
test_path = './TestFeatures/testcsv.csv'

testing(test_image_path)

ops.reset_default_graph()

# Parameters
learning_rate = 0.001
training_epochs = 1000

# Network Parameters
n_hidden_1 = 7 # 1st layer number of neurons
n_classes = 2 # no. of classes ( real or fake)

# tf Graph input
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], seed=2))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=3)),
    'out': tf.Variable(tf.random_normal([n_classes], seed=4))
}

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# For prediction
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))

# Initializing the variables
init = tf.global_variables_initializer()

evaluate(train_path, test_path)
