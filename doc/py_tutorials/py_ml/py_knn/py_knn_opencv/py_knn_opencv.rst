.. _KNN_OpenCV:

OCR of Hand-written Data using kNN
***********************************************

Goal
=======

In this chapter
    * We will use our knowledge on kNN to build a basic OCR application.
    * We will try with Digits and Alphabets data available that comes with OpenCV.


OCR of Hand-written Digits
============================

Our goal is to build an application which can read the handwritten digits. For this we need some train_data and test_data. OpenCV comes with an image `digits.png` (in the folder ``opencv/samples/python2/data/``) which has 5000 handwritten digits (500 for each digit). Each digit is a 20x20 image. So our first step is to split this image into 5000 different digits. For each digit, we flatten it into a single row with 400 pixels. That is our feature set, ie intensity values of all pixels. It is the simplest feature set we can create. We use first 250 samples of each digit as train_data, and next 250 samples as test_data. So let's prepare them first.
::

    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    img = cv2.imread('digits.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Now we prepare train_data and test_data.
    train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
    test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k,250)[:,np.newaxis]
    test_labels = train_labels.copy()

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train,train_labels)
    ret,result,neighbours,dist = knn.find_nearest(test,k=5)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print accuracy


So our basic OCR app is ready. This particular example gave me an accuracy of 91%. One option improve accuracy is to add more data for training, especially the wrong ones. So instead of finding this training data everytime I start application, I better save it, so that next time, I directly read this data from a file and start classification. You can do it with the help of some Numpy functions like np.savetxt, np.savez, np.load etc. Please check their docs for more details.
::

    # save the data
    np.savez('knn_data.npz',train=train, train_labels=train_labels)

    # Now load the data
    with np.load('knn_data.npz') as data:
        print data.files
        train = data['train']
        train_labels = data['train_labels']

In my system, it takes around 4.4 MB of memory. Since we are using intensity values (uint8 data) as features, it would be better to convert the data to np.uint8 first and then save it. It takes only 1.1 MB in this case. Then while loading, you can convert back into float32.

OCR of English Alphabets
===========================

Next we will do the same for English alphabets, but there is a slight change in data and feature set. Here, instead of images, OpenCV comes with a data file, ``letter-recognition.data`` in ``opencv/samples/cpp/`` folder. If you open it, you will see 20000 lines which may, on first sight, look like garbage. Actually, in each row, first column is an alphabet which is our label. Next 16 numbers following it are its different features. These features are obtained from `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml/>`_. You can find the details of these features in `this page <http://archive.ics.uci.edu/ml/datasets/Letter+Recognition>`_.

There are 20000 samples available, so we take first 10000 data as training samples and remaining 10000 as test samples. We should change the alphabets to ascii characters because we can't work with alphabets directly.
::

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the data, converters convert the letter to a number
    data= np.loadtxt('letter-recognition.data', dtype= 'float32', delimiter = ',',
                        converters= {0: lambda ch: ord(ch)-ord('A')})

    # split the data to two, 10000 each for train and test
    train, test = np.vsplit(data,2)

    # split trainData and testData to features and responses
    responses, trainData = np.hsplit(train,[1])
    labels, testData = np.hsplit(test,[1])

    # Initiate the kNN, classify, measure accuracy.
    knn = cv2.KNearest()
    knn.train(trainData, responses)
    ret, result, neighbours, dist = knn.find_nearest(testData, k=5)

    correct = np.count_nonzero(result == labels)
    accuracy = correct*100.0/10000
    print accuracy

It gives me an accuracy of 93.22%. Again, if you want to increase accuracy, you can iteratively add error data in each level.

Additional Resources
=======================

Exercises
=============
