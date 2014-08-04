Training Data
===================

.. highlight:: cpp

In machine learning algorithms there is notion of training data. Training data includes several components:

* A set of training samples. Each training sample is a vector of values (in Computer Vision it's sometimes referred to as feature vector). Usually all the vectors have the same number of components (features); OpenCV ml module assumes that. Each feature can be ordered (i.e. its values are floating-point numbers that can be compared with each other and strictly ordered, i.e. sorted) or categorical (i.e. its value belongs to a fixed set of values that can be integers, strings etc.).

* Optional set of responses corresponding to the samples. Training data with no responses is used in unsupervised learning algorithms that learn structure of the supplied data based on distances between different samples. Training data with responses is used in supervised learning algorithms, which learn the function mapping samples to responses. Usually the responses are scalar values, ordered (when we deal with regression problem) or categorical (when we deal with classification problem; in this case the responses are often called "labels"). Some algorithms, most noticeably Neural networks, can handle not only scalar, but also multi-dimensional or vector responses.

* Another optional component is the mask of missing measurements. Most algorithms require all the components in all the training samples be valid, but some other algorithms, such as decision tress, can handle the cases of missing measurements.

* In the case of classification problem user may want to give different weights to different classes. This is useful, for example, when
  * user wants to shift prediction accuracy towards lower false-alarm rate or higher hit-rate.
  * user wants to compensate for significantly different amounts of training samples from different classes.

* In addition to that, each training sample may be given a weight, if user wants the algorithm to pay special attention to certain training samples and adjust the training model accordingly.

* Also, user may wish not to use the whole training data at once, but rather use parts of it, e.g. to do parameter optimization via cross-validation procedure.

As you can see, training data can have rather complex structure; besides, it may be very big and/or not entirely available, so there is need to make abstraction for this concept. In OpenCV ml there is ``cv::ml::TrainData`` class for that.

TrainData
---------
.. ocv:class:: TrainData

Class encapsulating training data. Please note that the class only specifies the interface of training data, but not implementation. All the statistical model classes in ml take Ptr<TrainData>. In other words, you can create your own class derived from ``TrainData`` and supply smart pointer to the instance of this class into ``StatModel::train``.

TrainData::loadFromCSV
----------------------
Reads the dataset from a .csv file and returns the ready-to-use training data.

.. ocv:function:: Ptr<TrainData> loadFromCSV(const String& filename, int headerLineCount, int responseStartIdx=-1, int responseEndIdx=-1, const String& varTypeSpec=String(), char delimiter=',', char missch='?')

    :param filename: The input file name

    :param headerLineCount: The number of lines in the beginning to skip; besides the header, the function also skips empty lines and lines staring with '#'

    :param responseStartIdx: Index of the first output variable. If -1, the function considers the last variable as the response

    :param responseEndIdx: Index of the last output variable + 1. If -1, then there is single response variable at ``responseStartIdx``.

    :param varTypeSpec: The optional text string that specifies the variables' types. It has the format ``ord[n1-n2,n3,n4-n5,...]cat[n6,n7-n8,...]``. That is, variables from n1 to n2 (inclusive range), n3, n4 to n5 ... are considered ordered and n6, n7 to n8 ... are considered as categorical. The range [n1..n2] + [n3] + [n4..n5] + ... + [n6] + [n7..n8] should cover all the variables. If varTypeSpec is not specified, then algorithm uses the following rules:
        1. all input variables are considered ordered by default. If some column contains has non-numerical values, e.g. 'apple', 'pear', 'apple', 'apple', 'mango', the corresponding variable is considered categorical.
        2. if there are several output variables, they are all considered as ordered. Error is reported when non-numerical values are used.
        3. if there is a single output variable, then if its values are non-numerical or are all integers, then it's considered categorical. Otherwise, it's considered ordered.

    :param delimiter: The character used to separate values in each line.

    :param missch: The character used to specify missing measurements. It should not be a digit. Although it's a non-numerical value, it surely does not affect the decision of whether the variable ordered or categorical.

TrainData::create
-----------------
Creates training data from in-memory arrays.

.. ocv:function:: Ptr<TrainData> create(InputArray samples, int layout, InputArray responses, InputArray varIdx=noArray(), InputArray sampleIdx=noArray(), InputArray sampleWeights=noArray(), InputArray varType=noArray())

    :param samples: matrix of samples. It should have ``CV_32F`` type.

    :param layout: it's either ``ROW_SAMPLE``, which means that each training sample is a row of ``samples``, or ``COL_SAMPLE``, which means that each training sample occupies a column of ``samples``.

    :param responses: matrix of responses. If the responses are scalar, they should be stored as a single row or as a single column. The matrix should have type ``CV_32F`` or ``CV_32S`` (in the former case the responses are considered as ordered by default; in the latter case - as categorical)

    :param varIdx: vector specifying which variables to use for training. It can be an integer vector (``CV_32S``) containing 0-based variable indices or byte vector (``CV_8U``) containing a mask of active variables.

    :param sampleIdx: vector specifying which samples to use for training. It can be an integer vector (``CV_32S``) containing 0-based sample indices or byte vector (``CV_8U``) containing a mask of training samples.

    :param sampleWeights: optional vector with weights for each sample. It should have ``CV_32F`` type.

    :param varType: optional vector of type ``CV_8U`` and size <number_of_variables_in_samples> + <number_of_variables_in_responses>, containing types of each input and output variable. The ordered variables are denoted by value ``VAR_ORDERED``, and categorical - by ``VAR_CATEGORICAL``.


TrainData::getTrainSamples
--------------------------
Returns matrix of train samples

.. ocv:function:: Mat TrainData::getTrainSamples(int layout=ROW_SAMPLE, bool compressSamples=true, bool compressVars=true) const

    :param layout: The requested layout. If it's different from the initial one, the matrix is transposed.

    :param compressSamples: if true, the function returns only the training samples (specified by sampleIdx)

    :param compressVars: if true, the function returns the shorter training samples, containing only the active variables.

In current implementation the function tries to avoid physical data copying and returns the matrix stored inside TrainData (unless the transposition or compression is needed).


TrainData::getTrainResponses
----------------------------
Returns the vector of responses

.. ocv:function:: Mat TrainData::getTrainResponses() const

The function returns ordered or the original categorical responses. Usually it's used in regression algorithms.


TrainData::getClassLabels
----------------------------
Returns the vector of class labels

.. ocv:function:: Mat TrainData::getClassLabels() const

The function returns vector of unique labels occurred in the responses.


TrainData::getTrainNormCatResponses
-----------------------------------
Returns the vector of normalized categorical responses

.. ocv:function:: Mat TrainData::getTrainNormCatResponses() const

The function returns vector of responses. Each response is integer from 0 to <number of classes>-1. The actual label value can be retrieved then from the class label vector, see ``TrainData::getClassLabels``.

TrainData::setTrainTestSplitRatio
-----------------------------------
Splits the training data into the training and test parts

.. ocv:function:: void TrainData::setTrainTestSplitRatio(double ratio, bool shuffle=true)

The function selects a subset of specified relative size and then returns it as the training set. If the function is not called, all the data is used for training. Please, note that for each of ``TrainData::getTrain*`` there is corresponding ``TrainData::getTest*``, so that the test subset can be retrieved and processed as well.


Other methods
-------------
The class includes many other methods that can be used to access normalized categorical input variables, access training data by parts, so that does not have to fit into the memory etc.
