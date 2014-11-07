Statistical Models
==================

.. highlight:: cpp

.. index:: StatModel

StatModel
-----------
.. ocv:class:: StatModel

Base class for statistical models in OpenCV ML.


StatModel::train
------------------------
Trains the statistical model

.. ocv:function:: bool StatModel::train( const Ptr<TrainData>& trainData, int flags=0 )

.. ocv:function:: bool StatModel::train( InputArray samples, int layout, InputArray responses )

.. ocv:function:: Ptr<_Tp> StatModel::train(const Ptr<TrainData>& data, const _Tp::Params& p, int flags=0 )

.. ocv:function:: Ptr<_Tp> StatModel::train(InputArray samples, int layout, InputArray responses, const _Tp::Params& p, int flags=0 )

    :param trainData: training data that can be loaded from file using ``TrainData::loadFromCSV`` or created with ``TrainData::create``.

    :param samples: training samples

    :param layout: ``ROW_SAMPLE`` (training samples are the matrix rows) or ``COL_SAMPLE`` (training samples are the matrix columns)

    :param responses: vector of responses associated with the training samples.

    :param p: the stat model parameters.

    :param flags: optional flags, depending on the model. Some of the models can be updated with the new training samples, not completely overwritten (such as ``NormalBayesClassifier`` or ``ANN_MLP``).

There are 2 instance methods and 2 static (class) template methods. The first two train the already created model (the very first method must be overwritten in the derived classes). And the latter two variants are convenience methods that construct empty model and then call its train method.


StatModel::isTrained
-----------------------------
Returns true if the model is trained

.. ocv:function:: bool StatModel::isTrained()

The method must be overwritten in the derived classes.

StatModel::isClassifier
-----------------------------
Returns true if the model is classifier

.. ocv:function:: bool StatModel::isClassifier()

The method must be overwritten in the derived classes.

StatModel::getVarCount
-----------------------------
Returns the number of variables in training samples

.. ocv:function:: int StatModel::getVarCount()

The method must be overwritten in the derived classes.

StatModel::predict
------------------
Predicts response(s) for the provided sample(s)

.. ocv:function:: float StatModel::predict( InputArray samples, OutputArray results=noArray(), int flags=0 ) const

    :param samples: The input samples, floating-point matrix

    :param results: The optional output matrix of results.

    :param flags: The optional flags, model-dependent. Some models, such as ``Boost``, ``SVM`` recognize ``StatModel::RAW_OUTPUT`` flag, which makes the method return the raw results (the sum), not the class label.


StatModel::calcError
-------------------------
Computes error on the training or test dataset

.. ocv:function:: float StatModel::calcError( const Ptr<TrainData>& data, bool test, OutputArray resp ) const

    :param data: the training data

    :param test: if true, the error is computed over the test subset of the data, otherwise it's computed over the training subset of the data. Please note that if you loaded a completely different dataset to evaluate already trained classifier, you will probably want not to set the test subset at all with ``TrainData::setTrainTestSplitRatio`` and specify ``test=false``, so that the error is computed for the whole new set. Yes, this sounds a bit confusing.

    :param resp: the optional output responses.

The method uses ``StatModel::predict`` to compute the error. For regression models the error is computed as RMS, for classifiers - as a percent of missclassified samples (0%-100%).


StatModel::save
-----------------
Saves the model to a file.

.. ocv:function:: void StatModel::save( const String& filename )

In order to make this method work, the derived class must overwrite ``Algorithm::write(FileStorage& fs)``.

StatModel::load
-----------------
Loads model from the file

.. ocv:function:: Ptr<_Tp> StatModel::load( const String& filename )

This is static template method of StatModel. It's usage is following (in the case of SVM): ::

    Ptr<SVM> svm = StatModel::load<SVM>("my_svm_model.xml");

In order to make this method work, the derived class must overwrite ``Algorithm::read(const FileNode& fn)``.
