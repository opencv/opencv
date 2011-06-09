Statistical Models
==================

.. highlight:: cpp

.. index:: CvStatModel

CvStatModel
-----------
.. c:type:: CvStatModel

Base class for statistical models in ML ::

    class CvStatModel
    {
    public:
        /* CvStatModel(); */
        /* CvStatModel( const Mat& train_data ... ); */

        virtual ~CvStatModel();

        virtual void clear()=0;

        /* virtual bool train( const Mat& train_data, [int tflag,] ..., const
            Mat& responses, ...,
         [const Mat& var_idx,] ..., [const Mat& sample_idx,] ...
         [const Mat& var_type,] ..., [const Mat& missing_mask,]
            <misc_training_alg_params> ... )=0;
          */

        /* virtual float predict( const Mat& sample ... ) const=0; */

        virtual void save( const char* filename, const char* name=0 )=0;
        virtual void load( const char* filename, const char* name=0 )=0;

        virtual void write( CvFileStorage* storage, const char* name )=0;
        virtual void read( CvFileStorage* storage, CvFileNode* node )=0;
    };


In this declaration, some methods are commented off. These are methods for which there is no unified API (with the exception of the default constructor). However, there are many similarities in the syntax and semantics that are briefly described below in this section, as if they are part of the base class.

.. index:: CvStatModel::CvStatModel

.. _CvStatModel::CvStatModel:

CvStatModel::CvStatModel
------------------------
.. cpp:function:: CvStatModel::CvStatModel()

    Serves as a default constructor.

Each statistical model class in ML has a default constructor without parameters. This constructor is useful for a 2-stage model construction, when the default constructor is followed by ``train()`` or ``load()`` .

.. index:: CvStatModel::CvStatModel(...)

.. _CvStatModel::CvStatModel(...):

CvStatModel::CvStatModel(...)
-----------------------------
.. cpp:function:: CvStatModel::CvStatModel( const Mat& train_data ... )

    Serves as a training constructor.

Most ML classes provide a single-step constructor and train constructors. This constructor is equivalent to the default constructor, followed by the ``train()`` method with the parameters that are passed to the constructor.

.. index:: CvStatModel::~CvStatModel

.. _CvStatModel::~CvStatModel:

CvStatModel::~CvStatModel
-------------------------
.. cpp:function:: CvStatModel::~CvStatModel()

    Serves as a virtual destructor.

The destructor of the base class is declared as virtual. So, it is safe to write the following code: ::

    CvStatModel* model;
    if( use_svm )
        model = new CvSVM(... /* SVM params */);
    else
        model = new CvDTree(... /* Decision tree params */);
    ...
    delete model;


Normally, the destructor of each derived class does nothing. But in this instance, it calls the overridden method ``clear()`` that deallocates all the memory.

.. index:: CvStatModel::clear

.. _CvStatModel::clear:

CvStatModel::clear
------------------
.. cpp:function:: void CvStatModel::clear()

    Deallocates memory and resets the model state.

The method ``clear`` does the same job as the destructor: it deallocates all the memory occupied by the class members. But the object itself is not destructed and can be reused further. This method is called from the destructor, from the ``train`` methods of the derived classes, from the methods ``load()``,``read()`` , or even explicitly by the user.

.. index:: CvStatModel::save

.. _CvStatModel::save:

CvStatModel::save
-----------------
.. cpp:function:: void CvStatModel::save( const char* filename, const char* name=0 )

    Saves the model to a file.

The method ``save`` saves the complete model state to the specified XML or YAML file with the specified name or default name (which depends on a particular class). *Data persistence* functionality from ``CxCore`` is used.

.. index:: CvStatModel::load

.. _CvStatModel::load:

CvStatModel::load
-----------------
.. cpp:function:: void CvStatModel::load( const char* filename, const char* name=0 )

    Loads the model from a file.

The method ``load`` loads the complete model state with the specified name (or default model-dependent name) from the specified XML or YAML file. The previous model state is cleared by ``clear()`` .

**Note**:

The method is virtual, so any model can be loaded using this virtual method. However, unlike the C types of OpenCV that can be loaded using the generic
``cross{cvLoad}`` , the model type is required here to enable constructing an empty model beforehand.?? This limitation will be removed in the later ML versions.

.. index:: CvStatModel::write

.. _CvStatModel::write:

CvStatModel::write
------------------
.. cpp:function:: void CvStatModel::write( CvFileStorage* storage, const char* name )

    Writes the model to the file storage.

The method ``write`` stores the complete model state in the file storage with the specified name or default name (which depends on the particular class). The method is called by ``save()`` .

.. index:: CvStatModel::read

.. _CvStatModel::read:

CvStatModel::read
-----------------
.. cpp:function:: void CvStatMode::read( CvFileStorage* storage, CvFileNode* node )

    Reads the model from the file storage.

The method ``read`` restores the complete model state from the specified node of the file storage. Use the function
:ref:`GetFileNodeByName` to locate the node.

The previous model state is cleared by ``clear()`` .

.. index:: CvStatModel::train

.. _CvStatModel::train:

CvStatModel::train
------------------
.. cpp:function:: bool CvStatMode::train( const Mat& train_data, [int tflag,] ..., const Mat& responses, ...,     [const Mat& var_idx,] ..., [const Mat& sample_idx,] ...     [const Mat& var_type,] ..., [const Mat& missing_mask,] <misc_training_alg_params> ... )

    Trains the model.

The method trains the statistical model using a set of input feature vectors and the corresponding output values (responses). Both input and output vectors/values are passed as matrices. By default, the input feature vectors are stored as ``train_data`` rows, that is, all the components (features) of a training vector are stored continuously. However, some algorithms can handle the transposed representation when all values of each particular feature (component/input variable) over the whole input set are stored continuously. If both layouts are supported, the method includes the ``tflag`` parameter that specifies the orientation as follows:

* ``tflag=CV_ROW_SAMPLE``     The feature vectors are stored as rows.

* ``tflag=CV_COL_SAMPLE``     The feature vectors are stored as columns.

The ``train_data`` must have the ``CV_32FC1`` (32-bit floating-point, single-channel) format. Responses are usually stored in the 1D vector (a row or a column) of ``CV_32SC1`` (only in the classification problem) or ``CV_32FC1`` format, one value per input vector. Although, some algorithms, like various flavors of neural nets, take vector responses.

For classification problems, the responses are discrete class labels. For regression problems, the responses are values of the function to be approximated. Some algorithms can deal only with classification problems, some - only with regression problems, and some can deal with both problems. In the latter case, the type of output variable is either passed as a separate parameter or as the last element of the ``var_type`` vector:

* ``CV_VAR_CATEGORICAL``     The output values are discrete class labels.

* ``CV_VAR_ORDERED(=CV_VAR_NUMERICAL)``     The output values are ordered. This means that two different values can be compared as numbers, and this is a regression problem.

Types of input variables can be also specified using ``var_type`` . Most algorithms can handle only ordered input variables.

Many models in the ML may be trained on a selected feature subset, and/or on a selected sample subset of the training set. To make it easier for you, the method ``train`` usually includes the ``var_idx`` and ``sample_idx`` parameters. The former parameter identifies variables (features) of interest, and the latter one identifies samples of interest. Both vectors are either integer ( ``CV_32SC1`` ) vectors (lists of 0-based indices) or 8-bit ( ``CV_8UC1`` ) masks of active variables/samples. You may pass ``NULL`` pointers instead of either of the arguments, meaning that all of the variables/samples are used for training.

Additionally, some algorithms can handle missing measurements, that is, when certain features of certain training samples have unknown values (for example, they forgot to measure a temperature of patient A on Monday). The parameter ``missing_mask`` , an 8-bit matrix of the same size as ``train_data`` , is used to mark the missed values (non-zero elements of the mask).

Usually, the previous model state is cleared by ``clear()`` before running the training procedure. However, some algorithms may optionally update the model state with the new training data, instead of resetting it.

.. index:: CvStatModel::predict

.. _CvStatModel::predict:

CvStatModel::predict
--------------------
.. cpp:function:: float CvStatMode::predict( const Mat& sample[, <prediction_params>] ) const

    Predicts the response for a sample.

The method is used to predict the response for a new sample. In case of a classification, the method returns the class label. In case of a regression, the method returns the output function value. The input sample must have as many components as the ``train_data`` passed to ``train`` contains. If the ``var_idx`` parameter is passed to ``train`` , it is remembered and then is used to extract only the necessary components from the input sample in the method ``predict`` .

The suffix ``const`` means that prediction does not affect the internal model state, so the method can be safely called from within different threads.

