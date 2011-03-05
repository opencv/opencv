Statistical Models
==================

.. highlight:: cpp

.. index:: CvStatModel

CvStatModel
-----------
.. c:type:: CvStatModel

Base class for the statistical models in ML. ::

    class CvStatModel
    {
    public:
        /* CvStatModel(); */
        /* CvStatModel( const CvMat* train_data ... ); */

        virtual ~CvStatModel();

        virtual void clear()=0;

        /* virtual bool train( const CvMat* train_data, [int tflag,] ..., const
            CvMat* responses, ...,
         [const CvMat* var_idx,] ..., [const CvMat* sample_idx,] ...
         [const CvMat* var_type,] ..., [const CvMat* missing_mask,]
            <misc_training_alg_params> ... )=0;
          */

        /* virtual float predict( const CvMat* sample ... ) const=0; */

        virtual void save( const char* filename, const char* name=0 )=0;
        virtual void load( const char* filename, const char* name=0 )=0;

        virtual void write( CvFileStorage* storage, const char* name )=0;
        virtual void read( CvFileStorage* storage, CvFileNode* node )=0;
    };


In this declaration some methods are commented off. Actually, these are methods for which there is no unified API (with the exception of the default constructor), however, there are many similarities in the syntax and semantics that are briefly described below in this section, as if they are a part of the base class.

.. index:: CvStatModel::CvStatModel

.. _CvStatModel::CvStatModel:

CvStatModel::CvStatModel
------------------------
.. c:function:: CvStatModel::CvStatModel()

    Default constructor.

Each statistical model class in ML has a default constructor without parameters. This constructor is useful for 2-stage model construction, when the default constructor is followed by ``train()`` or ``load()`` .

.. index:: CvStatModel::CvStatModel(...)

.. _CvStatModel::CvStatModel(...):

CvStatModel::CvStatModel(...)
-----------------------------
.. c:function:: CvStatModel::CvStatModel( const CvMat* train_data ... )

    Training constructor.

Most ML classes provide single-step construct and train constructors. This constructor is equivalent to the default constructor, followed by the ``train()`` method with the parameters that are passed to the constructor.

.. index:: CvStatModel::~CvStatModel

.. _CvStatModel::~CvStatModel:

CvStatModel::~CvStatModel
-------------------------
.. c:function:: CvStatModel::~CvStatModel()

    Virtual destructor.

The destructor of the base class is declared as virtual, so it is safe to write the following code: ::

    CvStatModel* model;
    if( use_svm )
        model = new CvSVM(... /* SVM params */);
    else
        model = new CvDTree(... /* Decision tree params */);
    ...
    delete model;


Normally, the destructor of each derived class does nothing, but in this instance it calls the overridden method ``clear()`` that deallocates all the memory.

.. index:: CvStatModel::clear

.. _CvStatModel::clear:

CvStatModel::clear
------------------
.. c:function:: void CvStatModel::clear()

    Deallocates memory and resets the model state.

The method ``clear`` does the same job as the destructor; it deallocates all the memory occupied by the class members. But the object itself is not destructed, and can be reused further. This method is called from the destructor, from the ``train`` methods of the derived classes, from the methods ``load()``,``read()`` or even explicitly by the user.

.. index:: CvStatModel::save

.. _CvStatModel::save:

CvStatModel::save
-----------------
.. c:function:: void CvStatModel::save( const char* filename, const char* name=0 )

    Saves the model to a file.

The method ``save`` stores the complete model state to the specified XML or YAML file with the specified name or default name (that depends on the particular class). ``Data persistence`` functionality from CxCore is used.

.. index:: CvStatModel::load

.. _CvStatModel::load:

CvStatModel::load
-----------------
.. c:function:: void CvStatModel::load( const char* filename, const char* name=0 )

    Loads the model from a file.

The method ``load`` loads the complete model state with the specified name (or default model-dependent name) from the specified XML or YAML file. The previous model state is cleared by ``clear()`` .

Note that the method is virtual, so any model can be loaded using this virtual method. However, unlike the C types of OpenCV that can be loaded using the generic
\
cross{cvLoad}, here the model type must be known, because an empty model must be constructed beforehand. This limitation will be removed in the later ML versions.

.. index:: CvStatModel::write

.. _CvStatModel::write:

CvStatModel::write
------------------
.. c:function:: void CvStatModel::write( CvFileStorage* storage, const char* name )

    Writes the model to file storage.

The method ``write`` stores the complete model state to the file storage with the specified name or default name (that depends on the particular class). The method is called by ``save()`` .

.. index:: CvStatModel::read

.. _CvStatModel::read:

CvStatModel::read
-----------------
.. c:function:: void CvStatMode::read( CvFileStorage* storage, CvFileNode* node )

    Reads the model from file storage.

The method ``read`` restores the complete model state from the specified node of the file storage. The node must be located by the user using the function
:ref:`GetFileNodeByName` .

The previous model state is cleared by ``clear()`` .

.. index:: CvStatModel::train

.. _CvStatModel::train:

CvStatModel::train
------------------
.. c:function:: bool CvStatMode::train( const CvMat* train_data, [int tflag,] ..., const CvMat* responses, ...,     [const CvMat* var_idx,] ..., [const CvMat* sample_idx,] ...     [const CvMat* var_type,] ..., [const CvMat* missing_mask,] <misc_training_alg_params> ... )

    Trains the model.

The method trains the statistical model using a set of input feature vectors and the corresponding output values (responses). Both input and output vectors/values are passed as matrices. By default the input feature vectors are stored as ``train_data`` rows, i.e. all the components (features) of a training vector are stored continuously. However, some algorithms can handle the transposed representation, when all values of each particular feature (component/input variable) over the whole input set are stored continuously. If both layouts are supported, the method includes ``tflag`` parameter that specifies the orientation:

* ``tflag=CV_ROW_SAMPLE``     means that the feature vectors are stored as rows,

* ``tflag=CV_COL_SAMPLE``     means that the feature vectors are stored as columns.

The ``train_data`` must have a ``CV_32FC1`` (32-bit floating-point, single-channel) format. Responses are usually stored in the 1d vector (a row or a column) of ``CV_32SC1`` (only in the classification problem) or ``CV_32FC1`` format, one value per input vector (although some algorithms, like various flavors of neural nets, take vector responses).

For classification problems the responses are discrete class labels; for regression problems the responses are values of the function to be approximated. Some algorithms can deal only with classification problems, some - only with regression problems, and some can deal with both problems. In the latter case the type of output variable is either passed as separate parameter, or as a last element of ``var_type`` vector:

* ``CV_VAR_CATEGORICAL``     means that the output values are discrete class labels,

* ``CV_VAR_ORDERED(=CV_VAR_NUMERICAL)``     means that the output values are ordered, i.e. 2 different values can be compared as numbers, and this is a regression problem

The types of input variables can be also specified using ``var_type`` . Most algorithms can handle only ordered input variables.

Many models in the ML may be trained on a selected feature subset, and/or on a selected sample subset of the training set. To make it easier for the user, the method ``train`` usually includes ``var_idx`` and ``sample_idx`` parameters. The former identifies variables (features) of interest, and the latter identifies samples of interest. Both vectors are either integer ( ``CV_32SC1`` ) vectors, i.e. lists of 0-based indices, or 8-bit ( ``CV_8UC1`` ) masks of active variables/samples. The user may pass ``NULL`` pointers instead of either of the arguments, meaning that all of the variables/samples are used for training.

Additionally some algorithms can handle missing measurements, that is when certain features of certain training samples have unknown values (for example, they forgot to measure a temperature of patient A on Monday). The parameter ``missing_mask`` , an 8-bit matrix the same size as ``train_data`` , is used to mark the missed values (non-zero elements of the mask).

Usually, the previous model state is cleared by ``clear()`` before running the training procedure. However, some algorithms may optionally update the model state with the new training data, instead of resetting it.

.. index:: CvStatModel::predict

.. _CvStatModel::predict:

CvStatModel::predict
--------------------
.. c:function:: float CvStatMode::predict( const CvMat* sample[, <prediction_params>] ) const

    Predicts the response for the sample.

The method is used to predict the response for a new sample. In the case of classification the method returns the class label, in the case of regression - the output function value. The input sample must have as many components as the ``train_data`` passed to ``train`` contains. If the ``var_idx`` parameter is passed to ``train`` , it is remembered and then is used to extract only the necessary components from the input sample in the method ``predict`` .

The suffix "const" means that prediction does not affect the internal model state, so the method can be safely called from within different threads.

