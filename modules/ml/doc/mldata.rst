MLData
===================

.. highlight:: cpp

For the machine learning algorithms, the data set is often stored in a file of the ``.csv``-like format. The file contains a table of predictor and response values where each row of the table corresponds to a sample. Missing values are supported. The UC Irvine Machine Learning Repository (http://archive.ics.uci.edu/ml/) provides many data sets stored in such a format to the machine learning community. The class ``MLData`` is implemented to easily load the data for training one of the OpenCV machine learning algorithms. For float values, only the  ``'.'`` separator is supported.

CvMLData
--------
.. ocv:class:: CvMLData

Class for loading the data from a ``.csv`` file. 
::

    class CV_EXPORTS CvMLData
    {
    public:
        CvMLData();
        virtual ~CvMLData();

        int read_csv(const char* filename);

        const CvMat* get_values() const;
        const CvMat* get_responses();
        const CvMat* get_missing() const;

        void set_response_idx( int idx );
        int get_response_idx() const;

        
        void set_train_test_split( const CvTrainTestSplit * spl);
        const CvMat* get_train_sample_idx() const;
        const CvMat* get_test_sample_idx() const;
        void mix_train_and_test_idx();
        
        const CvMat* get_var_idx();
        void chahge_var_idx( int vi, bool state );

        const CvMat* get_var_types();
        void set_var_types( const char* str );
        
        int get_var_type( int var_idx ) const;
        void change_var_type( int var_idx, int type);
     
        void set_delimiter( char ch );
        char get_delimiter() const;

        void set_miss_ch( char ch );
        char get_miss_ch() const;
        
        const std::map<std::string, int>& get_class_labels_map() const;
        
    protected: 
        ... 
    };

CvMLData::read_csv
------------------
.. ocv:function:: int CvMLData::read_csv(const char* filename);

    Reads the data set from a ``.csv``-like ``filename`` file and stores all read values in a matrix. 
	
While reading the data, the method tries to define the type of variables (predictors and responses): ordered or categorical. If a value of the variable is not numerical (except for the label for a missing value), the type of the variable is set to ``CV_VAR_CATEGORICAL``. If all existing values of the variable are numerical, the type of the variable is set to ``CV_VAR_ORDERED``. So, the default definition of variables types works correctly for all cases except the case of a categorical variable with numerical class labeles. In this case, the type ``CV_VAR_ORDERED`` is set. You should change the type to ``CV_VAR_CATEGORICAL`` using the method :ocv:func:`CvMLData::change_var_type`. For categorical variables, a common map is built to convert a string class label to the numerical class label. Use :ocv:func:`CvMLData::get_class_labels_map` to obtain this map. 

Also, when reading the data, the method constructs the mask of missing values. For example, values are egual to `'?'`.

CvMLData::get_values
--------------------
.. ocv:function:: const CvMat* CvMLData::get_values() const;

    Returns a pointer to the matrix of predictor and response ``values``  or ``0`` if the data has not been loaded from the file yet. 
	
The row count of this matrix equals the sample count. The column count equals predictors ``+ 1`` for the response (if exists) count. This means that each row of the matrix contains values of one sample predictor and response. The matrix type is ``CV_32FC1``.

CvMLData::get_responses
-----------------------
.. ocv:function:: const CvMat* CvMLData::get_responses();

    Returns a pointer to the matrix of response values or throws an exception if the data has not been loaded from the file yet. 
	
This is a single-column matrix of the type ``CV_32FC1``. Its row count is equal to the sample count, one column and .

CvMLData::get_missing
---------------------
.. ocv:function:: const CvMat* CvMLData::get_missing() const;

    Returns a pointer to the mask matrix of missing values or throws an exception if the data has not been loaded from the file yet. 
	
This matrix has the same size as the  ``values`` matrix (see :ocv:func:`CvMLData::get_values`) and the type ``CV_8UC1``.

CvMLData::set_response_idx
--------------------------
.. ocv:function:: void CvMLData::set_response_idx( int idx );

    Sets the index of a response column in the ``values`` matrix (see :ocv:func:`CvMLData::get_values`) or throws an exception if the data has not been loaded from the file yet. 
	
The old response columns become predictors. If ``idx < 0``, there is no response.

CvMLData::get_response_idx
----------
.. ocv:function:: int CvMLData::get_response_idx() const;

    Gets the index of a response column in the ``values`` matrix (see :ocv:func:`CvMLData::get_values`) or throws an exception if the data has not been loaded from the file yet.
	
If ``idx < 0``, there is no response.
    

CvMLData::set_train_test_split
------------------------------
.. ocv:function:: void CvMLData::set_train_test_split( const CvTrainTestSplit * spl );
    
    Divides the read data set into two disjoint training and test subsets. 
	
This method sets parameters for such a split using ``spl`` (see :ocv:class:`CvTrainTestSplit`) or throws an exception if the data has not been loaded from the file yet. 

CvMLData::get_train_sample_idx
------------------------------
.. ocv:function:: const CvMat* CvMLData::get_train_sample_idx() const;

    Divides the data set into training and test subsets by setting a split (see :ocv:func:`CvMLData::set_train_test_split`).

The current method returns the matrix of sample indices for a training subset. This is a single-row  matrix of the type ``CV_32SC1``. If data split is not set, the method returns ``0``. If the data has not been loaded from the file yet, an exception is thrown.

CvMLData::get_test_sample_idx
-----------------------------
.. ocv:function:: const CvMat* CvMLData::get_test_sample_idx() const;
    
    Provides functionality similar to :ocv:func:`CvMLData::get_train_sample_idx` but for a test subset.
    
CvMLData::mix_train_and_test_idx
--------------------------------
.. ocv:function:: void CvMLData::mix_train_and_test_idx();
    
    Mixes the indices of training and test samples preserving sizes of training and test subsets if the data split is set by :ocv:func:`CvMLData::get_values`. If the data has not been loaded from the file yet, an exception is thrown.

CvMLData::get_var_idx
---------------------
.. ocv:function:: const CvMat* CvMLData::get_var_idx();
    
    Returns the indices of variables (columns) used in the ``values`` matrix (see :ocv:func:`CvMLData::get_values`). 

The function returns `0`` if the used subset is not set. It throws an exception if the data has not been loaded from the file yet. Returned matrix is a single-row matrix of the type ``CV_32SC1``. Its column count is equal to the size of the used variable subset.

CvMLData::chahge_var_idx
------------------------
.. ocv:function:: void CvMLData::chahge_var_idx( int vi, bool state );

    Controls the data set by changing the number of variables.??
	
By default, after reading the data set all variables in the ``values`` matrix (see :ocv:func:`CvMLData::get_values`) are used. But you may want to use only a subset of variables and include/exclude (depending on ``state`` value) a variable with the ``vi`` index from the used subset. If the data has not been loaded from the file yet, an exception is thrown.
    
CvMLData::get_var_types
-----------------------
.. ocv:function:: const CvMat* CvMLData::get_var_types();
    
	Returns a matrix of used variable types. 
	
The function returns a single-row matrix of the type ``CV_8UC1``. column count equel to used variables count and type . If data has not been loaded from file yet an exception is thrown.
    
CvMLData::set_var_types
-----------------------
.. ocv:function:: void CvMLData::set_var_types( const char* str );

    Sets variables types according to the given string ``str``. 
	
In the string, a variable type is followed by a list of variables indices. For example: ``"ord[0-17],cat[18]"``, ``"ord[0,2,4,10-12], cat[1,3,5-9,13,14]"``, ``"cat"`` (all variables are categorical), ``"ord"`` (all variables are ordered). 

CvMLData::get_var_type
----------------------
.. ocv:function:: int CvMLData::get_var_type( int var_idx ) const;

    Returns the type of a variable by the index ``var_idx`` ( ``CV_VAR_ORDERED`` or ``CV_VAR_CATEGORICAL``).
    
CvMLData::change_var_type
-------------------------
.. ocv:function:: void CvMLData::change_var_type( int var_idx, int type);
    
    Changes type of variable with index ``var_idx`` from existing type to ``type`` ( ``CV_VAR_ORDERED`` or ``CV_VAR_CATEGORICAL``).
     
CvMLData::set_delimiter
-----------------------
.. ocv:function:: void CvMLData::set_delimiter( char ch );

    Sets the delimiter for variable values in a file. For example: ``','`` (default), ``';'``, ``' '`` (space), or other characters. The float separator ``'.'`` is not allowed.

CvMLData::get_delimiter
-----------------------
.. ocv:function:: char CvMLData::get_delimiter() const;

    Gets the set delimiter character.

CvMLData::set_miss_ch
---------------------
.. ocv:function:: void CvMLData::set_miss_ch( char ch );

    Sets the character for a missing value. For example: ``'?'`` (default), ``'-'``. The float separator ``'.'`` is not allowed.

CvMLData::get_miss_ch
---------------------
.. ocv:function:: char CvMLData::get_miss_ch() const;

    Gets the character for a missing value.

CvMLData::get_class_labels_map
-------------------------------
.. ocv:function:: const std::map<std::string, int>& CvMLData::get_class_labels_map() const;

    Returns a map that converts string class labels to the numerical class labels. It can be used to get an original class label as in a file.

CvTrainTestSplit
----------------
.. ocv:class:: CvTrainTestSplit

Structure setting the split of a data set read by :ocv:class:`CvMLData`.
::

    struct CvTrainTestSplit
    {
        CvTrainTestSplit();
        CvTrainTestSplit( int train_sample_count, bool mix = true);
        CvTrainTestSplit( float train_sample_portion, bool mix = true);

        union
        {
            int count;
            float portion;
        } train_sample_part;
        int train_sample_part_mode;

        bool mix;
    };

There are two ways to construct a split:

* Set the training sample count (subset size) ``train_sample_count``. Other existing samples are located in a test subset. 

* Set a training sample portion in ``[0,..1]``. The flag ``mix`` is used to mix training and test samples indices when the split is set. Otherwise, the data set is split in the storing order: the first part of samples of a given size is a training subset, the second part is a test subset.
