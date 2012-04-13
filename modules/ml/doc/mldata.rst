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
Reads the data set from a ``.csv``-like ``filename`` file and stores all read values in a matrix. 

.. ocv:function:: int CvMLData::read_csv(const char* filename)

    :param filename: The input file name

While reading the data, the method tries to define the type of variables (predictors and responses): ordered or categorical. If a value of the variable is not numerical (except for the label for a missing value), the type of the variable is set to ``CV_VAR_CATEGORICAL``. If all existing values of the variable are numerical, the type of the variable is set to ``CV_VAR_ORDERED``. So, the default definition of variables types works correctly for all cases except the case of a categorical variable with numerical class labels. In this case, the type ``CV_VAR_ORDERED`` is set. You should change the type to ``CV_VAR_CATEGORICAL`` using the method :ocv:func:`CvMLData::change_var_type`. For categorical variables, a common map is built to convert a string class label to the numerical class label. Use :ocv:func:`CvMLData::get_class_labels_map` to obtain this map. 

Also, when reading the data, the method constructs the mask of missing values. For example, values are equal to `'?'`.

CvMLData::get_values
--------------------
Returns a pointer to the matrix of predictors and response values

.. ocv:function:: const CvMat* CvMLData::get_values() const

The method returns a pointer to the matrix of predictor and response ``values``  or ``0`` if the data has not been loaded from the file yet. 

The row count of this matrix equals the sample count. The column count equals predictors ``+ 1`` for the response (if exists) count. This means that each row of the matrix contains values of one sample predictor and response. The matrix type is ``CV_32FC1``.

CvMLData::get_responses
-----------------------
Returns a pointer to the matrix of response values

.. ocv:function:: const CvMat* CvMLData::get_responses()

The method returns a pointer to the matrix of response values or throws an exception if the data has not been loaded from the file yet. 

This is a single-column matrix of the type ``CV_32FC1``. Its row count is equal to the sample count, one column and .

CvMLData::get_missing
---------------------
Returns a pointer to the mask matrix of missing values

.. ocv:function:: const CvMat* CvMLData::get_missing() const

The method returns a pointer to the mask matrix of missing values or throws an exception if the data has not been loaded from the file yet. 

This matrix has the same size as the  ``values`` matrix (see :ocv:func:`CvMLData::get_values`) and the type ``CV_8UC1``.

CvMLData::set_response_idx
--------------------------
Specifies index of response column in the data matrix

.. ocv:function:: void CvMLData::set_response_idx( int idx )

The method sets the index of a response column in the ``values`` matrix (see :ocv:func:`CvMLData::get_values`) or throws an exception if the data has not been loaded from the file yet. 

The old response columns become predictors. If ``idx < 0``, there is no response.

CvMLData::get_response_idx
--------------------------
Returns index of the response column in the loaded data matrix

.. ocv:function:: int CvMLData::get_response_idx() const

The method returns the index of a response column in the ``values`` matrix (see :ocv:func:`CvMLData::get_values`) or throws an exception if the data has not been loaded from the file yet.

If ``idx < 0``, there is no response.
    

CvMLData::set_train_test_split
------------------------------
Divides the read data set into two disjoint training and test subsets. 

.. ocv:function:: void CvMLData::set_train_test_split( const CvTrainTestSplit * spl )

This method sets parameters for such a split using ``spl`` (see :ocv:class:`CvTrainTestSplit`) or throws an exception if the data has not been loaded from the file yet. 

CvMLData::get_train_sample_idx
------------------------------
Returns the matrix of sample indices for a training subset

.. ocv:function:: const CvMat* CvMLData::get_train_sample_idx() const

The method returns the matrix of sample indices for a training subset. This is a single-row  matrix of the type ``CV_32SC1``. If data split is not set, the method returns ``0``. If the data has not been loaded from the file yet, an exception is thrown.

CvMLData::get_test_sample_idx
-----------------------------
Returns the matrix of sample indices for a testing subset

.. ocv:function:: const CvMat* CvMLData::get_test_sample_idx() const

    
CvMLData::mix_train_and_test_idx
--------------------------------
Mixes the indices of training and test samples

.. ocv:function:: void CvMLData::mix_train_and_test_idx()
    
The method shuffles the indices of training and test samples preserving sizes of training and test subsets if the data split is set by :ocv:func:`CvMLData::get_values`. If the data has not been loaded from the file yet, an exception is thrown.

CvMLData::get_var_idx
---------------------
Returns the indices of the active variables in the data matrix

.. ocv:function:: const CvMat* CvMLData::get_var_idx()
    
The method returns the indices of variables (columns) used in the ``values`` matrix (see :ocv:func:`CvMLData::get_values`). 

It returns ``0`` if the used subset is not set. It throws an exception if the data has not been loaded from the file yet. Returned matrix is a single-row matrix of the type ``CV_32SC1``. Its column count is equal to the size of the used variable subset.

CvMLData::chahge_var_idx
------------------------
Enables or disables particular variable in the loaded data

.. ocv:function:: void CvMLData::chahge_var_idx( int vi, bool state )

By default, after reading the data set all variables in the ``values`` matrix (see :ocv:func:`CvMLData::get_values`) are used. But you may want to use only a subset of variables and include/exclude (depending on ``state`` value) a variable with the ``vi`` index from the used subset. If the data has not been loaded from the file yet, an exception is thrown.
    
CvMLData::get_var_types
-----------------------
Returns a matrix of the variable types. 

.. ocv:function:: const CvMat* CvMLData::get_var_types()
    
The function returns a single-row matrix of the type ``CV_8UC1``, where each element is set to either ``CV_VAR_ORDERED`` or ``CV_VAR_CATEGORICAL``. The number of columns is equal to the number of variables. If data has not been loaded from file yet an exception is thrown.
    
CvMLData::set_var_types
-----------------------
Sets the variables types in the loaded data.

.. ocv:function:: void CvMLData::set_var_types( const char* str )

In the string, a variable type is followed by a list of variables indices. For example: ``"ord[0-17],cat[18]"``, ``"ord[0,2,4,10-12], cat[1,3,5-9,13,14]"``, ``"cat"`` (all variables are categorical), ``"ord"`` (all variables are ordered). 

CvMLData::get_var_type
----------------------
Returns type of the specified variable

.. ocv:function:: int CvMLData::get_var_type( int var_idx ) const

The method returns the type of a variable by the index ``var_idx`` ( ``CV_VAR_ORDERED`` or ``CV_VAR_CATEGORICAL``).
    
CvMLData::change_var_type
-------------------------
Changes type of the specified variable

.. ocv:function:: void CvMLData::change_var_type( int var_idx, int type)
    
The method changes type of variable with index ``var_idx`` from existing type to ``type`` ( ``CV_VAR_ORDERED`` or ``CV_VAR_CATEGORICAL``).
     
CvMLData::set_delimiter
-----------------------
Sets the delimiter in the file used to separate input numbers

.. ocv:function:: void CvMLData::set_delimiter( char ch )

The method sets the delimiter for variables in a file. For example: ``','`` (default), ``';'``, ``' '`` (space), or other characters. The floating-point separator ``'.'`` is not allowed.

CvMLData::get_delimiter
-----------------------
Returns the currently used delimiter character.

.. ocv:function:: char CvMLData::get_delimiter() const


CvMLData::set_miss_ch
---------------------
Sets the character used to specify missing values

.. ocv:function:: void CvMLData::set_miss_ch( char ch )

The method sets the character used to specify missing values. For example: ``'?'`` (default), ``'-'``. The floating-point separator ``'.'`` is not allowed.

CvMLData::get_miss_ch
---------------------
Returns the currently used missing value character.

.. ocv:function:: char CvMLData::get_miss_ch() const

CvMLData::get_class_labels_map
-------------------------------
Returns a map that converts strings to labels.

.. ocv:function:: const std::map<std::string, int>& CvMLData::get_class_labels_map() const

The method returns a map that converts string class labels to the numerical class labels. It can be used to get an original class label as in a file.

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
