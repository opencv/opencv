MLData
===================

.. highlight:: cpp

For the machine learning algorithms usage it is often that data set is saved in file of format like .csv. The supported format file must contains the table of predictors and responses values, each row of the table must correspond to one sample. Missing values are supported. Famous UC Irvine Machine Learning Repository (http://archive.ics.uci.edu/ml/) provides many stored in such format data sets to the machine learning community. The class MLData has been implemented to ease the loading data for the training one of the existing in OpenCV machine learning algorithm. For float values only separator ``'.'`` is supported.

CvMLData
----------
.. ocv:class:: CvMLData

The class to load the data from .csv file. 
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
----------
.. ocv:function:: int CvMLData::read_csv(const char* filename);

    This method reads the data set from .csv-like file named ``filename`` and store all read values in one matrix. While reading the method tries to define variables (predictors and response) type: ordered or categorical. If some value of the variable is not a number (e.g. contains the letters) exept a label for missing value, then the type of the variable is set to ``CV_VAR_CATEGORICAL``. If all unmissing values of the variable are the numbers, then the type of the variable is set to ``CV_VAR_ORDERED``. So default definition of variables types works correctly for all cases except the case of categorical variable that has numerical class labeles. In such case the type ``CV_VAR_ORDERED`` will be set and user should change the type to ``CV_VAR_CATEGORICAL`` using method :ocv:func:`CvMLData::change_var_type`. For categorical variables the common map is built to convert string class label to the numerical class label and this map can be got by :ocv:func:`CvMLData::get_class_labels_map`. Also while reading the data the method constructs the mask of missing values (e.g. values are egual to `'?'`).

CvMLData::get_values
----------
.. ocv:function:: const CvMat* CvMLData::get_values() const;

    Returns the pointer to the predictors and responses ``values`` matrix or ``0`` if data has not been loaded from file yet. This matrix has rows count equal to samples count, columns count equal to predictors ``+ 1`` for response (if exist) count (i.e. each row of matrix is values of one sample predictors and response) and type ``CV_32FC1``.

CvMLData::get_responses
----------
.. ocv:function:: const CvMat* CvMLData::get_responses();

    Returns the pointer to the responses values matrix or throw exception if data has not been loaded from file yet. This matrix has rows count equal to samples count, one column and type ``CV_32FC1``.

CvMLData::get_missing
----------
.. ocv:function:: const CvMat* CvMLData::get_missing() const;

    Returns the pointer to the missing values mask matrix or throw exception if data has not been loaded from file yet. This matrix has the same size as ``values`` matrix (see :ocv:func:`CvMLData::get_values`) and type ``CV_8UC1``.

CvMLData::set_response_idx
----------
.. ocv:function:: void CvMLData::set_response_idx( int idx );

    Sets index of response column in ``values`` matrix (see :ocv:func:`CvMLData::get_values`) or throw exception if data has not been loaded from file yet. The old response column become pridictors. If ``idx < 0`` there will be no response.

CvMLData::get_response_idx
----------
.. ocv:function:: int CvMLData::get_response_idx() const;

    Gets response column index in ``values`` matrix (see :ocv:func:`CvMLData::get_values`), negative value there is no response or throw exception if data has not been loaded from file yet.
    

CvMLData::set_train_test_split
----------
.. ocv:function:: void set_train_test_split( const CvTrainTestSplit * spl );
    
    For different purposes it can be useful to devide the read data set into two disjoint subsets: training and test ones. This method sets parametes for such split (using ``spl``, see :ocv:class:`CvTrainTestSplit`) and make the data split or throw exception if data has not been loaded from file yet. 

CvMLData::get_train_sample_idx
----------
.. ocv:function:: const CvMat* CvMLData::get_train_sample_idx() const;

    The read data set can be devided on training and test data subsets by setting split (see :ocv:func:`CvMLData::set_train_test_split`). Current method returns the matrix of samples indices for training subset (this matrix has one row and type ``CV_32SC1``). If data split is not set then the method returns ``0``. If data has not been loaded from file yet an exception is thrown.

CvMLData::get_test_sample_idx
----------
.. ocv:function:: const CvMat* CvMLData::get_test_sample_idx() const;
    
    Analogically with :ocv:func:`CvMLData::get_train_sample_idx`, but for test subset.
    
CvMLData::mix_train_and_test_idx
----------
.. ocv:function:: void CvMLData::mix_train_and_test_idx();
    
    Mixes the indices of training and test samples preserving sizes of training and test subsets (if data split is set by :ocv:func:`CvMLData::get_values`). If data has not been loaded from file yet an exception is thrown.

CvMLData::get_var_idx
----------
.. ocv:function:: const CvMat* CvMLData::get_var_idx();
    
    Returns used variables (columns) indices in the ``values`` matrix (see :ocv:func:`CvMLData::get_values`), ``0`` if used subset is not set or throw exception if data has not been loaded from file yet. Returned matrix has one row, columns count equel to used variable subset size and type ``CV_32SC1``.

CvMLData::chahge_var_idx
----------
.. ocv:function:: void CvMLData::chahge_var_idx( int vi, bool state );

    By default after reading the data set all variables in ``values`` matrix (see :ocv:func:`CvMLData::get_values`) are used. But the user may want to use only subset of variables and can include on/off (depends on ``state`` value) a variable with ``vi`` index from used subset. If data has not been loaded from file yet an exception is thrown.
    
CvMLData::get_var_types
----------
.. ocv:function:: const CvMat* CvMLData::get_var_types();
    Returns matrix of used variable types. The matrix has one row, column count equel to used variables count and type ``CV_8UC1``. If data has not been loaded from file yet an exception is thrown.
    
CvMLData::set_var_types
----------
.. ocv:function:: void CvMLData::set_var_types( const char* str );

    Sets variables types according to given string ``str``. The better description of the supporting string format is several examples of it: ``"ord[0-17],cat[18]"``, ``"ord[0,2,4,10-12], cat[1,3,5-9,13,14]"``, ``"cat"`` (all variables are categorical), ``"ord"`` (all variables are ordered). That is after the variable type a list of such type variables indices is followed.

CvMLData::get_var_type
----------
.. ocv:function:: int CvMLData::get_var_type( int var_idx ) const;

    Returns type of variable by index ``var_idx`` ( ``CV_VAR_ORDERED`` or ``CV_VAR_CATEGORICAL``).
    
CvMLData::change_var_type
----------
.. ocv:function:: void CvMLData::change_var_type( int var_idx, int type);
    
    Changes type of variable with index ``var_idx`` from existing type to ``type`` ( ``CV_VAR_ORDERED`` or ``CV_VAR_CATEGORICAL``).
     
CvMLData::set_delimiter
----------
.. ocv:function:: void CvMLData::set_delimiter( char ch );

    Sets the delimiter for the variable values in file. E.g. ``','`` (default), ``';'``, ``' '`` (space) or other character (exapt float separator ``'.'``).

CvMLData::get_delimiter
----------
.. ocv:function:: char CvMLData::get_delimiter() const;

    Gets the set delimiter charecter.

CvMLData::set_miss_ch
----------
.. ocv:function:: void CvMLData::set_miss_ch( char ch );

    Sets the character denoting the missing of value. E.g. ``'?'`` (default), ``'-'``, etc (exapt float separator ``'.'``).

CvMLData::get_miss_ch
----------
.. ocv:function:: char CvMLData::get_miss_ch() const;

    Gets the character denoting the missing value.


CvTrainTestSplit
----------
.. ocv:class:: CvTrainTestSplit

The structure to set split of data set read by :ocv:class:`CvMLData`.
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

There are two ways to construct split. The first is by setting training sample count (subset size) ``train_sample_count``; other existing samples will be in test subset. The second is by setting training sample portion in ``[0,..1]``. The flag ``mix`` is used to mix training and test samples indices when split will be set, otherwise the data set will be devided in the storing order (first part of samples of given size is the training subset, other part is the test one).
