%typemap(javaimports) Mat "
/** Wrapper for the OpenCV Mat object. Good for passing around as a pointer to a Mat.
*/"

%typemap(javaimports) Size "
/** Wrapper for the OpenCV Size object. Good for setting dimensions of cv::Mat...
*/"

class Mat {
public:
 %immutable;
	int rows;
	int cols;
};

class Size{
public:
	Size(int width,int height);
	int width;
	int height;
	
};

template<class _Tp> class Ptr
{
public:
    //! empty constructor
    Ptr();
    //! take ownership of the pointer. The associated reference counter is allocated and set to 1
    Ptr(_Tp* _obj);
    //! calls release()
    ~Ptr();
    //! copy constructor. Copies the members and calls addref()
    Ptr(const Ptr& ptr);
    //! copy operator. Calls ptr.addref() and release() before copying the members
   // Ptr& operator = (const Ptr& ptr);
    //! increments the reference counter
    void addref();
    //! decrements the reference counter. If it reaches 0, delete_obj() is called
    void release();
    //! deletes the object. Override if needed
    void delete_obj();
    //! returns true iff obj==NULL
    bool empty() const;

    
    //! helper operators making "Ptr<T> ptr" use very similar to "T* ptr".
    _Tp* operator -> ();
   // const _Tp* operator -> () const;

   // operator _Tp* ();
  //  operator const _Tp*() const;
    
protected:
    _Tp* obj; //< the object pointer.
    int* refcount; //< the associated reference counter
};

%template(PtrMat) Ptr<Mat>;