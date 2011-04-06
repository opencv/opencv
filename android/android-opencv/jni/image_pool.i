

%typemap(javaimports) image_pool "
/** image_pool is used for keeping track of a pool of native images.  It stores images as cv::Mat's and
references them by an index.  It allows one to get a pointer to an underlying mat, and handles memory deletion.*/"


%javamethodmodifiers image_pool::getImage"
  /** gets a pointer to a stored image, by an index.  If the index is new, returns a null pointer
  	* @param idx the index in the pool that is associated with a cv::Mat
  	* @return the pointer to a cv::Mat, null pointer if the given idx is novel
    */
  public";
  
  
%javamethodmodifiers image_pool::deleteImage"
  /** deletes the image from the pool
  	* @param idx the index in the pool that is associated with a cv::Mat
    */
  public";
  
  
  
%javamethodmodifiers addYUVtoPool"
  /** adds a yuv
  	* @param idx the index in the pool that is associated with a cv::Mat
    */
  public";
  
%include "various.i"


%apply (char* BYTE) { (char *data)}; //byte[] to char*


%native (addYUVtoPool) void addYUVtoPool(image_pool* pool, char* data,int idx, int width, int height, bool grey);




%feature("director") image_pool;
class image_pool {
public:
	Mat getGrey(int i);
	Mat getImage(int i);
	void addImage(int i, Mat mat);
	void convertYUVtoColor(int i, Mat& out);
};

void RGB2BGR(const Mat& in, Mat& out);

