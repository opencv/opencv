
%typemap(javaimports) glcamera "
/** a class for doing the native rendering of images
this class renders using GL2 es, the native ndk version
This class is used by the GL2CameraViewer to do the rendering,
and is inspired by the gl2 example in the ndk samples
*/"



%javamethodmodifiers glcamera::init"
  /**  should be called onSurfaceChanged by the GLSurfaceView that is using this
  	*  as the drawing engine
  	* @param width the width of the surface view that this will be drawing to
    * @param width the height of the surface view that this will be drawing to
  	*
    */
  public";
  
%javamethodmodifiers glcamera::step"
  /**  should be called by GLSurfaceView.Renderer in the onDrawFrame method, as it
  handles the rendering of the opengl scene, and requires that the opengl context be
  valid.
 
  	*
    */
  public";
%javamethodmodifiers glcamera::drawMatToGL"
  /** copies an image from a pool and queues it for drawing in opengl.
  	*  this does transformation into power of two texture sizes
  	* @param idx the image index to copy
    * @param pool the image_pool to look up the image from
  	*
    */
  public";
  
class glcamera {
public:
     void init(int width, int height);
     void step();
     void drawMatToGL(int idx, image_pool* pool);
};

