*********************************************
Base Loader Callback Interface Implementation
*********************************************

.. highlight:: java
.. class:: BaseLoaderCallback

    Basic implementation of ``LoaderCallbackInterface``. Logic of this implementation is
    well-described by the following scheme:

.. image:: img/AndroidAppUsageModel.png

Using in Java Activity
----------------------

There is a very base code snippet implementing the async initialization with ``BaseLoaderCallback``.
See the "15-puzzle" OpenCV sample for details.

.. code-block:: java
    :linenos:

    public class MyActivity extends Activity implements HelperCallbackInterface
    {
    private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(this) {
       @Override
       public void onManagerConnected(int status) {
         switch (status) {
           case LoaderCallbackInterface.SUCCESS:
           {
              Log.i(TAG, "OpenCV loaded successfully");
              // Create and set View
              mView = new puzzle15View(mAppContext);
              setContentView(mView);
           } break;
           default:
           {
              super.onManagerConnected(status);
           } break;
         }
       }
    };

    /** Call on every application resume **/
    @Override
    protected void onResume()
    {
        Log.i(TAG, "Called onResume");
        super.onResume();

        Log.i(TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mOpenCVCallBack))
        {
            Log.e(TAG, "Cannot connect to OpenCV Manager");
        }
    }

Using in Service
----------------

Default ``BaseLoaderCallback`` implementation treats application context as ``Activity`` and calls
``Activity.finish()`` method to exit in case of initialization failure.
To override this behavior you need to override ``finish()`` method of ``BaseLoaderCallback`` class
and implement your own finalization method.
