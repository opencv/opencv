Video Analysis
==============

.. highlight:: cpp



gpu::VIBE_GPU
-------------
.. ocv:class:: gpu::VIBE_GPU

Class used for background/foreground segmentation. ::

    class VIBE_GPU
    {
    public:
        explicit VIBE_GPU(unsigned long rngSeed = 1234567);

        void initialize(const GpuMat& firstFrame, Stream& stream = Stream::Null());

        void operator()(const GpuMat& frame, GpuMat& fgmask, Stream& stream = Stream::Null());

        void release();

        ...
    };

The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [VIBE2011]_.



gpu::VIBE_GPU::VIBE_GPU
-----------------------
The constructor.

.. ocv:function:: gpu::VIBE_GPU::VIBE_GPU(unsigned long rngSeed = 1234567)

    :param rngSeed: Value used to initiate a random sequence.

Default constructor sets all parameters to default values.



gpu::VIBE_GPU::initialize
-------------------------
Initialize background model and allocates all inner buffers.

.. ocv:function:: void gpu::VIBE_GPU::initialize(const GpuMat& firstFrame, Stream& stream = Stream::Null())

    :param firstFrame: First frame from video sequence.

    :param stream: Stream for the asynchronous version.



gpu::VIBE_GPU::operator()
-------------------------
Updates the background model and returns the foreground mask

.. ocv:function:: void gpu::VIBE_GPU::operator()(const GpuMat& frame, GpuMat& fgmask, Stream& stream = Stream::Null())

    :param frame: Next video frame.

    :param fgmask: The output foreground mask as an 8-bit binary image.

    :param stream: Stream for the asynchronous version.



gpu::VIBE_GPU::release
----------------------
Releases all inner buffer's memory.

.. ocv:function:: void gpu::VIBE_GPU::release()




.. [VIBE2011] O. Barnich and M. Van D Roogenbroeck. *ViBe: A universal background subtraction algorithm for video sequences*. IEEE Transactions on Image Processing, 20(6) :1709-1724, June 2011
