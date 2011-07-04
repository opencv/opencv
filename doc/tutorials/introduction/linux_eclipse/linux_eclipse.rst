.. _Linux_Eclipse_Usage:

Using OpenCV with Eclipse (plugin CDT)
****************************************

.. note::
   For me at least, this works, is simple and quick. Suggestions are welcome

Prerequisites
===============

1. Having installed `Eclipse <http://www.eclipse.org/>`_ in your workstation (only the CDT plugin for C/C++ is needed). You can follow the following steps:

   * Go to the Eclipse site  

   * Download `Eclipse IDE for C/C++ Developers <http://www.eclipse.org/downloads/packages/eclipse-ide-cc-developers/heliossr2>`_ . Choose the link according to your workstation.

#. Having installed OpenCV. If not yet, go :ref:`here <Linux-Installation>`.

Making a project
=================

1. Start Eclipse. Just run the executable that comes in the folder. 

#. Go to **File -> New -> C/C++ Project**

   .. image:: images/a0.png
      :height: 400px 
      :alt: Eclipse Tutorial Screenshot 0
      :align: center

#. Choose a name for your project (i.e. DisplayImage). An **Empty Project** should be okay for this example. 

   .. image:: images/a1.png
      :height: 400px 
      :alt: Eclipse Tutorial Screenshot 1
      :align: center

#. Leave everything else by default. Press **Finish**. 

   .. image:: images/a2.png
      :height: 400px 
      :alt: Eclipse Tutorial Screenshot 2
      :align: center

#. Your project (in this case DisplayImage) should appear in the **Project Navigator** (usually at the left side of your window).

   .. image:: images/a3.png
      :height: 400px 
      :alt: Eclipse Tutorial Screenshot 3
      :align: center


#. Now, let's add a source file using OpenCV:

   * Right click on **DisplayImage** (in the Navigator). **New -> Folder** . 

     .. image:: images/a4.png
        :height: 400px 
        :alt: Eclipse Tutorial Screenshot 4
        :align: center

   * Name your folder **src** and then hit **Finish**

     .. image:: images/a5.png
        :height: 400px 
        :alt: Eclipse Tutorial Screenshot 5
        :align: center

   * Right click on your newly created **src** folder. Choose **New source file**:

     .. image:: images/a6.png
        :height: 400px 
        :alt: Eclipse Tutorial Screenshot 6
        :align: center

   * Call it **DisplayImage.cpp**. Hit **Finish**

     .. image:: images/a7.png
        :height: 400px 
        :alt: Eclipse Tutorial Screenshot 7
        :align: center

#. So, now you have a project with a empty .cpp file. Let's fill it with some sample code (in other words, copy and paste the snippet below):

   .. code-block:: cpp

      #include <cv.h>
      #include <highgui.h>

      using namespace cv;

      int main( int argc, char** argv )
      {
        Mat image;
        image = imread( argv[1], 1 );

        if( argc != 2 || !image.data )
          { 
            printf( "No image data \n" );
            return -1; 
          }

        namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
        imshow( "Display Image", image );

        waitKey(0);

        return 0;
      }

#. We are only missing one final step: To tell OpenCV where the OpenCV headers and libraries are. For this, do the following:

    * Go to  **Project-->Properties**

     .. image:: images/a8.png
        :height: 400px 
        :alt: Eclipse Tutorial Screenshot 8
        :align: center

    * In **C/C++ Build**, click on **Settings**. At the right, choose the **Tool Settings** Tab. Here we will enter the headers and libraries info:

      a. In **GCC C++ Compiler**, go to **Includes**. In **Include paths(-l)** you should include the path of the folder where opencv was installed. In our example, this is ``/usr/local/include/opencv``.

         .. image:: images/a9.png
            :height: 400px 
            :alt: Eclipse Tutorial Screenshot 9
            :align: center

         .. note::
            If you do not know where your opencv files are, open the **Terminal** and type: 

            .. code-block:: bash

               pkg-config --cflags opencv

            For instance, that command gave me this output:

            .. code-block:: bash

               -I/usr/local/include/opencv -I/usr/local/include 


      b. Now go to **GCC C++ Linker**,there you have to fill two spaces:

         First in **Library search path (-L)** you have to write the path to where the opencv libraries reside, in my case the path is:
         ::
          
            /usr/local/lib
          
         Then in **Libraries(-l)** add the OpenCV libraries that you may need. Usually just the 3 first on the list below are enough (for simple applications) . In my case, I am putting all of them since I plan to use the whole bunch:


         opencv_core      
         opencv_imgproc     
         opencv_highgui
         opencv_ml       
         opencv_video      
         opencv_features2d
         opencv_calib3d   
         opencv_objdetect   
         opencv_contrib
         opencv_legacy    
         opencv_flann

         .. image:: images/a10.png
             :height: 400px 
             :alt: Eclipse Tutorial Screenshot 10
             :align: center 
             
         If you don't know where your libraries are (or you are just psychotic and want to make sure the path is fine), type in **Terminal**:

         .. code-block:: bash
         
            pkg-config --libs opencv


         My output (in case you want to check) was:
         .. code-block:: bash
            
            -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann  

         Now you are done. Click **OK**

    * Your project should be ready to be built. For this, go to **Project->Build all**   

      .. image:: images/a11.png
         :height: 400px 
         :alt: Eclipse Tutorial Screenshot 11
         :align: center 

      In the Console you should get something like 

      .. image:: images/a12.png
         :height: 200px 
         :alt: Eclipse Tutorial Screenshot 12
         :align: center 

      If you check in your folder, there should be an executable there.

Running the executable
========================

So, now we have an executable ready to run. If we were to use the Terminal, we would probably do something like:

.. code-block:: bash

   cd <DisplayImage_directory>
   cd src
   ./DisplayImage ../images/HappyLittleFish.jpg

Assuming that the image to use as the argument would be located in <DisplayImage_directory>/images/HappyLittleFish.jpg. We can still do this, but let's do it from Eclipse:


#. Go to **Run->Run Configurations** 

   .. image:: images/a13.png
      :height: 300px 
      :alt: Eclipse Tutorial Screenshot 13
      :align: center 

#. Under C/C++ Application you will see the name of your executable + Debug (if not, click over C/C++ Application a couple of times). Select the name (in this case **DisplayImage Debug**). 

#. Now, in the right side of the window, choose the **Arguments** Tab. Write the path of the image file we want to open (path relative to the workspace/DisplayImage folder). Let's use **HappyLittleFish.jpg**:

   .. image:: images/a14.png
      :height: 300px 
      :alt: Eclipse Tutorial Screenshot 14
      :align: center 

#. Click on the **Apply** button and then in Run. An OpenCV window should pop up with the fish image (or whatever you used).

   .. image:: images/a15.png
      :alt: Eclipse Tutorial Screenshot 15
      :align: center 


#. Congratulations! You are ready to have fun with OpenCV using Eclipse.