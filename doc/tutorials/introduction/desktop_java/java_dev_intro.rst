
.. _Java_Dev_Intro:


Introduction to Java Development
********************************

Last updated: 20 January, 2013.

As of OpenCV 2.4.4, OpenCV supports desktop Java development using nearly the same interface as for Android development.
This guide will take you from the raw OpenCV source to a working Java application.
We will use the `Simple Build Tool (SBT) <http://www.scala-sbt.org/>`_ to build our Java application and to generate an Eclipse project.

For further reading after this guide, look at the Android tutorials. The interfaces for desktop Java and Android are nearly identical. You may also consult the `Java wiki page <http://code.opencv.org/projects/opencv/wiki/Java_API_howto>`_.

What we'll do in this guide
***************************

In this guide, we will:

* Download the OpenCV source from Github

* Build OpenCV and the Java wrappers and install OpenCV

* Create an SBT project and use it to generate an Eclipse project

* Copy the generated OpenCV jar into the SBT project and write a simple OpenCV application

The same process was used to create the samples in the samples/java folder of the OpenCV repository, so consult those files if you get lost.

Download the OpenCV source from Github
**************************************

The OpenCV repository is hosted on `Github <https://github.com/Itseez/opencv/>`_.

To download the repository, enter the following in a terminal:

        .. code-block:: bash

           git clone https://github.com/Itseez/opencv.git

This will create a directory "opencv". Enter it.

Build OpenCV and the Java wrappers and install OpenCV
*****************************************************

Java is currently only supported on the 2.4 git branch. This may change in the future, but for now you must switch to that branch:

        .. code-block:: bash

           git checkout 2.4

To build the Java wrapper, first make sure you have Java 1.6+ and Apache Ant installed. Then invoke cmake with the DBUILD_opencv_java flag turned ON:

        .. code-block:: bash

           mkdir build
           cd build
           cmake .. -DBUILD_opencv_java=ON

Examine the output of CMake and ensure "java" is one of the modules "To be built". If not, it's likely you're missing a dependency. You should troubleshoot by looking through the CMake output for any Java-related tools that aren't found and installing them.

Now build and install everything:

        .. code-block:: bash

           make -j8
           sudo make install

In addition to building and installing C++ libraries to your system, this will write a jar containing the Java interface to bin/opencv_<version>.jar. We'll use this jar later.

Create an SBT project and use it to generate an Eclipse project
*****************************************************************

Now we'll create a simple Java application using SBT. This serves as a brief introduction to those unfamiliar with this build tool. We're using SBT because it is particularly easy and powerful.

First, download and install SBT using the instructions `here <http://www.scala-sbt.org/>`_.

Next, navigate to a new directory where you'd like the application source to live (outside opencv). Let's call it "JavaSample" and create a directory for it:

        .. code-block:: bash

           cd <somewhere outside opencv>
           mkdir JavaSample

Now we will create the necessary folders and an SBT project:

        .. code-block:: bash

           cd JavaSample
           mkdir -p src/main/java # This is where SBT expects to find Java sources
           mkdir project # This is where the build definitions live

Now open project/build.scala in your favorite editor and paste the following. It defines your project:

        .. code-block:: scala

           import sbt._
	   import Keys._

	   object JavaSampleBuild extends Build {
	     def scalaSettings = Seq(
	       scalaVersion := "2.10.0",
	       scalacOptions ++= Seq(
	         "-optimize",
	         "-unchecked",
	         "-deprecation"
	       )
	     )

	     def buildSettings = 
	       Project.defaultSettings ++ 
	       scalaSettings

	     lazy val root = {
	       val settings = buildSettings ++ Seq(name := "JavaSample")
	       Project(id = "JavaSample", base = file("."), settings = settings)
	     }
	   }

Now edit project/plugins.sbt and paste the following. This will enable auto-generation of an Eclipse project:

        .. code-block:: scala

           addSbtPlugin("com.typesafe.sbteclipse" % "sbteclipse-plugin" % "2.1.0")

Now run sbt from the JavaSample root and from within SBT run "eclipse" to generate an eclipse project:

        .. code-block:: bash

           sbt # Starts the sbt console
           > eclipse # Running "eclipse" from within the sbt console

You should see something like this:

     .. image:: images/sbt_eclipse.png
        :alt: SBT output
        :align: center

You can now import the SBT project using "Import ... -> Existing projects into workspace" from Eclipse. Whether you actually do this is optional for the guide; we'll be using SBT to build the project, so if you choose to use Eclipse it will just be as a text editor.

To test everything is working, create a simple "Hello OpenCV" application. Do this by creating a file "src/main/java/HelloOpenCV.java" with the following contents:

        .. code-block:: java

           public class HelloOpenCV {
	     public static void main(String[] args) {
	       System.out.println("Hello, OpenCV");
	     }
	   }

Now execute "run" from the sbt console, or more concisely, run "sbt run" from the command line:

        .. code-block:: bash

           sbt run

You should see something like this:

     .. image:: images/sbt_run.png
        :alt: SBT run
        :align: center

Copy the OpenCV jar and write a simple application
********************************************************

Now we'll create a simple face detection application using OpenCV.

First, create a "lib/" folder and copy the OpenCV jar into it. By default, SBT adds jars in the lib folder to the Java library search path. You can optionally rerun "sbt eclipse" to update your Eclipse project.

        .. code-block:: bash

           mkdir lib
           cp <opencv_dir>/build/bin/opencv_<version>.jar lib/
           sbt eclipse

Next, create the directory src/main/resources and download this Lena image into it:

     .. image:: images/lena.bmp
        :alt: Lena
        :align: center

Make sure it's called "lena.bmp". Items in the resources directory are available to the Java application at runtime.

Next, copy lbpcascade_frontalface.xml into the resources directory:

        .. code-block:: bash

           cp <opencv_dir>/data/lbpcascades/lbpcascade_frontalface.xml src/main/resources/

Now modify src/main/java/HelloOpenCV.java so it contains the following Java code:

.. code-block:: java

   import org.opencv.core.Core;
   import org.opencv.core.Mat;
   import org.opencv.core.MatOfRect;
   import org.opencv.core.Point;
   import org.opencv.core.Rect;
   import org.opencv.core.Scalar;
   import org.opencv.highgui.Highgui;
   import org.opencv.objdetect.CascadeClassifier;
   
   //
   // Detects faces in an image, draws boxes around them, and writes the results
   // to "faceDetection.png".
   //
   class DetectFaceDemo {
     public void run() {
       System.out.println("\nRunning DetectFaceDemo");

       // Create a face detector from the cascade file in the resources
       // directory.
       CascadeClassifier faceDetector = new CascadeClassifier(getClass().getResource("/lbpcascade_frontalface.xml").getPath());
       Mat image = Highgui.imread(getClass().getResource("/lena.bmp").getPath());

       // Detect faces in the image.
       // MatOfRect is a special container class for Rect.
       MatOfRect faceDetections = new MatOfRect();
       faceDetector.detectMultiScale(image, faceDetections);

       System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

       // Draw a bounding box around each face.
       for (Rect rect : faceDetections.toArray()) {
           Core.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
       }

       // Save the visualized detection.
       String filename = "faceDetection.png";
       System.out.println(String.format("Writing %s", filename));
       Highgui.imwrite(filename, image);
     }
   }

   public class HelloOpenCV {
     public static void main(String[] args) {
       System.out.println("Hello, OpenCV");

       // Load the native library.
       System.loadLibrary("opencv_java");
       new DetectFaceDemo().run();
     }
   } 

Note the call to "System.loadLibrary("opencv_java")". This command must be executed exactly once per Java process prior to using any native OpenCV methods. If you don't call it, you will get UnsatisfiedLink errors. You will also get errors if you try to load OpenCV when it has already been loaded.

Now run the face detection app using "sbt run":

        .. code-block:: bash

           sbt run

You should see something like this:

     .. image:: images/sbt_run_face.png
        :alt: SBT run
        :align: center

It should also write the following image to faceDetection.png:

     .. image:: images/faceDetection.png
        :alt: Detected face
        :align: center
