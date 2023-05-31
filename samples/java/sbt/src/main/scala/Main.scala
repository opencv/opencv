/*
 * The main runner for the Java demos.
 * Demos whose name begins with "Scala" are written in the Scala language,
 * demonstrating the generic nature of the interface.
 * The other demos are in Java.
 * Currently, all demos are run, sequentially.
 *
 * You're invited to submit your own examples, in any JVM language of
 * your choosing so long as you can get them to build.
 */

import org.opencv.core.Core

object Main extends App {
  // We must load the native library before using any OpenCV functions.
  // You must load this library _exactly once_ per Java invocation.
  // If you load it more than once, you will get a java.lang.UnsatisfiedLinkError.
  System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

  ScalaCorrespondenceMatchingDemo.run()
  ScalaDetectFaceDemo.run()
  new DetectFaceDemo().run()
}
