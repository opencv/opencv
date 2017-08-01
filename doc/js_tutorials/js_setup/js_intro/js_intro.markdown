Introduction to OpenCV-JavaScript and Tutorials {#tutorial_js_intro}
=======================================

OpenCV
------

OpenCV was started at Intel in 1999 by **Gary Bradsky**, and the first release came out in 2000.
**Vadim Pisarevsky** joined Gary Bradsky to manage Intel's Russian software OpenCV team. In 2005,
OpenCV was used on Stanley, the vehicle that won the 2005 DARPA Grand Challenge. Later, its active
development continued under the support of Willow Garage with Gary Bradsky and Vadim Pisarevsky
leading the project. OpenCV now supports a multitude of algorithms related to Computer Vision and
Machine Learning and is expanding day by day.

OpenCV supports a wide variety of programming languages such as C++, Python, Java, JavaScript etc.,
and is available on different platforms including Windows, Linux, OS X, Android, iOS and Web Browsers.
Interfaces for high-speed GPU operations based on CUDA and OpenCL are also under active development.

OpenCV-JavaScript
-------------

Web is the most ubiquitous open computing platform. With HTML5 standards get implemented in every browser, web applications are able to render online video with HTML5 video tag, capture the video from web camera via WebRTC API and access each pixel of video frame via canvas API. With these huge multimedia contents available on web, web developers are in need of wide array of image and vision processing algorithms in JavaScript to build innovative applications, such as facial detection in live WebRTC streaming. And this requirement is even more essential for emerging usages on web, such as Web Virtual Reality and Augmented Reality (WebVR). All these use cases demand efficient implementations of computer-intensive vision kernels on web.

[Emscripten](http://kripken.github.io/emscripten-site) is an LLVM-to-JavaScript compiler. It takes LLVM bitcode - which can be generated from C/C++, using clang, and compiles that into a subset of JavaScript - asm.js. asm.js is an intermediate programming language designed to allow computer software written in languages such as C to be run as web applications while maintaining performance characteristics considerably better than standard JavaScript, the typical language used for such applications. The performance of asm.js is improved by limiting language features of JavaScript to those amenable to ahead-of-time optimization and other performance improvements.

OpenCV-JavaScript is the JavaScript API binding for OpenCV. It allows emerging web applications with multimedia processing to benefit from the wide variety of vision functions available in OpenCV. OpenCV-JavaScript leverages Emscripten, compiles OpenCV functions' implementation into asm.js and exposes JavaScript APIs for web applications use. The future version will support WebAssembly and other acceleration APIs available on Web.

OpenCV-JavaScript is based on [OpenCV.js](https://github.com/ucisysarch/opencvjs), the pioneer project initiated in Parallel Architectures and Systems Group at University of California Irvine.


OpenCV-JavaScript Tutorials
-----------------------

OpenCV introduces a new set of tutorials which will guide you through various functions available in
OpenCV-JavaScript. **This guide is mainly focused on OpenCV 3.x version**.

The purposes of OpenCV-JavaScript tutorials include
-# Help with adaptability of OpenCV in web development
-# Help the web community, developers and computer vision researcher to interactively access variety of web based OpenCV examples to help them understand specific vision algorithm.

As OpenCV-JavaScript is able to run directly inside browser, the OpenCV-JavaScript tutorial web pages are intuitive and interactive. For example, by using WebRTC API and evaluation of JavaScript code, it would allow developers to change the parameters of CV functions and even do live CV coding on web pages to see the results in real time.

Prior knowledge of JavaScript and web application development is recommended as they won't be covered in this guide.

Contributors
------------

Below is the list of contributors of OpenCV-JavaScript bindings and tutorials.

-#  Sajjad Taheri (Google Summer of Code 2017 mentor, University of California, Irvine)
-#  Congxiang Pan (Google Summer of Code 2017 intern, Shanghai Jiao Tong University)
-#  Wenyao Gan (Shanghai Jiao Tong University)
-#  Mohammad Reza Haghighat (Intel Corporation)
-#  Ningxin Hu (Intel Corporation)