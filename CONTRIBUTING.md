# Contribution guidelines

This file contain the basic steps you need to perform before opening an issue or pull request. Please make sure that you follow these guidelines.

## Making sure you are in the correct place

Before you open up anything on the OpenCV github page, be sure that you are at the right place with your problem.

* Check if your bug still exists. This can be done by building the latest [2.4 branch](https://github.com/Itseez/opencv/tree/2.4) or the [latest master branch](https://github.com/Itseez/opencv), and making sure that the error is still reproducable there. We do not fix bugs that only affect deprecated versions like OpenCV2.1 for example.

* If you have a question about the software, then this is **NOT** the right place. You should open up a question at the [OpenCV Q&A forum](http://answers.opencv.org/questions/). In order to post a decent question from the start, feel free to read the official [forum guidelines](http://answers.opencv.org/faq/).

## Opening up an issue

If you have found an actual bug, for example after discussing on the Q&A forum with some peer developers, then go ahead and open up an issue on this Github page. Make sure that you check the following things:

 1. Use the search button at the issues page on Github and make sure that there is not an issue open for your problem. If so add your information there as a response.
 2. If you have a bug related to the basic OpenCV repository, then open up an issue right [here](https://github.com/Itseez/opencv/issues/).
 3. If you have a bug related to the OpenCV contribution repository with new developed modules, then open up an issue right [here](https://github.com/Itseez/opencv_contrib/issues/).
 4. Make sure that you provide enough information. You should clearly state which OpenCV version you are using, what your system configuration is (OS, hardware specifications, extra libraries used, ...). The golden rule here is, the more the better.

Now wait until someone takes up your issue and tries to help you debug and solve the problem.

## Opening up a pull request

If you find the solution to the problem yourself, then you can provide your own pull request. A guide for doing so can be found [here](https://github.com/Itseez/opencv/wiki/How_to_contribute). Keep in mind the following points of attention:

* Make sure that the patch builds perfectly fine on your local OpenCV system.
* Provide a title that actually means something.
* Add a reference to the issue that is related to this pull request.

Now you can check the status of your pull request at the [OpenCV buildbot](http://pullrequest.opencv.org/#/summary/). If any error pops up, then please address it!

Now wait for an official OpenCV developer to help you out, provide comments on your fix and once everything is fine, it will all be merged.

