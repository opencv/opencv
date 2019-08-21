Launching Viz {#tutorial_launching_viz}
=============

Goal
----

In this tutorial you will learn how to

-   Open a visualization window.
-   Access a window by its name.
-   Start event loop.
-   Start event loop for a given amount of time.

Code
----

You can download the code from [here ](https://github.com/opencv/opencv/tree/3.4/samples/cpp/tutorial_code/viz/launching_viz.cpp).
@include samples/cpp/tutorial_code/viz/launching_viz.cpp

Explanation
-----------

Here is the general structure of the program:

-   Create a window.
    @code{.cpp}
    /// Create a window
    viz::Viz3d myWindow("Viz Demo");
    @endcode
-   Start event loop. This event loop will run until user terminates it by pressing **e**, **E**,
    **q**, **Q**.
    @code{.cpp}
    /// Start event loop
    myWindow.spin();
    @endcode
-   Access same window via its name. Since windows are implicitly shared, **sameWindow** is exactly
    the same with **myWindow**. If the name does not exist, a new window is created.
    @code{.cpp}
    /// Access window via its name
    viz::Viz3d sameWindow = viz::getWindowByName("Viz Demo");
    @endcode
-   Start a controlled event loop. Once it starts, **wasStopped** is set to false. Inside the while
    loop, in each iteration, **spinOnce** is called to prevent event loop from completely stopping.
    Inside the while loop, user can execute other statements including those which interact with the
    window.
    @code{.cpp}
    /// Event loop is over when pressed q, Q, e, E
    /// Start event loop once for 1 millisecond
    sameWindow.spinOnce(1, true);
    while(!sameWindow.wasStopped())
    {
        /// Interact with window

        /// Event loop for 1 millisecond
        sameWindow.spinOnce(1, true);
    }
    @endcode

Results
-------

Here is the result of the program.

![](images/window_demo.png)
