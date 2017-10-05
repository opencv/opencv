Creating a 3D histogram {#tutorial_histo3D}
================

Goal
----

In this tutorial you will learn how to

-   Create your own callback keyboard function for viz window.
-   Show your 3D histogram in a viz window.

Code
----

You can download the code from [here ](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/viz/histo3D.cpp).
@include samples/cpp/tutorial_code/viz/histo3D.cpp

Explanation
-----------

Here is the general structure of the program:

-   You can give full path to an image in command line
    @code{.cpp}
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    String nomFic = parser.get<String>(0);
    Mat img;
    if (nomFic.length() != 0)
    {
        img = imread(nomFic, IMREAD_COLOR);
        if (img.empty())
        {
            cout << "Image does not exist!";
            return 0;
        }
    }
    @endcode
    or without path, a synthetic image is generated with pixel values are a gaussian distribution center(60+/-10,40+/-5,50+/-20) in first quadrant,
    (160+/-20,10+/-5,50+/-10) in second quadrant, (90+/-10,100+/-20,50+/-20) in third quadrant, (100+/-10,10+/-5,150+/-40) in last quadrant.
    @code{.cpp}
    else
    {
        img = Mat(512,512,CV_8UC3);
        parser.printMessage();
        RNG r;
        r.fill(img(Rect(0, 0, 256, 256)), RNG::NORMAL, Vec3b(60, 40, 50), Vec3b(10, 5, 20));
        r.fill(img(Rect(256, 0, 256, 256)), RNG::NORMAL, Vec3b(160, 10, 50), Vec3b(20, 5, 10));
        r.fill(img(Rect(0, 256, 256, 256)), RNG::NORMAL, Vec3b(90, 100, 50), Vec3b(10, 20, 20));
        r.fill(img(Rect(256, 256, 256, 256)), RNG::NORMAL, Vec3b(100, 10, 150), Vec3b(10, 5, 40));
    }
    @endcode
    Image tridimensional histogram is calculated using opencv calcHist and normalize between 0 and 100.
    @code{.cpp}
    float hRange[] = { 0, 256 };
    const float* etendu[] = { hRange, hRange,hRange };
    int hBins = 32;
    int tailleHist[] = { hBins, hBins , hBins  };
    int canaux[] = { 2, 1,0 };
    calcHist(&img, 1, canaux, Mat(), h.histogram, 3, tailleHist, etendu, true, false);
    normalize(h.histogram, h.histogram, 100.0/(img.total()), 0, cv::NormTypes::NORM_MINMAX, -1, cv::Mat());
    minMaxIdx(h.histogram,NULL,&h.maxH,NULL,NULL);
    @endcode
    channel are 2, 1 and 0 to synchronise color with Viz axis color in objetc viz::WCoordinateSystem.

    A slidebar is inserted in image window. Init slidebar value is 90, it means that only histogram cell greater than 9/100000.0 (23 pixels for an 512X512 pixels) will be display.
    @code{.cpp}
    namedWindow("Image");
    imshow("Image",img);
    AddSlidebar("threshold","Image",0,100,h.seuil,&h.seuil, UpdateThreshold,&h);
    waitKey(30);
    @endcode
    We are ready to open a viz window with a callback function to capture keyboard event in viz window. Using Viz::spinonce enable keyboard event to be capture in imshow window too.
    @code{.cpp}
    h.fen3D = new viz::Viz3d("3D Histogram");
    h.nbWidget=0;
    h.fen3D->registerKeyboardCallback(KeyboardViz3d,&h);
    DrawHistogram3D(h);
    while (h.code!=27)
    {
        h.fen3D->spinOnce(1);
        if (h.status)
            DrawHistogram3D(h);
        if (h.code!=27)
            h.code= waitKey(30);
    }
    @endcode
    The function DrawHistogram3D processes histogram Mat to display it in a Viz window. Number of plan, row and column in three dimensional Mat ca be found using  this code :
    @code{.cpp}
    int planSize = h.histogram.step1(0);
    int cols = h.histogram.step1(1);
    int rows = planSize / cols;
    int plans = h.histogram.total() / planSize;
    h.fen3D->removeAllWidgets();
    h.nbWidget=0;
    if (h.nbWidget==0)
        h.fen3D->showWidget("Axis", viz::WCoordinateSystem(10));
    @endcode
    To get histogram value at a specific location we use at method with three arguments k, i and j where k is plane number, i row number and j column number.
    @code{.cpp}
    for (int k = 0; k < plans; k++)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double x = h.histogram.at<float>(k, i, j);
                if (x >= h.threshold)
                {
                    double r=std::max(x/h.maxH,0.1);
                    viz::WCube s(Point3d(k - r / 2, i - r / 2, j - r / 2), Point3d(k + r / 2, i + r / 2, j + r / 2), false, viz::Color(j / double(plans) * 255, i / double(rows) * 255, k / double(cols) * 255));
                    h.fen3D->showWidget(format("I3d%d", h.nbWidget++), s);
                }
            }
        }
    }
    @endcode

-   Callback function
    Principle are as mouse callback function. Key code pressed is in field code of class viz::KeyboardEvent.
    @code{.cpp}
    void  KeyboardViz3d(const viz::KeyboardEvent &w, void *t)
    {
       Histo3DData *x=(Histo3DData *)t;
       if (w.action)
         cout << "you pressed "<< w.symbol<< " in viz window "<<x->fen3D->getWindowName()<<"\n";
       x->code= w.code;
       switch (w.code) {
       case '/':
             x->status=true;
             x->threshold *= 0.9;
         break;
       case '*':
         x->status = true;
             x->threshold *= 1.1;
         break;

        }
       if (x->status)
       {
         cout <<  x->threshold << "\n";
         DrawHistogram3D(*x);
       }
    }
    @endcode

Results
-------

Here is the result of the program with no argument and threshold equal to 50.

![](images/histo50.png)
