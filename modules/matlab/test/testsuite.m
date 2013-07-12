% add the opencv bindings folder
addpath ..

%setup the tests
opencv_tests = OpenCVTest();

%run the tests
result = run(opencv_tests);

% shutdown
exit();
