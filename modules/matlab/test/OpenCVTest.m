% Matlab binding test cases
% Uses Matlab's builtin testing framework
classdef OpenCVTest < matlab.unittest.TestCase

  methods(Test)
    
    % check that std exception is thrown
    function stdException(testcase)
      try
        std_exception();
        testcase.verifyFail();
      catch
        % TODO: Catch more specific exception
        testcase.verifyTrue(true);
      end
    end

    % check that OpenCV exceptions are correctly caught
    function cvException(testcase)
      try
        cv_exception();
        testcase.verifyFail();
      catch
        % TODO: Catch more specific exception
        testcase.verifyTrue(true);
      end
    end

    % check that all exceptions are caught
    function allException(testcase)
      try
        exception();
        testcase.verifyFail();
      catch
        % TODO: Catch more specific exception
        testcase.verifyTrue(true);
      end
    end

    % check that a matrix is correctly filled with random numbers
    function randomFill(testcase)
      sz = [7 11];
      mat = zeros(sz);
      mat = cv.randn(mat, 0, 1);
      testcase.verifyEqual(size(mat), sz, 'Matrix should not change size');
      testcase.verifyNotEqual(mat, zeros(sz), 'Matrix should be nonzero');
    end

    % check that cv::waitKey waits for at least specified time
    function waitKey(testcase)
      tic();
      cv.waitKey(500);
      elapsed = toc();
      testcase.verifyGreaterThan(elapsed, 0.5, 'Elapsed time should be at least 0.5 seconds');
    end
  end
end
