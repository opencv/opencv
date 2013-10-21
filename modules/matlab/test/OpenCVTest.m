% Matlab binding test cases
% Uses Matlab's builtin testing framework
classdef OpenCVTest < matlab.unittest.TestCase

  methods(Test)

    % -------------------------------------------------------------------------
    % EXCEPTIONS
    % Check that errors and exceptions are thrown correctly
    % -------------------------------------------------------------------------

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

    % -------------------------------------------------------------------------
    % SIZES AND FILLS
    % Check that matrices are correctly filled and resized
    % -------------------------------------------------------------------------

    % check that a matrix is correctly filled with random numbers
    function randomFill(testcase)
      sz = [7 11];
      mat = zeros(sz);
      mat = cv.randn(mat, 0, 1);
      testcase.verifyEqual(size(mat), sz, 'Matrix should not change size');
      testcase.verifyNotEqual(mat, zeros(sz), 'Matrix should be nonzero');
    end

    function transpose(testcase)
      m = randn(19, 81);
      mt1 = transpose(m);
      mt2 = cv.transpose(m);
      testcase.verifyEqual(size(mt1), size(mt2), 'Matrix transposed to incorrect dimensionality');
      testcase.verifyLessThan(norm(mt1 - mt2), 1e-8, 'Too much precision lost in tranposition');
    end

    % multiple return
    function multipleReturn(testcase)
      A = randn(10);
      A = A'*A;
      [V1, D1] = eig(A); D1 = diag(D1);
      [~, D2, V2] = cv.eigen(A);
      testcase.verifyLessThan(norm(V1 - V2), 1e-6, 'Too much precision lost in eigenvectors');
      testcase.verifyLessThan(norm(D1 - D2), 1e-6, 'Too much precision lost in eigenvalues');
    end

    % complex output from SVD
    function complexOutputSVD(testcase)
      A = randn(10);
      [V1, D1] = eig(A);
      [~, D2, V2] = cv.eigen(A);
      testcase.verifyTrue(~isreal(V2) && size(V2,3) == 1, 'Output should be complex');
      testcase.verifyLessThan(norm(V1 - V2), 1e-6, 'Too much precision lost in eigenvectors');
    end

    % complex output from Fourier Transform
    function complexOutputFFT(testcase)
      A = randn(10);
      F1 = fft2(A);
      F2 = cv.dft(A, cv.DFT_COMPLEX_OUTPUT);
      testcase.verifyTrue(~isreal(F2) && size(F2,3) == 1, 'Output should be complex');
      testcase.verifyLessThan(norm(F1 - F2), 1e-6, 'Too much precision lost in eigenvectors');
    end

    % -------------------------------------------------------------------------
    % TYPE CASTS
    % Check that types are correctly cast
    % -------------------------------------------------------------------------

    % -------------------------------------------------------------------------
    % PRECISION
    % Check that basic operations are performed with sufficient precision
    % -------------------------------------------------------------------------

    % check that summing elements is within reasonable precision
    function sumElements(testcase)
      a = randn(5000);
      b = sum(a(:));
      c = cv.sum(a);
      testcase.verifyLessThan(norm(b - c), 1e-8, 'Matrix reduction with insufficient precision');
    end


    % check that adding two matrices is within reasonable precision
    function addPrecision(testcase)
      a = randn(50);
      b = randn(50);
      c = a+b;
      d = cv.add(a, b);
      testcase.verifyLessThan(norm(c - d), 1e-8, 'Matrices are added with insufficient precision');
    end

    % check that performing gemm is within reasonable precision
    function gemmPrecision(testcase)
      a = randn(10, 50);
      b = randn(50, 10);
      c = randn(10, 10);
      alpha = 2.71828;
      gamma = 1.61803;
      d = alpha*a*b + gamma*c;
      e = cv.gemm(a, b, alpha, c, gamma);
      testcase.verifyLessThan(norm(d - e), 1e-8, 'Matrices are multiplied with insufficient precision');
    end


    % -------------------------------------------------------------------------
    % MISCELLANEOUS
    % Miscellaneous tests
    % -------------------------------------------------------------------------

    % check that cv::waitKey waits for at least specified time
    function waitKey(testcase)
      tic();
      cv.waitKey(500);
      elapsed = toc();
      testcase.verifyGreaterThan(elapsed, 0.5, 'Elapsed time should be at least 0.5 seconds');
    end

    % check that highgui window can be created and destroyed
    function createAndDestroyWindow(testcase)
      try
        cv.namedWindow('test window');
      catch
        testcase.verifyFail('could not create window');
      end

      try
        cv.destroyWindow('test window');
      catch
        testcase.verifyFail('could not destroy window');
      end
      testcase.verifyTrue(true);
    end

  end
end
