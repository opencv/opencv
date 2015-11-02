 /*
 * solves equations x^4 - x^2 = 0, and x^2 - 2x + 1 = 0
 */

#include "opencv2/core/core.hpp"
#include <iostream>


int main(void)
{
        cv::Mat coefs = (cv::Mat_<float>(1,5) << 0, 0, -1, 0, 1 );
        std::cout << "x^4 - x^2 = 0\n\n" << "Coefficients: " << coefs << "\n" << std::endl;
        cv::Mat r;

        double prec;
        prec = cv::solvePoly(coefs, r);
        std::cout << "Preciseness = " << prec << std::endl;
        std::cout << "roots after 1000 iterations:\n" << r << "\n" << std::endl;

        prec = cv::solvePoly(coefs, r, 9999);
        std::cout << "Preciseness = " << prec << std::endl;
        std::cout << "roots after 9999 iterations:\n" << r << "\n" << std::endl;

        std::cout << "\n---------------------------------------\n" << std::endl;

        coefs = (cv::Mat_<float>(1,3) << 1, -2, 1 );
        std::cout << "x^2 - 2x + 1 = 0\n\n" << "Coefficients: " << coefs << "\n" << std::endl;

        prec = cv::solvePoly(coefs, r);
        std::cout << "Preciseness = " << prec << std::endl;
        std::cout << "roots after 1000 iterations:\n" << r << "\n" << std::endl;

        prec = cv::solvePoly(coefs, r, 9999);
        std::cout << "Preciseness = " << prec << std::endl;
        std::cout << "roots after 9999 iterations:\n" << r << "\n" << std::endl;

    return 0;
}