/*
 * =====================================================================================
 *
 *       Filename:  cpptypes.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  09/24/13 20:14:24
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <opencv_generated.hpp>
#include <cpptypes.hpp>

extern "C" {
string* std_create_string(char* s, int len) {

    return new string(*s, len);

}

vector_int* std_create_vector_int(int* is, size_t len) {

    return new vector_int(is, is + len);

}
}
