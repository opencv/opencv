/*
 * =====================================================================================
 *
 *       Filename:  cpptypes.hpp
 *
 *    Description:  Interface file for C++ types in std
 *
 *        Version:  1.0
 *        Created:  09/24/13 11:54:11
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Arjun Comar 
 *
 * =====================================================================================
 */

#include <opencv_generated.hpp>

extern "C" {
string* std_create_string(char* s, int len);
vector_int* std_create_vector_int(int* is, size_t len);
}
