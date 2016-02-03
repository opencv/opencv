/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmVR.h"
#include "gdcmAttribute.h"
#include "gdcmByteValue.h"

#include <sstream>
#include <iostream>
#include <iomanip>

#include <cmath> // atof
#include <algorithm> // std::rand()
#include <limits> // min_exponent10 and max_exponent10


// WARNING: The number of digits in exponent can be dependent from compiler.
//          gcc uses 2 digits if the exponent is < 100 and 3 digits if >=, but
//          some compilers (i.e. MSVC) may always use 3 digits in exponent.
//          If some other compiler with this behaviour is detected, should be
//          added here.
#if defined(_MSC_VER)
 #define ALWAYS_3_DIGITS_IN_EXPONENT
#endif


#ifdef ALWAYS_3_DIGITS_IN_EXPONENT
 #define MIN_NEGATIVE_EXP 6 //MSVC always use 3 digits in exponent.
#else
 #define MIN_NEGATIVE_EXP 5
#endif

template < typename Float >
std::string to_string ( Float data ) {

    std::stringstream in;
    // in.imbue(std::locale::classic()); // This is not required AFAIK

    unsigned long digits = 0;

    // 16 integer digits number or 15 integer digits negative number
    if ( (data >= 1e+15 && data < 1e16) || (data  <= -1e14 && data > -1e+15))
        in << std::fixed;
    else
    {
        digits = 15; // 16 - 1 (dot)

        // negative number
        if (data < 0)
            digits -= 1; // (minus)

        if (data != 0)
        {
            Float ldata = log10(fabs(data));

            // numbers that need std::scientific representation
            if ( ldata > 16 || (ldata > 15 && data < 0) || ldata < -MIN_NEGATIVE_EXP+1 )
            {
                in << std::scientific;
#ifdef ALWAYS_3_DIGITS_IN_EXPONENT
                digits -= 6; // (first digit + exponent)
#else
                digits -= 5; // (first digit + exponent)
                // 3 digits in exponent
                if ( ldata >= 100 || ldata < -99 )
                    digits -=1;
#endif
            }
            else if( ldata < 0){
				//since ldata is negative, to have the test pass,
				//the right casting has to be done to avoid a casting warning here
				unsigned long uldata = (unsigned long)(fabs(ldata)+1.0);
                digits -= uldata; // (zeros before first significant digit)
			}
        }
    }
    /*
    // I don't know if you really need this check
    unsigned long const max_digits =
    static_cast< unsigned long >(
    - std::log( std::numeric_limits<Float>::epsilon() )
    / std::log( 10.0 ) );
    digits = (digits > max_digits) ? max_digits : digits;
    */

    if ( in << std::dec << std::setprecision((int)digits) << data )
        return ( in.str() );
    else
        throw "Impossible Conversion"; // should not happen ...
}



bool checkerror(double d, std::string s)
{
	double theConverted = atof(s.c_str());
    double error = fabs(d - theConverted);

    int Log = (int)log10(fabs(d));
    int eo = ( Log - 14 );

    if ( Log <= -1 && Log >= -4 )
        eo = -13;
#ifdef ALWAYS_3_DIGITS_IN_EXPONENT
    else if ( Log >= 15 )
        eo = ( Log - 9);
#else
    else if ( Log >= 99 )
        eo = ( Log - 9 );
    else if ( Log >= 15 )
        eo = ( Log - 10);
#endif

    if (d<0)
        eo += 1;


    //if (error > pow(10., eo) )
	//pow will underflow at 10^-308, so errors lower than -308 will appear to be
	//larger than pow(10., eo), because the 'pow' result will be 0 in vs2010
	if (log10(error) > eo)
    {
        std::cout << "ERROR: Absoulte Error is too large (error = " << error << ", should be < " << pow(10., eo) << ")" << std::endl;
        return true;
    }
//    else if (error != 0.0) std::cout << "OK (error = " << error << ", is < " << pow(10, eo) << ")" << std::endl;

    return false;
}

bool checkerror(double d, std::string s, bool se)
{
    double error = fabs(d - atof( s.c_str() ));
    bool has_error = (error != 0);

    if (has_error)
    {
       std::cout << "\tError is: " << error;
    }
    std::cout << std::endl;

    if( has_error != se )
    {
        std::cout << "ERROR: has_error = " << has_error << " (should be " << se << ")" << std::endl;
        return true;
    }
    return checkerror(d,s);
}




/*
 d = double to test
 sz = size expected
 se = true if there should be an error
*/
bool singleTestDS(double d, int sz, bool se = false)
{
    bool fail = false;
    std::cout << "           -|----------------|-" << std::endl;
    std::string s = to_string<double>( d );
    std::cout << "  Result:    " << s << std::flush;

    if ( checkerror(d, s, se) )
        fail = true;

    assert(sz >= 0);
    if( s.size() != (unsigned int)sz )
    {
        std::cout << "ERROR: Size = " << s.size() << " (should be " << sz << ")" << std::endl;
        fail = true;
    }

    std::cout << std::endl;

    return fail;
}


#define TEST(x, y, z) { \
  std::cout << "  Testing:   " << #x << std::endl; \
  err_count += singleTestDS(x, y, z); \
  test_count++; }


/*
 * Test to make sure that double precision ieee 'double' is ok for DICOM VR = 'DS'
 */
int TestDS(int, char *[])
{
    int err_count = 0;
    int test_count = 0;

    TEST(               118.242525316066        , 16, false); // 3 digits + dot + 12 digits => 16 chars
    TEST(              -118.242525316066        , 16,  true); // minus + 3 digits + dot + 12 digits => 16 chars + ERROR
    TEST(               118.24252531606         , 15, false); // minus + 3 digits + dot + 11 digits => 16 chars
    TEST(              -118.24252531606         , 16, false); // minus + 3 digits + dot + 11 digits => 16 chars

    TEST(                 0.059303515816892     , 16,  true); // zero + dot + zero + 14 digits => 16 chars + ERROR
    TEST(                -0.059303515816892     , 16,  true); // minus + zero + dot + zero + 14 digits => 16 chars + ERROR
    TEST(                 0.05930351581689      , 16, false); // zero + dot + zero + 13 digits => 16 chars
    TEST(                -0.05930351581689      , 16,  true); // minus + zero + dot + zero + 13 digits => 16 chars + ERROR
    TEST(                 0.0593035158168       , 15, false); // zero + dot + zero + 12 digits => 15 chars
    TEST(                -0.0593035158168       , 16, false); // minus + zero + dot + zero + 12 digits => 16 chars

    TEST(                 0.00149700609543456   , 16,  true); // zero + dot + 2 zeros + 15 digits => 16 chars + ERROR
    TEST(                -0.00149700609543456   , 16,  true); // zero + dot + 2 zeros + 15 digits => 16 chars + ERROR
    TEST(                 0.0014970060954345    , 16,  true); // zero + dot + 2 zeros + 14 digits => 16 chars + ERROR
    TEST(                -0.0014970060954345    , 16,  true); // zero + dot + 2 zeros + 14 digits => 16 chars + ERROR
    TEST(                 0.001497006095434     , 16,  true); // zero + dot + 2 zeros + 13 digits => 16 chars + ERROR
    TEST(                -0.001497006095434     , 16,  true); // zero + dot + 2 zeros + 13 digits => 16 chars + ERROR
    TEST(                 0.00149700609543      , 16, false); // zero + dot + 2 zeros + 12 digits => 16 chars
    TEST(                -0.00149700609543      , 16,  true); // zero + dot + 2 zeros + 12 digits => 16 chars + ERROR
    TEST(                 0.0014970060954       , 15, false); // zero + dot + 2 zeros + 11 digits => 15 chars
    TEST(                -0.0014970060954       , 16, false); // zero + dot + 2 zeros + 11 digits => 16 chars
    TEST(                 0.000593035158168     , 16,  true); // zero + dot + 3 zeros + 12 digits => 16 chars + ERROR
    TEST(                 5.93035158168e-04     , 16,  true); // same number: cannot fit in 16 chars even in scientific notation (17 chars)
    TEST(                 0.00059303515816      , 16, false); // zero + dot + 3 zeros + 11 digits => 16 chars
    TEST(                -0.00059303515816      , 16,  true); // minus + zero + dot + 3 zeros + 11 digits => 16 chars + ERROR
    TEST(                -5.9303515816e-04      , 16,  true); // same number: cannot fit in 16 chars even in scientific notation (17 chars)
    TEST(                -0.0005930351581       , 16, false); // minus + zero + dot + 3 zeros + 10 digits => 16 chars

    TEST(                 0.0000593035158168    , 16,  true); // zero + dot + 4 zeros + 12 digits => 16 chars (w/ scientific notation) + ERROR
    TEST(                 0.00005930351581      , 16, false); // zero + dot + 4 zeros + 10 digits => 16 chars
    TEST(                -0.000059303515816     , 16,  true); // minus + zero + dot + 4 zeros + 10 digits => 16 chars (w/ scientific notation) + ERROR
    TEST(                -0.0000593035158       , 16, false); // minus + zero + dot + 4 zeros + 10 digits => 16 chars

#ifdef ALWAYS_3_DIGITS_IN_EXPONENT
    TEST(                 0.000059303515816     , 16, true); // zero + dot + 4 zeros + 11 digits => 16 chars (w/ scientific notation) + ERROR
    TEST(                -0.00005930351581      , 16, true); // minus + zero + dot + 4 zeros + 10 digits => 16 chars (w/ scientific notation) + ERROR
#else
    TEST(                 0.000059303515816     , 16, false); // zero + dot + 4 zeros + 11 digits => 16 chars (w/ scientific notation)
    TEST(                -0.00005930351581      , 16, false); // minus + zero + dot + 4 zeros + 10 digits => 16 chars (w/ scientific notation)
#endif

    TEST(      123456789012.1                   , 14, false); // 12 digits + dot + 1 digit => 14 chars
    TEST(     -123456789012.1                   , 15, false); // minus + 12 digits + dot + 1 digit => 14 chars
    TEST(     1234567890123.1                   , 15, false); // 13 digits + dot + 1 digit => 15 chars
    TEST(    -1234567890123.1                   , 16, false); // minus + 13 digits + dot + 1 digit => 15 chars
    TEST(     1234567890123.12                  , 16, false); // 13 digits + dot + 2 digit => 16 chars
    TEST(    -1234567890123.12                  , 16,  true); // minus + 13 digits + dot + 2 digit => 16 chars + ERROR
    TEST(     1234567890123.123                 , 16,  true); // 13 digits + dot + 3 digit => 16 chars + ERROR
    TEST(    -1234567890123.123                 , 16,  true); // minus + 13 digits + dot + 3 digit => 16 chars + ERROR

//    TEST(     12345678901234                    , 14, false); // 14 digits => 14 chars
    TEST(     12345678901234.                   , 14, false); // same number
    TEST(     12345678901234.0                  , 14, false); // same number
    TEST(    1.2345678901234e+13                , 14, false); // same number
//    TEST(    -12345678901234                    , 15, false); // minus + 14 digits => 15 chars
    TEST(    -12345678901234.                   , 15, false); // same number
    TEST(    -12345678901234.0                  , 15, false); // same number
    TEST(   -1.2345678901234e+13                , 15, false); // same number

    TEST(     12345678901234.1                  , 16, false); // 14 digits + dot + 1 digit => 16 chars
    TEST(    -12345678901234.1                  , 15,  true); // minus + 14 digits + dot + 1 digit => 15 chars + ERROR
    TEST(     12345678901234.12                 , 16,  true); // 14 digits + dot + 2 digit => 16 chars + ERROR
    TEST(    -12345678901234.12                 , 15,  true); // minus + 15 digits + dot + 1 digit => 15 chars + ERROR

//    TEST(    123456789012345                    , 15, false); // 15 digit => 15 chars
    TEST(    123456789012345.                   , 15, false); // same number
    TEST(    123456789012345.0                  , 15, false); // same number
    TEST(   1.23456789012345e+14                , 15, false); // same number

//    TEST(   -123456789012345                    , 16, false); // minus + 15 digit => 16 chars
    TEST(   -123456789012345.                   , 16, false); // same number
    TEST(   -123456789012345.0                  , 16, false); // same number
    TEST(  -1.23456789012345e+14                , 16, false); // same number

    TEST(    123456789012345.1                  , 15,  true); // 15 digits + dot + 1 digit => 15 chars + ERROR
    TEST(   -123456789012345.1                  , 16,  true); // minus + 15 digits + dot + 1 digit => 16 chars + ERROR

//    TEST(   1234567890123456                    , 16, false); // 16 digits => 16 chars
    TEST(   1234567890123456.                   , 16, false); // same number
    TEST(   1234567890123456.0                  , 16, false); // same number
    TEST(  1.234567890123456e+15                , 16, false); // same number
//    TEST(  -1234567890123456                    , 16,  true); // minus + 6 digits => 16 chars
    TEST(  -1234567890123456.                   , 16,  true); // same number
    TEST(  -1234567890123456.0                  , 16,  true); // same number
    TEST( -1.234567890123456e+15                , 16,  true); // same number

    TEST(   1234567890123456.2                  , 16,  true); // 16 digits + dot + 1 digit => 16 chars + ERROR
    TEST(  -1234567890123456.2                  , 16,  true); // minus + 16 digits + dot + 1 digit => 16 chars + ERROR

//    TEST(  12345678901234567                    , 16,  true); // 17 digits => 16 chars (w/ scientific notation) + ERROR
    TEST(  12345678901234567.                   , 16,  true); // same number
    TEST( 1.2345678901234567e+16                , 16,  true); // same number
//    TEST( -12345678901234567                    , 16,  true); // minus + 17 digits => 16 chars (w/ scientific notation) + ERROR
    TEST( -12345678901234567.                   , 16,  true); // same number
    TEST(-1.2345678901234567e+16                , 16,  true); // same number

//    TEST( 123456789012345678                    , 16,  true); // 18 digits => 16 chars (w/ scientific notation) + ERROR
    TEST( 123456789012345678.                   , 16,  true); // same number
    TEST(1.23456789012345678e+17                , 16,  true); // same number
//    TEST(-123456789012345678                    , 16,  true); // minus + 18 digits => 16 chars (w/ scientific notation) + ERROR
    TEST(-123456789012345678.                   , 16,  true); // same number
    TEST(-1.23456789012345678e+17               , 16,  true); // same number

//    TEST( 1234567890123456789                   , 16,  true); // 19 digits => 16 chars (w/ scientific notation) + ERROR
    TEST( 1234567890123456789.                  , 16,  true); // same number
    TEST(1.234567890123456789e+18               , 16,  true); // same number
//    TEST(-1234567890123456789                   , 16,  true); // minus + 19 digits => 16 chars (w/ scientific notation) + ERROR
    TEST(-1234567890123456789.                  , 16,  true); // same number
    TEST(-1.234567890123456789e+18              , 16,  true); // same number

    TEST(1.2345678901234567891e+19              , 16,  true);
    TEST(-1.2345678901234567891e+19             , 16,  true);
    TEST(1.23456789012345678901e+20             , 16,  true);
    TEST(-1.23456789012345678901e+20            , 16,  true);

    TEST(1.23456789012345678901e+99             , 16,  true);
    TEST(-1.23456789012345678901e+99            , 16,  true);
    TEST(1.23456789012345678901e+100            , 16,  true);
    TEST(-1.23456789012345678901e+100           , 16,  true);

    TEST(    100000000000000.                   , 15, false); // 15 digits => 15 chars
    TEST(   -100000000000000.                   , 16, false); // minus + 15 digits => 15 chars
    TEST(    999999999999999.                   , 15, false); // 15 digits => 15 chars
    TEST(   -999999999999999.                   , 16, false); // minus + 15 digits => 15 chars
    TEST(   1000000000000000.                   , 16, false); // 16 chars
    TEST(              1e+15                    , 16, false); // same number
    TEST(   9999999999999998.                   , 16, false); // 16 chars
    TEST(  -9999999999999998.                   , 16,  true); // minus + 16 chars
    TEST(  -9999999990099999.                   , 16,  true);
    TEST( -10000000000000000.                   , 16, false); // minus + 17 chars => 16 digits (w/ scientific notation)

#ifdef ALWAYS_3_DIGITS_IN_EXPONENT
    TEST(  10000000000000000.                   ,  6, false); // 17 chars => 6 digits (w/ scientific notation)
    TEST(               1e16                    ,  6, false);
    TEST(  -1000000000000000.                   ,  7, false); // minus + 7 chars (w/ scientific notation)
    TEST(             -1e+15                    ,  7, false); // same number
#else
    TEST(  10000000000000000.                   ,  5, false); // 17 chars => 5 digits (w/ scientific notation)
    TEST(               1e16                    ,  5, false);
    TEST(  -1000000000000000.                   ,  6, false); // minus + 7 chars (w/ scientific notation)
    TEST(             -1e+15                    ,  6, false); // same number
#endif

    TEST(              -1e16                    , 16, false);
    TEST(               1e17                    , 16, false);
    TEST(              -1e17                    , 16, false);
    TEST(               1e18                    , 16, false);
    TEST(              -1e18                    , 16, false);
    TEST(               1e19                    , 16, false);
    TEST(              -1e19                    , 16, false);
    TEST(               1e20                    , 16, false);
    TEST(              -1e20                    , 16, false);

    //TEST(                  0                    ,  1, false); // zero => 1 char (MM: cannot execute this test with ftrapv)
    TEST(                  1                    ,  1, false); // 1 digit => 1 char

    TEST(    9.9999999999e-4                    , 16, false);

#ifdef ALWAYS_3_DIGITS_IN_EXPONENT
    TEST(               1e-5                    ,  6, false);
#else
    TEST(               1e-5                    , 16, false);
#endif

    TEST(             5.1e-4                    ,  7, false);

#ifdef ALWAYS_3_DIGITS_IN_EXPONENT
    TEST(             5.1e-5                    ,  8, false);
#else
    TEST(             5.1e-5                    , 16, false);
#endif

    TEST(             5.1e-6                    , 16, false);
    TEST(             5.1e-7                    , 16, false);
    TEST(             5.1e-8                    , 16, false);
    TEST(             5.1e-9                    , 16, false);
    TEST(            5.1e-10                    , 16, false);
    TEST(            5.1e-11                    , 16, false);

    TEST(              1e+99                    , 16, false);
    TEST(             -1e+99                    , 16, false);
    TEST(             1e+100                    , 16, false);
    TEST(            -1e+100                    , 16, false);
    TEST(        3.4584e+100                    , 16, false);
    TEST(       -3.4584e+100                    , 16, false);
    TEST(        3.4584e+101                    , 16, false);
    TEST(       -3.4584e+101                    , 16, false);
    TEST(              1e-99                    , 16, false);
    TEST(             -1e-99                    , 16, false);
    TEST(             1e-100                    , 16, false);
    TEST(            -1e-100                    , 16, false);
    TEST(        3.4584e-100                    , 16, false);
    TEST(       -3.4584e-100                    , 16, false);
    TEST(        3.4584e-101                    , 16, false);
    TEST(       -3.4584e-101                    , 16, false);


// Tests failing due to double precision

    if (1234567890123456.1 != 1234567890123456.)
    {  TEST(   1234567890123456.1                  , 16,  true); } // 16 digits + dot + 1 digit => 16 chars + ERROR
    else
    {  TEST(   1234567890123456.1                  , 16, false); } // 16 digits + dot + 1 digit => 16 chars + NO ERROR


    if ( 9999999999999999. != 1e+16 )
    {  TEST(   9999999999999999.                   , 16, false); } // 16 chars => 16 digits
    else
    {  TEST(   9999999999999999.                   ,  5, false); } // 16 chars => 5 digits (w/ scientific notation)

    if ( -9999999998999999. != -9.999999999e+15 )
    {  TEST(  -9999999998999999.                   , 16,  true); }
    else
    {  TEST(  -9999999998999999.                   , 16, false); }


    std::cout << "---> Failed test(s): " << err_count << " of " << test_count << std::endl << std::endl;

// ---- RANDOM TESTS:

    const unsigned int random_count = 100000;
    int random_err_count = 0;
    int min_exp = std::numeric_limits<double>::min_exponent10;
    int max_exp = std::numeric_limits<double>::max_exponent10;

    std::cout << "Running " << random_count << " random tests." << std::endl << std::endl;
    for (unsigned int i = 0; i<random_count; i++)
    {
        // Create something that looks like a random double
        int rand_exp =  ( ((int)( std::rand() / (double)(RAND_MAX) ) ) * (max_exp - min_exp) ) + min_exp;
        double rand = static_cast<double>(std::rand()) * pow(10., rand_exp);
        if (rand != rand) {i--; continue;} // nan
        if (rand == std::numeric_limits<double>::infinity()) {i--; continue;} // inf

        std::string s = to_string( rand );
        //std::cout << s;
        if (s.size() > 16 || !s.compare("inf") || !s.compare("nan") )
        {
//            std::cout << "\t--- FAIL" << std::endl;
            random_err_count += 1;
            continue;
        }

        if ( checkerror(rand, s) )
            random_err_count += 1;
//        else
//            std::cout << "\t--- OK" << std::endl;
    }

    std::cout << "---> Failed random test(s): " << random_err_count << " of " << random_count << std::endl << std::endl;

    return err_count + random_err_count;
}
