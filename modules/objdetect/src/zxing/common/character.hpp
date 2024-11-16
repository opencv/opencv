//
//  Character.hpp
//  ZXing
//
//  Created by skylook on 9/28/14.
//  Copyright (c) 2014 Valiantliu. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <ctype.h>

#ifndef ZXing_Character_h
#define ZXing_Character_h

namespace zxing
{
    
    class Character
    {
    public:
        static char toUpperCase(char c)
        {
            return toupper(c);
        };
        
        static bool isDigit(char c)
        {

            if (c < '0' || c > '9')
            {
                return false;
            }
            
            return true;

            // return isdigit(c);
        };
        
        static int digit(char c, int radix)
        {
            // return digit(c, radix);

            if (c >= '0' && c<= '9') {
                return (int)(c - '0');
            }
            
            if (c >= 'a' && c<= 'z' && c< (radix + 'a' - 10)) {
                return (int)(c - 'a' + 10);
            }
            
            if (c >= 'A' && c<= 'Z' && c< (radix + 'A' - 10)) {
                return (int)(c - 'A' + 10);
            }
            
            return -1;
        }
    };
}

#endif
