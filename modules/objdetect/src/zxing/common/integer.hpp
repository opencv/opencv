#include <iostream>

#ifndef ZXing_Integer_h
#define ZXing_Integer_h


namespace zxing
{

class Integer
{
public:
	static int parseInt(Ref<String> strInteger)
	{
        int integer = parseInt(strInteger->getText());
		return integer;
	}
    
    static int parseInt(std::string strInteger)
    {
        int integer = 0;
        
        integer = atoi(strInteger.c_str());

        return integer;
    }
};

}

#endif
