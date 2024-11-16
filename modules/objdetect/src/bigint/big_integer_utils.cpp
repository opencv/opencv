#include "big_integer_utils.hpp"
#include "big_unsigned_in_abase.hpp"

std::string bigUnsignedToString(const BigUnsigned &x)
{
    return std::string(BigUnsignedInABase(x, 10));
}

std::string bigIntegerToString(const BigInteger &x)
{
    return (x.getSign() == BigInteger::negative)
    ? (std::string("-") + bigUnsignedToString(x.getMagnitude()))
    : (bigUnsignedToString(x.getMagnitude()));
}

BigUnsigned stringToBigUnsigned(const std::string &s)
{
    return BigUnsigned(BigUnsignedInABase(s, 10));
}

BigInteger stringToBigInteger(const std::string &s)
{
    // Recognize a sign followed by a BigUnsigned.
    return (s[0] == '-') ? BigInteger(stringToBigUnsigned(s.substr(1, s.length() - 1)), BigInteger::negative)
    : (s[0] == '+') ? BigInteger(stringToBigUnsigned(s.substr(1, s.length() - 1)))
    : BigInteger(stringToBigUnsigned(s));
}

std::ostream &operator <<(std::ostream &os, const BigUnsigned &x)
{
    BigUnsignedInABase::Base base;
    long osFlags = os.flags();
    if (osFlags & os.dec)
        base = 10;
    else if (osFlags & os.hex)
    {
        base = 16;
        if (osFlags & os.showbase)
            os << "0x";
    }
    else if (osFlags & os.oct)
    {
        base = 8;
        if (osFlags & os.showbase)
            os << '0';
    }
    else
        throw "std::ostream << BigUnsigned: Could not determine the desired base from output-stream flags";
    std::string s = std::string(BigUnsignedInABase(x, base));
    os << s;
    return os;
}

std::ostream &operator <<(std::ostream &os, const BigInteger &x)
{
    if (x.getSign() == BigInteger::negative)
        os << '-';
    os << x.getMagnitude();
    return os;
}
