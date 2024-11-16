#ifndef BIGINTEGER_H
#define BIGINTEGER_H

#include "big_unsigned.hpp"

/* A BigInteger object represents a signed integer of size limited only by
 * available memory.  BigUnsigneds support most mathematical operators and can
 * be converted to and from most primitive integer types.
 *
 * A BigInteger is just an aggregate of a BigUnsigned and a sign.  (It is no
 * longer derived from BigUnsigned because that led to harmful implicit
 * conversions.) */
class BigInteger {

public:
	typedef BigUnsigned::Blk Blk;
	typedef BigUnsigned::Index Index;
	typedef BigUnsigned::CmpRes CmpRes;
	static const CmpRes
		less    = BigUnsigned::less   ,
		equal   = BigUnsigned::equal  ,
		greater = BigUnsigned::greater;
	// Enumeration for the sign of a BigInteger.
	enum Sign { negative = -1, zero = 0, positive = 1 };

protected:
	Sign sign;
	BigUnsigned mag;

public:
	// Constructs zero.
	BigInteger() : sign(zero), mag() {}

	// Copy constructor
	BigInteger(const BigInteger &x) : sign(x.sign), mag(x.mag) {};

	// Assignment operator
	void operator=(const BigInteger &x);

	// Constructor that copies from a given array of blocks with a sign.
	BigInteger(const Blk *b, Index blen, Sign s);

	// Nonnegative constructor that copies from a given array of blocks.
	BigInteger(const Blk *b, Index blen) : mag(b, blen) {
		sign = mag.isZero() ? zero : positive;
	}

	// Constructor from a BigUnsigned and a sign
	BigInteger(const BigUnsigned &x, Sign s);

	// Nonnegative constructor from a BigUnsigned
	BigInteger(const BigUnsigned &x) : mag(x) {
		sign = mag.isZero() ? zero : positive;
	}

	// Constructors from primitive integer types
	BigInteger(unsigned long  x);
	BigInteger(         long  x);
	BigInteger(unsigned int   x);
	BigInteger(         int   x);
	BigInteger(unsigned short x);
	BigInteger(         short x);

	/* Converters to primitive integer types
	 * The implicit conversion operators caused trouble, so these are now
	 * named. */
	unsigned long  toUnsignedLong () const;
	long           toLong         () const;
	unsigned int   toUnsignedInt  () const;
	int            toInt          () const;
	unsigned short toUnsignedShort() const;
	short          toShort        () const;
protected:
	// Helper
	template <class X> X convertToUnsignedPrimitive() const;
	template <class X, class UX> X convertToSignedPrimitive() const;
public:

	// ACCESSORS
	Sign getSign() const { return sign; }
	/* The client can't do any harm by holding a read-only reference to the
	 * magnitude. */
	const BigUnsigned &getMagnitude() const { return mag; }

	// Some accessors that go through to the magnitude
	Index getLength() const { return mag.getLength(); }
	Index getCapacity() const { return mag.getCapacity(); }
	Blk getBlock(Index i) const { return mag.getBlock(i); }
	bool isZero() const { return sign == zero; }  // A bit special

	// COMPARISONS

	// Compares this to x like Perl's <=>
	CmpRes compareTo(const BigInteger &x) const;

	// Ordinary comparison operators
	bool operator ==(const BigInteger &x) const {
		return sign == x.sign && mag == x.mag;
	}
	bool operator !=(const BigInteger &x) const { return !operator ==(x); };
	bool operator < (const BigInteger &x) const { return compareTo(x) == less  ; }
	bool operator <=(const BigInteger &x) const { return compareTo(x) != greater; }
	bool operator >=(const BigInteger &x) const { return compareTo(x) != less  ; }
	bool operator > (const BigInteger &x) const { return compareTo(x) == greater; }

	// OPERATORS -- See the discussion in BigUnsigned.hpp.
	void add     (const BigInteger &a, const BigInteger &b);
	void subtract(const BigInteger &a, const BigInteger &b);
	void multiply(const BigInteger &a, const BigInteger &b);
	/* See the comment on BigUnsigned::divideWithRemainder.  Semantics
	 * differ from those of primitive integers when negatives and/or zeros
	 * are involved. */
	void divideWithRemainder(const BigInteger &b, BigInteger &q);
	void negate(const BigInteger &a);
	
	/* Bitwise operators are not provided for BigIntegers.  Use
	 * getMagnitude to get the magnitude and operate on that instead. */

	BigInteger operator +(const BigInteger &x) const;
	BigInteger operator -(const BigInteger &x) const;
	BigInteger operator *(const BigInteger &x) const;
	BigInteger operator /(const BigInteger &x) const;
	BigInteger operator %(const BigInteger &x) const;
	BigInteger operator -() const;

	void operator +=(const BigInteger &x);
	void operator -=(const BigInteger &x);
	void operator *=(const BigInteger &x);
	void operator /=(const BigInteger &x);
	void operator %=(const BigInteger &x);
	void flipSign();

	// INCREMENT/DECREMENT OPERATORS
	void operator ++(  );
	void operator ++(int);
	void operator --(  );
	void operator --(int);
};

// NORMAL OPERATORS
/* These create an object to hold the result and invoke
 * the appropriate put-here operation on it, passing
 * this and x.  The new object is then returned. */
inline BigInteger BigInteger::operator +(const BigInteger &x) const {
	BigInteger ans;
	ans.add(*this, x);
	return ans;
}
inline BigInteger BigInteger::operator -(const BigInteger &x) const {
	BigInteger ans;
	ans.subtract(*this, x);
	return ans;
}
inline BigInteger BigInteger::operator *(const BigInteger &x) const {
	BigInteger ans;
	ans.multiply(*this, x);
	return ans;
}
inline BigInteger BigInteger::operator /(const BigInteger &x) const {
	if (x.isZero()) throw "BigInteger::operator /: division by zero";
	BigInteger q, r;
	r = *this;
	r.divideWithRemainder(x, q);
	return q;
}
inline BigInteger BigInteger::operator %(const BigInteger &x) const {
	if (x.isZero()) throw "BigInteger::operator %: division by zero";
	BigInteger q, r;
	r = *this;
	r.divideWithRemainder(x, q);
	return r;
}
inline BigInteger BigInteger::operator -() const {
	BigInteger ans;
	ans.negate(*this);
	return ans;
}

/*
 * ASSIGNMENT OPERATORS
 * 
 * Now the responsibility for making a temporary copy if necessary
 * belongs to the put-here operations.  See Assignment Operators in
 * BigUnsigned.hpp.
 */
inline void BigInteger::operator +=(const BigInteger &x) {
	add(*this, x);
}
inline void BigInteger::operator -=(const BigInteger &x) {
	subtract(*this, x);
}
inline void BigInteger::operator *=(const BigInteger &x) {
	multiply(*this, x);
}
inline void BigInteger::operator /=(const BigInteger &x) {
	if (x.isZero()) throw "BigInteger::operator /=: division by zero";
	/* The following technique is slightly faster than copying *this first
	 * when x is large. */
	BigInteger q;
	divideWithRemainder(x, q);
	// *this contains the remainder, but we overwrite it with the quotient.
	*this = q;
}
inline void BigInteger::operator %=(const BigInteger &x) {
	if (x.isZero()) throw "BigInteger::operator %=: division by zero";
	BigInteger q;
	// Mods *this by x.  Don't care about quotient left in q.
	divideWithRemainder(x, q);
}
// This one is trivial
inline void BigInteger::flipSign() {
	sign = Sign(-sign);
}

#endif
