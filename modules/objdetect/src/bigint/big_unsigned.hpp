#ifndef BIGUNSIGNED_H
#define BIGUNSIGNED_H

#include "numberlike_array.hpp"

/* A BigUnsigned object represents a nonnegative integer of size limited only by
 * available memory.  BigUnsigneds support most mathematical operators and can
 * be converted to and from most primitive integer types.
 *
 * The number is stored as a NumberlikeArray of unsigned longs as if it were
 * written in base 256^sizeof(unsigned long).  The least significant block is
 * first, and the length is such that the most significant block is nonzero. */
class BigUnsigned : protected NumberlikeArray<unsigned long> {

public:
	// Enumeration for the result of a comparison.
	enum CmpRes { less = -1, equal = 0, greater = 1 };

	// BigUnsigneds are built with a Blk type of unsigned long.
	typedef unsigned long Blk;

	typedef NumberlikeArray<Blk>::Index Index;
	using NumberlikeArray<Blk>::N;

protected:
	// Creates a BigUnsigned with a capacity; for internal use.
	BigUnsigned(int, Index c) : NumberlikeArray<Blk>(0, c) {}

	// Decreases len to eliminate any leading zero blocks.
	void zapLeadingZeros() { 
		while (len > 0 && blk[len - 1] == 0)
			len--;
	}

public:
	// Constructs zero.
	BigUnsigned() : NumberlikeArray<Blk>() {}

	// Copy constructor
	BigUnsigned(const BigUnsigned &x) : NumberlikeArray<Blk>(x) {}

	// Assignment operator
	void operator=(const BigUnsigned &x) {
		NumberlikeArray<Blk>::operator =(x);
	}

	// Constructor that copies from a given array of blocks.
	BigUnsigned(const Blk *b, Index blen) : NumberlikeArray<Blk>(b, blen) {
		// Eliminate any leading zeros we may have been passed.
		zapLeadingZeros();
	}

	// Destructor.  NumberlikeArray does the delete for us.
	~BigUnsigned() {}
	
	// Constructors from primitive integer types
	BigUnsigned(unsigned long  x);
	BigUnsigned(         long  x);
	BigUnsigned(unsigned int   x);
	BigUnsigned(         int   x);
	BigUnsigned(unsigned short x);
	BigUnsigned(         short x);
protected:
	// Helpers
	template <class X> void initFromPrimitive      (X x);
	template <class X> void initFromSignedPrimitive(X x);
public:

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
	// Helpers
	template <class X> X convertToSignedPrimitive() const;
	template <class X> X convertToPrimitive      () const;
public:

	// BIT/BLOCK ACCESSORS

	// Expose these from NumberlikeArray directly.
	using NumberlikeArray<Blk>::getCapacity;
	using NumberlikeArray<Blk>::getLength;

	/* Returns the requested block, or 0 if it is beyond the length (as if
	 * the number had 0s infinitely to the left). */
	Blk getBlock(Index i) const { return i >= len ? 0 : blk[i]; }
	/* Sets the requested block.  The number grows or shrinks as necessary. */
	void setBlock(Index i, Blk newBlock);

	// The number is zero if and only if the canonical length is zero.
	bool isZero() const { return NumberlikeArray<Blk>::isEmpty(); }

	/* Returns the length of the number in bits, i.e., zero if the number
	 * is zero and otherwise one more than the largest value of bi for
	 * which getBit(bi) returns true. */
	Index bitLength() const;
	/* Get the state of bit bi, which has value 2^bi.  Bits beyond the
	 * number's length are considered to be 0. */
	bool getBit(Index bi) const {
		return (getBlock(bi / N) & (Blk(1) << (bi % N))) != 0;
	}
	/* Sets the state of bit bi to newBit.  The number grows or shrinks as
	 * necessary. */
	void setBit(Index bi, bool newBit);

	// COMPARISONS

	// Compares this to x like Perl's <=>
	CmpRes compareTo(const BigUnsigned &x) const;

	// Ordinary comparison operators
	bool operator ==(const BigUnsigned &x) const {
		return NumberlikeArray<Blk>::operator ==(x);
	}
	bool operator !=(const BigUnsigned &x) const {
		return NumberlikeArray<Blk>::operator !=(x);
	}
	bool operator < (const BigUnsigned &x) const { return compareTo(x) == less  ; }
	bool operator <=(const BigUnsigned &x) const { return compareTo(x) != greater; }
	bool operator >=(const BigUnsigned &x) const { return compareTo(x) != less  ; }
	bool operator > (const BigUnsigned &x) const { return compareTo(x) == greater; }

	/*
	 * BigUnsigned and BigInteger both provide three kinds of operators.
	 * Here ``big-integer'' refers to BigInteger or BigUnsigned.
	 *
	 * (1) Overloaded ``return-by-value'' operators:
	 *     +, -, *, /, %, unary -, &, |, ^, <<, >>.
	 * Big-integer code using these operators looks identical to code using
	 * the primitive integer types.  These operators take one or two
	 * big-integer inputs and return a big-integer result, which can then
	 * be assigned to a BigInteger variable or used in an expression.
	 * Example:
	 *     BigInteger a(1), b = 1;
	 *     BigInteger c = a + b;
	 *
	 * (2) Overloaded assignment operators:
	 *     +=, -=, *=, /=, %=, flipSign, &=, |=, ^=, <<=, >>=, ++, --.
	 * Again, these are used on big integers just like on ints.  They take
	 * one writable big integer that both provides an operand and receives a
	 * result.  Most also take a second read-only operand.
	 * Example:
	 *     BigInteger a(1), b(1);
	 *     a += b;
	 *
	 * (3) Copy-less operations: `add', `subtract', etc.
	 * These named methods take operands as arguments and store the result
	 * in the receiver (*this), avoiding unnecessary copies and allocations.
	 * `divideWithRemainder' is special: it both takes the dividend from and
	 * stores the remainder into the receiver, and it takes a separate
	 * object in which to store the quotient.  NOTE: If you are wondering
	 * why these don't return a value, you probably mean to use the
	 * overloaded return-by-value operators instead.
	 * 
	 * Examples:
	 *     BigInteger a(43), b(7), c, d;
	 *
	 *     c = a + b;   // Now c == 50.
	 *     c.add(a, b);  // Same effect but without the two copies.
	 *
	 *     c.divideWithRemainder(b, d);
	 *     // 50 / 7; now d == 7 (quotient) and c == 1 (remainder).
	 *
	 *     // ``Aliased'' calls now do the right thing using a temporary
	 *     // copy, but see note on `divideWithRemainder'.
	 *     a.add(a, b); 
	 */

	// COPY-LESS OPERATIONS

	// These 8: Arguments are read-only operands, result is saved in *this.
	void add(const BigUnsigned &a, const BigUnsigned &b);
	void subtract(const BigUnsigned &a, const BigUnsigned &b);
	void multiply(const BigUnsigned &a, const BigUnsigned &b);
	void bitAnd(const BigUnsigned &a, const BigUnsigned &b);
	void bitOr(const BigUnsigned &a, const BigUnsigned &b);
	void bitXor(const BigUnsigned &a, const BigUnsigned &b);
	/* Negative shift amounts translate to opposite-direction shifts,
	 * except for -2^(8*sizeof(int)-1) which is unimplemented. */
	void bitShiftLeft(const BigUnsigned &a, int b);
	void bitShiftRight(const BigUnsigned &a, int b);

	/* `a.divideWithRemainder(b, q)' is like `q = a / b, a %= b'.
	 * / and % use semantics similar to Knuth's, which differ from the
	 * primitive integer semantics under division by zero.  See the
	 * implementation in BigUnsigned.cpp for details.
	 * `a.divideWithRemainder(b, a)' throws an exception: it doesn't make
	 * sense to write quotient and remainder into the same variable. */
	void divideWithRemainder(const BigUnsigned &b, BigUnsigned &q);

	/* `divide' and `modulo' are no longer offered.  Use
	 * `divideWithRemainder' instead. */

	// OVERLOADED RETURN-BY-VALUE OPERATORS
	BigUnsigned operator +(const BigUnsigned &x) const;
	BigUnsigned operator -(const BigUnsigned &x) const;
	BigUnsigned operator *(const BigUnsigned &x) const;
	BigUnsigned operator /(const BigUnsigned &x) const;
	BigUnsigned operator %(const BigUnsigned &x) const;
	/* OK, maybe unary minus could succeed in one case, but it really
	 * shouldn't be used, so it isn't provided. */
	BigUnsigned operator &(const BigUnsigned &x) const;
	BigUnsigned operator |(const BigUnsigned &x) const;
	BigUnsigned operator ^(const BigUnsigned &x) const;
	BigUnsigned operator <<(int b) const;
	BigUnsigned operator >>(int b) const;

	// OVERLOADED ASSIGNMENT OPERATORS
	void operator +=(const BigUnsigned &x);
	void operator -=(const BigUnsigned &x);
	void operator *=(const BigUnsigned &x);
	void operator /=(const BigUnsigned &x);
	void operator %=(const BigUnsigned &x);
	void operator &=(const BigUnsigned &x);
	void operator |=(const BigUnsigned &x);
	void operator ^=(const BigUnsigned &x);
	void operator <<=(int b);
	void operator >>=(int b);

	/* INCREMENT/DECREMENT OPERATORS
	 * To discourage messy coding, these do not return *this, so prefix
	 * and postfix behave the same. */
	void operator ++(  );
	void operator ++(int);
	void operator --(  );
	void operator --(int);

	// Helper function that needs access to BigUnsigned internals
	friend Blk getShiftedBlock(const BigUnsigned &num, Index x,
			unsigned int y);

	// See BigInteger.cpp.
	template <class X>
	friend X convertBigUnsignedToPrimitiveAccess(const BigUnsigned &a);
};

/* Implementing the return-by-value and assignment operators in terms of the
 * copy-less operations.  The copy-less operations are responsible for making
 * any necessary temporary copies to work around aliasing. */

inline BigUnsigned BigUnsigned::operator +(const BigUnsigned &x) const {
	BigUnsigned ans;
	ans.add(*this, x);
	return ans;
}
inline BigUnsigned BigUnsigned::operator -(const BigUnsigned &x) const {
	BigUnsigned ans;
	ans.subtract(*this, x);
	return ans;
}
inline BigUnsigned BigUnsigned::operator *(const BigUnsigned &x) const {
	BigUnsigned ans;
	ans.multiply(*this, x);
	return ans;
}
inline BigUnsigned BigUnsigned::operator /(const BigUnsigned &x) const {
	if (x.isZero()) throw "BigUnsigned::operator /: division by zero";
	BigUnsigned q, r;
	r = *this;
	r.divideWithRemainder(x, q);
	return q;
}
inline BigUnsigned BigUnsigned::operator %(const BigUnsigned &x) const {
	if (x.isZero()) throw "BigUnsigned::operator %: division by zero";
	BigUnsigned q, r;
	r = *this;
	r.divideWithRemainder(x, q);
	return r;
}
inline BigUnsigned BigUnsigned::operator &(const BigUnsigned &x) const {
	BigUnsigned ans;
	ans.bitAnd(*this, x);
	return ans;
}
inline BigUnsigned BigUnsigned::operator |(const BigUnsigned &x) const {
	BigUnsigned ans;
	ans.bitOr(*this, x);
	return ans;
}
inline BigUnsigned BigUnsigned::operator ^(const BigUnsigned &x) const {
	BigUnsigned ans;
	ans.bitXor(*this, x);
	return ans;
}
inline BigUnsigned BigUnsigned::operator <<(int b) const {
	BigUnsigned ans;
	ans.bitShiftLeft(*this, b);
	return ans;
}
inline BigUnsigned BigUnsigned::operator >>(int b) const {
	BigUnsigned ans;
	ans.bitShiftRight(*this, b);
	return ans;
}

inline void BigUnsigned::operator +=(const BigUnsigned &x) {
	add(*this, x);
}
inline void BigUnsigned::operator -=(const BigUnsigned &x) {
	subtract(*this, x);
}
inline void BigUnsigned::operator *=(const BigUnsigned &x) {
	multiply(*this, x);
}
inline void BigUnsigned::operator /=(const BigUnsigned &x) {
	if (x.isZero()) throw "BigUnsigned::operator /=: division by zero";
	/* The following technique is slightly faster than copying *this first
	 * when x is large. */
	BigUnsigned q;
	divideWithRemainder(x, q);
	// *this contains the remainder, but we overwrite it with the quotient.
	*this = q;
}
inline void BigUnsigned::operator %=(const BigUnsigned &x) {
	if (x.isZero()) throw "BigUnsigned::operator %=: division by zero";
	BigUnsigned q;
	// Mods *this by x.  Don't care about quotient left in q.
	divideWithRemainder(x, q);
}
inline void BigUnsigned::operator &=(const BigUnsigned &x) {
	bitAnd(*this, x);
}
inline void BigUnsigned::operator |=(const BigUnsigned &x) {
	bitOr(*this, x);
}
inline void BigUnsigned::operator ^=(const BigUnsigned &x) {
	bitXor(*this, x);
}
inline void BigUnsigned::operator <<=(int b) {
	bitShiftLeft(*this, b);
}
inline void BigUnsigned::operator >>=(int b) {
	bitShiftRight(*this, b);
}

/* Templates for conversions of BigUnsigned to and from primitive integers.
 * BigInteger.cpp needs to instantiate convertToPrimitive, and the uses in
 * BigUnsigned.cpp didn't do the trick; I think g++ inlined convertToPrimitive
 * instead of generating linkable instantiations.  So for consistency, I put
 * all the templates here. */

// CONSTRUCTION FROM PRIMITIVE INTEGERS

/* Initialize this BigUnsigned from the given primitive integer.  The same
 * pattern works for all primitive integer types, so I put it into a template to
 * reduce code duplication.  (Don't worry: this is protected and we instantiate
 * it only with primitive integer types.)  Type X could be signed, but x is
 * known to be nonnegative. */
template <class X>
void BigUnsigned::initFromPrimitive(X x) {
	if (x == 0)
		;  // NumberlikeArray already initialized us to zero.
	else {
		// Create a single block.  blk is NULL; no need to delete it.
		cap = 1;
		blk = new Blk[1];
		len = 1;
		blk[0] = Blk(x);
	}
}

/* Ditto, but first check that x is nonnegative.  I could have put the check in
 * initFromPrimitive and let the compiler optimize it out for unsigned-type
 * instantiations, but I wanted to avoid the warning stupidly issued by g++ for
 * a condition that is constant in *any* instantiation, even if not in all. */
template <class X>
void BigUnsigned::initFromSignedPrimitive(X x) {
	if (x < 0)
		throw "BigUnsigned constructor: "
			"Cannot construct a BigUnsigned from a negative number";
	else
		initFromPrimitive(x);
}

// CONVERSION TO PRIMITIVE INTEGERS

/* Template with the same idea as initFromPrimitive.  This might be slightly
 * slower than the previous version with the masks, but it's much shorter and
 * clearer, which is the library's stated goal. */
template <class X>
X BigUnsigned::convertToPrimitive() const {
	if (len == 0)
		// The number is zero; return zero.
		return 0;
	else if (len == 1) {
		// The single block might fit in an X.  Try the conversion.
		X x = X(blk[0]);
		// Make sure the result accurately represents the block.
		if (Blk(x) == blk[0])
			// Successful conversion.
			return x;
		// Otherwise fall through.
	}
	throw "BigUnsigned::to<Primitive>: "
		"Value is too big to fit in the requested type";
}

/* Wrap the above in an x >= 0 test to make sure we got a nonnegative result,
 * not a negative one that happened to convert back into the correct nonnegative
 * one.  (E.g., catch incorrect conversion of 2^31 to the long -2^31.)  Again,
 * separated to avoid a g++ warning. */
template <class X>
X BigUnsigned::convertToSignedPrimitive() const {
	X x = convertToPrimitive<X>();
	if (x >= 0)
		return x;
	else
		throw "BigUnsigned::to(Primitive): "
			"Value is too big to fit in the requested type";
}

#endif
