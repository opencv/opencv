#ifndef BIGINTEGERALGORITHMS_H
#define BIGINTEGERALGORITHMS_H

#include "big_integer.hpp"

/* Some mathematical algorithms for big integers.
 * This code is new and, as such, experimental. */

// Returns the greatest common divisor of a and b.
BigUnsigned gcd(BigUnsigned a, BigUnsigned b);

/* Extended Euclidean algorithm.
 * Given m and n, finds gcd g and numbers r, s such that r*m + s*n == g. */
void extendedEuclidean(BigInteger m, BigInteger n,
		BigInteger &g, BigInteger &r, BigInteger &s);

/* Returns the multiplicative inverse of x modulo n, or throws an exception if
 * they have a common factor. */
BigUnsigned modinv(const BigInteger &x, const BigUnsigned &n);

// Returns (base ^ exponent) % modulus.
BigUnsigned modexp(const BigInteger &base, const BigUnsigned &exponent,
		const BigUnsigned &modulus);

#endif
