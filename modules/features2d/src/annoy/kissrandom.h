#ifndef ANNOY_KISSRANDOM_H
#define ANNOY_KISSRANDOM_H

#if defined(_MSC_VER) && _MSC_VER == 1500
typedef unsigned __int32    uint32_t;
typedef unsigned __int64    uint64_t;
#else
#include <stdint.h>
#endif

namespace Annoy {

// KISS = "keep it simple, stupid", but high quality random number generator
// http://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf -> "Use a good RNG and build it into your code"
// http://mathforum.org/kb/message.jspa?messageID=6627731
// https://de.wikipedia.org/wiki/KISS_(Zufallszahlengenerator)

// 32 bit KISS
struct Kiss32Random {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t c;

  static const uint32_t default_seed = 123456789;
#if __cplusplus < 201103L
  typedef uint32_t seed_type;
#endif

  // seed must be != 0
  Kiss32Random(uint32_t seed = default_seed) {
    x = seed;
    y = 362436000;
    z = 521288629;
    c = 7654321;
  }

  uint32_t kiss() {
    // Linear congruence generator
    x = 69069 * x + 12345;

    // Xor shift
    y ^= y << 13;
    y ^= y >> 17;
    y ^= y << 5;

    // Multiply-with-carry
    uint64_t t = 698769069ULL * z + c;
    c = t >> 32;
    z = (uint32_t) t;

    return x + y + z;
  }
  inline int flip() {
    // Draw random 0 or 1
    return kiss() & 1;
  }
  inline size_t index(size_t n) {
    // Draw random integer between 0 and n-1 where n is at most the number of data points you have
    return kiss() % n;
  }
  inline void set_seed(uint32_t seed) {
    x = seed;
  }
};

// 64 bit KISS. Use this if you have more than about 2^24 data points ("big data" ;) )
struct Kiss64Random {
  uint64_t x;
  uint64_t y;
  uint64_t z;
  uint64_t c;

  static const uint64_t default_seed = 1234567890987654321ULL;
#if __cplusplus < 201103L
  typedef uint64_t seed_type;
#endif

  // seed must be != 0
  Kiss64Random(uint64_t seed = default_seed) {
    x = seed;
    y = 362436362436362436ULL;
    z = 1066149217761810ULL;
    c = 123456123456123456ULL;
  }

  uint64_t kiss() {
    // Linear congruence generator
    z = 6906969069LL*z+1234567;

    // Xor shift
    y ^= (y<<13);
    y ^= (y>>17);
    y ^= (y<<43);

    // Multiply-with-carry (uint128_t t = (2^58 + 1) * x + c; c = t >> 64; x = (uint64_t) t)
    uint64_t t = (x<<58)+c;
    c = (x>>6);
    x += t;
    c += (x<t);

    return x + y + z;
  }
  inline int flip() {
    // Draw random 0 or 1
    return kiss() & 1;
  }
  inline size_t index(size_t n) {
    // Draw random integer between 0 and n-1 where n is at most the number of data points you have
    return kiss() % n;
  }
  inline void set_seed(uint64_t seed) {
    x = seed;
  }
};

}

#endif
// vim: tabstop=2 shiftwidth=2
