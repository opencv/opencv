#include "big_unsigned.hpp"

// Memory management definitions have moved to the bottom of NumberlikeArray.hpp.

// The templates used by these constructors and converters are at the bottom of
// BigUnsigned.hpp.

BigUnsigned::BigUnsigned(unsigned long  x) { initFromPrimitive      (x); }
BigUnsigned::BigUnsigned(unsigned int   x) { initFromPrimitive      (x); }
BigUnsigned::BigUnsigned(unsigned short x) { initFromPrimitive      (x); }
BigUnsigned::BigUnsigned(         long  x) { initFromSignedPrimitive(x); }
BigUnsigned::BigUnsigned(         int   x) { initFromSignedPrimitive(x); }
BigUnsigned::BigUnsigned(         short x) { initFromSignedPrimitive(x); }

unsigned long  BigUnsigned::toUnsignedLong () const { return convertToPrimitive      <unsigned long >(); }
unsigned int   BigUnsigned::toUnsignedInt  () const { return convertToPrimitive      <unsigned int  >(); }
unsigned short BigUnsigned::toUnsignedShort() const { return convertToPrimitive      <unsigned short>(); }
long           BigUnsigned::toLong         () const { return convertToSignedPrimitive<         long >(); }
int            BigUnsigned::toInt          () const { return convertToSignedPrimitive<         int  >(); }
short          BigUnsigned::toShort        () const { return convertToSignedPrimitive<         short>(); }

// BIT/BLOCK ACCESSORS

void BigUnsigned::setBlock(Index i, Blk newBlock)
{
    if (newBlock == 0)
    {
        if (i < len)
        {
            blk[i] = 0;
            zapLeadingZeros();
        }
        // If i >= len, no effect.
    }
    else
    {
        if (i >= len)
        {
            // The nonzero block extends the number.
            allocateAndCopy(i+1);
            // Zero any added blocks that we aren't setting.
            for (Index j = len; j < i; j++)
                blk[j] = 0;
            len = i+1;
        }
        blk[i] = newBlock;
    }
}

/* Evidently the compiler wants BigUnsigned:: on the return type because, at
 * that point, it hasn't yet parsed the BigUnsigned:: on the name to get the
 * proper scope. */
BigUnsigned::Index BigUnsigned::bitLength() const {
    if (isZero())
        return 0;
    else
    {
        Blk leftmostBlock = getBlock(len - 1);
        Index leftmostBlockLen = 0;
        while (leftmostBlock != 0)
        {
            leftmostBlock >>= 1;
            leftmostBlockLen++;
        }
        return leftmostBlockLen + (len - 1) * N;
    }
}

void BigUnsigned::setBit(Index bi, bool newBit)
{
    Index blockI = bi / N;
    Blk block = getBlock(blockI), mask = Blk(1) << (bi % N);
    block = newBit ? (block | mask) : (block & ~mask);
    setBlock(blockI, block);
}

// COMPARISON
BigUnsigned::CmpRes BigUnsigned::compareTo(const BigUnsigned &x) const
{
    // A bigger length implies a bigger number.
    if (len < x.len)
        return less;
    else if (len > x.len)
        return greater;
    else
    {
        // Compare blocks one by one from left to right.
        Index i = len;
        while (i > 0)
        {
            i--;
            if (blk[i] == x.blk[i])
                continue;
            else if (blk[i] > x.blk[i])
                return greater;
            else
                return less;
        }
        // If no blocks differed, the numbers are equal.
        return equal;
    }
}

// COPY-LESS OPERATIONS

/*
 * On most calls to copy-less operations, it's safe to read the inputs little by
 * little and write the outputs little by little.  However, if one of the
 * inputs is coming from the same variable into which the output is to be
 * stored (an "aliased" call), we risk overwriting the input before we read it.
 * In this case, we first compute the result into a temporary BigUnsigned
 * variable and then copy it into the requested output variable *this.
 * Each put-here operation uses the DTRT_ALIASED macro (Do The Right Thing on
 * aliased calls) to generate code for this check.
 *
 * I adopted this approach on 2007.02.13 (see Assignment Operators in
 * BigUnsigned.hpp).  Before then, put-here operations rejected aliased calls
 * with an exception.  I think doing the right thing is better.
 *
 * Some of the put-here operations can probably handle aliased calls safely
 * without the extra copy because (for example) they process blocks strictly
 * right-to-left.  At some point I might determine which ones don't need the
 * copy, but my reasoning would need to be verified very carefully.  For now
 * I'll leave in the copy.
 */
#define DTRT_ALIASED(cond, op) \
if (cond) { \
BigUnsigned tmpThis; \
tmpThis.op; \
*this = tmpThis; \
return; \
}



void BigUnsigned::add(const BigUnsigned &a, const BigUnsigned &b)
{
    DTRT_ALIASED(this == &a || this == &b, add(a, b));
    // If one argument is zero, copy the other.
    if (a.len == 0)
    {
        operator =(b);
        return;
    }
    else if (b.len == 0)
    {
        operator =(a);
        return;
    }
    // Some variables...
    // Carries in and out of an addition stage
    bool carryIn, carryOut;
    Blk temp;
    Index i;
    // a2 points to the longer input, b2 points to the shorter
    const BigUnsigned *a2, *b2;
    if (a.len >= b.len)
    {
        a2 = &a;
        b2 = &b;
    }
    else
    {
        a2 = &b;
        b2 = &a;
    }
    // Set prelimiary length and make room in this BigUnsigned
    len = a2->len + 1;
    allocate(len);
    // For each block index that is present in both inputs...
    for (i = 0, carryIn = false; i < b2->len; i++)
    {
        // Add input blocks
        temp = a2->blk[i] + b2->blk[i];
        // If a rollover occurred, the result is less than either input.
        // This test is used many times in the BigUnsigned code.
        carryOut = (temp < a2->blk[i]);
        // If a carry was input, handle it
        if (carryIn)
        {
            temp++;
            carryOut |= (temp == 0);
        }
        blk[i] = temp;  // Save the addition result
        carryIn = carryOut;  // Pass the carry along
    }
    // If there is a carry left over, increase blocks until
    // one does not roll over.
    for (; i < a2->len && carryIn; i++)
    {
        temp = a2->blk[i] + 1;
        carryIn = (temp == 0);
        blk[i] = temp;
    }
    // If the carry was resolved but the larger number
    // still has blocks, copy them over.
    for (; i < a2->len; i++)
        blk[i] = a2->blk[i];
    // Set the extra block if there's still a carry, decrease length otherwise
    if (carryIn)
        blk[i] = 1;
    else
        len--;
}

void BigUnsigned::subtract(const BigUnsigned &a, const BigUnsigned &b)
{
    DTRT_ALIASED(this == &a || this == &b, subtract(a, b));
    if (b.len == 0)
    {
        // If b is zero, copy a.
        operator =(a);
        return;
    }
    else if (a.len < b.len)
        // If a is shorter than b, the result is negative.
        throw "BigUnsigned::subtract: "
        "Negative result in unsigned calculation";
    // Some variables...
    bool borrowIn, borrowOut;
    Blk temp;
    Index i;
    // Set preliminary length and make room
    len = a.len;
    allocate(len);
    // For each block index that is present in both inputs...
    for (i = 0, borrowIn = false; i < b.len; i++)
    {
        temp = a.blk[i] - b.blk[i];
        // If a reverse rollover occurred,
        // the result is greater than the block from a.
        borrowOut = (temp > a.blk[i]);
        // Handle an incoming borrow
        if (borrowIn)
        {
            borrowOut |= (temp == 0);
            temp--;
        }
        blk[i] = temp;  // Save the subtraction result
        borrowIn = borrowOut;  // Pass the borrow along
    }
    // If there is a borrow left over, decrease blocks until
    // one does not reverse rollover.
    for (; i < a.len && borrowIn; i++)
    {
        borrowIn = (a.blk[i] == 0);
        blk[i] = a.blk[i] - 1;
    }
    /* If there's still a borrow, the result is negative.
     * Throw an exception, but zero out this object so as to leave it in a
     * predictable state. */
    if (borrowIn)
    {
        len = 0;
        throw "BigUnsigned::subtract: Negative result in unsigned calculation";
    }
    else
    {
        // Copy over the rest of the blocks
        for (; i < a.len; i++)
            blk[i] = a.blk[i];
    }
    // Zap leading zeros
    zapLeadingZeros();
}

/*
 * About the multiplication and division algorithms:
 *
 * I searched unsucessfully for fast C++ built-in operations like the `b_0'
 * and `c_0' Knuth describes in Section 4.3.1 of ``The Art of Computer
 * Programming'' (replace `place' by `Blk'):
 *
 *    ``b_0[:] multiplication of a one-place integer by another one-place
 *      integer, giving a two-place answer;
 *
 *    ``c_0[:] division of a two-place integer by a one-place integer,
 *      provided that the quotient is a one-place integer, and yielding
 *      also a one-place remainder.''
 *
 * I also missed his note that ``[b]y adjusting the word size, if
 * necessary, nearly all computers will have these three operations
 * available'', so I gave up on trying to use algorithms similar to his.
 * A future version of the library might include such algorithms; I
 * would welcome contributions from others for this.
 *
 * I eventually decided to use bit-shifting algorithms.  To multiply `a'
 * and `b', we zero out the result.  Then, for each `1' bit in `a', we
 * shift `b' left the appropriate amount and add it to the result.
 * Similarly, to divide `a' by `b', we shift `b' left varying amounts,
 * repeatedly trying to subtract it from `a'.  When we succeed, we note
 * the fact by setting a bit in the quotient.  While these algorithms
 * have the same O(n^2) time complexity as Knuth's, the ``constant factor''
 * is likely to be larger.
 *
 * Because I used these algorithms, which require single-block addition
 * and subtraction rather than single-block multiplication and division,
 * the innermost loops of all four routines are very similar.  Study one
 * of them and all will become clear.
 */

/*
 * This is a little inline function used by both the multiplication
 * routine and the division routine.
 *
 * `getShiftedBlock' returns the `x'th block of `num << y'.
 * `y' may be anything from 0 to N - 1, and `x' may be anything from
 * 0 to `num.len'.
 *
 * Two things contribute to this block:
 *
 * (1) The `N - y' low bits of `num.blk[x]', shifted `y' bits left.
 *
 * (2) The `y' high bits of `num.blk[x-1]', shifted `N - y' bits right.
 *
 * But we must be careful if `x == 0' or `x == num.len', in
 * which case we should use 0 instead of (2) or (1), respectively.
 *
 * If `y == 0', then (2) contributes 0, as it should.  However,
 * in some computer environments, for a reason I cannot understand,
 * `a >> b' means `a >> (b % N)'.  This means `num.blk[x-1] >> (N - y)'
 * will return `num.blk[x-1]' instead of the desired 0 when `y == 0';
 * the test `y == 0' handles this case specially.
 */
inline BigUnsigned::Blk getShiftedBlock(const BigUnsigned &num,
                                        BigUnsigned::Index x, unsigned int y) {
    BigUnsigned::Blk part1 = (x == 0 || y == 0) ? 0 : (num.blk[x - 1] >> (BigUnsigned::N - y));
    BigUnsigned::Blk part2 = (x == num.len) ? 0 : (num.blk[x] << y);
    return part1 | part2;
}

void BigUnsigned::multiply(const BigUnsigned &a, const BigUnsigned &b) {
    DTRT_ALIASED(this == &a || this == &b, multiply(a, b));
    // If either a or b is zero, set to zero.
    if (a.len == 0 || b.len == 0)
    {
        len = 0;
        return;
    }
    /*
     * Overall method:
     *
     * Set this = 0.
     * For each 1-bit of `a' (say the `i2'th bit of block `i'):
     *    Add `b << (i blocks and i2 bits)' to *this.
     */
    // Variables for the calculation
    Index i, j, k;
    unsigned int i2;
    Blk temp;
    bool carryIn, carryOut;
    // Set preliminary length and make room
    len = a.len + b.len;
    allocate(len);
    // Zero out this object
    for (i = 0; i < len; i++)
        blk[i] = 0;
    // For each block of the first number...
    for (i = 0; i < a.len; i++)
    {
        // For each 1-bit of that block...
        for (i2 = 0; i2 < N; i2++)
        {
            if ((a.blk[i] & (Blk(1) << i2)) == 0)
                continue;
            /*
             * Add b to this, shifted left i blocks and i2 bits.
             * j is the index in b, and k = i + j is the index in this.
             *
             * `getShiftedBlock', a short inline function defined above,
             * is now used for the bit handling.  It replaces the more
             * complex `bHigh' code, in which each run of the loop dealt
             * immediately with the low bits and saved the high bits to
             * be picked up next time.  The last run of the loop used to
             * leave leftover high bits, which were handled separately.
             * Instead, this loop runs an additional time with j == b.len.
             * These changes were made on 2005.01.11.
             */
            for (j = 0, k = i, carryIn = false; j <= b.len; j++, k++)
            {
                /*
                 * The body of this loop is very similar to the body of the first loop
                 * in `add', except that this loop does a `+=' instead of a `+'.
                 */
                temp = blk[k] + getShiftedBlock(b, j, i2);
                carryOut = (temp < blk[k]);
                if (carryIn)
                {
                    temp++;
                    carryOut |= (temp == 0);
                }
                blk[k] = temp;
                carryIn = carryOut;
            }
            // No more extra iteration to deal with `bHigh'.
            // Roll-over a carry as necessary.
            for (; carryIn; k++)
            {
                blk[k]++;
                carryIn = (blk[k] == 0);
            }
        }
    }
    // Zap possible leading zero
    if (blk[len - 1] == 0)
        len--;
}

/*
 * DIVISION WITH REMAINDER
 * This monstrous function mods *this by the given divisor b while storing the
 * quotient in the given object q; at the end, *this contains the remainder.
 * The seemingly bizarre pattern of inputs and outputs was chosen so that the
 * function copies as little as possible (since it is implemented by repeated
 * subtraction of multiples of b from *this).
 *
 * "modWithQuotient" might be a better name for this function, but I would
 * rather not change the name now.
 */
void BigUnsigned::divideWithRemainder(const BigUnsigned &b, BigUnsigned &q)
{
    /* Defending against aliased calls is more complex than usual because we
     * are writing to both *this and q.
     *
     * It would be silly to try to write quotient and remainder to the
     * same variable.  Rule that out right away. */
    if (this == &q)
        throw "BigUnsigned::divideWithRemainder: Cannot write quotient and remainder into the same variable";
    /* Now *this and q are separate, so the only concern is that b might be
     * aliased to one of them.  If so, use a temporary copy of b. */
    if (this == &b || &q == &b)
    {
        BigUnsigned tmpB(b);
        divideWithRemainder(tmpB, q);
        return;
    }

    /*
     * Knuth's definition of mod (which this function uses) is somewhat
     * different from the C++ definition of % in case of division by 0.
     *
     * We let a / 0 == 0 (it doesn't matter much) and a % 0 == a, no
     * exceptions thrown.  This allows us to preserve both Knuth's demand
     * that a mod 0 == a and the useful property that
     * (a / b) * b + (a % b) == a.
     */
    if (b.len == 0)
    {
        q.len = 0;
        return;
    }

    /*
     * If *this.len < b.len, then *this < b, and we can be sure that b doesn't go into
     * *this at all.  The quotient is 0 and *this is already the remainder (so leave it alone).
     */
    if (len < b.len)
    {
        q.len = 0;
        return;
    }

    // At this point we know (*this).len >= b.len > 0.  (Whew!)

    /*
     * Overall method:
     *
     * For each appropriate i and i2, decreasing:
     *    Subtract (b << (i blocks and i2 bits)) from *this, storing the
     *      result in subtractBuf.
     *    If the subtraction succeeds with a nonnegative result:
     *        Turn on bit i2 of block i of the quotient q.
     *        Copy subtractBuf back into *this.
     *    Otherwise bit i2 of block i remains off, and *this is unchanged.
     *
     * Eventually q will contain the entire quotient, and *this will
     * be left with the remainder.
     *
     * subtractBuf[x] corresponds to blk[x], not blk[x+i], since 2005.01.11.
     * But on a single iteration, we don't touch the i lowest blocks of blk
     * (and don't use those of subtractBuf) because these blocks are
     * unaffected by the subtraction: we are subtracting
     * (b << (i blocks and i2 bits)), which ends in at least `i' zero
     * blocks. */
    // Variables for the calculation
    Index i, j, k;
    unsigned int i2;
    Blk temp;
    bool borrowIn, borrowOut;

    /*
     * Make sure we have an extra zero block just past the value.
     *
     * When we attempt a subtraction, we might shift `b' so
     * its first block begins a few bits left of the dividend,
     * and then we'll try to compare these extra bits with
     * a nonexistent block to the left of the dividend.  The
     * extra zero block ensures sensible behavior; we need
     * an extra block in `subtractBuf' for exactly the same reason.
     */
    Index origLen = len;  // Save real length.
    /* To avoid an out-of-bounds access in case of reallocation, allocate
     * first and then increment the logical length. */
    allocateAndCopy(len + 1);
    len++;
    blk[origLen] = 0;  // Zero the added block.

    // subtractBuf holds part of the result of a subtraction; see above.
    Blk *subtractBuf = new Blk[len];

    // Set preliminary length for quotient and make room
    q.len = origLen - b.len + 1;
    q.allocate(q.len);
    // Zero out the quotient
    for (i = 0; i < q.len; i++)
        q.blk[i] = 0;

    // For each possible left-shift of b in blocks...
    i = q.len;
    while (i > 0)
    {
        i--;
        // For each possible left-shift of b in bits...
        // (Remember, N is the number of bits in a Blk.)
        q.blk[i] = 0;
        i2 = N;
        while (i2 > 0)
        {
            i2--;
            /*
             * Subtract b, shifted left i blocks and i2 bits, from *this,
             * and store the answer in subtractBuf.  In the for loop, `k == i + j'.
             *
             * Compare this to the middle section of `multiply'.  They
             * are in many ways analogous.  See especially the discussion
             * of `getShiftedBlock'.
             */
            for (j = 0, k = i, borrowIn = false; j <= b.len; j++, k++)
            {
                temp = blk[k] - getShiftedBlock(b, j, i2);
                borrowOut = (temp > blk[k]);
                if (borrowIn)
                {
                    borrowOut |= (temp == 0);
                    temp--;
                }
                // Since 2005.01.11, indices of `subtractBuf' directly match those of `blk', so use `k'.
                subtractBuf[k] = temp;
                borrowIn = borrowOut;
            }
            // No more extra iteration to deal with `bHigh'.
            // Roll-over a borrow as necessary.
            for (; k < origLen && borrowIn; k++)
            {
                borrowIn = (blk[k] == 0);
                subtractBuf[k] = blk[k] - 1;
            }
            /*
             * If the subtraction was performed successfully (!borrowIn),
             * set bit i2 in block i of the quotient.
             *
             * Then, copy the portion of subtractBuf filled by the subtraction
             * back to *this.  This portion starts with block i and ends--
             * where?  Not necessarily at block `i + b.len'!  Well, we
             * increased k every time we saved a block into subtractBuf, so
             * the region of subtractBuf we copy is just [i, k).
             */
            if (!borrowIn)
            {
                q.blk[i] |= (Blk(1) << i2);
                while (k > i)
                {
                    k--;
                    blk[k] = subtractBuf[k];
                }
            }
        }
    }
    // Zap possible leading zero in quotient
    if (q.blk[q.len - 1] == 0)
        q.len--;
    // Zap any/all leading zeros in remainder
    zapLeadingZeros();
    // Deallocate subtractBuf.
    // (Thanks to Brad Spencer for noticing my accidental omission of this!)
    delete [] subtractBuf;
}

/* BITWISE OPERATORS
 * These are straightforward blockwise operations except that they differ in
 * the output length and the necessity of zapLeadingZeros. */

void BigUnsigned::bitAnd(const BigUnsigned &a, const BigUnsigned &b)
{
    DTRT_ALIASED(this == &a || this == &b, bitAnd(a, b));
    // The bitwise & can't be longer than either operand.
    len = (a.len >= b.len) ? b.len : a.len;
    allocate(len);
    Index i;
    for (i = 0; i < len; i++)
        blk[i] = a.blk[i] & b.blk[i];
    zapLeadingZeros();
}

void BigUnsigned::bitOr(const BigUnsigned &a, const BigUnsigned &b)
{
    DTRT_ALIASED(this == &a || this == &b, bitOr(a, b));
    Index i;
    const BigUnsigned *a2, *b2;
    if (a.len >= b.len)
    {
        a2 = &a;
        b2 = &b;
    }
    else
    {
        a2 = &b;
        b2 = &a;
    }
    allocate(a2->len);
    for (i = 0; i < b2->len; i++)
        blk[i] = a2->blk[i] | b2->blk[i];
    for (; i < a2->len; i++)
        blk[i] = a2->blk[i];
    len = a2->len;
    // Doesn't need zapLeadingZeros.
}

void BigUnsigned::bitXor(const BigUnsigned &a, const BigUnsigned &b)
{
    DTRT_ALIASED(this == &a || this == &b, bitXor(a, b));
    Index i;
    const BigUnsigned *a2, *b2;
    if (a.len >= b.len)
    {
        a2 = &a;
        b2 = &b;
    }
    else
    {
        a2 = &b;
        b2 = &a;
    }
    allocate(a2->len);
    for (i = 0; i < b2->len; i++)
        blk[i] = a2->blk[i] ^ b2->blk[i];
    for (; i < a2->len; i++)
        blk[i] = a2->blk[i];
    len = a2->len;
    zapLeadingZeros();
}

void BigUnsigned::bitShiftLeft(const BigUnsigned &a, int b)
{
    DTRT_ALIASED(this == &a, bitShiftLeft(a, b));
    if (b < 0)
    {
        if (b << 1 == 0)
            throw "BigUnsigned::bitShiftLeft: "
            "Pathological shift amount not implemented";
        else
        {
            bitShiftRight(a, -b);
            return;
        }
    }
    Index shiftBlocks = b / N;
    unsigned int shiftBits = b % N;
    // + 1: room for high bits nudged left into another block
    len = a.len + shiftBlocks + 1;
    allocate(len);
    Index i, j;
    for (i = 0; i < shiftBlocks; i++)
        blk[i] = 0;
    for (j = 0, i = shiftBlocks; j <= a.len; j++, i++)
        blk[i] = getShiftedBlock(a, j, shiftBits);
    // Zap possible leading zero
    if (blk[len - 1] == 0)
        len--;
}

void BigUnsigned::bitShiftRight(const BigUnsigned &a, int b)
{
    DTRT_ALIASED(this == &a, bitShiftRight(a, b));
    if (b < 0)
    {
        if (b << 1 == 0)
            throw "BigUnsigned::bitShiftRight: "
            "Pathological shift amount not implemented";
        else
        {
            bitShiftLeft(a, -b);
            return;
        }
    }
    // This calculation is wacky, but expressing the shift as a left bit shift
    // within each block lets us use getShiftedBlock.
    Index rightShiftBlocks = (b + N - 1) / N;
    unsigned int leftShiftBits = N * rightShiftBlocks - b;
    // Now (N * rightShiftBlocks - leftShiftBits) == b
    // and 0 <= leftShiftBits < N.
    if (rightShiftBlocks >= a.len + 1)
    {
        // All of a is guaranteed to be shifted off, even considering the left
        // bit shift.
        len = 0;
        return;
    }
    // Now we're allocating a positive amount.
    // + 1: room for high bits nudged left into another block
    len = a.len + 1 - rightShiftBlocks;
    allocate(len);
    Index i, j;
    for (j = rightShiftBlocks, i = 0; j <= a.len; j++, i++)
        blk[i] = getShiftedBlock(a, j, leftShiftBits);
    // Zap possible leading zero
    if (blk[len - 1] == 0)
        len--;
}

// INCREMENT/DECREMENT OPERATORS

// Prefix increment
void BigUnsigned::operator ++()
{
    Index i;
    bool carry = true;
    for (i = 0; i < len && carry; i++) {
        blk[i]++;
        carry = (blk[i] == 0);
    }
    if (carry)
    {
        // Allocate and then increase length, as in divideWithRemainder
        allocateAndCopy(len + 1);
        len++;
        blk[i] = 1;
    }
}

// Postfix increment: same as prefix
void BigUnsigned::operator ++(int) {
    operator ++();
}

// Prefix decrement
void BigUnsigned::operator --() {
    if (len == 0)
        throw "BigUnsigned::operator --(): Cannot decrement an unsigned zero";
    Index i;
    bool borrow = true;
    for (i = 0; borrow; i++) {
        borrow = (blk[i] == 0);
        blk[i]--;
    }
    // Zap possible leading zero (there can only be one)
    if (blk[len - 1] == 0)
        len--;
}

// Postfix decrement: same as prefix
void BigUnsigned::operator --(int) {
    operator --();
}
