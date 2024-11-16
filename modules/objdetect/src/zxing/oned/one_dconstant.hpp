#ifndef __CODE_CONSTANT_H__
#define __CODE_CONSTANT_H__

#include "../zxing.hpp"

#define LEN(v) ((int)(sizeof(v)/sizeof(v[0])))
#define VECTOR_INIT(v) v, v + sizeof(v)/sizeof(v[0])

namespace zxing
{
    namespace oned
    {
        namespace constant
        {
            namespace Code128
            {
                const int CODE_SHIFT = 98;

                const int CODE_CODE_C = 99;
                const int CODE_CODE_B = 100;
                const int CODE_CODE_A = 101;

                const int CODE_FNC_1 = 102;
                const int CODE_FNC_2 = 97;
                const int CODE_FNC_3 = 96;
                const int CODE_FNC_4_A = 101;
                const int CODE_FNC_4_B = 100;

                const int CODE_START_A = 103;
                const int CODE_START_B = 104;
                const int CODE_START_C = 105;
                const int CODE_STOP = 106;

                // Dummy characters used to specify control characters in input
                const char ESCAPE_FNC_1 = 0xf1;
                const char ESCAPE_FNC_2 = 0xf2;
                const char ESCAPE_FNC_3 = 0xf3;
                const char ESCAPE_FNC_4 = 0xf4;
                
                const int CODE_PATTERNS_ONE_SIZE = 6;

                const int CODE_PATTERNS_LENGTH = 107;
                const int CODE_PATTERNS[CODE_PATTERNS_LENGTH][6] = {
                    {2, 1, 2, 2, 2, 2}, /* 0 */
                    {2, 2, 2, 1, 2, 2},
                    {2, 2, 2, 2, 2, 1},
                    {1, 2, 1, 2, 2, 3},
                    {1, 2, 1, 3, 2, 2},
                    {1, 3, 1, 2, 2, 2}, /* 5 */
                    {1, 2, 2, 2, 1, 3},
                    {1, 2, 2, 3, 1, 2},
                    {1, 3, 2, 2, 1, 2},
                    {2, 2, 1, 2, 1, 3},
                    {2, 2, 1, 3, 1, 2}, /* 10 */
                    {2, 3, 1, 2, 1, 2},
                    {1, 1, 2, 2, 3, 2},
                    {1, 2, 2, 1, 3, 2},
                    {1, 2, 2, 2, 3, 1},
                    {1, 1, 3, 2, 2, 2}, /* 15 */
                    {1, 2, 3, 1, 2, 2},
                    {1, 2, 3, 2, 2, 1},
                    {2, 2, 3, 2, 1, 1},
                    {2, 2, 1, 1, 3, 2},
                    {2, 2, 1, 2, 3, 1}, /* 20 */
                    {2, 1, 3, 2, 1, 2},
                    {2, 2, 3, 1, 1, 2},
                    {3, 1, 2, 1, 3, 1},
                    {3, 1, 1, 2, 2, 2},
                    {3, 2, 1, 1, 2, 2}, /* 25 */
                    {3, 2, 1, 2, 2, 1},
                    {3, 1, 2, 2, 1, 2},
                    {3, 2, 2, 1, 1, 2},
                    {3, 2, 2, 2, 1, 1},
                    {2, 1, 2, 1, 2, 3}, /* 30 */
                    {2, 1, 2, 3, 2, 1},
                    {2, 3, 2, 1, 2, 1},
                    {1, 1, 1, 3, 2, 3},
                    {1, 3, 1, 1, 2, 3},
                    {1, 3, 1, 3, 2, 1}, /* 35 */
                    {1, 1, 2, 3, 1, 3},
                    {1, 3, 2, 1, 1, 3},
                    {1, 3, 2, 3, 1, 1},
                    {2, 1, 1, 3, 1, 3},
                    {2, 3, 1, 1, 1, 3}, /* 40 */
                    {2, 3, 1, 3, 1, 1},
                    {1, 1, 2, 1, 3, 3},
                    {1, 1, 2, 3, 3, 1},
                    {1, 3, 2, 1, 3, 1},
                    {1, 1, 3, 1, 2, 3}, /* 45 */
                    {1, 1, 3, 3, 2, 1},
                    {1, 3, 3, 1, 2, 1},
                    {3, 1, 3, 1, 2, 1},
                    {2, 1, 1, 3, 3, 1},
                    {2, 3, 1, 1, 3, 1}, /* 50 */
                    {2, 1, 3, 1, 1, 3},
                    {2, 1, 3, 3, 1, 1},
                    {2, 1, 3, 1, 3, 1},
                    {3, 1, 1, 1, 2, 3},
                    {3, 1, 1, 3, 2, 1}, /* 55 */
                    {3, 3, 1, 1, 2, 1},
                    {3, 1, 2, 1, 1, 3},
                    {3, 1, 2, 3, 1, 1},
                    {3, 3, 2, 1, 1, 1},
                    {3, 1, 4, 1, 1, 1}, /* 60 */
                    {2, 2, 1, 4, 1, 1},
                    {4, 3, 1, 1, 1, 1},
                    {1, 1, 1, 2, 2, 4},
                    {1, 1, 1, 4, 2, 2},
                    {1, 2, 1, 1, 2, 4}, /* 65 */
                    {1, 2, 1, 4, 2, 1},
                    {1, 4, 1, 1, 2, 2},
                    {1, 4, 1, 2, 2, 1},
                    {1, 1, 2, 2, 1, 4},
                    {1, 1, 2, 4, 1, 2}, /* 70 */
                    {1, 2, 2, 1, 1, 4},
                    {1, 2, 2, 4, 1, 1},
                    {1, 4, 2, 1, 1, 2},
                    {1, 4, 2, 2, 1, 1},
                    {2, 4, 1, 2, 1, 1}, /* 75 */
                    {2, 2, 1, 1, 1, 4},
                    {4, 1, 3, 1, 1, 1},
                    {2, 4, 1, 1, 1, 2},
                    {1, 3, 4, 1, 1, 1},
                    {1, 1, 1, 2, 4, 2}, /* 80 */
                    {1, 2, 1, 1, 4, 2},
                    {1, 2, 1, 2, 4, 1},
                    {1, 1, 4, 2, 1, 2},
                    {1, 2, 4, 1, 1, 2},
                    {1, 2, 4, 2, 1, 1}, /* 85 */
                    {4, 1, 1, 2, 1, 2},
                    {4, 2, 1, 1, 1, 2},
                    {4, 2, 1, 2, 1, 1},
                    {2, 1, 2, 1, 4, 1},
                    {2, 1, 4, 1, 2, 1}, /* 90 */
                    {4, 1, 2, 1, 2, 1},
                    {1, 1, 1, 1, 4, 3},
                    {1, 1, 1, 3, 4, 1},
                    {1, 3, 1, 1, 4, 1},
                    {1, 1, 4, 1, 1, 3}, /* 95 */
                    {1, 1, 4, 3, 1, 1},
                    {4, 1, 1, 1, 1, 3},
                    {4, 1, 1, 3, 1, 1},
                    {1, 1, 3, 1, 4, 1},
                    {1, 1, 4, 1, 3, 1}, /* 100 */
                    {3, 1, 1, 1, 4, 1},
                    {4, 1, 1, 1, 3, 1},
                    {2, 1, 1, 4, 1, 2},
                    {2, 1, 1, 2, 1, 4},
                    {2, 1, 1, 2, 3, 2}, /* 105 */
                    {2, 3, 3, 1, 1, 1},
                };
            }

            namespace UPCEAN{
                
                /**
                 * Start/end guard pattern.
                 */
                const int START_END_PATTERN_[] = {1, 1, 1};
                const int START_END_PATTERN_LEN = LEN(START_END_PATTERN_);
                
                /**
                 * Pattern marking the middle of a UPC/EAN pattern, separating the two halves.
                 */
                const int MIDDLE_PATTERN_[] = {1, 1, 1, 1, 1};
                const int MIDDLE_PATTERN_LEN = LEN(MIDDLE_PATTERN_);
                
                const int L_AND_G_PATTERNS_ONE_SIZE = 4;
                
                /**
                 * "Odd", or "L" patterns used to encode UPC/EAN digits.
                 */
                const int L_PATTERNS_[][4] = {
                    {3, 2, 1, 1},  // 0
                    {2, 2, 2, 1},  // 1
                    {2, 1, 2, 2},  // 2
                    {1, 4, 1, 1},  // 3
                    {1, 1, 3, 2},  // 4
                    {1, 2, 3, 1},  // 5
                    {1, 1, 1, 4},  // 6
                    {1, 3, 1, 2},  // 7
                    {1, 2, 1, 3},  // 8
                    {3, 1, 1, 2}  // 9
                };
                const int L_PATTERNS_LEN = LEN(L_PATTERNS_);
                
                /**
                 * "G" patterns used to encode UPC/EAN digits. -- Valiantliu
                 */
                const int G_PATTERNS_[][4] = {
					{1, 1, 2, 3},  // 10 reversed 0
					{1, 2, 2, 2},  // 11 reversed 1
					{2, 2, 1, 2},  // 12 reversed 2
					{1, 1, 4, 1},  // 13 reversed 3
					{2, 3, 1, 1},  // 14 reversed 4
					{1, 3, 2, 1},  // 15 reversed 5
					{4, 1, 1, 1},  // 16 reversed 6
					{2, 1, 3, 1},  // 17 reversed 7
					{3, 1, 2, 1},  // 18 reversed 8
					{2, 1, 1, 3}  // 19 reversed 9
                };
                const int G_PATTERNS_LEN = LEN(G_PATTERNS_);
                /**
                 * As above but also including the "even", or "G" patterns used to encode UPC/EAN digits.
                 */
                const int L_AND_G_PATTERNS_[][4] = {
                    {3, 2, 1, 1},  // 0
                    {2, 2, 2, 1},  // 1
                    {2, 1, 2, 2},  // 2
                    {1, 4, 1, 1},  // 3
                    {1, 1, 3, 2},  // 4
                    {1, 2, 3, 1},  // 5
                    {1, 1, 1, 4},  // 6
                    {1, 3, 1, 2},  // 7
                    {1, 2, 1, 3},  // 8
                    {3, 1, 1, 2},  // 9
                    {1, 1, 2, 3},  // 10 reversed 0
                    {1, 2, 2, 2},  // 11 reversed 1
                    {2, 2, 1, 2},  // 12 reversed 2
                    {1, 1, 4, 1},  // 13 reversed 3
                    {2, 3, 1, 1},  // 14 reversed 4
                    {1, 3, 2, 1},  // 15 reversed 5
                    {4, 1, 1, 1},  // 16 reversed 6
                    {2, 1, 3, 1},  // 17 reversed 7
                    {3, 1, 2, 1},  // 18 reversed 8
                    {2, 1, 1, 3}  // 19 reversed 9
                };
                const int L_AND_G_PATTERNS_LEN = LEN(L_AND_G_PATTERNS_);
            }

            namespace EAN13 {
                const int FIRST_DIGIT_ENCODINGS[10] = {
                    0x00, 0x0B, 0x0D, 0xE, 0x13, 0x19, 0x1C, 0x15, 0x16, 0x1A
                };
            }
            
            namespace Code39{
                const char ALPHABET[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. *$/+%";
                
                /**
                 * These represent the encodings of characters, as patterns of wide and narrow
                 * bars.
                 * The 9 least-significant bits of each int correspond to the pattern of wide
                 * and narrow, with 1s representing "wide" and 0s representing narrow.
                 */
                const int CHARACTER_ENCODINGS_LEN = 44;
                const int CHARACTER_ENCODINGS[CHARACTER_ENCODINGS_LEN] = {
                    0x034, 0x121, 0x061, 0x160, 0x031, 0x130, 0x070, 0x025, 0x124, 0x064,  // 0-9
                    0x109, 0x049, 0x148, 0x019, 0x118, 0x058, 0x00D, 0x10C, 0x04C, 0x01C,  // A-J
                    0x103, 0x043, 0x142, 0x013, 0x112, 0x052, 0x007, 0x106, 0x046, 0x016,  // K-T
                    0x181, 0x0C1, 0x1C0, 0x091, 0x190, 0x0D0, 0x085, 0x184, 0x0C4, 0x094,  // U-*
                    0x0A8, 0x0A2, 0x08A, 0x02A // $-%
                };
                
                const int ASTERISK_ENCODING = 0x094;
                const char ALPHABET_STRING[] =
                "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. *$/+%";
                
                const std::string alphabet_string (ALPHABET_STRING);
            }
            
            namespace Code93{
                char const ALPHABET[] =
                "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%abcd*";
                const std::string ALPHABET_STRING (ALPHABET);
                
                /**
                 * These represent the encodings of characters, as patterns of wide and narrow bars.
                 * The 9 least-significant bits of each int correspond to the pattern of wide and narrow.
                 */
                int const CHARACTER_ENCODINGS[] = {
                    0x114, 0x148, 0x144, 0x142, 0x128, 0x124, 0x122, 0x150, 0x112, 0x10A,  // 0-9
                    0x1A8, 0x1A4, 0x1A2, 0x194, 0x192, 0x18A, 0x168, 0x164, 0x162, 0x134,  // A-J
                    0x11A, 0x158, 0x14C, 0x146, 0x12C, 0x116, 0x1B4, 0x1B2, 0x1AC, 0x1A6,  // K-T
                    0x196, 0x19A, 0x16C, 0x166, 0x136, 0x13A,  // U-Z
                    0x12E, 0x1D4, 0x1D2, 0x1CA, 0x16E, 0x176, 0x1AE,  // - - %
                    0x126, 0x1DA, 0x1D6, 0x132, 0x15E,  // Control chars? $-*
                };
                int const CHARACTER_ENCODINGS_LENGTH =
                (int)sizeof(CHARACTER_ENCODINGS)/sizeof(CHARACTER_ENCODINGS[0]);
                const int ASTERISK_ENCODING = CHARACTER_ENCODINGS[47];
            }

			namespace Code25 {
				int const START_PATTERN[4] = {
					1,1,1,1
				};
				int const END_PATTERN[4] = {
					2,1,1
				};
                const int START_PATTERN_A[] = {1, 1};
                const int START_PATTERN_B[] = {1, 1};
				int const NUMBER_ENCODINGS_LEN = 10;
				int const NUMBER_ENCODINGS[] = {
					0x6, 0x11, 0x9, 0x18, 0x5, 0x14, 0xC, 0x3, 0x12, 0xA // 0-9
				};
			}
            
            namespace ITF{
                // Writer
                const int START_PATTERN[] = {1, 1, 1, 1};
                const int END_PATTERN[] = {3, 1, 1};
                const int END_PATTERN_LEN = 3;
                
                // Reader
                const int W = 3;  // Pixel width of a wide line
                const int N = 1;  // Pixed width of a narrow line
                
                const int DEFAULT_ALLOWED_LENGTHS_[] =
                { 48, 44, 24, 20, 18, 16, 14, 12, 10, 8, 6 };
                const ArrayRef<int> DEFAULT_ALLOWED_LENGTHS (new Array<int>(VECTOR_INIT(DEFAULT_ALLOWED_LENGTHS_)));
                
                /**
                 * Start/end guard pattern.
                 *
                 * Note: The end pattern is reversed because the row is reversed before
                 * searching for the END_PATTERN
                 */
                const int START_PATTERN_[] = {N, N, N, N};
                const int START_PATTERN_LEN = 4;
                const std::vector<int> START_PATTERN_VECTOR (VECTOR_INIT(START_PATTERN_));
                
                const int END_PATTERN_REVERSED_[] = {N, N, W};
                const int END_PATTERN_REVERSED_LEN = 3;
                const std::vector<int> END_PATTERN_REVERSED (VECTOR_INIT(END_PATTERN_REVERSED_));
                
                /**
                 * Patterns of Wide / Narrow lines to indicate each digit
                 */
                const int PATTERNS[][5] = {
                    {N, N, W, W, N},  // 0
                    {W, N, N, N, W},  // 1
                    {N, W, N, N, W},  // 2
                    {W, W, N, N, N},  // 3
                    {N, N, W, N, W},  // 4
                    {W, N, W, N, N},  // 5
                    {N, W, W, N, N},  // 6
                    {N, N, N, W, W},  // 7
                    {W, N, N, W, N},  // 8
                    {N, W, N, W, N}  // 9
                };
                
                const int PATTERN_ONE_LEN = 5;
                
            }
            
            namespace CodaBar{
                // Writer
                const char START_END_CHARS[4] = {'A', 'B', 'C', 'D'};
                const char ALT_START_END_CHARS[4] = {'T', 'N', '*', 'E'};
                const char CHARS_WHICH_ARE_TEN_LENGTH_EACH_AFTER_DECODED[4] = {'/', ':', '+', '.'};
                const char DEFAULT_GUARD = START_END_CHARS[0];
                
                // Reader
                char const ALPHABET_STRING[] = "0123456789-$:/.+ABCD";
                char const* const ALPHABET = ALPHABET_STRING;

                const int ALPHABET_LENGTH = strlen(ALPHABET);
                /**
                * These represent the encodings of characters, as patterns of wide and narrow bars. The 7 least-significant bits of
                * each int correspond to the pattern of wide and narrow, with 1s representing "wide" and 0s representing narrow.
                */
                const int CHARACTER_ENCODINGS[] = {
                0x003, 0x006, 0x009, 0x060, 0x012, 0x042, 0x021, 0x024, 0x030, 0x048,  // 0-9
                0x00c, 0x018, 0x045, 0x051, 0x054, 0x015, 0x01A, 0x029, 0x00B, 0x00E,  // -$:/.+ABCD
                };

                // minimal number of characters that should be present (inclusing start and stop characters)
                // under normal circumstances this should be set to 3, but can be set higher
                // as a last-ditch attempt to reduce false positives.
                const int MIN_CHARACTER_LENGTH = 3;

                // official start and end patterns
                const char STARTEND_ENCODING[] = {'A', 'B', 'C', 'D', 0};
                // some codabar generator allow the codabar string to be closed by every
                // character. This will cause lots of false positives!

                // some industries use a checksum standard but this is not part of the original codabar standard
                // for more information see : http:// www.mecsw.com/specs/codabar.html
            }
        }
    }
}
#endif
