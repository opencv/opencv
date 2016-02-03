// 
// (C) Jan de Vaan 2007-2010, all rights reserved. See the accompanying "License.txt" for licensed use. 
// 


#ifndef CHARLS_CONTEXT
#define CHARLS_CONTEXT


//
// JlsContext: a JPEG-LS context with it's current statistics.
//
struct JlsContext
{
public:
	JlsContext() 
	{}

 	JlsContext(LONG a) :
		A(a),
		B(0),
		C(0),
		N(1)
	{
	}

	LONG A;
	LONG B;
	short C;
	short N;

	inlinehint LONG GetErrorCorrection(LONG k) const
	{
		if (k != 0)
			return 0;

		return BitWiseSign(2 * B + N - 1);
	}
	

	inlinehint void UpdateVariables(LONG errorValue, LONG NEAR, LONG NRESET)
	{
		ASSERT(N != 0);

		// For performance work on copies of A,B,N (compiler will use registers).
		int b = B + errorValue * (2 * NEAR + 1); 
		int a = A + abs(errorValue);
		int n = N;

		ASSERT(a < 65536 * 256);
		ASSERT(abs(b) < 65536 * 256);
		
		if (n == NRESET) 
		{
			a = a >> 1;
			b = b >> 1;
			n = n >> 1;
		}

		n = n + 1;
		
		if (b + n <= 0) 
		{
			b = b + n;
			if (b <= -n)
			{
				b = -n + 1;
			}
			C = _tableC[C - 1];
		} 
		else  if (b > 0) 
		{
			b = b - n;				
			if (b > 0)
			{
				b = 0;
			}
			C = _tableC[C + 1];
		}
		A = a;
		B = b;
		N = (short)n;
		ASSERT(N != 0);
	}



	inlinehint LONG GetGolomb() const
	{
		LONG Ntest	= N;
		LONG Atest	= A;
		LONG k = 0;
		for(; (Ntest << k) < Atest; k++) 
		{ 
			ASSERT(k <= 32); 
		};
		return k;
	}

	static signed char* CreateTableC()
	{
		static std::vector<signed char> rgtableC;
		
		rgtableC.reserve(256 + 2);

		rgtableC.push_back(-128);	
		for (int i = -128; i < 128; i++)
		{
			rgtableC.push_back(char(i));	
		}
		rgtableC.push_back(127);	
		
		signed char* pZero = &rgtableC[128 + 1];	
		ASSERT(pZero[0] == 0);
		return pZero;
	}
private:

	static signed char* _tableC;
};

#endif
