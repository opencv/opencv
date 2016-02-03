// 
// (C) Jan de Vaan 2007-2010, all rights reserved. See the accompanying "License.txt" for licensed use. 
// 


#ifndef CHARLS_CONTEXTRUNMODE
#define CHARLS_CONTEXTRUNMODE

// Implements statistical modelling for the run mode context.
// Computes model dependent parameters like the golomb code lengths

struct CContextRunMode 
{
	CContextRunMode(LONG a, LONG nRItype, LONG nReset) :
		A(a),
		N(1),	
		Nn(0),
		_nRItype(nRItype),
		_nReset((BYTE)nReset)
	{
	}

	LONG A;
	BYTE N;
	BYTE Nn;
	LONG _nRItype;
	BYTE _nReset;

	CContextRunMode()
	{}


	inlinehint LONG GetGolomb() const
	{
		LONG Ntest	= N;
		LONG TEMP	= A + (N >> 1) * _nRItype;
		LONG k = 0;
		for(; Ntest < TEMP; k++) 
		{ 
			Ntest <<= 1;
			ASSERT(k <= 32); 
		};
		return k;
	}


	void UpdateVariables(LONG Errval, LONG EMErrval)
	{		
		if (Errval < 0)
		{
			Nn = Nn + 1;
		}
		A = A + ((EMErrval + 1 - _nRItype) >> 1);
		if (N == _nReset) 
		{
			A = A >> 1;
			N = N >> 1;
			Nn = Nn >> 1;
		}
		N = N + 1;
	}

	inlinehint LONG ComputeErrVal(LONG temp, LONG k)
	{
		bool map = temp & 1;

		LONG errvalabs = (temp + map) / 2;

		if ((k != 0 || (2 * Nn >= N)) == map)
		{
			ASSERT(map == ComputeMap(-errvalabs, k));
			return -errvalabs;
		}

		ASSERT(map == ComputeMap(errvalabs, k));	
		return errvalabs;
	}


	bool ComputeMap(LONG Errval, LONG k) const
	{
		if ((k == 0) && (Errval > 0) && (2 * Nn < N))
			return 1;

		else if ((Errval < 0) && (2 * Nn >= N))
			return 1;		 

		else if ((Errval < 0) && (k != 0))
			return 1;

		return 0;
	}


	inlinehint LONG ComputeMapNegativeE(LONG k) const
	{
		return  k != 0 || (2 * Nn >= N );
	}
};

#endif
