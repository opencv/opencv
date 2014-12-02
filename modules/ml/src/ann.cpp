/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

namespace cv { namespace ml {

ANN::Params::Params()
{
	// BP parameters
	bpDWScale = 0.1; bpMomentScale = 0.95;

	// RPROP parameters
	rpDW0 = 0.1; rpDWPlus = 1.2; rpDWMinus = 0.5; rpDWMin = 1E-6; rpDWMax = 50.0;
}

ANN::Params::Params(const Mat& _layerSizes, int _activateFunc, float _fparam1, float _fparam2,
                        TermCriteria _termCrit, int _trainMethod, float _bpDWScale, float _bpMomentScale, 
                        float _rpDW0, float _rpDWPlus, float _rpDWMinus, float _rpDWMin, float _rpDWMax)
{
	// general parameters
	layerSizes = _layerSizes;
	activateFunc = _activateFunc;
	fparam1 = _fparam1;
	fparam2 = _fparam2;
	termCrit = _termCrit;
	trainMethod = _trainMethod;

	// BP parameters
	bpDWScale = _bpDWScale; bpMomentScale = _bpMomentScale;

	// RPROP parameters
	rpDW0 = _rpDW0; rpDWPlus = _rpDWPlus; rpDWMinus = _rpDWMinus; rpDWMin = _rpDWMin; rpDWMax = _rpDWMax;
}

class ANNImpl : public ANN
{

#ifndef BP
	#define BP ANN::Params::BACKPROP
#endif
#ifndef RPROP
	#define RPROP ANN::Params::RPROP
#endif
#define regstr(pointer, string) { pointer = new char[sizeof(string)]; strcpy(pointer, string); }
#define dregstr(pointer) { delete pointer; }

public:
	/**
	*  @brief  	Neural networks' constructor procedure.
	*/
	ANNImpl()
	{
		// register strings
		regstr(StringSampleSizeMissMatch, "inputs / outputs size should equal, with one vector per row. that is, inputs.rows == outputs.rows.");
		regstr(StringInputVectorSizeImproper, "size of input vector should equal to quantity of neurons in input layer. that is, inputs.cols == neurons[0].");
		regstr(StringOutputVectorSizeImproper, "size of output vector should equal to quantity of neurons in output layer. that is, outputs.cols == neurons[neurons.size() - 1].");
		regstr(StringLayerSizesImproper, "size of layerSizes in create() should be either a uint matrix of size (1, n) or a uint vector of length n, where n > 0. the n is quantity of layers.");
		regstr(StringSampleWeightsSizeImproper, "sample weights matrix should be either empty or size of (1, n), where n is sample size. empty is allowed where the average weights is autoly assigned.");
	}
	ANNImpl(const Params &_Params)
	{
		setParams(_Params);
		ANNImpl();
	}

	/**
	*  @brief  	Neural networks' destructor procedure.
	*/
	~ANNImpl()
	{
		// dregister strings
		dregstr(StringSampleSizeMissMatch);
		dregstr(StringInputVectorSizeImproper);
		dregstr(StringOutputVectorSizeImproper);
		dregstr(StringLayerSizesImproper);
		dregstr(StringSampleWeightsSizeImproper);
	}

	/**
	*  @brief  	Neural networks' creation procedure.
	*  @param	Neurons  Quantity of neurons on each layer. It structurally defines a feed-forward 
	*        	         neural network.
	*/
	int create(const Mat &layerSizes, int activateFunc = SIGMOID_SYM, \
		float fparam1 = 0.667, float fparam2 = 1.716)
	{
		uint r = layerSizes.rows, c = layerSizes.cols;
		if(r != 1 || c <= 0)
			CV_Error(CV_StsBadArg, StringLayerSizesImproper);
		std::vector<uint> v(c);
		for(uint i = 0; i < c; i++)
			v[i] = (uint)layerSizes.at<int>(0, i);
		create(v);
		setActivation(activateFunc, fparam2, fparam1);
		return 0;
	}
	int create(const std::vector<uint> &neurons)
	{
		L = neurons.size() - 1;
		C = neurons;

		I.assign(C[0] + 1, 0);
		D.assign(C[L] + 1, 0);
		O.assign(C[L] + 1, 0);

		std::vector<float> Space;
		V.assign(L + 1, Space);
		Y.assign(L + 1, Space);
		S.assign(L + 1, Space);
		for(uint i = 0; i < L + 1; i++)
		{
			V[i].assign(C[i] + 1, 0);
			Y[i].assign(C[i] + 1, 0);
			S[i].assign(C[i] + 1, 0);
		}

		R = 0.1;// BP parameters
		m = 0.95;
		InitializeBPMemory();

		U0 = 0.1;// RPROP parameters
		Ru = 1.2;
		Rl = 0.5;
		Umn = 1E-6;
		Umx = 50.0;
		InitializeRPROPMemory();

		setEpoches(100);
		setAlgorithm(BP);
		setActivation(SIGMOID_SYM, 1.716, 0.667);
		return 0;
	}

	/**
	*  @brief  	Neural networks' training procedure
	*  @param	inputs   the sample input vectors
	*        	outputs  the expected output vectors
	*        	input    one sample input vector
	*        	output   one expected output vector
	*        	epoches  epoches of the training procedure
	*/
	int train(const Mat& inputs, const Mat& outputs, const Mat& sampleWeights = Mat())
	{
		if(inputs.rows != outputs.rows)
			CV_Error(CV_StsBadArg, StringSampleSizeMissMatch);
		if(inputs.cols != (int)C[0])
			CV_Error(CV_StsOutOfRange, StringInputVectorSizeImproper);
		if(outputs.cols != (int)C[L])
			CV_Error(CV_StsOutOfRange, StringOutputVectorSizeImproper);
		if(!sampleWeights.empty() && (sampleWeights.rows != 1 || sampleWeights.cols != inputs.rows))
			CV_Error(CV_StsOutOfRange, StringSampleWeightsSizeImproper);

		// initialize _E
		_E = -1.0;

		uint n = inputs.rows;
		for(uint i = 0; i < epoches; i++)
		{
			// initialize E = 0
			E = 0.0;

			// randomize order (otherwise ANN will even try to fit the time sequence)
			std::vector<uint> order = randvec(0, n - 1, 1);

			for(uint j = 0; j < n; j++)
			{
				std::vector<float> input = getrow(inputs, order[j]);
				std::vector<float> output = getrow(outputs, order[j]);
				trainonce(input, output);

				if(!sampleWeights.empty())
					reweights(sampleWeights.at<float>(0, order[j]));

				// dE = 1/2 * ||D - O||^2
				E = E + 0.5 * sqdist(D, O);
			}

			// update learning speed to ensure convergence
			if (_E == -1.0)
			{
				_E = E;
				continue;// first epoch
			}
			if (E > _E)
				R = R / 2;

			// print gradient descent data
			//printf("SquareError=%lf, LearningSpeed=%lf.\n", E, R);

			// save E to _E
			_E = E;
		}
		return 0;
	}
	int trainonce(std::vector<float> input, std::vector<float> output)
	{
		// load sample input
		I[0] = -1.0;
		for(uint i = 0; i < input.size(); i++)
			I[i + 1] = input[i];

		// load expected output
		D[0] = -1.0;
		for(uint i = 0; i < output.size(); i++)
			D[i + 1] = output[i];

		// training
		switch(algorithm)
		{
			case BP:
				ActivateBPAlgorithm();
				break;
			case RPROP:
				ActivateRPROPAlgorithm();
				break;
		}
		return 0;
	}
	int reweights(float sampleweight)
	{
		// rolling back weighting and perform reweighting
		switch(algorithm)
		{
			case BP:
				for(uint l = 1; l < L + 1; l++)
					W[l - 1] = W[l - 1] - dW[l - 1] + sampleweight * dW[l - 1];
				break;
			case RPROP:
				for(uint l = 1; l < L + 1; l++)
					W[l - 1] = W[l - 1] - _dW1[l - 1] + sampleweight * _dW1[l - 1];
				break;
		}
		return 0;
	}

	/**
	*  @brief  	Neural networks' evaluation procedure
	*  @param	inputs   the sample input vectors
	*        	outputs  the predicted output vectors
	*       	input    one sample input vector
	*        	output   one predicted output vector
	*/
	Mat predict(const Mat &inputs)
	{
		if(inputs.cols != (int)C[0])
			CV_Error(CV_StsOutOfRange, StringInputVectorSizeImproper);
		Mat outputs = Mat(inputs.rows, C[L], CV_32F);
		predict(inputs, outputs);
		return outputs;
	}
	int predict(const Mat &inputs, Mat &outputs)
	{
		if(inputs.rows != outputs.rows)
			CV_Error(CV_StsBadArg, StringSampleSizeMissMatch);
		if(inputs.cols != (int)C[0])
			CV_Error(CV_StsOutOfRange, StringInputVectorSizeImproper);
		if(outputs.cols != (int)C[L])
			CV_Error(CV_StsOutOfRange, StringOutputVectorSizeImproper);
		uint n = inputs.rows;
		for(uint i = 0; i < n; i++)
		{
			std::vector<float> input = getrow(inputs, i);
			std::vector<float> output(C[L]);
			predictonce(input, output);
			setrow(outputs, i, output);
		}
		return 0;
	}
	int predictonce(std::vector<float> input, std::vector<float> &output)
	{
		// load sample input
		I[0] = -1.0;
		for(uint i = 0; i < input.size(); i++)
			I[i + 1] = input[i];

		// predicting
		switch(algorithm)
		{
			case BP:
				ForwardCalculation();
				break;
			case RPROP:
				ForwardCalculation();
				break;
		}

		// return actual output
		for(uint i = 0; i < output.size(); i++)
			output[i] = O[i + 1];
		return 0;
	}

private:
	/**
	*  @brief  	Activates one epoch of BP algorithm
	*/
	void ActivateBPAlgorithm()
	{
		ForwardCalculation();			// forward procedure
		BackwardCalculation();			// backward procedure
		WeightMatricsUpdate();			// update weight matrics
	}

	/**
	*  @brief  	Activates one epoch of RPROP algorithm
	*/
	void ActivateRPROPAlgorithm()
	{
		ForwardCalculation();			// forward procedure
		BackwardCalculation();			// backward procedure
		WeightMatricsUpdate();			// update weight matrics
		FirstOrderWeightMatricsUpdate();// update first order weight matrics
	}

	/**
	*  @brief  	BP-Core: Activates forward procedure
	*/
	void ForwardCalculation()
	{
		// layer 0 output I
		Y[0] = I;

		// l = 1, 2 ... L
		for(uint l = 1; l < L + 1; l++)
		{
			// linear combination cofficients Vl = Wl,l-1 * Yl-1
			Mat _YM = v2m(Y[l - 1]);
			Mat _YT = _YM.t();
			Mat _AM = W[l - 1] * _YT;
			Mat _AT = _AM.t();
			V[l] = m2v(_AT);

			// layer l output Yl = f(Vl)
			Mat _VM = v2m(V[l]);
			Mat _FV = fm(_VM, actfun, a, b);
			Y[l] = m2v(_FV);
		}

		// layer l output OL
		O = Y[L];
	}

	/**
	*  @brief  	BP-Core: Activates backward procedure
	*/
	void BackwardCalculation()
	{
		// layer l local gradients SL = (DL - OL) .* f'(VL)
		Mat _DM = v2m(D);
		Mat _OM = v2m(O);
		Mat _AM1 = _DM - _OM;
		Mat _VM1 = v2m(V[L]);
		Mat _dFV1 = dfm(_VM1, actfun, a, b);
		Mat _AM2 = _AM1.mul(_dFV1);
		S[L] = m2v(_AM2);

		// l = L-1 ... 2, 1
		for(uint l = L - 1; l > 0; l--)
		{
			// layer l local gradients Sl = f'(Vl) .* (Wl+1,l(T) * Sl+1)
			Mat _VM2 = v2m(V[l]);
			Mat _dFV2 = dfm(_VM2, actfun, a, b);
			Mat _WT = W[l].t();
			Mat _SM = v2m(S[l + 1]);
			Mat _ST = _SM.t();
			Mat _AM3 = _WT * _ST;
			Mat _AM3T = _AM3.t();
			Mat _AM4 = _dFV2.mul(_AM3T);
			S[l] = m2v(_AM4);
		}
	}

	/**
	*  @brief  	BP-Core: Updates weight matrics
	*/
	void WeightMatricsUpdate()
	{
		// l = 1, 2 ... L
		for(uint l = 1; l < L + 1; l++)
		{
			// dWts = R * St * Ys(T)
			uint s = l - 1;
			uint t = l;
			Mat _SM = v2m(S[t]);
			Mat _ST = _SM.t();
			Mat _YM = v2m(Y[s]);
			Mat _AM = _ST * _YM;
			Mat _dWts = R * _AM;

			// Wts = Wts + m * dWts(t) + (1 - m) * dWts(t - 1)
			Mat _mdWts = m * _dWts + (1 - m) * _uW[l - 1];
			W[l - 1] = W[l - 1] + _mdWts;
			_uW[l - 1] = _mdWts;

			// memorize the derivative in case of RPROP use
			dW[l - 1] = _mdWts;
		}
	}

	/**
	*  @brief  	RPROP-Core: Updates first-order weight matrics
	*/
	void FirstOrderWeightMatricsUpdate()
	{
		// l = 1, 2 ... L
		for(uint l = 1; l < L + 1; l++)
		{
			// rolling back BP and perform RPROP
			W[l - 1] = W[l - 1] - dW[l - 1];

			// SIG = _dW .* dW
			Mat SIG = _dW[l - 1].mul(dW[l - 1]);

			// branch A (SIG > 0)
			Mat UA = bound(Ru * _U[l - 1], Umn, Umx);
			Mat dW1A = - sign(dW[l - 1]).mul(UA);

			// branch B (SIG < 0)
			Mat UB = bound(Rl * _U[l - 1], Umn, Umx);
			Mat dW1B = - _dW1[l - 1];

			// branch C (SIG == 0)
			Mat UC = bound(_U[l - 1], Umn, Umx);
			Mat dW1C = - sign(dW[l - 1]).mul(UC);

			// update update-value matrics (entry operation)
			Mat Zero = Mat::zeros(dW[l - 1].rows, dW[l - 1].cols, CV_32F);
			_dW[l - 1] = enop(dW[l - 1], Zero, dW[l - 1], SIG);
			_U[l - 1] = enop(UA, UB, UC, SIG);
			_dW1[l - 1] = enop(dW1A, dW1B, dW1C, SIG);

			// Wts = Wts + dW1ts
			W[l - 1] = W[l - 1] + _dW1[l - 1];
		}
	}

	/**
	*  @brief  	Initialize memory of neural networks for BP algorithm
	*/
	void InitializeBPMemory()
	{
		PreCreateWeightMatrics();
		if (!IW.empty())
			CreatePartialCopyWeightMatrics(W, IW);
		else
			CreateRandomizedWeightMatrics(W);
	}

	/**
	*  @brief  	Precreates weight matrics, allocates the memory of neural networks
	*  this is a sub function of InitializeBPMemory()
	*/
	void PreCreateWeightMatrics()
	{
		for(uint i = 0; i < L; i++)
		{
			// i + 1 is t layer，i is s layer. 
			// s layer has one more threshold neuron coANNects to t layer
			Mat WSpace = Mat(C[i + 1] + 1, C[i] + 1, CV_32F, 0.f);
			W.push_back(WSpace);
			Mat _uWSpace = Mat(C[i + 1] + 1, C[i] + 1, CV_32F, 0.f);
			_uW.push_back(_uWSpace);
		}
	}

	/**
	*  @brief  	Sets weight matrics to user defined initial weight matrics values
	*  @param	Weights   weight matrics
	*  @param	Initials  Initial weight matrics
	*  this is also a sub function of InitializeBPMemory()
	*/
	void CreatePartialCopyWeightMatrics(std::vector<Mat> &Weights, std::vector<Mat> Initials)
	{
		for(uint i = 0; i < L; i++)
		{
			for(uint j = 0; j < C[i + 1] + 1; j++)
			{
				for(uint k = 0; k < C[i] + 1; k++)
				{
					// threshold of next layer
					if (j == 0)
						Weights[i].at<float>(j, k) = 0.0;
					// threshold of previous layer
					else if (j != 0 && k == 0)
						Weights[i].at<float>(j, k) = InitialWeight(C[i] + 1);
					else
						Weights[i].at<float>(j, k) = Initials[i].at<float>(j - 1, k - 1);
				}
			}
		}
	}

	/**
	*  @brief  	Creates weight matrics using autoly generated randomized weight matrics
	*  @param	Weights   weight matrics
	*  this is as well a sub function of InitializeBPMemory()
	*/
	void CreateRandomizedWeightMatrics(std::vector<Mat> &Weights)
	{
		for(uint i = 0; i < L; i++)
		{
			for(uint j = 0; j < C[i + 1] + 1; j++)
			{
				for(uint k = 0; k < C[i] + 1; k++)
				{
					// threshold of next layer
					if (j == 0)
						Weights[i].at<float>(j, k) = 0.0;
					else
						Weights[i].at<float>(j, k) = InitialWeight(C[i] + 1);
				}
			}
		}
	}

	/**
	*  @brief  	Creates a random value
	*  @param	F	distribution parameter
	*  the pdf (probabilistic density function) is f(x) = 1 / 6 * sqrt(F), where x belongs
	*  to [-3 / sqrt(F), 3 / sqrt(F)], and F is quantity of neurons in that layer
	*/
	float InitialWeight(float F)
	{
		float _a = -3.0 / sqrt(F);
		float _b = +3.0 / sqrt(F);
		return avgrand(_a, _b);
	}

	/**
	*  @brief  	Initialize high-level memory of neural networks for RPROP algorithm
	*/
	void InitializeRPROPMemory()
	{
		CreateUpdateValueMatrics();
	}

	/**
	*  @brief  	Creates update-value matrics, allocates the high-level memory of neural networks
	*  this is a sub funtion of InitializeRPROPMemory()
	*/
	void CreateUpdateValueMatrics()
	{
		for(uint i = 0; i < L; i++)
		{
			// update of weight matrics
			Mat dWSpace = Mat(C[i + 1] + 1, C[i] + 1, CV_32F, 0.f);
			dW.push_back(dWSpace);
			// update of weight matrics of previous epoch
			Mat _dWSpace = Mat(C[i + 1] + 1, C[i] + 1, CV_32F, 0.f);
			_dW.push_back(_dWSpace);
			// update of first-order weight matrics of previous epoch
			Mat _dW1Space = Mat(C[i + 1] + 1, C[i] + 1, CV_32F, 0.f);
			_dW1.push_back(_dW1Space);
			// update-value matrics of previous epoch
			Mat _USpace = Mat(C[i + 1] + 1, C[i] + 1, CV_32F, U0);
			_U.push_back(_USpace);
		}
	}

	/**
	*  @brief  	Calculate the square distance between two vectors
	*  @param	v1  the vector 1
	*       	v2  the vector 2
	*/
	float sqdist(std::vector<float> v1, std::vector<float> v2)
	{
		float dval = 0.f;
		for(uint i = 0; i < v1.size(); i++)
		{
			float dist = abs(v1[i] - v2[i]);
			dval += (dist * dist);
		}
		return dval;
	}

	/**
	*  @brief  	Convert a vector of length l to a matrix of size (1, l)
	*  @param	v  the vector to convert
	*/
	Mat v2m(std::vector<float> v)
	{
		uint l = v.size();
		Mat M = Mat(1, l, CV_32F);
		setrow(M, 0, v);
		return M;
	}

	/**
	*  @brief  	Convert a matrix of size (1, l) to a vector of length l
	*  @param	M  the matrix to convert
	*/
	std::vector<float> m2v(Mat M)
	{
		return getrow(M, 0);
	}

	/**
	*  @brief  	Get r row of a matrix of size (n, l) to a vector of length l
	*  @param	M  the original matrix
	*        	r  the row required
	*/
	std::vector<float> getrow(Mat M, uint r)
	{
		uint l = M.cols;
		std::vector<float> v(l);
		for(uint i = 0; i < l; i++)
			v[i] = M.at<float>(r, i);
		return v;
	}

	/**
	*  @brief  	Set r row of a matrix of size (n, l) to a vector of length l
	*  @param	M  the original matrix
	*        	r  the row required
	*        	v  the vector that the row sets to
	*/
	int setrow(Mat M, uint r, std::vector<float> v)
	{
		uint l = v.size();
		for(uint i = 0; i < l; i++)
			M.at<float>(r, i) = v[i];
		return 0;
	}
	
	/**
	*  @brief  	Returns a vector, its entries are continuous integers, but its order is randomized
	*  @param	start   start of integer
	*  	     	end     end of integer
	*  	     	degree  how random are these integers
	*/
	std::vector<uint> randvec(uint start, uint end, uint degree)
	{
		uint len = end - start + 1;
		std::vector<uint> vec(len);
		for(uint i = start; i <= end; i++)
			vec[i] = i;
		uint time = degree * len;
		for(uint j = 0; j < time; j++)
			swap(vec, randint(start, end), randint(start, end));
		return vec;
	}

	/**
	*  @brief  	Swap two entries in a vector
	*  @param	vec  the vector
	*        	_a   entry 1
	*  	     	_b   entry 2
	*/
	int swap(std::vector<uint> &vec, int _a, int _b)
	{
		if(_a == _b)
			return 0;
		int ival = vec[_a];
		vec[_a] = vec[_b];
		vec[_b] = ival;
		return 0;
	}

	/**
	*  @brief  	Returns random integer
	*  @param	_a  lower limit
	*  	     	_b  upper limit
	*/
	uint randint(int _a, int _b)
	{
		float dval = avgrand(_a, _b + 1);
		uint ival = dval >= _b + 1 ? _b : floor(dval);
		return ival;
	}

	/**
	*  @brief  	Returns a random number
	*  @param	_a  lower limit
	*  	     	_b  upper limit
	*/
	float avgrand(float _a, float _b)
	{
		float dval = (float)rand() / (float)INT_MAX;
		return dval * (_b - _a) + _a;
	}

	/**
	*  @brief  	Returns a bounded matrix
	*  @param	M     the matrix to calculate
	*        	llim  lower limit
	*        	ulim  upper limit
	*/
	Mat bound(Mat M, float llim, float ulim)
	{
		Mat B = Mat(M.rows, M.cols, CV_32F, 0.f);
		for(int i = 0; i < M.rows; i++)
		{
			for(int j = 0; j < M.cols; j++)
			{
				float mval = M.at<float>(i, j);
				float bval = 0.f;
				if(mval > ulim) bval = ulim;
				else if(mval < llim) bval = llim;
				else bval = mval;
				B.at<float>(i, j) = bval;
			}
		}
		return B;
	}

	/**
	*  @brief  	Returns a sign matrix
	*  @param	M  the matrix to calculate
	*/
	Mat sign(Mat M)
	{
		Mat _S = Mat(M.rows, M.cols, CV_32F, 0.f);
		for(int i = 0; i < M.rows; i++)
		{
			for(int j = 0; j < M.cols; j++)
			{
				float mval = M.at<float>(i, j);
				float sval = 0.f;
				if(mval > 0) sval = 1.f;
				else if(mval < 0) sval = -1.f;
				else sval = 0.f;
				_S.at<float>(i, j) = sval;
			}
		}
		return _S;
	}

	/**
	*  @brief  	entry operation. in each of the corresponding entries of matrics,
				if SIG > 0 returns A, if SIG < 0 returns B, if SIG = 0 returns C
	*  @param	_A    value matrix A
	*        	_B    value matrix B
	*        	_C    value matrix C
	*        	_SIG  sign matrix SIG
	*/
	Mat enop(Mat _A, Mat _B, Mat _C, Mat _SIG)
	{
		Mat M = Mat(_SIG.rows, _SIG.cols, CV_32F, 0.f);
		for(int i = 0; i < _SIG.rows; i++)
		{
			for(int j = 0; j < _SIG.cols; j++)
			{
				float sig = _SIG.at<float>(i, j);
				if(sig > 0) M.at<float>(i, j) = _A.at<float>(i, j);
				else if(sig < 0) M.at<float>(i, j) = _B.at<float>(i, j);
				else M.at<float>(i, j) = _C.at<float>(i, j);
			}
		}
		return M;
	}

	/**
	*  @brief  	apply function computation to each entry of the matrix
	*  @param	M  the source matrix
	*  @optional
    *                    2a
    *   sigmoid(x) = ----------- - a
    *                1 + e^(-bx)
	*	(recommended parameters : a = 1.716, b = 0.667)
	*/
	float sigmoid(float _x, float _a, float _b)
	{
		return (2.0 * _a) / (1.0 + exp(-_b * _x)) - _a;
	}
	/*
    *                b
    * dsigmoid(x) = -- (a + tanh(x))(a - tanh(x))
    *               2a
	*	(recommended parameters : a = 1.716, b = 0.667)
	*/
	float dsigmoid(float _x, float _a, float _b)
	{
		return (_b / (2.0 * _a)) * (_a + sigmoid(_x, _a, _b)) * (_a - sigmoid(_x, _a, _b));
	}
	//	activation function collection
	Mat fm(Mat M, uint _actfun, float _a, float _b)
	{
		switch(_actfun)
		{
			case IDENTITY:
			// ...
			break;
			case SIGMOID_SYM:
			for(int i = 0; i < M.rows; i++)
				for(int j = 0; j < M.cols; j++)
					M.at<float>(i, j) = sigmoid(M.at<float>(i, j), _a, _b);
			break;
			case GAUSSIAN:
			// ...
			break;
		}
		return M;
	}
	//	derivative activation function collection
	Mat dfm(Mat M, uint _actfun, float _a, float _b)
	{
		switch(_actfun)
		{
			case IDENTITY:
			// ...
			break;
			case SIGMOID_SYM:
			for(int i = 0; i < M.rows; i++)
				for(int j = 0; j < M.cols; j++)
					M.at<float>(i, j) = dsigmoid(M.at<float>(i, j), _a, _b);
			break;
			case GAUSSIAN:
			// ...
			break;
		}
		return M;
	}

public:
	// set parameters

	/**
	*  @brief  	Sets initial weight matrics (optional).
	*  @param	Weights  The initial weight matrics. If is empty then the randomized 
						 weight matrics are autoly generated 
	*  @sample
	*    float w10[2][2] = {{0.5, 0.5}, {0.5, 0.5}}; // weight matrix between layer 1 and 0
	*    float w21[1][2] = {{0.5, 0.5}}; // between layer 2 and 1
	*    Mat W10 = Mat(2, 2, CV_32F, w10);
	*    Mat W21 = Mat(1, 2, CV_32F, w21);
	*    std::vector<Mat> W;
	*    W.push_back(W10);
	*    W.push_back(W21);
	*    setInitialWeightMatrics(W);
	*/
	int setInitialWeightMatrics(std::vector<Mat> Weights)
	{
		IW = Weights;
		return 0;
	}

	/**
	*  @brief  	Sets parameters of BP learning algorithm
	*  @param	bpDWScale	   learning speed (learning rate), belongs to [0, 1]
	*           bpMomentScale  learning momentum, belongs to [0, 1]
	*/
	int setBPParameters(float bpDWScale, float bpMomentScale)
	{
		R = bpDWScale;
		m = bpMomentScale;
		return 0;
	}

	/**
	*  @brief  	Sets parameters of RPROP learning algorithm
	*  @param	rpDW0	   initial update-values
	*           rpDWPlus   update-value increasing parameter R+
	*           rpDWMinus  update-value decreasing parameter R-
	*           rpDWMin    update-value minimum
	*           rpDWMax    update-value maximum
	*/
	int setRPROPParameters(float rpDW0, float rpDWPlus, float rpDWMinus, \
		float rpDWMin, float rpDWMax)
	{
		U0 = rpDW0;
		Ru = rpDWPlus;
		Rl = rpDWMinus;
		Umn = rpDWMin;
		Umx = rpDWMax;
		return 0;
	}

	/**
	*  @brief  	Sets activation function parameters
	*  @param	_actfun  activation function
	*			_a	activation function parameter 1
	*  			_b	activation function parameter 2
	*/
	int setActivation(uint _actfun, float _a, float _b)
	{
		actfun = _actfun;
		a = _a;
		b = _b;
		return 0;
	}

	/**
	*  @brief  	Sets learning algorithm.
	*  @param	_algorithm  learning algorithm
	*/
	int setAlgorithm(uint _algorithm)
	{
		algorithm = _algorithm;
		return 0;
	}

	/**
	*  @brief  	Sets epoches of training.
	*  @param	_epoches  epoches of training, an epoch is defined as a cycle that 
				all samples are used to train the network once.
	*/
	int setEpoches(uint _epoches)
	{
		epoches = _epoches;
		return 0;
	}

	// get parameters

	int getLayerCount()
	{
		return L + 1;
	}
	std::vector<uint> getLayerSizes()
	{
		return C;
	}
	std::vector<Mat> getWeights()
	{
		return W;
	}
	int getVarCount() const
	{
		return 0;
	}
	bool isTrained() const
	{
		return true;
	}
	bool isClassifier() const
	{
		return false;
	}
	float predict(InputArray _I, OutputArray _O, int _p) const
	{
		if(_I.kind() == _O.kind() && _p == -1)
			_p = 0;
		return 0.0f;
	}
	String getDefaultModelName() const
	{
		return "nn";
	}
	Mat getWeights(int i) const
	{
		return W[i];
	}
	void setParams(const Params &_Params)
	{
		create(_Params.layerSizes, _Params.activateFunc, _Params.fparam1, _Params.fparam2);
		// _Params.termCrit;// reserved
		setAlgorithm(_Params.trainMethod);
		setBPParameters(_Params.bpDWScale, _Params.bpMomentScale);
		setRPROPParameters(_Params.rpDW0, _Params.rpDWPlus, _Params.rpDWMinus, \
			_Params.rpDWMin, _Params.rpDWMax);
	}
	Params getParams() const
	{
		Params _Params;
		_Params.layerSizes = Mat::zeros(1, C.size(), CV_32S);
		for(uint i = 0; i < C.size(); i++)
			_Params.layerSizes.at<int>(0, i) = (int)C[i];
		_Params.activateFunc = actfun;
		_Params.fparam1 = b;
		_Params.fparam2 = a;
		_Params.trainMethod = algorithm;
		_Params.bpDWScale = R;
		_Params.bpMomentScale = m;
		_Params.rpDW0 = U0;
		_Params.rpDWPlus = Ru;
		_Params.rpDWMinus = Rl;
		_Params.rpDWMin = Umn;
		_Params.rpDWMax = Umx;
		return _Params;
	}

private:
	// structural parameters
	uint L;                 // quantity of hidden layers           L
	std::vector<uint> C;    // quantity of neurons on each layer   {C0, C1 ... CL}
	
	// memory parameters (BP)
	std::vector<Mat> W;     // weight matrics, the memory of ANN   {W10, W21... WLL-1}
	std::vector<Mat> IW;	// initial weight matrics (optional)
	std::vector<Mat> _uW;	// update of weight matrics of previous epoch

	// high-level memory parameters (RPROP)
	std::vector<Mat> dW;	// update of weight matrics
	std::vector<Mat> _dW;   // update of weight matrics of previous epoch
	std::vector<Mat> _dW1;	// update of first-order weight matrics of previous epoch
	std::vector<Mat> _U;    // update-value matrics of previous epoch

	// data parameters
	std::vector<float> I;  // current input vector     length of C0 + 1, the 1 is for threshold
	std::vector<float> D;  // expected output vector   length of CL + 1, like above
	std::vector<float> O;  // actual output vector     length of CL + 1, like above

	// learning parameters (BP)
	float a;               // activation function parameter
	float b;               // activation function parameter
	float R;               // learning speed, belongs to [0, 1]
	float m;				// learning momentum, belongs to [0, 1]
	uint actfun;            // the activation function
	uint algorithm;         // the learning algorithm
	uint epoches;			// epoches of training

	// high-level learning parameters (RPROP)
	float U0;				// initial update-values
	float Ru;				// update-value increasing parameter R+
	float Rl;				// update-value decreasing parameter R-
	float Umn;				// update-value minimum
	float Umx;				// update-value maximum

	// intermediate parameters
	float E;                             // accumulation error of this epoch
	float _E;                            // accumulation error of previous epoch
	std::vector<std::vector<float> > V;  // linear combination coefficients of each layer
	std::vector<std::vector<float> > Y;  // actual output of each layer
	std::vector<std::vector<float> > S;  // local gradients of each layer

	// strings that used
	char *StringSampleSizeMissMatch;
	char *StringInputVectorSizeImproper;
	char *StringOutputVectorSizeImproper;
	char *StringLayerSizesImproper;
	char *StringSampleWeightsSizeImproper;
};

Ptr<ANN> ANN::create(const Params &params)
{
	Ptr<ANNImpl> ann = makePtr<ANNImpl>(params);
	return ann;
}

}}

