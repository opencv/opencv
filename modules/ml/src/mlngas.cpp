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

#include "_ml.h"

CvNeuralGas::CvNeuralGas() {

}

CvNeuralGas::CvNeuralGas( CvMat* _distr, unsigned int _total_nodes, unsigned int _max_iterations, float _lambda0, float _lambdaT, float _epsilon0, float _epsilonT ) {

    default_model_name = "neuralgas";

    distribution = _distr;
    total_nodes = _total_nodes;
    max_iterations = _max_iterations;
    iteration = 0;
    lambda0 = _lambda0;
    lambdaT = _lambdaT;
    epsilon0 = _epsilon0;
    epsilonT = _epsilonT;
    input = NULL;
    nodes = new std::vector<CvGasNode *>;
}

CvNeuralGas::CvNeuralGas( const cv::Mat& _distr, unsigned int _total_nodes, unsigned int _max_iterations, float _lambda0, float _lambdaT, float _epsilon0, float _epsilonT ) {

    default_model_name = "neuralgas";

    distribution = new CvMat(_distr);
    total_nodes = _total_nodes;
    max_iterations = _max_iterations;
    iteration = 0;
    lambda0 = _lambda0;
    lambdaT = _lambdaT;
    epsilon0 = _epsilon0;
    epsilonT = _epsilonT;
    input = NULL;
    nodes = new std::vector<CvGasNode *>;
}

CvNeuralGas::~CvNeuralGas() {
    clear( );
}

bool CvNeuralGas::init() {
    bool ok = true;

    CV_FUNCNAME( "CvNeuralGas::init" );

    __BEGIN__;

    int x = 0;
    int y = 0;

    // Create nodes.
    for( unsigned int i=0; i<total_nodes; i++ ) {
        CvGasNode* node = new CvGasNode();

        x = rng.next() % (distribution->width - 1);
        y = rng.next() % (distribution->height - 1);

        node->id = i;
        node->rank = 0;
        node->ref_vector = cvGet2D( distribution, y, x );
        node->distance = 0.0;

        nodes->push_back( node );
    }


    __END__;

    return ok;
}

bool CvNeuralGas::train_auto() {

    CV_FUNCNAME( "CvNeuralGas::train_auto" );
    __BEGIN__;

    while( iteration < max_iterations ) {
        if( train() == false )
            return false;
    }

    __END__;
    return true;
}

bool CvNeuralGas::train( cv::Scalar& _input ) {
    CvScalar* input = new CvScalar( _input );
    return train( input );
}

bool CvNeuralGas::train( CvScalar* _input ) {

    CV_FUNCNAME( "CvNeuralGas::train" );
    __BEGIN__;

    //if( input != NULL )
    //    delete input;

    if( _input != NULL ) {
        input = _input;
    } else {
        // peak random
        int x = rng.next() % (distribution->width - 1);
        int y = rng.next()  % (distribution->height - 1);

        input = &(cvGet2D( distribution, y, x ));
    }

    // Calculate the distance of each node`s reference vector from the
    // projected input vector.
    double temp = 0.0;
    double val = 0.0;
    for( unsigned long int i=0; i<total_nodes; i++ ) {
        CvGasNode* curr = nodes->at( i );
        curr->distance = 0.0;

        CvScalar* ref_vector = &(curr->ref_vector);
        for( int x=0; x<4; x++ ) {
            val = input->val[x] - ref_vector->val[x];
            temp += pow( val, 2.0 );
        }

        curr->distance = sqrt( temp );

        temp = 0.0;
        val = 0.0;
    }

    //Sort the nodes based on their distance.
    std::sort( nodes->begin(), nodes->end(), Compare);

    //Fetch the bmu/smu/wmu.
    bmu = nodes->at( 0 );
    smu = nodes->at( 1 );
    wmu = nodes->at( total_nodes - 1 );

    // Adapt the nodes.
    double epsilon_t = epsilon0 * pow( ( epsilonT / epsilon0 ), (float)iteration/max_iterations );
    double sqr_sigma = lambda0 * pow( ( lambdaT / lambda0 ), (float)iteration/max_iterations );

    for( unsigned long int i=0; i<total_nodes; i++ ) {
        CvGasNode* curr = nodes->at( i );
        curr->rank = -i;

        double h = exp( ((double)curr->rank) / sqr_sigma );

        CvScalar* ref_vector = &(curr->ref_vector);
        CvScalar delta;

        for(int x=0;x<4;x++){
            delta.val[x] = (input->val[x] - ref_vector->val[x]) * h * epsilon_t;
            ref_vector->val[x] += delta.val[x];
        }
    }

    iteration++;

    __END__;
    return true;
}

void CvNeuralGas::clear() {
    bmu = NULL;
    smu = NULL;
    wmu = NULL;

    for( unsigned int i=0; i<total_nodes; i++ ) {
        delete( nodes->at( i ) );
    }

    delete( nodes );
    delete( distribution );
}

std::vector<CvGasNode*>* CvNeuralGas::get_nodes() const {
    return nodes;
}

unsigned int CvNeuralGas::get_iteration() const {
    return iteration;
}

unsigned int CvNeuralGas::get_max_iterations() const {
    return max_iterations;
}

CvGasNode* CvNeuralGas::get_bmu() const {
    return bmu;
}

CvGasNode* CvNeuralGas::get_smu() const {
    return smu;
}

CvGasNode* CvNeuralGas::get_wmu() const {
    return wmu;
}

CvScalar* CvNeuralGas::get_input() const {
    return input;
}

unsigned int CvNeuralGas::get_total_nodes() const {
    return total_nodes;
}
