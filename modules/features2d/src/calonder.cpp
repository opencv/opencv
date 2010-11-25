//*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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
#include <cstdio>
#include <iostream>
#include <fstream>

using namespace std;

class CSMatrixGenerator {
public:
   typedef enum { PDT_GAUSS=1, PDT_BERNOULLI, PDT_DBFRIENDLY } PHI_DISTR_TYPE;
   ~CSMatrixGenerator();
   static float* getCSMatrix(int m, int n, PHI_DISTR_TYPE dt);     // do NOT free returned pointer


private:
   static float *cs_phi_;    // matrix for compressive sensing
   static int cs_phi_m_, cs_phi_n_;
};

float* CSMatrixGenerator::getCSMatrix(int m, int n, PHI_DISTR_TYPE dt)
{
   assert(m <= n);

   if (cs_phi_m_!=m || cs_phi_n_!=n || cs_phi_==NULL) {
      if (cs_phi_) delete [] cs_phi_;
      cs_phi_ = new float[m*n];
   }

   #if 0 // debug - load the random matrix from a file (for reproducability of results)
      //assert(m == 176);
      //assert(n == 500);
      //const char *phi = "/u/calonder/temp/dim_red/kpca_phi.txt";
      const char *phi = "/u/calonder/temp/dim_red/debug_phi.txt";
      std::ifstream ifs(phi);
      for (size_t i=0; i<m*n; ++i) {
         if (!ifs.good()) {
            printf("[ERROR] RandomizedTree::makeRandomMeasMatrix: problem reading '%s'\n", phi);
            exit(0);
         }
         ifs >> cs_phi[i];
      }
      ifs.close();

      static bool warned=false;
      if (!warned) {
         printf("[NOTE] RT: reading %ix%i PHI matrix from '%s'...\n", m, n, phi);
         warned=true;
      }

      return;
   #endif

   float *cs_phi = cs_phi_;

   if (m == n) {
      // special case - set to 0 for safety
      memset(cs_phi, 0, m*n*sizeof(float));
      printf("[WARNING] %s:%i: square CS matrix (-> no reduction)\n", __FILE__, __LINE__);
   }
   else {
       cv::RNG rng(23);

      // par is distr param, cf 'Favorable JL Distributions' (Baraniuk et al, 2006)
      if (dt == PDT_GAUSS) {
         float par = (float)(1./m);
         for (int i=0; i<m*n; ++i)
            *cs_phi++ = (float)rng.gaussian(par);
      }
      else if (dt == PDT_BERNOULLI) {
         float par = (float)(1./sqrt((float)m));
         for (int i=0; i<m*n; ++i)
            *cs_phi++ = (rng(2)==0 ? par : -par);
      }
      else if (dt == PDT_DBFRIENDLY) {
         float par = (float)sqrt(3./m);
         for (int i=0; i<m*n; ++i) {
            int r = rng(6);
            *cs_phi++ = (r==0 ? par : (r==1 ? -par : 0.f));
         }
      }
      else
         throw("PHI_DISTR_TYPE not implemented.");
   }

   return cs_phi_;
}

CSMatrixGenerator::~CSMatrixGenerator()
{
   if (cs_phi_) delete [] cs_phi_;
   cs_phi_ = NULL;
}

float *CSMatrixGenerator::cs_phi_   = NULL;
int    CSMatrixGenerator::cs_phi_m_ = 0;
int    CSMatrixGenerator::cs_phi_n_ = 0;


inline void addVec(int size, const float* src1, const float* src2, float* dst)
{
  while(--size >= 0) {
    *dst = *src1 + *src2;
    ++dst; ++src1; ++src2;
  }
}


// sum up 50 byte vectors of length 176
// assume 4 bits max for input vector values
// final shift is 2 bits right
// temp buffer should be twice as long as signature
// sig and buffer need not be initialized
inline void sum_50t_176c(uchar **pp, uchar *sig, unsigned short *temp)
{
#if CV_SSE2
  __m128i acc, *acc1, *acc2, *acc3, *acc4, tzero;
  __m128i *ssig, *ttemp;

  ssig = (__m128i *)sig;
  ttemp = (__m128i *)temp;

  // empty ttemp[]
  tzero = _mm_set_epi32(0, 0, 0, 0);
  for (int i=0; i<22; i++)
    ttemp[i] = tzero;

  for (int j=0; j<48; j+=16)
    {
      // empty ssig[]
      for (int i=0; i<11; i++)
    ssig[i] = tzero;

      for (int i=j; i<j+16; i+=4) // 4 columns at a time, to 16
    {
      acc1 = (__m128i *)pp[i];
      acc2 = (__m128i *)pp[i+1];
      acc3 = (__m128i *)pp[i+2];
      acc4 = (__m128i *)pp[i+3];

      // add next four columns
      acc = _mm_adds_epu8(acc1[0],acc2[0]);
      acc = _mm_adds_epu8(acc,acc3[0]);
      acc = _mm_adds_epu8(acc,acc4[1]);
      ssig[0] = _mm_adds_epu8(acc,ssig[0]);
      // add four columns
      acc = _mm_adds_epu8(acc1[1],acc2[1]);
      acc = _mm_adds_epu8(acc,acc3[1]);
      acc = _mm_adds_epu8(acc,acc4[1]);
      ssig[1] = _mm_adds_epu8(acc,ssig[1]);
      // add four columns
      acc = _mm_adds_epu8(acc1[2],acc2[2]);
      acc = _mm_adds_epu8(acc,acc3[2]);
      acc = _mm_adds_epu8(acc,acc4[2]);
      ssig[2] = _mm_adds_epu8(acc,ssig[2]);
      // add four columns
      acc = _mm_adds_epu8(acc1[3],acc2[3]);
      acc = _mm_adds_epu8(acc,acc3[3]);
      acc = _mm_adds_epu8(acc,acc4[3]);
      ssig[3] = _mm_adds_epu8(acc,ssig[3]);
      // add four columns
      acc = _mm_adds_epu8(acc1[4],acc2[4]);
      acc = _mm_adds_epu8(acc,acc3[4]);
      acc = _mm_adds_epu8(acc,acc4[4]);
      ssig[4] = _mm_adds_epu8(acc,ssig[4]);
      // add four columns
      acc = _mm_adds_epu8(acc1[5],acc2[5]);
      acc = _mm_adds_epu8(acc,acc3[5]);
      acc = _mm_adds_epu8(acc,acc4[5]);
      ssig[5] = _mm_adds_epu8(acc,ssig[5]);
      // add four columns
      acc = _mm_adds_epu8(acc1[6],acc2[6]);
      acc = _mm_adds_epu8(acc,acc3[6]);
      acc = _mm_adds_epu8(acc,acc4[6]);
      ssig[6] = _mm_adds_epu8(acc,ssig[6]);
      // add four columns
      acc = _mm_adds_epu8(acc1[7],acc2[7]);
      acc = _mm_adds_epu8(acc,acc3[7]);
      acc = _mm_adds_epu8(acc,acc4[7]);
      ssig[7] = _mm_adds_epu8(acc,ssig[7]);
      // add four columns
      acc = _mm_adds_epu8(acc1[8],acc2[8]);
      acc = _mm_adds_epu8(acc,acc3[8]);
      acc = _mm_adds_epu8(acc,acc4[8]);
      ssig[8] = _mm_adds_epu8(acc,ssig[8]);
      // add four columns
      acc = _mm_adds_epu8(acc1[9],acc2[9]);
      acc = _mm_adds_epu8(acc,acc3[9]);
      acc = _mm_adds_epu8(acc,acc4[9]);
      ssig[9] = _mm_adds_epu8(acc,ssig[9]);
      // add four columns
      acc = _mm_adds_epu8(acc1[10],acc2[10]);
      acc = _mm_adds_epu8(acc,acc3[10]);
      acc = _mm_adds_epu8(acc,acc4[10]);
      ssig[10] = _mm_adds_epu8(acc,ssig[10]);
    }

      // unpack to ttemp buffer and add
      ttemp[0] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[0],tzero),ttemp[0]);
      ttemp[1] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[0],tzero),ttemp[1]);
      ttemp[2] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[1],tzero),ttemp[2]);
      ttemp[3] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[1],tzero),ttemp[3]);
      ttemp[4] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[2],tzero),ttemp[4]);
      ttemp[5] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[2],tzero),ttemp[5]);
      ttemp[6] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[3],tzero),ttemp[6]);
      ttemp[7] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[3],tzero),ttemp[7]);
      ttemp[8] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[4],tzero),ttemp[8]);
      ttemp[9] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[4],tzero),ttemp[9]);
      ttemp[10] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[5],tzero),ttemp[10]);
      ttemp[11] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[5],tzero),ttemp[11]);
      ttemp[12] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[6],tzero),ttemp[12]);
      ttemp[13] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[6],tzero),ttemp[13]);
      ttemp[14] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[7],tzero),ttemp[14]);
      ttemp[15] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[7],tzero),ttemp[15]);
      ttemp[16] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[8],tzero),ttemp[16]);
      ttemp[17] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[8],tzero),ttemp[17]);
      ttemp[18] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[9],tzero),ttemp[18]);
      ttemp[19] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[9],tzero),ttemp[19]);
      ttemp[20] = _mm_add_epi16(_mm_unpacklo_epi8(ssig[10],tzero),ttemp[20]);
      ttemp[21] = _mm_add_epi16(_mm_unpackhi_epi8(ssig[10],tzero),ttemp[21]);
    }

  // create ssignature from 16-bit result
  ssig[0] =_mm_packus_epi16(_mm_srai_epi16(ttemp[0],2),_mm_srai_epi16(ttemp[1],2));
  ssig[1] =_mm_packus_epi16(_mm_srai_epi16(ttemp[2],2),_mm_srai_epi16(ttemp[3],2));
  ssig[2] =_mm_packus_epi16(_mm_srai_epi16(ttemp[4],2),_mm_srai_epi16(ttemp[5],2));
  ssig[3] =_mm_packus_epi16(_mm_srai_epi16(ttemp[6],2),_mm_srai_epi16(ttemp[7],2));
  ssig[4] =_mm_packus_epi16(_mm_srai_epi16(ttemp[8],2),_mm_srai_epi16(ttemp[9],2));
  ssig[5] =_mm_packus_epi16(_mm_srai_epi16(ttemp[10],2),_mm_srai_epi16(ttemp[11],2));
  ssig[6] =_mm_packus_epi16(_mm_srai_epi16(ttemp[12],2),_mm_srai_epi16(ttemp[13],2));
  ssig[7] =_mm_packus_epi16(_mm_srai_epi16(ttemp[14],2),_mm_srai_epi16(ttemp[15],2));
  ssig[8] =_mm_packus_epi16(_mm_srai_epi16(ttemp[16],2),_mm_srai_epi16(ttemp[17],2));
  ssig[9] =_mm_packus_epi16(_mm_srai_epi16(ttemp[18],2),_mm_srai_epi16(ttemp[19],2));
  ssig[10] =_mm_packus_epi16(_mm_srai_epi16(ttemp[20],2),_mm_srai_epi16(ttemp[21],2));
#else
  CV_Error( CV_StsNotImplemented, "Not supported without SSE2" );
#endif
}

namespace cv
{
RandomizedTree::RandomizedTree()
  : posteriors_(NULL), posteriors2_(NULL)
{
}

RandomizedTree::~RandomizedTree()
{
   freePosteriors(3);
}

void RandomizedTree::createNodes(int num_nodes, RNG &rng)
{
  nodes_.reserve(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    nodes_.push_back( RTreeNode((uchar)rng(RandomizedTree::PATCH_SIZE),
                                (uchar)rng(RandomizedTree::PATCH_SIZE),
                                (uchar)rng(RandomizedTree::PATCH_SIZE),
                                (uchar)rng(RandomizedTree::PATCH_SIZE)) );
  }
}

int RandomizedTree::getIndex(uchar* patch_data) const
{
  int index = 0;
  for (int d = 0; d < depth_; ++d) {
    int child_offset = nodes_[index](patch_data);
    index = 2*index + 1 + child_offset;
  }
  return (int)(index - nodes_.size());
}

void RandomizedTree::train(std::vector<BaseKeypoint> const& base_set,
                           RNG &rng, int depth, int views, size_t reduced_num_dim,
                           int num_quant_bits)
{
  PatchGenerator make_patch;
  train(base_set, rng, make_patch, depth, views, reduced_num_dim, num_quant_bits);
}

void RandomizedTree::train(std::vector<BaseKeypoint> const& base_set,
                           RNG &rng, PatchGenerator &make_patch,
                           int depth, int views, size_t reduced_num_dim,
                           int num_quant_bits)
{
  init((int)base_set.size(), depth, rng);

  Mat patch;

  // Estimate posterior probabilities using random affine views
  std::vector<BaseKeypoint>::const_iterator keypt_it;
  int class_id = 0;
  Size patchSize(PATCH_SIZE, PATCH_SIZE);
  for (keypt_it = base_set.begin(); keypt_it != base_set.end(); ++keypt_it, ++class_id) {
    for (int i = 0; i < views; ++i) {
      make_patch( Mat(keypt_it->image), Point(keypt_it->x, keypt_it->y ), patch, patchSize, rng );
      IplImage iplPatch = patch;
      addExample(class_id, getData(&iplPatch));
    }
  }

  finalize(reduced_num_dim, num_quant_bits);
}

void RandomizedTree::allocPosteriorsAligned(int num_leaves, int num_classes)
{
  freePosteriors(3);

  posteriors_ = new float*[num_leaves]; //(float**) malloc(num_leaves*sizeof(float*));
  for (int i=0; i<num_leaves; ++i) {
    posteriors_[i] = (float*)cvAlloc(num_classes*sizeof(posteriors_[i][0]));
    memset(posteriors_[i], 0, num_classes*sizeof(float));
  }

  posteriors2_ = new uchar*[num_leaves];
  for (int i=0; i<num_leaves; ++i) {
    posteriors2_[i] = (uchar*)cvAlloc(num_classes*sizeof(posteriors2_[i][0]));
    memset(posteriors2_[i], 0, num_classes*sizeof(uchar));
  }

  classes_ = num_classes;
}

void RandomizedTree::freePosteriors(int which)
{
   if (posteriors_ && (which&1)) {
      for (int i=0; i<num_leaves_; ++i)
         if (posteriors_[i])
            cvFree( &posteriors_[i] );
      delete [] posteriors_;
      posteriors_ = NULL;
   }

   if (posteriors2_ && (which&2)) {
      for (int i=0; i<num_leaves_; ++i)
         cvFree( &posteriors2_[i] );
      delete [] posteriors2_;
      posteriors2_ = NULL;
   }

   classes_ = -1;
}

void RandomizedTree::init(int num_classes, int depth, RNG &rng)
{
  depth_ = depth;
  num_leaves_ = 1 << depth;        // 2**d
  int num_nodes = num_leaves_ - 1; // 2**d - 1

  // Initialize probabilities and counts to 0
  allocPosteriorsAligned(num_leaves_, num_classes);      // will set classes_ correctly
  for (int i = 0; i < num_leaves_; ++i)
    memset((void*)posteriors_[i], 0, num_classes*sizeof(float));
  leaf_counts_.resize(num_leaves_);

  for (int i = 0; i < num_leaves_; ++i)
    memset((void*)posteriors2_[i], 0, num_classes*sizeof(uchar));

  createNodes(num_nodes, rng);
}

void RandomizedTree::addExample(int class_id, uchar* patch_data)
{
  int index = getIndex(patch_data);
  float* posterior = getPosteriorByIndex(index);
  ++leaf_counts_[index];
  ++posterior[class_id];
}

// returns the p% percentile of data (length n vector)
static float percentile(float *data, int n, float p)
{
   assert(n>0);
   assert(p>=0 && p<=1);
   std::vector<float> vec(data, data+n);
   std::sort(vec.begin(), vec.end());
   int ix = (int)(p*(n-1));
   return vec[ix];
}

void RandomizedTree::finalize(size_t reduced_num_dim, int num_quant_bits)
{
   // Normalize by number of patches to reach each leaf
   for (int index = 0; index < num_leaves_; ++index) {
      float* posterior = posteriors_[index];
      assert(posterior != NULL);
      int count = leaf_counts_[index];
      if (count != 0) {
         float normalizer = 1.0f / count;
         for (int c = 0; c < classes_; ++c) {
            *posterior *= normalizer;
            ++posterior;
         }
      }
   }
   leaf_counts_.clear();

   // apply compressive sensing
   if ((int)reduced_num_dim != classes_)
      compressLeaves(reduced_num_dim);
   else {
      static bool notified = false;
      if (!notified)
         printf("\n[OK] NO compression to leaves applied, dim=%i\n", (int)reduced_num_dim);
      notified = true;
   }

   // convert float-posteriors to char-posteriors (quantization step)
   makePosteriors2(num_quant_bits);
}

void RandomizedTree::compressLeaves(size_t reduced_num_dim)
{
   static bool warned = false;
   if (!warned) {
      printf("\n[OK] compressing leaves with phi %i x %i\n", (int)reduced_num_dim, (int)classes_);
      warned = true;
   }

   static bool warned2 = false;
   if ((int)reduced_num_dim == classes_) {
     if (!warned2)
       printf("[WARNING] RandomizedTree::compressLeaves:  not compressing because reduced_dim == classes()\n");
     warned2 = true;
     return;
   }

   // DO NOT FREE RETURNED POINTER
   float *cs_phi = CSMatrixGenerator::getCSMatrix((int)reduced_num_dim, classes_, CSMatrixGenerator::PDT_BERNOULLI);

   float *cs_posteriors = new float[num_leaves_ * reduced_num_dim];         // temp, num_leaves_ x reduced_num_dim
   for (int i=0; i<num_leaves_; ++i) {
      float *post = getPosteriorByIndex(i);
      float *prod = &cs_posteriors[i*reduced_num_dim];
      Mat A( (int)reduced_num_dim, classes_, CV_32FC1, cs_phi );
      Mat X( classes_, 1, CV_32FC1, post );
      Mat Y( (int)reduced_num_dim, 1, CV_32FC1, prod );
      Y = A*X;
   }

   // copy new posteriors
   freePosteriors(3);
   allocPosteriorsAligned(num_leaves_, (int)reduced_num_dim);
   for (int i=0; i<num_leaves_; ++i)
      memcpy(posteriors_[i], &cs_posteriors[i*reduced_num_dim], reduced_num_dim*sizeof(float));
   classes_ = (int)reduced_num_dim;

   delete [] cs_posteriors;
}

void RandomizedTree::makePosteriors2(int num_quant_bits)
{
   int N = (1<<num_quant_bits) - 1;

   float perc[2];
   estimateQuantPercForPosteriors(perc);

   assert(posteriors_ != NULL);
   for (int i=0; i<num_leaves_; ++i)
      quantizeVector(posteriors_[i], classes_, N, perc, posteriors2_[i]);

   // printf("makePosteriors2 quantization bounds: %.3e, %.3e (num_leaves=%i, N=%i)\n",
   //        perc[0], perc[1], num_leaves_, N);
}

void RandomizedTree::estimateQuantPercForPosteriors(float perc[2])
{
   // _estimate_ percentiles for this tree
   // TODO: do this more accurately
   assert(posteriors_ != NULL);
   perc[0] = perc[1] = .0f;
   for (int i=0; i<num_leaves_; i++) {
      perc[0] += percentile(posteriors_[i], classes_, GET_LOWER_QUANT_PERC());
      perc[1] += percentile(posteriors_[i], classes_, GET_UPPER_QUANT_PERC());
   }
   perc[0] /= num_leaves_;
   perc[1] /= num_leaves_;
}


float* RandomizedTree::getPosterior(uchar* patch_data)
{
  return const_cast<float*>(const_cast<const RandomizedTree*>(this)->getPosterior(patch_data));
}

const float* RandomizedTree::getPosterior(uchar* patch_data) const
{
  return getPosteriorByIndex( getIndex(patch_data) );
}

uchar* RandomizedTree::getPosterior2(uchar* patch_data)
{
  return const_cast<uchar*>(const_cast<const RandomizedTree*>(this)->getPosterior2(patch_data));
}

const uchar* RandomizedTree::getPosterior2(uchar* patch_data) const
{
  return getPosteriorByIndex2( getIndex(patch_data) );
}

void RandomizedTree::quantizeVector(float *vec, int dim, int N, float bnds[2], int clamp_mode)
{
   float map_bnd[2] = {0.f,(float)N};          // bounds of quantized target interval we're mapping to
   for (int k=0; k<dim; ++k, ++vec) {
      *vec = float(int((*vec - bnds[0])/(bnds[1] - bnds[0])*(map_bnd[1] - map_bnd[0]) + map_bnd[0]));
      // 0: clamp both, lower and upper values
      if (clamp_mode == 0)      *vec = (*vec<map_bnd[0]) ? map_bnd[0] : ((*vec>map_bnd[1]) ? map_bnd[1] : *vec);
      // 1: clamp lower values only
      else if (clamp_mode == 1) *vec = (*vec<map_bnd[0]) ? map_bnd[0] : *vec;
      // 2: clamp upper values only
      else if (clamp_mode == 2) *vec = (*vec>map_bnd[1]) ? map_bnd[1] : *vec;
      // 4: no clamping
      else if (clamp_mode == 4) ; // yep, nothing
      else {
         printf("clamp_mode == %i is not valid (%s:%i).\n", clamp_mode, __FILE__, __LINE__);
         exit(1);
      }
   }

}

void RandomizedTree::quantizeVector(float *vec, int dim, int N, float bnds[2], uchar *dst)
{
   int map_bnd[2] = {0, N};          // bounds of quantized target interval we're mapping to
   int tmp;
   for (int k=0; k<dim; ++k) {
      tmp = int((*vec - bnds[0])/(bnds[1] - bnds[0])*(map_bnd[1] - map_bnd[0]) + map_bnd[0]);
      *dst = (uchar)((tmp<0) ? 0 : ((tmp>N) ? N : tmp));
      ++vec;
      ++dst;
   }
}


void RandomizedTree::read(const char* file_name, int num_quant_bits)
{
  std::ifstream file(file_name, std::ifstream::binary);
  read(file, num_quant_bits);
  file.close();
}

void RandomizedTree::read(std::istream &is, int num_quant_bits)
{
  is.read((char*)(&classes_), sizeof(classes_));
  is.read((char*)(&depth_), sizeof(depth_));

  num_leaves_ = 1 << depth_;
  int num_nodes = num_leaves_ - 1;

  nodes_.resize(num_nodes);
  is.read((char*)(&nodes_[0]), num_nodes * sizeof(nodes_[0]));

  //posteriors_.resize(classes_ * num_leaves_);
  //freePosteriors(3);
  //printf("[DEBUG] reading: %i leaves, %i classes\n", num_leaves_, classes_);
  allocPosteriorsAligned(num_leaves_, classes_);
  for (int i=0; i<num_leaves_; i++)
    is.read((char*)posteriors_[i], classes_ * sizeof(*posteriors_[0]));

  // make char-posteriors from float-posteriors
  makePosteriors2(num_quant_bits);
}

void RandomizedTree::write(const char* file_name) const
{
  std::ofstream file(file_name, std::ofstream::binary);
  write(file);
  file.close();
}

void RandomizedTree::write(std::ostream &os) const
{
  if (!posteriors_) {
    printf("WARNING: Cannot write float posteriors (posteriors_ = NULL).\n");
    return;
  }

  os.write((char*)(&classes_), sizeof(classes_));
  os.write((char*)(&depth_), sizeof(depth_));

  os.write((char*)(&nodes_[0]), nodes_.size() * sizeof(nodes_[0]));
  for (int i=0; i<num_leaves_; i++) {
    os.write((char*)posteriors_[i], classes_ * sizeof(*posteriors_[0]));
  }
}


void RandomizedTree::savePosteriors(std::string url, bool append)
{
   std::ofstream file(url.c_str(), (append?std::ios::app:std::ios::out));
   for (int i=0; i<num_leaves_; i++) {
      float *post = posteriors_[i];
      char buf[20];
      for (int i=0; i<classes_; i++) {
         sprintf(buf, "%.10e", *post++);
         file << buf << ((i<classes_-1) ? " " : "");
      }
      file << std::endl;
   }
   file.close();
}

void RandomizedTree::savePosteriors2(std::string url, bool append)
{
   std::ofstream file(url.c_str(), (append?std::ios::app:std::ios::out));
   for (int i=0; i<num_leaves_; i++) {
      uchar *post = posteriors2_[i];
      for (int i=0; i<classes_; i++)
         file << int(*post++) << (i<classes_-1?" ":"");
      file << std::endl;
   }
   file.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RTreeClassifier::RTreeClassifier()
  : classes_(0)
{
  posteriors_ = NULL;
}

void RTreeClassifier::train(std::vector<BaseKeypoint> const& base_set,
                            RNG &rng, int num_trees, int depth,
                            int views, size_t reduced_num_dim,
                            int num_quant_bits)
{
  PatchGenerator make_patch;
  train(base_set, rng, make_patch, num_trees, depth, views, reduced_num_dim, num_quant_bits);
}

// Single-threaded version of train(), with progress output
void RTreeClassifier::train(std::vector<BaseKeypoint> const& base_set,
                            RNG &rng, PatchGenerator &make_patch, int num_trees,
                            int depth, int views, size_t reduced_num_dim,
                            int num_quant_bits)
{
  if (reduced_num_dim > base_set.size()) {
    printf("INVALID PARAMS in RTreeClassifier::train: reduced_num_dim{%i} > base_set.size(){%i}\n",
           (int)reduced_num_dim, (int)base_set.size());
    return;
  }

  num_quant_bits_ = num_quant_bits;
  classes_ = (int)reduced_num_dim; // base_set.size();
  original_num_classes_ = (int)base_set.size();
  trees_.resize(num_trees);

  printf("[OK] Training trees: base size=%i, reduced size=%i\n", (int)base_set.size(), (int)reduced_num_dim);

  int count = 1;
  printf("[OK] Trained 0 / %i trees", num_trees);  fflush(stdout);
  for( int ti = 0; ti < num_trees; ti++ ) {
    trees_[ti].train(base_set, rng, make_patch, depth, views, reduced_num_dim, num_quant_bits_);
    printf("\r[OK] Trained %i / %i trees", count++, num_trees);
    fflush(stdout);
  }

  printf("\n");
  countZeroElements();
  printf("\n\n");
}

void RTreeClassifier::getSignature(IplImage* patch, float *sig) const
{
  // Need pointer to 32x32 patch data
  uchar buffer[RandomizedTree::PATCH_SIZE * RandomizedTree::PATCH_SIZE];
  uchar* patch_data;
  if (patch->widthStep != RandomizedTree::PATCH_SIZE) {
    //printf("[INFO] patch is padded, data will be copied (%i/%i).\n",
    //       patch->widthStep, RandomizedTree::PATCH_SIZE);
    uchar* data = getData(patch);
    patch_data = buffer;
    for (int i = 0; i < RandomizedTree::PATCH_SIZE; ++i) {
      memcpy((void*)patch_data, (void*)data, RandomizedTree::PATCH_SIZE);
      data += patch->widthStep;
      patch_data += RandomizedTree::PATCH_SIZE;
    }
    patch_data = buffer;
  }
  else {
    patch_data = getData(patch);
  }

  memset((void*)sig, 0, classes_ * sizeof(float));
  std::vector<RandomizedTree>::const_iterator tree_it;

  // get posteriors
  float **posteriors = new float*[trees_.size()];  // TODO: move alloc outside this func
  float **pp = posteriors;
  for (tree_it = trees_.begin(); tree_it != trees_.end(); ++tree_it, pp++) {
    *pp = const_cast<float*>(tree_it->getPosterior(patch_data));
    assert(*pp != NULL);
  }

  // sum them up
  pp = posteriors;
  for (tree_it = trees_.begin(); tree_it != trees_.end(); ++tree_it, pp++)
    addVec(classes_, sig, *pp, sig);

  delete [] posteriors;
  posteriors = NULL;

  // full quantization (experimental)
  #if 0
    int n_max = 1<<8 - 1;
    int sum_max = (1<<4 - 1)*trees_.size();
    int shift = 0;
    while ((sum_max>>shift) > n_max) shift++;

    for (int i = 0; i < classes_; ++i) {
      sig[i] = int(sig[i] + .5) >> shift;
      if (sig[i]>n_max) sig[i] = n_max;
    }

    static bool warned = false;
    if (!warned) {
      printf("[WARNING] Using full quantization (RTreeClassifier::getSignature)! shift=%i\n", shift);
      warned = true;
    }
  #else
    // TODO: get rid of this multiply (-> number of trees is known at train
    // time, exploit it in RandomizedTree::finalize())
    float normalizer = 1.0f / trees_.size();
    for (int i = 0; i < classes_; ++i)
      sig[i] *= normalizer;
  #endif
}

void RTreeClassifier::getSignature(IplImage* patch, uchar *sig) const
{
  // Need pointer to 32x32 patch data
  uchar buffer[RandomizedTree::PATCH_SIZE * RandomizedTree::PATCH_SIZE];
  uchar* patch_data;
  if (patch->widthStep != RandomizedTree::PATCH_SIZE) {
    //printf("[INFO] patch is padded, data will be copied (%i/%i).\n",
    //       patch->widthStep, RandomizedTree::PATCH_SIZE);
    uchar* data = getData(patch);
    patch_data = buffer;
    for (int i = 0; i < RandomizedTree::PATCH_SIZE; ++i) {
      memcpy((void*)patch_data, (void*)data, RandomizedTree::PATCH_SIZE);
      data += patch->widthStep;
      patch_data += RandomizedTree::PATCH_SIZE;
    }
    patch_data = buffer;
  } else {
    patch_data = getData(patch);
  }

  std::vector<RandomizedTree>::const_iterator tree_it;

  // get posteriors
  if (posteriors_ == NULL)
    {
      posteriors_ = (uchar**)cvAlloc( trees_.size()*sizeof(posteriors_[0]) );
      ptemp_ = (unsigned short*)cvAlloc( classes_*sizeof(ptemp_[0]) );
    }
  /// @todo What is going on in the next 4 lines?
  uchar **pp = posteriors_;
  for (tree_it = trees_.begin(); tree_it != trees_.end(); ++tree_it, pp++)
    *pp = const_cast<uchar*>(tree_it->getPosterior2(patch_data));
  pp = posteriors_;

#if 1
     // SSE2 optimized code
     sum_50t_176c(pp, sig, ptemp_);    // sum them up
#else
     static bool warned = false;

     memset((void*)sig, 0, classes_ * sizeof(sig[0]));
     unsigned short *sig16 = new unsigned short[classes_];           // TODO: make member, no alloc here
     memset((void*)sig16, 0, classes_ * sizeof(sig16[0]));
     for (tree_it = trees_.begin(); tree_it != trees_.end(); ++tree_it, pp++)
       addVec(classes_, sig16, *pp, sig16);

     // squeeze signatures into an uchar
     const bool full_shifting = true;
     int shift;
     if (full_shifting) {
        float num_add_bits_f = log((float)trees_.size())/log(2.f);     // # additional bits required due to summation
        int num_add_bits = int(num_add_bits_f);
        if (num_add_bits_f != float(num_add_bits)) ++num_add_bits;
        shift = num_quant_bits_ + num_add_bits - 8*sizeof(uchar);
//shift = num_quant_bits_ + num_add_bits - 2;
//shift = 6;
        if (shift>0)
          for (int i = 0; i < classes_; ++i)
            sig[i] = (sig16[i] >> shift);      // &3 cut off all but lowest 2 bits, 3(dec) = 11(bin)

        if (!warned)
           printf("[OK] RTC: quantizing by FULL RIGHT SHIFT, shift = %i\n", shift);
     }
     else {
        printf("[ERROR] RTC: not implemented!\n");
        exit(0);
     }

     if (!warned)
        printf("[WARNING] RTC: unoptimized signature computation\n");
     warned = true;
#endif
}


void RTreeClassifier::getSparseSignature(IplImage *patch, float *sig, float thresh) const
{
   getFloatSignature(patch, sig);
   for (int i=0; i<classes_; ++i, sig++)
      if (*sig < thresh) *sig = 0.f;
}

int RTreeClassifier::countNonZeroElements(float *vec, int n, double tol)
{
   int res = 0;
   while (n-- > 0)
      res += (fabs(*vec++) > tol);
   return res;
}

void RTreeClassifier::read(const char* file_name)
{
  std::ifstream file(file_name, std::ifstream::binary);
  read(file);
  file.close();
}

void RTreeClassifier::read(std::istream &is)
{
  int num_trees = 0;
  is.read((char*)(&num_trees), sizeof(num_trees));
  is.read((char*)(&classes_), sizeof(classes_));
  is.read((char*)(&original_num_classes_), sizeof(original_num_classes_));
  is.read((char*)(&num_quant_bits_), sizeof(num_quant_bits_));

  if (num_quant_bits_<1 || num_quant_bits_>8) {
    printf("[WARNING] RTC: suspicious value num_quant_bits_=%i found; setting to %i.\n",
           num_quant_bits_, (int)DEFAULT_NUM_QUANT_BITS);
    num_quant_bits_ = DEFAULT_NUM_QUANT_BITS;
  }

  trees_.resize(num_trees);
  std::vector<RandomizedTree>::iterator tree_it;

  for (tree_it = trees_.begin(); tree_it != trees_.end(); ++tree_it) {
    tree_it->read(is, num_quant_bits_);
  }

  printf("[OK] Loaded RTC, quantization=%i bits\n", num_quant_bits_);

  countZeroElements();
}

void RTreeClassifier::write(const char* file_name) const
{
  std::ofstream file(file_name, std::ofstream::binary);
  write(file);
  file.close();
}

void RTreeClassifier::write(std::ostream &os) const
{
  int num_trees = (int)trees_.size();
  os.write((char*)(&num_trees), sizeof(num_trees));
  os.write((char*)(&classes_), sizeof(classes_));
  os.write((char*)(&original_num_classes_), sizeof(original_num_classes_));
  os.write((char*)(&num_quant_bits_), sizeof(num_quant_bits_));
printf("RTreeClassifier::write: num_quant_bits_=%i\n", num_quant_bits_);

  std::vector<RandomizedTree>::const_iterator tree_it;
  for (tree_it = trees_.begin(); tree_it != trees_.end(); ++tree_it)
    tree_it->write(os);
}

void RTreeClassifier::saveAllFloatPosteriors(std::string url)
{
  printf("[DEBUG] writing all float posteriors to %s...\n", url.c_str());
  for (int i=0; i<(int)trees_.size(); ++i)
    trees_[i].savePosteriors(url, (i==0 ? false : true));
  printf("[DEBUG] done\n");
}

void RTreeClassifier::saveAllBytePosteriors(std::string url)
{
  printf("[DEBUG] writing all byte posteriors to %s...\n", url.c_str());
  for (int i=0; i<(int)trees_.size(); ++i)
    trees_[i].savePosteriors2(url, (i==0 ? false : true));
  printf("[DEBUG] done\n");
}


void RTreeClassifier::setFloatPosteriorsFromTextfile_176(std::string url)
{
   std::ifstream ifs(url.c_str());

   for (int i=0; i<(int)trees_.size(); ++i) {
      int num_classes = trees_[i].classes_;
      assert(num_classes == 176);     // TODO: remove this limitation (arose due to SSE2 optimizations)
      for (int k=0; k<trees_[i].num_leaves_; ++k) {
         float *post = trees_[i].getPosteriorByIndex(k);
         for (int j=0; j<num_classes; ++j, ++post)
            ifs >> *post;
         assert(ifs.good());
      }
   }
   classes_ = 176;

   //setQuantization(num_quant_bits_);

   ifs.close();
   printf("[EXPERIMENTAL] read entire tree from '%s'\n", url.c_str());
}


float RTreeClassifier::countZeroElements()
{
   size_t flt_zeros = 0;
   size_t ui8_zeros = 0;
   size_t num_elem = trees_[0].classes();
   for (int i=0; i<(int)trees_.size(); ++i)
      for (int k=0; k<(int)trees_[i].num_leaves_; ++k) {
         float *p = trees_[i].getPosteriorByIndex(k);
         uchar *p2 = trees_[i].getPosteriorByIndex2(k);
         assert(p); assert(p2);
         for (int j=0; j<num_elem; ++j, ++p, ++p2) {
            if (*p == 0.f) flt_zeros++;
            if (*p2 == 0) ui8_zeros++;
         }
      }
   num_elem = trees_.size()*trees_[0].num_leaves_*num_elem;
   float flt_perc = 100.f*flt_zeros/num_elem;
   float ui8_perc = 100.f*ui8_zeros/num_elem;
   printf("[OK] RTC: overall %i/%i (%.3f%%) zeros in float leaves\n", flt_zeros, num_elem, flt_perc);
   printf("          overall %i/%i (%.3f%%) zeros in uint8 leaves\n", ui8_zeros, num_elem, ui8_perc);

   return flt_perc;
}

void RTreeClassifier::setQuantization(int num_quant_bits)
{
   for (int i=0; i<(int)trees_.size(); ++i)
      trees_[i].applyQuantization(num_quant_bits);

   printf("[OK] signature quantization is now %i bits (before: %i)\n", num_quant_bits, num_quant_bits_);
   num_quant_bits_ = num_quant_bits;
}

void RTreeClassifier::discardFloatPosteriors()
{
   for (int i=0; i<(int)trees_.size(); ++i)
      trees_[i].discardFloatPosteriors();
   printf("[OK] RTC: discarded float posteriors of all trees\n");
}

}
