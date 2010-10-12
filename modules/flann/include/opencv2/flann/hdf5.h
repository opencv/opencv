/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/


#ifndef IO_H_
#define IO_H_

#include <H5Cpp.h>

#include "opencv2/flann/matrix.h"



#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

namespace cvflann 
{


namespace {

template<typename T>
PredType get_hdf5_type()
{
	throw FLANNException("Unsupported type for IO operations");
}

template<> PredType get_hdf5_type<char>() { return PredType::NATIVE_CHAR; }
template<> PredType get_hdf5_type<unsigned char>() { return PredType::NATIVE_UCHAR; }
template<> PredType get_hdf5_type<short int>() { return PredType::NATIVE_SHORT; }
template<> PredType get_hdf5_type<unsigned short int>() { return PredType::NATIVE_USHORT; }
template<> PredType get_hdf5_type<int>() { return PredType::NATIVE_INT; }
template<> PredType get_hdf5_type<unsigned int>() { return PredType::NATIVE_UINT; }
template<> PredType get_hdf5_type<long>() { return PredType::NATIVE_LONG; }
template<> PredType get_hdf5_type<unsigned long>() { return PredType::NATIVE_ULONG; }
template<> PredType get_hdf5_type<float>() { return PredType::NATIVE_FLOAT; }
template<> PredType get_hdf5_type<double>() { return PredType::NATIVE_DOUBLE; }
template<> PredType get_hdf5_type<long double>() { return PredType::NATIVE_LDOUBLE; }

}


template<typename T>
void save_to_file(const cvflann::Matrix<T>& flann_dataset, const std::string& filename, const std::string& name)
{
	// Try block to detect exceptions raised by any of the calls inside it
	try
	{
		/*
		 * Turn off the auto-printing when failure occurs so that we can
		 * handle the errors appropriately
		 */
		Exception::dontPrint();

		/*
		 * Create a new file using H5F_ACC_TRUNC access,
		 * default file creation properties, and default file
		 * access properties.
		 */
		H5File file( filename, H5F_ACC_TRUNC );

		/*
		 * Define the size of the array and create the data space for fixed
		 * size dataset.
		 */
		hsize_t     dimsf[2];              // dataset dimensions
		dimsf[0] = flann_dataset.rows;
		dimsf[1] = flann_dataset.cols;
		DataSpace dataspace( 2, dimsf );

		/*
		 * Create a new dataset within the file using defined dataspace and
		 * datatype and default dataset creation properties.
		 */
		DataSet dataset = file.createDataSet( name, get_hdf5_type<T>(), dataspace );

		/*
		 * Write the data to the dataset using default memory space, file
		 * space, and transfer properties.
		 */
		dataset.write( flann_dataset.data, get_hdf5_type<T>() );
	}  // end of try block
	catch( H5::Exception& error )
	{
		error.printError();
		throw FLANNException(error.getDetailMsg());
	}
}


template<typename T>
void load_from_file(cvflann::Matrix<T>& flann_dataset, const std::string& filename, const std::string& name)
{
	try
	{
		Exception::dontPrint();

		H5File file( filename, H5F_ACC_RDONLY );
		DataSet dataset = file.openDataSet( name );

		/*
		 * Check the type used by the dataset matches
		 */
		if ( !(dataset.getDataType()==get_hdf5_type<T>())) {
			throw FLANNException("Dataset matrix type does not match the type to be read.");
		}

		/*
		 * Get dataspace of the dataset.
		 */
		DataSpace dataspace = dataset.getSpace();

		/*
		 * Get the dimension size of each dimension in the dataspace and
		 * display them.
		 */
		hsize_t dims_out[2];
		dataspace.getSimpleExtentDims( dims_out, NULL);
		
		flann_dataset.rows = dims_out[0];
		flann_dataset.cols = dims_out[1];
		flann_dataset.data = new T[flann_dataset.rows*flann_dataset.cols];

		dataset.read( flann_dataset.data, get_hdf5_type<T>() );
	}  // end of try block
	catch( H5::Exception &error )
	{
		error.printError();
		throw FLANNException(error.getDetailMsg());
	}
}


} // namespace cvflann

#endif /* IO_H_ */
