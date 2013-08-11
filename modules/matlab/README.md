OpenCV Matlab Code Generator
============================
This module contains a code generator to automatically produce Matlab mex wrappers for other modules within the OpenCV library. Once compiled and added to the Matlab path, this gives users the ability to call OpenCV methods natively from within Matlab.


Build
-----
The Matlab code generator is fully integrated into the OpenCV build system. If cmake finds a Matlab installation available on the host system while configuring OpenCV, it will attempt to generate Matlab wrappers for all OpenCV modules. If cmake is having trouble finding your Matlab installation, you can explicitly point it to the root by defining the `MATLAB_ROOT_DIR` variable. For example, on a Mac you could type:

	cmake -DMATLAB_ROOT_DIR=/Applications/MATLAB_R2013a.app ..


Install
-------
In order to use the bindings, you will need to add them to the Matlab path. The path to add is:

1. ${CMAKE_BUILD_DIR}/modules/matlab if you are working from the build tree, or
2. ${CMAKE_INSTALL_PREFIX}/matlab if you have installed OpenCV

In Matlab, simply run:

    addpath('/path/to/opencv/matlab/');


Run
---
Once you've added the bindings directory to the Matlab path, you can start using them straight away! OpenCV calls need to be prefixed with a 'cv' qualifier, to disambiguate them from Matlab methods of the same name. For example, to compute the dft of a matrix, you might do the following:

    % load an image (Matlab)
    I = imread('cameraman.tif');

    % compute the DFT (OpenCV)
    If = cv.dft(I, cv.DFT_COMPLEX_OUTPUT);

As you can see, both OpenCV methods and constants can be used with 'cv' qualification. You can also call:

    help cv.dft

to get help on the purpose and call signature of a particular method. You can also call

    help cv

to get general help regarding the OpenCV bindings. If you ever run into issues with the bindings, you can call

	cv.buildInformation();
	
to get a printout of diagnostic information pertaining to your particular build of OS, OpenCV and Matlab. It is useful to submit this information alongside a bug report to the OpenCV team.

------------------------------------------------------------------


Developer
=========
The following sections contain information for developers seeking to use, understand or extend the Matlab bindings. The bindings are generated in python using a powerful templating engine called Jinja2. Because Matlab mex gateways have a common structure, they are well suited to templatization. There are separate templates for formatting C++ classes, Matlab classes, C++ functions, constants (enums) and documentation.

The task of the generator is two-fold:

1. To parse the OpenCV headers and build a semantic tree that can be fed to the template engine
2. To define type conversions between C++/OpenCV and Matlab types

Once a source file has been generated for each OpenCV definition, and type conversions have been written, compiling the files into mex gateways (shared objects) and linking to the OpenCV libraries is trivial.


File layout
-----------
opencv/modules/matlab (this module)  

* CMakeLists.txt (main cmake configuration file)
* README.md (this file)
* compile.cmake (the cmake help script for compiling generated source code)
* generator (the folder containing generator code)
  * jinja2 (the binding templating engine)
  * filters.py (template filters)
  * gen_matlab.py (the binding generator control script)
  * parse_tree.py (python class to refactor the hdr_parser.py output)
  * templates (the raw templates for populating classes, constants, functions and docs)
* include (C++ headers for the bindings)
  * mxarray.hpp (C++ OOP-style interface for Matlab mxArray* class)
  * bridge.hpp (type conversions)
  * map.hpp (hash map interface for instance storage and method lookup)
* io (FileStorage interface for .mat files)
* test (generator, compiler and binding test scripts)
  
  
Call Tree
---------
The cmake call tree can be broken into 3 main components:

1. configure time
2. build time
3. install time

**Find Matlab (configure)**  
The first thing to do is discover a Matlab installation on the host system. This is handled by the `OpenCVFindMatlab.cmake` in `opencv/cmake`. On Windows machines it searches the registry and path, while on *NIX machines it searches a set of canonical install paths. Once Matlab has been found, a number of variables are defined, such as the path to the mex compiler, the mex libraries, the mex include paths, the architectural extension, etc.

**Test the generator (configure)**  
Attempt to produce a source file for a simple definition. This tests whether python and pythonlibs are correctly invoked on the host.

**Test the mex compiler (configure)**  
Attempt to compile a simple definition using the mex compiler. A mex file is actually just a shared object with a special exported symbol `_mexFunction` which serves as the entry-point to the function. As such, the mex compiler is just a set of scripts configuring the system compiler. In most cases this is the same as the OpenCV compiler, but *could* be different. The test checks whether the mex and generator includes can be found, the system libraries can be linked and the passed compiler flags are compatible.

If any of the configure time tests fail, the bindings will be disabled, but the main OpenCV configure will continue without error. The configuration summary will contain the block:

	Matlab
		mex:					/Applications/MATLAB_R2013a.app/bin/mex
		compiler/generator:		Not working (bindings will not be generated)
			
**Generate the sources (build)**  
Given a set of modules (the intersection of the OpenCV modules being built and the matlab module optional dependencies), the `CppHeaderParser()` from `opencv/modules/python/src2/hdr_parser.py` will parse the module headers and produce a set of definitions.

The `ParseTree()` from `opencv/modules/matlab/generator/parse_tree.py` takes this set of definitions and refactors them into a semantic tree better suited to templatization. For example, a trivial definition from the header parser may look something like:

	[fill, void, ['/S'], [cv::Mat&, mat, '', ['/I', '/O']]]
	
The equivalent refactored output may look like:

	Function 
		name   = 'fill'
		rtype  = 'void'
		static = True
		req = 
			Argument
				name    = 'mat'
				type    = 'cv::Mat'
				ref     = '&'
				I       = True
				O       = True
				default = ''

The added semantics (Namespace, Class, Function, Argument, name, etc) makes it much easier for the templating engine to parse, slice and populate definitions.

Once the definitions have been parsed, `gen_matlab.py` passes each definition to the template engine with the appropriate template (class, function, enum, doc) and the filled template gets written to the `${CMAKE_CURRENT_BUILD_DIR}/src` directory.

The generator relies upon a proxy object called `generate.proxy` to determine when the sources are out of date and need to be re-generated.

**Compile the sources (build)**  
Once the sources have been generated, they are compiled by the mex compiler. The `compile.cmake` script in `opencv/modules/matlab/` takes responsibility for iterating over each source file in `${CMAKE_CURRENT_BUILD_DIR}/src` and compiling it with the passed includes and OpenCV libraries. 

The flags used to compile the main OpenCV libraries are also forwarded to the mex compiler. So if, for example, you compiled OpenCV with SSE support, the mex bindings will also use SSE. Likewise, if you compile OpenCV in debug mode, the bindings will link to the debug version of the libraries.

Importantly, the mex compiler includes the `mxarray.hpp`, `bridge.hpp` and `map.hpp` files from the `opencv/modules/matlab/include` directory. `mxarray.hpp` defines a `MxArray` class which wraps Matlab's `mxArray*` type in a more friendly OOP-syle interface. `bridge.hpp` defines a `Bridge` class which is able to perform type conversions between Matlab types and std/OpenCV types. It can be extended with new definitions using the plugin interface described in that file. 

**Install the files (install)**  
At install time, the mex files are put into place and their linkages updated.


Jinja2
------
Jinja2 is a powerful templating engine, similar to python's builtin `string.Template` class but implementing the model-view-controller paradigm:

**view.py**

	<title>{{ title }}</title>
	<ul>
	{% for user in users %}
		<li><a href="{{ user.url }}">{{ user.username | sanitize }}</a></li>
	{% endfor %}
	</ul>
	
**model.py**

	class User(object):
		__init__(self):
			self.username = ''
			self.url = ''

	def sanitize(text):
		"""Filter for escaping html tags to prevent code injection"""
		
**controller.py**

	def populate(users):
		# initialize jinja
		jtemplate = jinja2.Environment(loader=FileSystemLoader())
		
		# add the filters to the engine
		jtemplate['sanitize'] = sanitize
		
		# get the view
		template = jtemplate.get_template('view')
		
		# populate the template with a list of User objects
		populated = template.render(title='all users', users=users)
		
		# write to file
		with open('users.html', 'wb') as f:
			f.write(populated)

File Reference
--------------
**gen_matlab.py**  
gen_matlab has the following call signature:

	gen_matlab.py --hdrparser path/to/hdr_parser/dir
				  --rstparser path/to/rst_parser/dir
				  --moduleroot path/to/opencv/modules
				  --modules core imgproc highgui etc
				  --extra namespace=/additional/header/to/parse
				  --outdir /path/to/place/generated/src
				  
**build_info.py**  
build_info has the following call signature:

	build_info.py --os operating_system_string
				  --arch bitness processor
				  --compiler id version
				  --mex_arch arch_string
				  --mex_script /path/to/mex/script
				  --cxx_flags -list -of -flags -to -passthrough
				  --opencv_version version_string
				  --commit commit_hash_if_using_git
				  --modules core imgproc highgui etc
				  --configuration Debug/Release
				  --outdir path/to/place/build/info

**parse_tree.py**  
To build a parse tree, first parse a set of headers, then invoke the parse tree to refactor the output:

	# parse a set of definitions into a dictionary of namespaces
	parser = CppHeaderParser()
	ns['core'] = parser.parse('path/to/opencv/core.hpp')
	
	# refactor into a semantic tree
	parse_tree = ParseTree()
	parse_tree.build(ns)
	
	# iterate over the tree
	for namespace in parse_tree.namespaces:
		for clss in namespace.classes:
			# do stuff
		for method in namespace.methods:
			# do stuff

**mxarray.hpp**
mxarray.hpp defines a class called `MxArray` which provides an OOP-style interface for Matlab's homogeneous `mxArray*` type. To create an `MxArray`, you can either inherit an existing array

	MxArray mat(prhs[0]);
	
or create a new array

	MxArray mat(5, 5, Matlab::Traits<double>::ScalarType);
	MxArray mat = MxArray::Matrix<double>(5, 5);
	
The default constructor allocates a `0 x 0` array. Once you have encapculated an `mxArray` you can access its properties through member functions:

	mat.rows();
	mat.cols();
	mat.size();
	mat.channels();
	mat.isComplex();
	mat.isNumeric();
	mat.isLogical();
	mat.isClass();
	mat.className();
	mat.real();
	mat.imag();
	etc…
	
The MxArray object uses scoped memory management. If you wish to pass an MxArray back to Matlab (as a lhs pointer), you need to explicitly release ownership of the array so that it is not destroyed when it leaves scope:

	plhs[0] = mat.releaseOwnership();

**bridge.hpp**  
The bridge interface provides a `Bridge` class which provides type conversion between std/OpenCV and Matlab types. A type conversion must provide the following:

	Bridge& operator=(const MyObject&);
	MyObject toMyObject();
	operator MyObject();
	
The binding generator will then automatically call the conversion operators (either explicitly or implicitly) if your `MyObject` class is encountered as an input or return from a parsed definition.