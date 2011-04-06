/*
 * These typemaps provide support for sharing data between JNI and JVM code
 * using NIO direct buffers. It is the responsibility of the JVM code to
 * allocate a direct buffer of the appropriate size.
 *
 * Example use:

 * Wrapping:
 * %include "buffers.i" 
 * %apply int* BUFF {int* buffer}
 * int read_foo_int(int* buffer);
 *
 * Java:
 * IntBuffer buffer = IntBuffer.allocateDirect(nInts*4).order(ByteOrder.nativeOrder()).asIntBuffer();
 * Example.read_foo_int(buffer);
 *

 * The following typemaps are defined:
 * void* BUFF           <--> javax.nio.Buffer
 * char* BUFF           <--> javax.nio.ByteBuffer
 * char* CBUFF          <--> javax.nio.CharBuffer
 * unsigned char* INBUFF/OUTBUFF  <--> javax.nio.ShortBuffer
 * short* BUFF          <--> javax.nio.ShortBuffer 
 * unsigned short* INBUFF/OUTBUFF <--> javax.nio.IntBuffer
 * int* BUFF            <--> javax.nio.IntBuffer
 * unsigned int* INBUFF/OUTBUFF   <--> javax.nio.LongBuffer
 * long* BUFF           <--> javax.nio.IntBuffer
 * unsigned long* INBUFF/OUTBUF  <--> javax.nio.LongBuffer
 * long long* BUFF      <--> javax.nio.LongBuffer
 * float* BUFF          <--> javax.nio.FloatBuffer
 * double* BUFF         <--> javax.nio.DoubleBuffer
 *
 * Note the potential for data loss in the conversion from 
 * the C type 'unsigned long' to the signed Java long type.
 * Hopefully, I can implement a workaround with BigNumber in the future.
 *
 * The use of ByteBuffer vs CharBuffer for the char* type should
 * depend on the type of data. In general you'll probably
 * want to use CharBuffer for actual text data.
 */
/*
 * This macro is used to define the nio buffers for primitive types.
 */
%define NIO_BUFFER_TYPEMAP(CTYPE, LABEL, BUFFERTYPE)
%typemap(jni) CTYPE* LABEL "jobject"
%typemap(jtype) CTYPE* LABEL "BUFFERTYPE"
%typemap(jstype) CTYPE* LABEL "BUFFERTYPE"
%typemap(javain, 
	pre="    assert $javainput.isDirect() : \"Buffer must be allocated direct.\";") CTYPE* LABEL "$javainput"
%typemap(javaout) CTYPE* LABEL {
    return $jnicall;
}
%typemap(in) CTYPE* LABEL {
  $1 = (CTYPE*)(jenv)->GetDirectBufferAddress( $input);
  if ($1 == NULL) {
    SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "Unable to get address of direct buffer. Buffer must be allocated direct.");
  }
}
%typemap(memberin) CTYPE* LABEL {
  if ($input) {
    $1 = $input;
  } else {
    $1 = 0;
  }
}
%typemap(freearg) CTYPE* LABEL ""
%enddef

NIO_BUFFER_TYPEMAP(void, BUFF, java.nio.Buffer);
NIO_BUFFER_TYPEMAP(char, BUFF, java.nio.ByteBuffer);
NIO_BUFFER_TYPEMAP(char, CBUFF, java.nio.CharBuffer);
/*NIO_BUFFER_TYPEMAP(unsigned char, BUFF, java.nio.ShortBuffer);*/
NIO_BUFFER_TYPEMAP(short, BUFF, java.nio.ShortBuffer);
NIO_BUFFER_TYPEMAP(unsigned short, BUFF, java.nio.IntBuffer);
NIO_BUFFER_TYPEMAP(int, BUFF, java.nio.IntBuffer);
NIO_BUFFER_TYPEMAP(unsigned int, BUFF, java.nio.LongBuffer);
NIO_BUFFER_TYPEMAP(long, BUFF, java.nio.IntBuffer);
NIO_BUFFER_TYPEMAP(unsigned long, BUFF, java.nio.LongBuffer);
NIO_BUFFER_TYPEMAP(long long, BUFF, java.nio.LongBuffer);
NIO_BUFFER_TYPEMAP(float, BUFF, java.nio.FloatBuffer);
NIO_BUFFER_TYPEMAP(double, BUFF, java.nio.DoubleBuffer);
#undef NIO_BUFFER_TYPEMAP


%define UNSIGNED_NIO_BUFFER_TYPEMAP(CTYPE, BSIZE, BUFFERTYPE, PACKFCN, UNPACKFCN)
%typemap(jni) CTYPE* INBUFF "jobject"
%typemap(jtype) CTYPE* INBUFF "java.nio.ByteBuffer"
%typemap(jstype) CTYPE* INBUFF "BUFFERTYPE"
%typemap(javain, 
	pre="    java.nio.ByteBuffer tmp$javainput = PACKFCN($javainput, true);") CTYPE* INBUFF "tmp$javainput"
%typemap(javaout) CTYPE* INBUFF {
    return $jnicall;
}
%typemap(in) CTYPE* INBUFF {
  $1 = (jenv)->GetDirectBufferAddress($input);
  if ($1 == NULL) {
    SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "Unable to get address of direct buffer. Buffer must be allocated direct.");
  }
}
%typemap(memberin) CTYPE* INBUFF {
  if ($input) {
    $1 = $input;
  } else {
    $1 = 0;
  }
}
%typemap(freearg) CTYPE* INBUFF ""

%typemap(jni) CTYPE* OUTBUFF "jobject"
%typemap(jtype) CTYPE* OUTBUFF "java.nio.ByteBuffer"
%typemap(jstype) CTYPE* OUTBUFF "BUFFERTYPE"
%typemap(javain, 
	pre="    java.nio.ByteBuffer tmp$javainput = java.nio.ByteBuffer.allocateDirect($javainput.capacity()*BSIZE).order($javainput.order());",
        post="       UNPACKFCN(tmp$javainput, $javainput);") CTYPE* OUTBUFF "tmp$javainput"
%typemap(javaout) CTYPE* OUTBUFF {
    return $jnicall;
}
%typemap(in) CTYPE* OUTBUFF {
  $1 = (jenv)->GetDirectBufferAddress( $input);
  if ($1 == NULL) {
    SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "Unable to get address of direct buffer. Buffer must be allocated direct.");
  }
}
%typemap(memberin) CTYPE* OUTBUFF {
  if ($input) {
    $1 = $input;
  } else {
    $1 = 0;
  }
}
%typemap(freearg) CTYPE* OUTBUFF ""
%enddef

UNSIGNED_NIO_BUFFER_TYPEMAP(unsigned char, 1, java.nio.ShortBuffer, permafrost.hdf.libhdf.BufferUtils.packUChar, permafrost.hdf.libhdf.BufferUtils.unpackUChar);
UNSIGNED_NIO_BUFFER_TYPEMAP(unsigned short, 2, java.nio.IntBuffer, permafrost.hdf.libhdf.BufferUtils.packUShort, permafrost.hdf.libhdf.BufferUtils.unpackUShort);
UNSIGNED_NIO_BUFFER_TYPEMAP(unsigned int, 4, java.nio.LongBuffer, permafrost.hdf.libhdf.BufferUtils.packUInt, permafrost.hdf.libhdf.BufferUtils.unpackUInt);
UNSIGNED_NIO_BUFFER_TYPEMAP(unsigned long, 4, java.nio.LongBuffer, permafrost.hdf.libhdf.BufferUtils.packUInt, permafrost.hdf.libhdf.BufferUtils.unpackUInt);

/*
%typemap(jni) unsigned char* BUFF "jobject"
%typemap(jtype) unsigned char* BUFF "java.nio.ByteBuffer"
%typemap(jstype) unsigned char* BUFF "java.nio.ShortBuffer"
%typemap(javain, 
	pre="    java.nio.ByteBuffer tmp$javainput = permafrost.hdf.libhdf.BufferUtils.packUChar($javainput, true);",
        post="      permafrost.hdf.libhdf.BufferUtils.unpackUChar(tmp$javainput, $javainput);") unsigned char* BUFF "tmp$javainput"
%typemap(javaout) unsigned char* BUFF {
    return $jnicall;
}
%typemap(in) unsigned char* BUFF {
  $1 = (const char*)(jenv)->GetDirectBufferAddress( $input);
  if ($1 == NULL) {
    SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "Unable to get address of direct buffer. Buffer must be allocated direct.");
  }
}
%typemap(memberin) unsigned char* BUFF {
  if ($input) {
    $1 = $input;
  } else {
    $1 = 0;
  }
}
%typemap(freearg) unsigned char* BUFF ""
*/

#undef UNSIGNED_NIO_BUFFER_TYPEMAP