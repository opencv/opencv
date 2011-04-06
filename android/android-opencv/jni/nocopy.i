/* 
 * int *INTARRAY  typemaps. 
 * These are input typemaps for mapping a Java int[] array to a C int array.
 * Note that as a Java array is used and thus passeed by reference, the C routine 
 * can return data to Java via the parameter.
 *
 * Example usage wrapping:
 *   void foo((int *INTARRAY, int INTARRAYSIZE);
 *  
 * Java usage:
 *   byte b[] = new byte[20];
 *   modulename.foo(b);
 */

%typemap(in) (int *INTARRAY, int INTARRAYSIZE) {
    $1 = (int *) JCALL2(GetIntArrayElements, jenv, $input, 0); 
    jsize sz = JCALL1(GetArrayLength, jenv, $input);
    $2 = (int)sz;
}

%typemap(argout) (int *INTARRAY, int INTARRAYSIZE) {
    JCALL3(ReleaseIntArrayElements, jenv, $input, (jint *) $1, 0); 
}


/* Prevent default freearg typemap from being used */
%typemap(freearg) (int *INTARRAY, int INTARRAYSIZE) ""

%typemap(jni) (int *INTARRAY, int INTARRAYSIZE) "jintArray"
%typemap(jtype) (int *INTARRAY, int INTARRAYSIZE) "int[]"
%typemap(jstype) (int *INTARRAY, int INTARRAYSIZE) "int[]"
%typemap(javain) (int *INTARRAY, int INTARRAYSIZE) "$javainput"




