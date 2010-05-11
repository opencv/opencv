/* This file is to prevent problems with swig <= 1.3.25 which use
   PyInt_AS_LONG and may cause some corruption on RedHat systems.

   Include it in every .i file who generate a C/C++ file */

%runtime%{
/* define the PyAPI_FUNC macro if it doesn't exist, for example with Python
   version below 2.3... But not really tested... */
#ifndef PyAPI_FUNC
#       define PyAPI_FUNC(RTYPE) RTYPE
#endif

/* remove the PyInt_AS_LONG if defined, as this cause problems on RedHat */
#ifdef PyInt_AS_LONG
#undef PyInt_AS_LONG
#endif

/* wrapper to the better function PyInt_AsLong, removing problems
   with RedHat (I hope) */
long PyInt_AS_LONG (PyObject *obj) {
    return PyInt_AsLong (obj);
}

/* remove the PyFloat_AS_DOUBLE if defined, to prevent errors */
#ifdef PyFloat_AS_DOUBLE
#undef PyFloat_AS_DOUBLE
#endif

/* wrapper to the better function PyFloat_AS_DOUBLE, to prevent errors */
double PyFloat_AS_DOUBLE (PyObject *obj) {
    return PyFloat_AsDouble (obj);
}
%}

