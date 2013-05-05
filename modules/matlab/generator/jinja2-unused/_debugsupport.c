/**
 * jinja2._debugsupport
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * C implementation of `tb_set_next`.
 *
 * :copyright: (c) 2010 by the Jinja Team.
 * :license: BSD.
 */

#include <Python.h>


static PyObject*
tb_set_next(PyObject *self, PyObject *args)
{
	PyTracebackObject *tb, *old;
	PyObject *next;

	if (!PyArg_ParseTuple(args, "O!O:tb_set_next", &PyTraceBack_Type, &tb, &next))
		return NULL;
	if (next == Py_None)
		next = NULL;
	else if (!PyTraceBack_Check(next)) {
		PyErr_SetString(PyExc_TypeError,
				"tb_set_next arg 2 must be traceback or None");
		return NULL;
	}
	else
		Py_INCREF(next);

	old = tb->tb_next;
	tb->tb_next = (PyTracebackObject*)next;
	Py_XDECREF(old);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyMethodDef module_methods[] = {
	{"tb_set_next", (PyCFunction)tb_set_next, METH_VARARGS,
	 "Set the tb_next member of a traceback object."},
	{NULL, NULL, 0, NULL}		/* Sentinel */
};


#if PY_MAJOR_VERSION < 3

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_debugsupport(void)
{
	Py_InitModule3("jinja2._debugsupport", module_methods, "");
}

#else /* Python 3.x module initialization */

static struct PyModuleDef module_definition = {
        PyModuleDef_HEAD_INIT,
	"jinja2._debugsupport",
	NULL,
	-1,
	module_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__debugsupport(void)
{
	return PyModule_Create(&module_definition);
}

#endif
