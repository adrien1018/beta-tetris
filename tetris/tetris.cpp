#include "game_py.h"
#include "train_py.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>

static PyMethodDef py_tetris_module_methods[] = {
#ifdef DEBUG_METHODS
  {"GetClearCol", (PyCFunction)GetClearCol, METH_NOARGS,
   "Print line clear column stats"},
#endif
  {nullptr},
};
static PyModuleDef py_tetris_module = {
  PyModuleDef_HEAD_INIT,
  "tetris",
  "Tetris module",
  -1,
  py_tetris_module_methods,
};

PyMODINIT_FUNC PyInit_tetris() {
  import_array();
  if (PyType_Ready(&py_tetris_class) < 0 ||
      PyType_Ready(&py_training_manager_class) < 0) return nullptr;

  PyObject *m = PyModule_Create(&py_tetris_module);
  if (m == nullptr) return nullptr;

  PyObject *all = Py_BuildValue("[s,s]", "Tetris", "TrainingManager");
#ifdef DEBUG_METHODS
  PyList_Append(all, Py_BuildValue("s", "GetClearCol"));
#endif
  Py_INCREF(&py_tetris_class);
  Py_INCREF(&py_training_manager_class);

  if (PyModule_AddObject(m, "Tetris", (PyObject*)&py_tetris_class) < 0 ||
      PyModule_AddObject(m, "TrainingManager", (PyObject*)&py_training_manager_class) < 0 ||
      PyModule_AddObject(m, "__all__", all) < 0) {
    Py_DECREF(&py_tetris_class);
    Py_DECREF(&py_training_manager_class);
    Py_DECREF(m);
    Py_CLEAR(all);
    return nullptr;
  }
  return m;
}
