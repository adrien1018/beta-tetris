#include "train.h"
#include "train_py.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>

static void TrainingManagerDealloc(TrainingManager* self) {
  self->~TrainingManager();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* TrainingManagerNew(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  TrainingManager* self = (TrainingManager*)type->tp_alloc(type, 0);
  // leave initialization to __init__
  return (PyObject*)self;
}

static int TrainingManagerInit(TrainingManager* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"freeze_multiplier", nullptr};
  int freeze = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p", (char**)kwlist, &freeze)) {
    return -1;
  }
  new(self) TrainingManager(freeze);
  return 0;
}

static PyObject* TrainingManager_getitem(TrainingManager* self, PyObject* args) {
  int idx;
  if (!PyArg_ParseTuple(args, "i", &idx)) {
    return nullptr;
  }
  return Py_BuildValue("O", &(*self)[idx]);
}

static PyObject* TrainingManager_SetResetParams(TrainingManager* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"pre_trans", "penalty_multiplier", "reward_ratio", nullptr};
  ResetParams params;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|fdd", (char**)kwlist,
        &params.pre_trans, &params.penalty_multiplier, &params.reward_ratio)) {
    return nullptr;
  }
  self->SetResetParams(params);
  Py_RETURN_NONE;
}

static PyObject* TrainingManager_Init(TrainingManager* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"size", "seed", nullptr};
  int size;
  unsigned long long seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|K", (char**)kwlist, &size, &seed)) {
    return nullptr;
  }
  auto state = self->Init(size, seed);
  npy_intp dims_state[] = {size, std::tuple_size<Tetris::State>::value, Tetris::kN, Tetris::kM};
  PyObject* ret = PyArray_SimpleNew(4, dims_state, NPY_FLOAT32);
  memcpy(PyArray_DATA((PyArrayObject*)ret), state.data(), sizeof(Tetris::State) * size);
  return ret;
}

static PyObject* TrainingManager_ResetGame(TrainingManager* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"index", nullptr};
  int idx;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", (char**)kwlist, &idx)) {
    return nullptr;
  }
  self->ResetGame(idx);
  Py_RETURN_NONE;
}

static PyObject* MapToDict(const decltype(TrainingManager::ActionResult::info)::value_type& mp) {
  PyObject* py_dict = PyDict_New();
  for (const auto& [key, value] : mp) {
    PyObject* py_key = PyUnicode_FromString(key.c_str());
    if (value.index() == 0) {
      PyObject* py_value = PyLong_FromLong(std::get<int>(value));
      PyDict_SetItem(py_dict, py_key, py_value);
      Py_DECREF(py_value);
    } else {
      PyObject* py_value = PyFloat_FromDouble(std::get<double>(value));
      PyDict_SetItem(py_dict, py_key, py_value);
      Py_DECREF(py_value);
    }
    Py_DECREF(py_key);
  }
  return py_dict;
}

static PyObject* TrainingManager_Step(TrainingManager* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"actions", nullptr};
  PyObject* py_arr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &py_arr)) {
    return nullptr;
  }

  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(py_arr);
  if (!PyArray_Check(np_array)) {
    PyErr_SetString(PyExc_TypeError, "Argument must be a NumPy array.");
    return nullptr;
  }
  if (PyArray_TYPE(np_array) != NPY_INT) {
    PyArray_Descr* dtype = PyArray_DescrFromType(NPY_INT32);
    np_array = (PyArrayObject*)PyArray_CastToType(np_array, dtype, NPY_ARRAY_DEFAULT);
    if (!np_array) {
      PyErr_SetString(PyExc_TypeError, "NumPy array conversion failed.");
      return nullptr;
    }
  }

  int* data = static_cast<int*>(PyArray_DATA(np_array));
  std::vector<int> actions(data, data + PyArray_SIZE(np_array));

  auto result = self->Step(actions);
  if (result.state.empty()) Py_RETURN_NONE;

  long N = result.state.size();
  npy_intp dims_state[] = {N, std::tuple_size<Tetris::State>::value, Tetris::kN, Tetris::kM};
  PyObject* ret_state = PyArray_SimpleNew(4, dims_state, NPY_FLOAT32);
  memcpy(PyArray_DATA((PyArrayObject*)ret_state), result.state.data(), sizeof(Tetris::State) * N);

  npy_intp dims_reward[] = {N, 2};
  PyObject* ret_reward = PyArray_SimpleNew(2, dims_reward, NPY_FLOAT64);
  memcpy(PyArray_DATA((PyArrayObject*)ret_reward), result.reward.data(), sizeof(double) * 2 * N);

  npy_intp dims_over[] = {N};
  PyObject* ret_over = PyArray_SimpleNew(1, dims_over, NPY_INT8);
  memcpy(PyArray_DATA((PyArrayObject*)ret_over), result.is_over.data(), N);

  PyObject* ret_info = PyList_New(result.info.size());
  for (size_t i = 0; i < result.info.size(); i++) {
    PyList_SET_ITEM(ret_info, i, MapToDict(result.info[i]));
  }

  PyObject* ret = PyTuple_Pack(4, ret_state, ret_reward, ret_over, ret_info);
  Py_DECREF(ret_state);
  Py_DECREF(ret_reward);
  Py_DECREF(ret_over);
  Py_DECREF(ret_info);

  return ret;
}

static PyObject* TrainingManager_GetState(TrainingManager* self, PyObject* Py_UNUSED(ignored)) {
  auto mp = self->GetState();
  PyObject* dict = PyDict_New();
  for (const auto& [key, value] : mp) {
    PyObject* py_key = Py_BuildValue("(iii)",
        key.hz_mode, key.step_points, key.drought_mode);
    PyObject* py_value = Py_BuildValue("(dl)", value.first, value.second);
    PyDict_SetItem(dict, py_key, py_value);
    Py_DECREF(py_key);
    Py_DECREF(py_value);
  }
  return dict;
}

static PyObject* TrainingManager_LoadState(TrainingManager* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"state", nullptr};
  PyObject* state;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &state)) {
    return nullptr;
  }
  if (!PyDict_Check(state)) {
    PyErr_SetString(PyExc_TypeError, "Argument must be a dictionary.");
    return nullptr;
  }
  TrainingManager::StateMap mp;
  PyObject *key, *value;
  for (Py_ssize_t pos = 0; PyDict_Next(state, &pos, &key, &value);) {
    NormalizingParams nkey;
    std::pair<double, int64_t> nval;
    if (!PyArg_ParseTuple(key, "iii",
        &nkey.hz_mode, &nkey.step_points, &nkey.drought_mode)) {
      return nullptr;
    }
    if (!PyArg_ParseTuple(value, "dl", &nval.first, &nval.second)) {
      return nullptr;
    }
    mp.emplace(nkey, nval);
  }
  self->LoadState(std::move(mp));
  Py_RETURN_NONE;
}

static PyMethodDef py_training_manager_class_methods[] = {
    {"__getitem__", (PyCFunction)TrainingManager_getitem,
     METH_VARARGS, "Get a Tetris instance"},
    {"SetResetParams", (PyCFunction)TrainingManager_SetResetParams,
     METH_VARARGS | METH_KEYWORDS, "Set parameters for resetting games"},
    {"Init", (PyCFunction)TrainingManager_Init,
     METH_VARARGS | METH_KEYWORDS, "Initialize a pool with some agents"},
    {"ResetGame", (PyCFunction)TrainingManager_ResetGame,
     METH_VARARGS | METH_KEYWORDS, "Reset a specific agent"},
    {"Step", (PyCFunction)TrainingManager_Step,
     METH_VARARGS | METH_KEYWORDS, "Make a step on all agents and return training information"},
    {"GetState", (PyCFunction)TrainingManager_GetState,
     METH_NOARGS, "Get current manager state"},
    {"LoadState", (PyCFunction)TrainingManager_LoadState,
     METH_VARARGS | METH_KEYWORDS, "Load manager state"},
    {nullptr}};

PyTypeObject py_training_manager_class = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "tetris.TrainingManager", // tp_name
    sizeof(TrainingManager), // tp_basicsize
    0,                       // tp_itemsize
    (destructor)TrainingManagerDealloc, // tp_dealloc
    0,                       // tp_print
    0,                       // tp_getattr
    0,                       // tp_setattr
    0,                       // tp_reserved
    0,                       // tp_repr
    0,                       // tp_as_number
    0,                       // tp_as_sequence
    0,                       // tp_as_mapping
    0,                       // tp_hash
    0,                       // tp_call
    0,                       // tp_str
    0,                       // tp_getattro
    0,                       // tp_setattro
    0,                       // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "Maintain multiple environments for training with reward normalization", // tp_doc
    0,                       // tp_traverse
    0,                       // tp_clear
    0,                       // tp_richcompare
    0,                       // tp_weaklistoffset
    0,                       // tp_iter
    0,                       // tp_iternext
    py_training_manager_class_methods, // tp_methods
    0,                       // tp_members
    0,                       // tp_getset
    0,                       // tp_base
    0,                       // tp_dict
    0,                       // tp_descr_get
    0,                       // tp_descr_set
    0,                       // tp_dictoffset
    (initproc)TrainingManagerInit, // tp_init
    0,                       // tp_alloc
    TrainingManagerNew,      // tp_new
};
