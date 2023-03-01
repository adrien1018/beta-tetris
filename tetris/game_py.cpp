#include "game.h"
#include "game_py.h"

#include <stdexcept>
#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>

static void TetrisDealloc(Tetris* self) {
  self->~Tetris();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* TetrisNew(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  Tetris* self = (Tetris*)type->tp_alloc(type, 0);
  // leave initialization to __init__
  return (PyObject*)self;
}

static int TetrisInit(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"seed", nullptr};
  unsigned long long seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|K", (char**)kwlist, &seed)) {
    return -1;
  }
  new(self) Tetris(seed);
  return 0;
}

static PyObject* Tetris_IsOver(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong((long)self->IsOver());
}

static PyObject* Tetris_InputPlacement(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"rotate", "x", "y", "training", nullptr};
  int rotate, x, y, training = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii|p", (char**)kwlist, &rotate,
                                   &x, &y, &training)) {
    return nullptr;
  }
  std::pair<double, double> reward = self->InputPlacement({rotate, x, y}, training);
  PyObject* r1 = PyFloat_FromDouble(reward.first);
  PyObject* r2 = PyFloat_FromDouble(reward.second);
  PyObject* ret = PyTuple_Pack(2, r1, r2);
  Py_DECREF(r1);
  Py_DECREF(r2);
  return ret;
}

static PyObject* Tetris_SetPreviousPlacement(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"rotate", "x", "y", nullptr};
  int rotate, x, y;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii", (char**)kwlist, &rotate, &x, &y)) {
    return nullptr;
  }
  return PyBool_FromLong(self->SetPreviousPlacement({rotate, x, y}));
}

static int ParsePieceID(PyObject* obj) {
  if (PyUnicode_Check(obj)) {
    if (PyUnicode_GET_LENGTH(obj) < 1) {
      PyErr_SetString(PyExc_KeyError, "Invalid piece symbol.");
      return -1;
    }
    switch (PyUnicode_READ_CHAR(obj, 0)) {
      case 'T': return 0;
      case 'J': return 1;
      case 'Z': return 2;
      case 'O': return 3;
      case 'S': return 4;
      case 'L': return 5;
      case 'I': return 6;
      default: {
        PyErr_SetString(PyExc_KeyError, "Invalid piece symbol.");
        return -1;
      }
    }
  } else if (PyLong_Check(obj)) {
    long x = PyLong_AsLong(obj);
    if (x < 0 || x >= 7) {
      PyErr_SetString(PyExc_IndexError, "Piece ID out of range.");
      return -1;
    }
    return x;
  } else {
    PyErr_SetString(PyExc_TypeError, "Invalid type for piece.");
    return -1;
  }
}

static PyObject* Tetris_SetNowPiece(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"piece", nullptr};
  PyObject* obj;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &obj)) {
    return nullptr;
  }
  int piece = ParsePieceID(obj);
  if (piece < 0) return nullptr;
  return PyBool_FromLong(self->SetNowPiece(piece));
}

static PyObject* Tetris_SetNextPiece(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"piece", nullptr};
  PyObject* obj;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &obj)) {
    return nullptr;
  }
  int piece = ParsePieceID(obj);
  if (piece < 0) return nullptr;
  self->SetNextPiece(piece);
  Py_RETURN_NONE;
}

static bool ParseField(PyObject* obj, Tetris::Field& f) {
  if (!PyList_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "Expect List[List[int]].");
    return false;
  }
  if (PyList_Size(obj) < Tetris::kN) {
    PyErr_SetString(PyExc_IndexError, "Incorrect list size.");
    return false;
  }
  for (size_t i = 0; i < Tetris::kN; i++) {
    PyObject* row = PyList_GetItem(obj, i);
    if (!PyList_Check(row)) {
      PyErr_SetString(PyExc_TypeError, "Expect List[List[int]].");
      return false;
    }
    if (PyList_Size(row) < Tetris::kM) {
      PyErr_SetString(PyExc_IndexError, "Incorrect list size.");
      return false;
    }
    for (size_t j = 0; j < Tetris::kM; j++) {
      PyObject* item = PyList_GetItem(row, j);
      if (!PyLong_Check(item)) {
        PyErr_SetString(PyExc_TypeError, "Expect List[List[int]].");
        return false;
      }
      long x = PyLong_AsLong(item);
      f[i][j] = x != 0;
    }
  }
  return true;
}

static PyObject* Tetris_SetState(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {
    "field", "now_piece", "next_piece", "now_rotate", "now_x", "now_y", "lines",
    "score", "pieces", "prev_drop_time", "prev_misdrop", "prev_micro", nullptr
  };
  PyObject *field_obj, *now_piece_obj, *next_piece_obj;
  Tetris::Position pos;
  int lines, score, pieces;
  double prev_drop_time = 1000;
  int prev_misdrop = 0, prev_micro = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOiiiiii|dpp", (char**)kwlist,
                                   &field_obj, &now_piece_obj, &next_piece_obj,
                                   &pos.rotate, &pos.x, &pos.y, &lines, &score,
                                   &pieces, &prev_drop_time, &prev_misdrop,
                                   &prev_micro)) {
    return nullptr;
  }
  Tetris::Field f;
  if (!ParseField(field_obj, f)) return nullptr;
  int now_piece = ParsePieceID(now_piece_obj);
  if (now_piece < 0) return nullptr;
  int next_piece = ParsePieceID(next_piece_obj);
  if (next_piece < 0) return nullptr;
  return PyBool_FromLong(self->SetState(f, now_piece, next_piece, pos, lines,
                                        score, pieces, prev_drop_time,
                                        prev_misdrop, prev_micro));
}

static PyObject* Tetris_GetState(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  Tetris::State state = self->GetState();
  npy_intp dims[] = {state.size(), Tetris::kN, Tetris::kM};
  PyObject* ret = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
  memcpy(PyArray_DATA((PyArrayObject*)ret), state.data(), sizeof(state));
  return ret;
}

static PyObject* Tetris_StateShape(void*, PyObject* Py_UNUSED(ignored)) {
  PyObject* dim1 = PyLong_FromLong(std::tuple_size<Tetris::State>::value);
  PyObject* dim2 = PyLong_FromLong(Tetris::kN);
  PyObject* dim3 = PyLong_FromLong(Tetris::kM);
  PyObject* ret = PyTuple_Pack(3, dim1, dim2, dim3);
  Py_DECREF(dim1);
  Py_DECREF(dim2);
  Py_DECREF(dim3);
  return ret;
}

static PyObject* Tetris_ResetGame(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {
      "start_level", "hz_avg", "hz_dev", "microadj_delay", "start_lines",
      "drought_mode", "step_points", "penalty_multiplier", "target_column",
      nullptr
  };
  GameParams params;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iddiipidi", (char**)kwlist,
        &params.start_level, &params.hz_avg, &params.hz_dev,
        &params.microadj_delay, &params.start_lines, &params.drought_mode,
        &params.step_points, &params.penalty_multiplier, &params.target_column)) {
    return nullptr;
  }
  self->ResetGame(params);
  Py_RETURN_NONE;
}

static PyObject* Tetris_GetScore(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->GetScore());
}

static PyObject* Tetris_GetLines(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->GetLines());
}

static PyObject* Tetris_GetTetrisStat(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  auto stat = self->GetTetrisStat();
  PyObject* r1 = PyLong_FromLong(stat.first);
  PyObject* r2 = PyLong_FromLong(stat.second);
  PyObject* ret = PyTuple_Pack(2, r1, r2);
  Py_DECREF(r1);
  Py_DECREF(r2);
  return ret;
}

static inline PyObject* FramePyObject(const Tetris::FrameInput& f) {
  PyObject* ret = PyDict_New();
  PyObject *v1 = PyBool_FromLong(f.a), *v2 = PyBool_FromLong(f.b),
           *v3 = PyBool_FromLong(f.l), *v4 = PyBool_FromLong(f.r);
  PyDict_SetItemString(ret, "A", v1);
  PyDict_SetItemString(ret, "B", v2);
  PyDict_SetItemString(ret, "left", v3);
  PyDict_SetItemString(ret, "right", v4);
  Py_DECREF(v1);
  Py_DECREF(v2);
  Py_DECREF(v3);
  Py_DECREF(v4);
  return ret;
}

static PyObject* SequencePyObject(const Tetris::FrameSequence& fseq) {
  PyObject* ret = PyList_New(fseq.seq.size());
  for (size_t i = 0; i < fseq.seq.size(); i++) {
    PyObject* item = FramePyObject(fseq.seq[i]);
    PyList_SetItem(ret, i, item); // steal ref
  }
  return ret;
}

static PyObject* Tetris_GetPlannedSequence(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"truncate", nullptr};
  int truncate = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p", (char**)kwlist, &truncate)) {
    return nullptr;
  }
  return SequencePyObject(self->GetPlannedSequence(truncate));
}

static PyObject* Tetris_GetMicroadjSequence(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"truncate", nullptr};
  int truncate = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p", (char**)kwlist, &truncate)) {
    return nullptr;
  }
  return SequencePyObject(self->GetMicroadjSequence(truncate));
}

static PyObject* StatesPyObject(const std::vector<Tetris::State>& states) {
  if (states.size() == 0) return nullptr;
  constexpr size_t kItemSize = sizeof(Tetris::State);
  npy_intp dims[] = {(long)states.size(), (long)states[0].size(), Tetris::kN, Tetris::kM};
  PyObject* ret = PyArray_SimpleNew(4, dims, NPY_FLOAT32);
  uint8_t* dest = (uint8_t*)PyArray_DATA((PyArrayObject*)ret);
  for (size_t i = 0; i < states.size(); i++) {
    memcpy(dest + kItemSize * i, states[i].data(), kItemSize);
  }
  return ret;
}

static PyObject* StatesPyObjectWithNextPiece(const std::vector<Tetris::State>& states) {
  if (states.size() == 0) return nullptr;
  constexpr size_t kItemSize = sizeof(Tetris::State);
  npy_intp dims[] = {(long)states.size() * 7, (long)states[0].size(), Tetris::kN, Tetris::kM};
  PyObject* ret = PyArray_SimpleNew(4, dims, NPY_FLOAT32);
  uint8_t* dest = (uint8_t*)PyArray_DATA((PyArrayObject*)ret);
  for (size_t i = 0; i < states.size(); i++) {
    Tetris::State tmp_state = states[i];
    float* misc = (float*)tmp_state[14].data();
    // 7-14: next / 7(if place_stage_)
    for (size_t j = 7; j <= 14; j++) misc[j] = 0;
    for (size_t j = 0; j < 7; j++) {
      misc[j + 7] = 1;
      memcpy(dest + kItemSize * (i * 7 + j), tmp_state.data(), kItemSize);
    }
  }
  return ret;
}

static std::vector<std::vector<int>> PyObjectToMoveTable(PyObject* obj) {
  if (!PyList_Check(obj)) throw std::runtime_error("");
  size_t n = PyList_Size(obj);
  std::vector<std::vector<int>> ret(n);
  for (size_t i = 0; i < n; i++) {
    PyObject* row = PyList_GetItem(obj, i);
    if (!PyList_Check(row)) throw std::runtime_error("");
    size_t m = PyList_Size(row);
    ret[i].resize(m);
    for (size_t j = 0; j < m; j++) {
      PyObject* item = PyList_GetItem(row, j);
      if (!PyLong_Check(item)) throw std::runtime_error("");
      ret[i][j] = PyLong_AsLong(item);
    }
  }
  return ret;
}

static std::vector<double> PyObjectToValueTable(PyObject* obj) {
  if (!PyList_Check(obj)) throw std::runtime_error("");
  size_t n = PyList_Size(obj);
  std::vector<double> ret(n);
  for (size_t i = 0; i < n; i++) {
    PyObject* item = PyList_GetItem(obj, i);
    if (!PyFloat_Check(item)) throw std::runtime_error("");
    ret[i] = PyFloat_AsDouble(item);
  }
  return ret;
}

static PyObject* Tetris_Search(Tetris* self, PyObject* args, PyObject* kwds) {
  // place stage should be false (microadj phase)
  static const char *kwlist[] = {"func", "first_gain", nullptr};
  // func(states: ndarray, place_stage: bool, return_value: bool) ->
  //     Union[List[List[int]], List[float]]
  // return likely policy list (r*200+x*10+y) if return_value == false, values otherwise
  PyObject* func;
  double first_gain = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|d", (char**)kwlist, &func, &first_gain)) {
    return nullptr;
  }
  if (!PyCallable_Check(func)) {
    PyErr_SetString(PyExc_TypeError, "func must be callable");
    return nullptr;
  }
  if (self->GetPlaceStage()) {
    PyErr_SetString(PyExc_ValueError, "place stage incorrect");
    return nullptr;
  }
  const int* next_piece_dist = self->GetNextPieceDistribution();
  int next_piece_denom = 0;
  for (int i = 0; i < Tetris::kT; i++) next_piece_denom += next_piece_dist[i];

  struct Result {
    Tetris game;
    Tetris::FrameSequence adj, nxt;
    double reward;
  };
  auto ToPlacement = [](int x) {
    return Tetris::Position{x / 200, x / 10 % 20, x % 10};
  };
  std::vector<Result> all_games;
  try {
    // First (micro search)
    PyObject* state_arr = StatesPyObject({self->GetState()});
    PyObject* arglist = Py_BuildValue("(OOO)", state_arr, Py_False, Py_False);
    PyObject* result = PyObject_CallObject(func, arglist);
    Py_DECREF(state_arr);
    Py_DECREF(arglist);
    if (!result) return nullptr;

    std::vector<Tetris> first_games;
    std::vector<Tetris::State> first_state;
    std::vector<std::pair<Tetris::FrameSequence, double>> first_steps;
    auto moves = PyObjectToMoveTable(result);
    Py_DECREF(result);
    if (moves.size() != 1) {
      PyErr_SetString(PyExc_ValueError, "Incorrect policy length");
      return nullptr;
    }
    bool flag = true;
    for (int move : moves[0]) {
      Tetris tmp_game = *self;
      double reward = tmp_game.InputPlacement(ToPlacement(move)).first;
      if (!tmp_game.GetPlaceStage() || tmp_game.IsOver()) continue;
      first_steps.push_back({tmp_game.GetMicroadjSequence(), reward + flag * first_gain});
      first_state.push_back(tmp_game.GetState());
      first_games.push_back(std::move(tmp_game));
      flag = false;
    }
    if (first_games.empty()) Py_RETURN_NONE;

    // Second (next search)
    state_arr = StatesPyObject(first_state);
    arglist = Py_BuildValue("(OOO)", state_arr, Py_True, Py_False);
    result = PyObject_CallObject(func, arglist);
    Py_DECREF(state_arr);
    Py_DECREF(arglist);
    if (!result) return nullptr;

    moves = PyObjectToMoveTable(result);
    Py_DECREF(result);
    if (moves.size() != first_games.size()) {
      PyErr_SetString(PyExc_ValueError, "Incorrect policy length");
      return nullptr;
    }
    for (size_t i = 0; i < first_games.size(); i++) {
      bool flag = true;
      for (int move : moves[i]) {
        Result res;
        res.game = first_games[i];
        res.reward = res.game.InputPlacement(ToPlacement(move)).first + first_steps[i].second + flag * first_gain;
        if (res.game.GetPlaceStage() || res.game.IsOver()) continue;
        res.adj = first_steps[i].first;
        res.nxt = res.game.GetPlannedSequence();
        all_games.push_back(std::move(res));
        flag = false;
      }
    }
    if (all_games.empty()) Py_RETURN_NONE;
  } catch (std::runtime_error&) {
    PyErr_SetString(PyExc_ValueError, "func should return List[List[int]]");
    return nullptr;
  }

  std::vector<double> expected_rewards;
  try {
    std::vector<Tetris::State> states;
    for (auto& i : all_games) states.push_back(i.game.GetState());

    PyObject* state_arr = StatesPyObjectWithNextPiece(states);
    PyObject* arglist = Py_BuildValue("(OOO)", state_arr, Py_False, Py_True);
    PyObject* result = PyObject_CallObject(func, arglist);
    Py_DECREF(state_arr);
    Py_DECREF(arglist);
    if (!result) return nullptr;

    auto values = PyObjectToValueTable(result);
    Py_DECREF(result);
    if (values.size() != all_games.size() * 7) {
      PyErr_SetString(PyExc_ValueError, "Incorrect value length");
      return nullptr;
    }
    for (size_t i = 0; i < all_games.size(); i++) {
      double sum = 0;
      for (size_t j = 0; j < 7; j++) {
        sum += values[i * 7 + j] * next_piece_dist[j];
      }
      expected_rewards.push_back(all_games[i].reward + sum / next_piece_denom);
    }
  } catch (std::runtime_error&) {
    PyErr_SetString(PyExc_ValueError, "func should return List[float]");
    return nullptr;
  }

  size_t idx =
      std::max_element(expected_rewards.begin(), expected_rewards.end()) -
      expected_rewards.begin();
  auto& res = all_games[idx];
  *self = res.game;
  PyObject* ret1 = SequencePyObject(res.adj);
  PyObject* ret2 = SequencePyObject(res.nxt);
  PyObject* ret = PyTuple_Pack(2, ret1, ret2);
  Py_DECREF(ret1);
  Py_DECREF(ret2);
  return ret;
}

#ifdef DEBUG_METHODS

static PyObject* Tetris_PrintState(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  self->PrintState();
  Py_RETURN_NONE;
}

static PyObject* Tetris_PrintField(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  self->PrintState(true);
  Py_RETURN_NONE;
}

static PyObject* Tetris_PrintAllState(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  self->PrintAllState();
  Py_RETURN_NONE;
}

static PyObject* GetClearCol_(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  npy_intp dims[] = {4, Tetris::kM};
  PyObject* ret = PyArray_SimpleNew(2, dims, NPY_UINT64);
  memcpy(PyArray_DATA((PyArrayObject*)ret), clear_col_count, sizeof(clear_col_count));
  return ret;
}

PyCFunction GetClearCol = (PyCFunction)GetClearCol_;

#endif // DEBUG_METHODS

static PyMethodDef py_tetris_class_methods[] = {
    {"IsOver", (PyCFunction)Tetris_IsOver, METH_NOARGS,
     "Check whether the game is over"},
    {"InputPlacement", (PyCFunction)Tetris_InputPlacement,
     METH_VARARGS | METH_KEYWORDS, "Input a placement and return the reward"},
    {"GetState", (PyCFunction)Tetris_GetState, METH_NOARGS, "Get state array"},
    {"StateShape", (PyCFunction)Tetris_StateShape, METH_NOARGS | METH_STATIC,
     "Get shape of state array (static)"},
    {"ResetGame", (PyCFunction)Tetris_ResetGame, METH_VARARGS | METH_KEYWORDS,
     "Reset a game using given parameters"},
    {"GetLines", (PyCFunction)Tetris_GetLines, METH_NOARGS, "Get lines"},
    {"GetScore", (PyCFunction)Tetris_GetScore, METH_NOARGS, "Get score"},
    {"GetTetrisStat", (PyCFunction)Tetris_GetTetrisStat, METH_NOARGS, "Get tetris statistics"},
    {"GetPlannedSequence", (PyCFunction)Tetris_GetPlannedSequence,
     METH_VARARGS | METH_KEYWORDS, "Get planned frame input sequence"},
    {"GetMicroadjSequence", (PyCFunction)Tetris_GetMicroadjSequence,
     METH_VARARGS | METH_KEYWORDS, "Get microadjustment frame input sequence"},
    {"SetPreviousPlacement", (PyCFunction)Tetris_SetPreviousPlacement,
     METH_VARARGS | METH_KEYWORDS, "Set actual placement"},
    {"SetNowPiece", (PyCFunction)Tetris_SetNowPiece,
     METH_VARARGS | METH_KEYWORDS, "Set the current piece (at game start)"},
    {"SetNextPiece", (PyCFunction)Tetris_SetNextPiece,
     METH_VARARGS | METH_KEYWORDS, "Set the next piece"},
    {"SetState", (PyCFunction)Tetris_SetState, METH_VARARGS | METH_KEYWORDS,
     "Set the game board & state"},
    {"Search", (PyCFunction)Tetris_Search, METH_VARARGS | METH_KEYWORDS,
     "Search for the best move and make it"},
#ifdef DEBUG_METHODS
    {"PrintState", (PyCFunction)Tetris_PrintState, METH_NOARGS,
     "Print state array"},
    {"PrintField", (PyCFunction)Tetris_PrintField, METH_NOARGS,
     "Print current field"},
    {"PrintAllState", (PyCFunction)Tetris_PrintAllState, METH_NOARGS,
     "Print all internal state"},
#endif
    {nullptr}};

PyTypeObject py_tetris_class = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "tetris.Tetris",         // tp_name
    sizeof(Tetris),          // tp_basicsize
    0,                       // tp_itemsize
    (destructor)TetrisDealloc, // tp_dealloc
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
    "Tetris class",          // tp_doc
    0,                       // tp_traverse
    0,                       // tp_clear
    0,                       // tp_richcompare
    0,                       // tp_weaklistoffset
    0,                       // tp_iter
    0,                       // tp_iternext
    py_tetris_class_methods, // tp_methods
    0,                       // tp_members
    0,                       // tp_getset
    0,                       // tp_base
    0,                       // tp_dict
    0,                       // tp_descr_get
    0,                       // tp_descr_set
    0,                       // tp_dictoffset
    (initproc)TetrisInit,    // tp_init
    0,                       // tp_alloc
    TetrisNew,               // tp_new
};
