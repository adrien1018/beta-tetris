#ifndef TRAIN_H_
#define TRAIN_H_

#include "params.h"
#include "game.h"

#include <cstdint>
#include <random>
#include <vector>
#include <variant>
#include <unordered_map>

#ifndef NO_PYTHON
#include "python.h"
#endif

struct ResetParams {
  float pre_trans;
  double penalty_multiplier;
  double reward_ratio;

  ResetParams() : pre_trans(0.5), penalty_multiplier(1), reward_ratio(1) {}
};


class TrainingManager {
 public:
#ifndef NO_PYTHON
  PyObject_HEAD
#endif

  struct ActionResult {
    std::vector<Tetris::State> state;
    std::vector<std::array<double, 2>> reward;
    std::vector<uint8_t> is_over;
    std::vector<std::unordered_map<std::string, std::variant<int, double>>> info;
  };
  using StateMap = std::unordered_map<NormalizingParams, std::pair<double, int64_t>>;
 private:
  StateMap avg_infor_;
  std::vector<Tetris> envs_;
  std::vector<double> reward_multiplier_;
  std::vector<double> tot_rewards_;
  std::vector<double> tot_sq_rewards_;
  std::vector<int> tot_length_;
  std::vector<int> unchanged_pieces_;
  std::vector<GameParams> cur_params_;
  ResetParams reset_params_;
  int64_t total_length_;
  std::mt19937_64 gen_;
  bool freeze_;

  static constexpr int kKeepParamPieces_ = 384;
  static constexpr double kPerStepNorm_ = 0.05;

 public:

  TrainingManager(bool freeze = false) : total_length_(0), freeze_(freeze) {}

  Tetris& operator[](size_t idx) { return envs_[idx]; }

  void SetResetParams(const ResetParams& params) { reset_params_ = params; }
  std::vector<Tetris::State> Init(size_t sz, uint64_t seed);
  void ResetGame(size_t idx);
  ActionResult Step(const std::vector<int>& actions);
  const StateMap& GetState() const { return avg_infor_; }
  void LoadState(StateMap&& st);
};

#endif // TRAIN_H_
