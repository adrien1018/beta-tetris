#include "train.h"

#include <cmath>

template <class T>
static GameParams GetRandomParams(const ResetParams& params, T& rng_) {
  constexpr double hz_table[] = {12, 13.5, 15, 20, 30};
  constexpr double start_level_table[] = {18, 19, 29};
  constexpr int adj_delay_table[] = {8, 16, 21, 25, 61};
  constexpr double step_points_table[] = {40, 200, 3000};

  using IntRand = std::uniform_int_distribution<int>;
  using RealRand = std::uniform_real_distribution<float>;

  GameParams ret;
  int hz_ind = IntRand(0, 4)(rng_);
  ret.start_level = start_level_table[IntRand(0, 2)(rng_)];
  ret.hz_avg = hz_table[hz_ind];
  ret.hz_dev = ret.hz_avg >= 29 || IntRand(0, 2)(rng_) ? 0 : int(ret.hz_avg) / 6 * 0.5;
  ret.microadj_delay = adj_delay_table[IntRand(0, 4)(rng_)];
  ret.step_points = step_points_table[std::discrete_distribution<int>(
      {1, params.reward_ratio, std::pow(params.reward_ratio, 1.5)})(rng_)];
  ret.start_lines = 0;
  ret.drought_mode = IntRand(0, 2)(rng_) == 0;
  ret.target_column = RandTargetColumn(rng_);
  ret.penalty_multiplier = params.penalty_multiplier;

  if (IntRand(0, 3)(rng_) == 0) ret.target_column = -2;
  if (IntRand(0, 1)(rng_)) {
    ret.hz_avg = IntRand(0, 1)(rng_) ? 12 : 20;
    ret.microadj_delay = 21;
  }
  if (RealRand(0, 1)(rng_) < params.pre_trans) {
    // pre-transition training
    int rnd = IntRand(0, 19)(rng_), rnd2 = IntRand(0, 3)(rng_);
    if (ret.start_level == 18) {
      if (rnd >= 13) {
        ret.start_lines = 205 + rnd2; // 230-25
      } else {
        ret.start_lines = 105 + rnd2; // 130-25
      }
    } else if (ret.start_level == 19) {
      ret.start_lines = 205 + rnd2;
    }
  }
  /*
  printf("LVL%d %dln col%d (%.1lf,%.1lf)hz %dfr %dpts dr%d\n", ret.start_level, ret.start_lines, ret.target_column,
      ret.hz_avg, ret.hz_dev, ret.microadj_delay, ret.step_points, (int)ret.drought_mode);
  */
  return ret;
}

std::vector<Tetris::State> TrainingManager::Init(size_t sz, uint64_t seed) {
  gen_.seed(seed);
  envs_.clear();
  envs_.reserve(sz);
  for (size_t i = 0; i < sz; i++) envs_.emplace_back(gen_());
  reward_multiplier_.assign(sz, 1.0);
  tot_rewards_.assign(sz, 0.0);
  tot_sq_rewards_.assign(sz, 0.0);
  tot_length_.assign(sz, 0);
  unchanged_pieces_.assign(sz, 0);
  for (size_t i = 0; i < sz; i++) {
    cur_params_.emplace_back(GetRandomParams(reset_params_, gen_));
  }

  std::vector<Tetris::State> ret(sz);
  for (size_t i = 0; i < envs_.size(); i++) ret[i] = envs_[i].GetState();
  return ret;
}

void TrainingManager::ResetGame(size_t idx) {
  unchanged_pieces_[idx] += tot_length_[idx] / 2;
  if (unchanged_pieces_[idx] >= kKeepParamPieces_) {
    cur_params_[idx] = GetRandomParams(reset_params_, gen_);
    unchanged_pieces_[idx] = 0;
  }
  auto key = cur_params_[idx].GetNormalizingParams();
  auto& avg = avg_infor_.emplace(key, std::make_pair(kPerStepNorm_, 0)).first->second;
  reward_multiplier_[idx] = kPerStepNorm_ / avg.first;
  envs_[idx].ResetGame(cur_params_[idx]);
}

TrainingManager::ActionResult TrainingManager::Step(const std::vector<int>& actions) {
  if (actions.size() != envs_.size()) return {};
  ActionResult ret;
  ret.state.resize(envs_.size());
  ret.reward.resize(envs_.size());
  ret.is_over.resize(envs_.size());
  for (size_t i = 0; i < envs_.size(); i++) {
    int r = actions[i] / 200, x = actions[i] / 10 % 20, y = actions[i] % 10;
    std::pair<double, double> reward = envs_[i].InputPlacement({r, x, y}, true);
    ret.reward[i][0] = reward_multiplier_[i] * reward.first;
    ret.reward[i][1] = reward.second;
    ret.is_over[i] = envs_[i].IsOver();
    tot_rewards_[i] += ret.reward[i][0];
    tot_sq_rewards_[i] += ret.reward[i][0] * ret.reward[i][0];
    tot_length_[i]++;

    if (ret.is_over[i]) {
      ret.info.emplace_back();
      auto& info = ret.info.back();
      auto stat = envs_[i].GetTetrisStat();
      info.emplace("reward", tot_rewards_[i]);
      info.emplace("length", tot_length_[i] * 0.5);
      info.emplace("score", envs_[i].GetScore());
      info.emplace("lines", envs_[i].GetLines());
      info.emplace("trt", stat.first);
      info.emplace("rtrt", stat.second);

      auto key = cur_params_[i].GetNormalizingParams();
      auto& avg = avg_infor_.emplace(key, std::make_pair(kPerStepNorm_, 0)).first->second;
      avg.second += tot_length_[i];
      total_length_ += tot_length_[i];
      double ratio = 8192. / tot_length_[i];
      // make it proceed to average faster when the frequency is low
      if (total_length_ > 30 * avg.second) ratio /= (double)total_length_ / avg.second / 30;
      double alpha = 1. / (5 + ratio);
      double avg_reward = std::max(std::sqrt(tot_sq_rewards_[i] / tot_length_[i]), 1e-3);
      avg.first = avg.first * (1 - alpha) + avg_reward * alpha;
      /*
      printf("(%d %d %d %d %d %d):%lf %.3lf %.1lf\n",
          key.start_level, key.hz_mode, key.step_points, key.target_column_mode, key.start_line_mode, key.drought_mode,A
          avg_reward, kPerStepNorm_ / avg.first, (double)total_length_ / avg.second);
      */
      ResetGame(i);
      tot_rewards_[i] = 0.0;
      tot_sq_rewards_[i] = 0.0;
      tot_length_[i] = 0;
    }

    ret.state[i] = envs_[i].GetState();
  }
  return ret;
}
