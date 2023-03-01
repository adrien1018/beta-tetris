#ifndef PARAMS_H_
#define PARAMS_H_

#include <random>

struct NormalizingParams {
  int start_level;
  int hz_mode;
  int step_points;
  int target_column_mode;
  int start_line_mode;
  int drought_mode;

  bool operator==(const NormalizingParams&) const = default;
  bool operator!=(const NormalizingParams&) const = default;
};

namespace std {

template <>
struct hash<NormalizingParams> {
  size_t operator()(const NormalizingParams& x) const {
    size_t a = 0;
    a += (x.start_level + x.hz_mode * 37 + x.drought_mode * 2235) * 6093894908103630263L;
    a ^= a >> 32;
    a += (x.step_points + x.target_column_mode + x.start_line_mode * 5) * 3691759670834417381L;
    a ^= a >> 28;
    return a;
  }
};

} // namespace std

template <class T>
int RandTargetColumn(T& rng_) {
  int ret = std::uniform_int_distribution<int>(-3, 11)(rng_);
  if (ret > 9) ret = 9;
  if (ret < -1) ret = -1;
  return ret;
}

struct GameParams {
  int start_level;
  double hz_avg, hz_dev;
  int microadj_delay;
  int start_lines;
  bool drought_mode;
  int step_points;
  double penalty_multiplier;
  int target_column;

  GameParams() :
      start_level(18), hz_avg(12), hz_dev(0),
      microadj_delay(21), start_lines(0), drought_mode(false),
      step_points(0), penalty_multiplier(1), target_column(-1) {}

  NormalizingParams GetNormalizingParams() const;
};

#endif // PARAMS_H_
