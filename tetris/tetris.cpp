#include <cstdint>
#include <array>
#include <queue>
#include <random>
#include <vector>
#include <utility>
#include <algorithm>

#define PY_SSIZE_T_CLEAN
#if __has_include(<python3.8/Python.h>)
#include <python3.8/Python.h>
#elif __has_include(<python3.9/Python.h>)
#include <python3.9/Python.h>
#else
#include <Python.h>
#endif

class Tetris {
 public:
  PyObject_HEAD

  static constexpr int kN = 20, kM = 10, kT = 7;
  using Field = std::array<std::array<bool, kM>, kN>;

  enum class MoveType { kA, kB, kL, kR };
  struct Move {
    int height_start, height_end;
    MoveType type;
  };
  struct MoveSequence {
    int rotation; // -1(A)~2(B)
    int move; // -5(L)~4(R)
    // final movements (that is, movements can't be done at start, usually spins and tucks)
    std::vector<Move> fin;
  };
  struct PlacementSequence {
    MoveSequence start, micro;
    bool use_micro; // if true, microadjustment is added and the final movements are overrided
  };

 private:
  // # RNG
  std::mt19937_64 rng_;

  // # Game state
  int start_level_, level_, lines_, score_, pieces_;
  int now_, next_; // piece
  Field field_;

  // # Parameters
  // Movement model
  double hz_avg_, hz_dev_;
  // The actual hz is sampled from NormalDistribution(hz_avg_, hz_dev_)
  double first_tap_max_; // in milliseconds
  // The actual first-tap delay is sampled from UniformDistribution(0, first_tap_max_)
  // Note that for DAS we can set hz_avg_ = 10, hz_dev_ = 0 and first_tap_max_ = 0

  double microadj_delay_; // in milliseconds
  // For simplicity, we assume a fixed microadjustment delay

  // Misdrop model
  double base_misdrop_rate_; // base misdrop rate
  double misdrop_param_; // in milliseconds
  // The actual base misdrop rate will be ((misdrop_param_ / x)^1.5+1)*base_misdrop_rate_,
  //   where x is the dropping time of the previous piece
  //   (this simulates that it is harder to get
  // Misdrop multiplier for each misdrop types
  //   (the multiplier will be multiplied with the base misdrop rate to get the actual misdrop rate):
  //   1. Missing L/R tap (for each tap count)
  static constexpr double kMissLRTapMultiplier_[] = {0.0, 0.4, 1.0, 1.0, 1.5, 2.5};
  //   2. Missing A/B tap
  static constexpr double kMissABTapMultiplier_ = 1.0;
  //   3. Additional L/R tap (for each tap count)
  static constexpr double kAddLRTapMultiplier_[] = {0.0, 1.0, 1.0, 1.0, 0.4, 0.0};
  //   4. Random placement (that is, sample a feasible piece placement at random)
  static constexpr double kRandomPlacementMultipler_ = 0.2;
  //   5. Miss tuck
  static constexpr double kMissTuckMultiplier_ = 1.0;
  //   6. Miss spin
  static constexpr double kMissSpinMultiplier_ = 2.0;
  //   7. Miss spin-tuck
  static constexpr double kMissSpinTuckMultiplier_ = 6.0;
  //   8. Miss microadjustment
  static constexpr double kMissMicroMultiplier_ = 2.0;
  //   9. The next piece after any misdrop will have higher misdrop rate
  static constexpr double kAfterMisdropMultiplier_ = 2.0;
  // Again, for simplicity, we fix all the multipliers here.
  // I assigned these multipliers at my will; a future work may be analyzing
  //   human-played games to get a more promising misdrop model.
  double prev_drop_time_; // in milliseconds
  bool prev_misdrop_;

  // Reward model
  int target_; // in points
  static constexpr double kRewardMultiplierBeforeTarget_ = 1e-5;
  static constexpr double kRewardMultiplierAfterTarget_ = 5e-6;
  static constexpr double kTargetReward = 50;
  // The agent will get 1e-5 reward per point before reaching the target
  //   (that is, 10 per max-out), and get 50 reward immediately when reaching
  //   the target; after that, 5e-6 reward per point.
  // We use this to guide the agent toward the appropriate aggression to reach
  //   the point target.
  static constexpr double kInfeasibleReward = -0.2;
  // Provide a large reward deduction if the agent makes an infeasible placement
  // "Infeasible" placements are those cannot be done by +3σ tapping speeds
  //   (750-in-1 chance) and without misdrop
  static constexpr double kMisdropReward = -0.06;
  // Provide a small reward deduction each time the agent makes an misdrop;
  //   this can guide the agent to avoid high-risk movements

  // # Game constants
  static constexpr int kTransitionProb_[kT][kT] = {
    {1, 5, 6, 5, 5, 5, 5},
    {6, 1, 5, 5, 5, 5, 5},
    {5, 6, 1, 5, 5, 5, 5},
    {5, 5, 5, 2, 5, 5, 5},
    {5, 5, 5, 5, 2, 5, 5},
    {6, 5, 5, 5, 5, 1, 5},
    {5, 5, 5, 5, 6, 5, 1},
  };
  using Poly_ = std::array<std::pair<int, int>, 4>;
  static const std::vector<Poly_> kBlocks_[kT];

  void SpawnPiece_() {
    int next = std::discrete_distribution<int>(
        kTransitionProb_[next_], kTransitionProb_[next_] + kT)(rng_);
    now_ = next_;
    next_ = next;
  }

 public:
  Tetris(uint64_t seed = 0) : rng_(seed) {
    ResetGame(18);
  }

  void ResetGame(int start_level, double hz_avg = 10, double hz_dev = 0,
                 double first_tap_max = 0, double microadj_delay = 300,
                 double base_misdrop_rate = 5e-3, double misdrop_param = 250,
                 int target = 1000000) {
    start_level_ = start_level;
    level_ = start_level;
    lines_ = 0;
    score_ = 0;
    next_ = 0;
    pieces_ = 0;
    SpawnPiece_();
    SpawnPiece_();
    for (auto& i : field_) std::fill(i.begin(), i.end(), false);
    hz_avg_ = hz_avg;
    hz_dev_ = hz_dev;
    first_tap_max_ = first_tap_max;
    microadj_delay_ = microadj_delay;
    base_misdrop_rate_ = base_misdrop_rate;
    misdrop_param_ = misdrop_param;
    prev_misdrop_ = false;
    prev_drop_time_ = 1800;
    target_ = target;
  }

  static void PlaceField(Field& field, int piece, int cx, int cy, int rotate) {
    auto& pl = kBlocks_[piece][rotate];
    for (auto& i : pl) {
      int nx = cx + i.first, ny = cy + i.second;
      if (nx >= kN || ny >= kM || nx < 0 || ny < 0) continue;
      field[nx][ny] = true;
    }
    int i = kN - 1, j = kN - 1;
    for (; i >= 0; i--, j--) {
      bool flag = true;
      for (int y = 0; y < kM; y++) flag &= field[i][y];
      if (flag) {
        j++;
      } else if (i != j) {
        for (int y = 0; y < kM; y++) field[j][y] = field[i][y];
      }
    }
    int ans = j + 1;
    for (; j >= 0; j--) {
      for (int y = 0; y < kM; y++) field[j][y] = false;
    }
  }
};

decltype(Tetris::kBlocks_) Tetris::kBlocks_ = {
    {{{{1, 0}, {0, 0}, {0, 1}, {0, -1}}}, // T
     {{{1, 0}, {0, 0}, {-1, 0}, {0, -1}}},
     {{{0, -1}, {0, 0}, {0, 1}, {-1, 0}}},
     {{{1, 0}, {0, 0}, {0, 1}, {-1, 0}}}},
    {{{{0, -1}, {0, 0}, {0, 1}, {1, 1}}}, // J
     {{{-1, 0}, {0, 0}, {1, -1}, {1, 0}}},
     {{{-1, -1}, {0, -1}, {0, 0}, {0, 1}}},
     {{{-1, 0}, {-1, 1}, {0, 0}, {1, 0}}}},
    {{{{0, -1}, {0, 0}, {1, 0}, {1, 1}}}, // Z
     {{{-1, 1}, {0, 0}, {0, 1}, {1, 0}}}},
    {{{{0, -1}, {0, 0}, {1, -1}, {1, 0}}}}, // O
    {{{{0, 0}, {0, 1}, {1, -1}, {1, 0}}}, // S
     {{{-1, 0}, {0, 0}, {0, 1}, {1, 1}}}},
    {{{{0, -1}, {0, 0}, {0, 1}, {1, -1}}}, // L
     {{{-1, -1}, {-1, 0}, {0, 0}, {1, 0}}},
     {{{-1, 1}, {0, -1}, {0, 0}, {0, 1}}},
     {{{-1, 0}, {0, 0}, {1, 0}, {1, 1}}}},
    {{{{0, -2}, {0, -1}, {0, 0}, {0, 1}}}, // I
     {{{-2, 0}, {-1, 0}, {0, 0}, {1, 0}}}}};
decltype(Tetris::kTransitionProb_) Tetris::kTransitionProb_;

#ifndef DEBUG

#else

int main() {
  Tetris t;
}

#endif
