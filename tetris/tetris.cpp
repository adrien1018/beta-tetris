#include <cstdint>
#include <array>
#include <queue>
#include <random>
#include <vector>
#include <utility>
#include <algorithm>
#include <unordered_map>

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
    bool valid;
    std::vector<Move> start;
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
  int now_piece_, next_piece_; // piece
  Field field_;

  // # Parameters
  // Movement model
  double hz_avg_, hz_dev_;
  // The actual hz is sampled from NormalDistribution(hz_avg_, hz_dev_)
  // Note that for default DAS we can set hz_avg_ = 10, hz_dev_ = 0
  double first_tap_max_; // in milliseconds
  // The actual first-tap delay is sampled from UniformDistribution(0, first_tap_max_)
  // DAS's effect:
  //  (1) Try to quicktap if the placement is infeasible
  //  (2) The next piece after microadjustment / spins / spintucks / quicktap
  //      would have more delay (specifically, (8/3) / hz_avg_) because of
  //      uncharged DAS
  //  (3) Use DAS(Miss|Add)LRTapMultiplier instead
  bool das_;

  // For simplicity, we assume a fixed microadjustment delay (the first
  //   microadjustment input would be done at exactly this time)
  double microadj_delay_; // in milliseconds

  // Misdrop model
  double base_misdrop_rate_; // base misdrop rate
  double misdrop_param_time_; // in milliseconds
  double misdrop_param_pow_; // in milliseconds
  // The actual base misdrop rate will be ((misdrop_param_time_ / x)^misdrop_param_pow_+1)*base_misdrop_rate_,
  //   where x is min(dropping time of the previous piece, current piece)
  //   (this simulates that it is more likely to misdrop with less thinking time
  //    or less dropping time)
  // Misdrop multiplier for each misdrop types
  //   (the multiplier will be multiplied with the base misdrop rate to get the
  //    actual misdrop rate):
  //   1. Missing L/R tap (for each tap count)
  static constexpr double kMissLRTapMultiplier_[] = {0.0, 0.4, 1.0, 1.0, 1.5, 2.5};
  static constexpr double kDASMissLRTapMultiplier_[] = {0.0, 0.4, 0.6, 0.6, 0.6, 0.6};
  //   2. Missing A/B tap
  static constexpr double kMissABTapMultiplier_ = 1.0;
  //   3. Additional L/R tap (for each tap count)
  static constexpr double kAddLRTapMultiplier_[] = {0.0, 1.0, 1.0, 1.0, 0.4, 0.0};
  static constexpr double kDASAddLRTapMultiplier_[] = {0.0, 1.0, 1.0, 1.0, 1.0, 0.0};
  //   4. Miss tuck
  static constexpr double kMissTuckMultiplier_ = 2.0;
  //   5. Miss spin
  static constexpr double kMissSpinMultiplier_ = 4.0;
  //   6. Miss spin-tuck
  static constexpr double kMissSpinTuckMultiplier_ = 7.0;
  //   7. Miss (each) microadjustment
  static constexpr double kMissMicroMultiplier_ = 3.0;
  //   8. The next piece after any misdrop will have higher misdrop rate
  static constexpr double kAfterMisdropMultiplier_ = 2.0;
  //   9. The next piece after any microadjustment will have higher misdrop rate
  static constexpr double kAfterMircoMultiplier_ = 1.3;
  // Note: The training process would try suboptimal moves, so we don't simulate
  //   misdrops due to incorrect decision.
  // Again, for simplicity, we fix all the multipliers here.
  // I assigned these multipliers at my will, so some of them would differ a lot
  //   from human behavior. A future work can be analyzing human-played games to
  //   get a more promising misdrop model.
  double prev_drop_time_; // in milliseconds
  bool prev_misdrop_, prev_micro_, das_not_charged_;

  // Reward model
  int target_; // in points
  static constexpr double kRewardMultiplierBeforeTarget_ = 6e-6;
  static constexpr double kRewardMultiplierAfterTarget_ = 3e-6;
  static constexpr double kTargetReward = 100;
  // The agent will get 6e-6 reward per point before reaching the target
  //   (that is, 6 per max-out), and get 100 reward immediately when reaching
  //   the target; after that, 3e-6 reward per point.
  // We use this to guide the agent toward the appropriate aggression to
  //   maximize the probability of reaching the point target.
  static constexpr double kInfeasibleReward = -0.2;
  // Provide a large reward deduction if the agent makes an infeasible placement
  // "Infeasible" placements are those cannot be done by +3σ tapping speeds
  //   (750-in-1 chance) and without misdrop
  static constexpr double kMisdropReward = -0.06;
  // Provide a small reward deduction each time the agent makes an misdrop;
  //   this can guide the agent to avoid high-risk movements

  // # Game constants
  // Piece generation probabilities
  static constexpr int kTransitionProb_[kT][kT] = {
  // T  J  Z  O  S  L  I (next)
    {1, 5, 6, 5, 5, 5, 5}, // T (current)
    {6, 1, 5, 5, 5, 5, 5}, // J
    {5, 6, 1, 5, 5, 5, 5}, // Z
    {5, 5, 5, 2, 5, 5, 5}, // O
    {5, 5, 5, 5, 2, 5, 5}, // S
    {6, 5, 5, 5, 5, 1, 5}, // L
    {5, 5, 5, 5, 6, 5, 1}, // I
  };
  // Tetraminos
  using Poly_ = std::array<std::pair<int, int>, 4>;
  static const std::vector<Poly_> kBlocks_[kT]; // definition outside
  // Score
  static constexpr int kScoreBase_[] = {0, 40, 100, 300, 1200};
  // Level
  static constexpr int kLinesBeforeLevelUp_[] = {
      10, 20, 30, 40, 50, 60, 70, 80, 90, 100, // 0-9
      100, 100, 100, 100, 100, 100, 110, 120, 130, 140, // 10-19
      150, 160, 170, 180, 190, 200, 200, 200, 200, 200, // 20-29
  };
  // Timing
  static constexpr int kFramesPerDrop_[] = {
      48, 43, 38, 33, 28, 23, 18, 13, 8, 6, // 0-9
      5, 5, 5, 4, 4, 4, 3, 3, 3, 2, // 10-19
      2, 2, 2, 2, 2, 2, 2, 2, 2, 1, // 20-29
  };
  static constexpr int kBaseDelay_ = 10;
  static constexpr int kNotGroundDelay_ = 2;
  static constexpr int kLineClearDelay_ = 20;
  // 1000 / (30 * 525 / 1.001 * 455 / 2 / 2 / (341 * 262 - 0.5) * 3)
  static constexpr double kFrameLength = 655171. / 39375;

  void SpawnPiece_() {
    int next = std::discrete_distribution<int>(
        kTransitionProb_[next_piece_], kTransitionProb_[next_piece_] + kT)(rng_);
    now_piece_ = next_piece_;
    next_piece_ = next;
  }

  static bool IsGround_(int piece, int x, int rotate) {
    auto& pl = kBlocks_[piece][rotate];
    for (auto& i : pl) {
      if (x + i.first == kN - 1) return true;
    }
    return false;
  }

  static double GetDropTime_(int piece, int x, int rotate, int level,
                             bool clear) {
    return (x + 1 + kBaseDelay_ +
            (IsGround_(piece, x, rotate) ? 0 : kNotGroundDelay_) +
            (clear ? kLineClearDelay_ : 0)) *
           kFrameLength;
  }

  template <class T>
  using CMap_ = std::vector<std::array<std::array<T, kM + 2>, kN + 2>>;
  using Map_ = CMap_<uint8_t>;

  static Map_ GetMap_(const Field& field, int poly) {
    const size_t R = kBlocks_[poly].size();
    Map_ ret(R, Map_::value_type{});
    for (size_t r = 0; r < R; r++) {
      auto& pl = kBlocks_[poly][r];
      for (int x = 0; x < kN; x++) {
        for (int y = 0; y < kM; y++) {
          bool flag = true;
          for (int i = 0; i < 4; i++) {
            int nx = pl[i].first + x, ny = pl[i].second + y;
            if (ny < 0 || nx >= kN || ny >= kM || (nx >= 0 && field[nx][ny])) {
              flag = false;
              break;
            }
          }
          ret[r][x + 1][y + 1] = flag;
        }
      }
    }
    return ret;
  }

  // Dijkstra for input sequence generation
  struct Weight_ {
    uint16_t step, weight, height;
    bool operator<(const Weight_ x) const { return Value_() < x.Value_(); }
   private:
    uint64_t Value_() const {
      return (uint64_t)step << 32 | (uint64_t)weight << 16 | height;
    }
  };

  struct Node_ {
    // dir: 1(down) 2(rotateL) 3(rotateR) 4(left) 5(right)
    int r, x, y, dir, n;
    Weight_ w;
    bool operator<(const Node_& a) const { return a.w < w; }
  };

  // Highest possible move
  static Map_ Dijkstra_(const Map_& v, int cx, int cy, int rotate) {
    const int R = v.size();
    ++cx, ++cy; // start
    Map_ ret(R, Map_::value_type{});
    if (!v[rotate][cx][cy]) return ret;
    CMap_<Weight_> d(v.size());
    for (auto& i : d) for (auto& j : i) for (auto& k : j) k = {16384, 0, 0};
    std::priority_queue<Node_> pq;
    pq.push({rotate, cx, cy, 0, 0, {0, 0, 0}});
    d[rotate][cx][cy] = {0, 0, 0};
    while (!pq.empty()) {
      Node_ nd = pq.top();
      pq.pop();
      if (d[nd.r][nd.x][nd.y] < nd.w) continue;
      ret[nd.r][nd.x][nd.y] = nd.dir;
      // Move as high as possible
      Weight_ wp = {(uint16_t)(nd.w.step + 1), nd.w.weight,
                    (uint16_t)(nd.w.height + nd.x)};
      auto Relax = [&](int r, int x, int y, Weight_ w, uint8_t dir) {
        w.weight += dir;
        if (v[r][x][y] > 0 && w < d[r][x][y]) {
          pq.push({r, x, y, dir, 0, w});
          d[r][x][y] = w;
        }
      };
      Relax(nd.r, nd.x + 1, nd.y, nd.w, 1);
      // Try rotate before move
      if (R != 1) {
        int r1 = nd.r == R - 1 ? 0 : nd.r + 1;
        int r2 = nd.r == 0 ? R - 1 : nd.r - 1;
        Relax(r1, nd.x, nd.y, wp, 2);
        Relax(r2, nd.x, nd.y, wp, 3);
      }
      Relax(nd.r, nd.x, nd.y - 1, wp, 4);
      Relax(nd.r, nd.x, nd.y + 1, wp, 5);
    }
    return ret;
  }

  // Lowest possible move (constrained to a specific move sequence)
  static Map_ Dijkstra_(const Map_& v, int cx, int cy, int rotate,
                        const std::vector<std::pair<int, MoveType>>& moves) {
    const int R = v.size(), N = moves.size();
    ++cx, ++cy; // start
    Map_ ret(R, Map_::value_type{});
    if (!v[rotate][cx][cy]) return ret;
    std::vector<CMap_<Weight_>> d(N + 1, CMap_<Weight_>(v.size()));
    for (auto& r : d) {
      for (auto& i : r) for (auto& j : i) for (auto& k : j) k = {16384, 0, 0};
    }
    std::priority_queue<Node_> pq;
    pq.push({rotate, cx, cy, 0, 0, {0, 0, 0}});
    d[0][rotate][cx][cy] = {0, 0, 0};
    while (!pq.empty()) {
      Node_ nd = pq.top();
      pq.pop();
      if (d[nd.n][nd.r][nd.x][nd.y] < nd.w) continue;
      ret[nd.r][nd.x][nd.y] = nd.dir;
      // Move as low as possible
      Weight_ wp = {(uint16_t)(nd.w.step + 1), nd.w.weight,
                    (uint16_t)(nd.w.height + kN - nd.x)};
      auto Relax = [&](int r, int x, int y, int n, Weight_ w, uint8_t dir) {
        w.weight += dir;
        if (v[r][x][y] > 0 && w < d[n][r][x][y]) {
          pq.push({r, x, y, dir, n, w});
          d[n][r][x][y] = w;
        }
      };
      Relax(nd.r, nd.x + 1, nd.y, nd.n, nd.w, 1);
      if (nd.n == N) continue;
      int r1 = nd.r == R - 1 ? 0 : nd.r + 1;
      int r2 = nd.r == 0 ? R - 1 : nd.r - 1;
      switch (moves[nd.n].second) {
        case MoveType::kA: Relax(r1, nd.x, nd.y, nd.n + 1, wp, 2); break;
        case MoveType::kB: Relax(r2, nd.x, nd.y, nd.n + 1, wp, 3); break;
        case MoveType::kL: Relax(nd.r, nd.x, nd.y - 1, nd.n + 1, wp, 4); break;
        case MoveType::kR: Relax(nd.r, nd.x, nd.y + 1, nd.n + 1, wp, 5); break;
      }
    }
    return ret;
  }

  static std::vector<std::pair<int, MoveType>> MovesFromMap_(
      Map_&& mp, int x, int y, int r) {
    int R = mp.size();
    ++x, ++y;
    std::vector<std::pair<int, MoveType>> ret;
    static constexpr MoveType lookup[] = {
      MoveType::kL, MoveType::kL, // not used
      MoveType::kA, MoveType::kB, MoveType::kL, MoveType::kR,
    };
    while (mp[r][x][y]) {
      if (mp[r][x][y] != 1) ret.push_back({x - 1, lookup[mp[r][x][y]]});
      switch (mp[r][x][y]) {
        case 1: x--; break;
        case 2: r = r == 0 ? R - 1 : r - 1; break;
        case 3: r = r == R - 1 ? 0 : r + 1; break;
        case 4: y++; break;
        case 5: y--; break;
        default: x = y = 0;
      }
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
  }


 public:
  Tetris(uint64_t seed = 0) : rng_(seed) {
    ResetGame(18);
  }

  void ResetGame(int start_level, double hz_avg = 10, double hz_dev = 0,
                 bool das = true, double first_tap_max = 0,
                 double microadj_delay = 400, double base_misdrop_rate = 5e-3,
                 double misdrop_param_time = 250,
                 double misdrop_param_pow = 1.5, int target = 1000000) {
    start_level_ = start_level;
    level_ = start_level;
    lines_ = 0;
    score_ = 0;
    next_piece_ = 0;
    pieces_ = 0;
    SpawnPiece_();
    SpawnPiece_();
    for (auto& i : field_) std::fill(i.begin(), i.end(), false);
    hz_avg_ = hz_avg;
    hz_dev_ = hz_dev;
    das_ = das;
    first_tap_max_ = first_tap_max;
    microadj_delay_ = microadj_delay;
    base_misdrop_rate_ = base_misdrop_rate;
    misdrop_param_time_ = misdrop_param_time;
    misdrop_param_pow_ = misdrop_param_pow;
    prev_drop_time_ = 1800;
    prev_misdrop_ = false;
    prev_micro_ = false;
    das_not_charged_ = false;
    target_ = target;
  }

  static int PlaceField(Field& field, int piece, int cx, int cy, int rotate) {
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
        field[j] = field[i];
      }
    }
    int ans = j + 1;
    for (; j >= 0; j--) std::fill(field[j].begin(), field[j].end(), false);
    return ans;
  }

  /// For training

  /// For evaluating
  void SetNowPiece(int piece) { now_piece_ = piece; }
  void SetNextPiece(int piece) { next_piece_ = piece; }



  // Helpers

  using PlaceMap = std::vector<std::array<std::array<bool, kM>, kN>>;

  static PlaceMap GetPlacements(const Field& field, int piece) {
    Map_ mp = Dijkstra_(GetMap_(field, piece), 0, 5, 0);
    PlaceMap ret(mp.size(), PlaceMap::value_type{});
    for (size_t r = 0; r < mp.size(); r++) {
      for (size_t x = 0; x < kN; x++) {
        for (size_t y = 0; y < kM; y++) {
          ret[r][x][y] = mp[r][x + 1][y + 1] && !mp[r][x + 2][y + 1];
        }
      }
    }
    return ret;
  }

  static MoveSequence GetMoveSequence(
      const Field& field, int piece, int start_x, int start_y, int start_rotate,
      int end_x, int end_y, int end_rotate) {
    MoveSequence ret{};
    if (start_x == end_x && start_y == end_y && start_rotate == end_rotate) {
      ret.valid = true;
      return ret;
    }

    Map_ mp = GetMap_(field, piece);
    Map_ mp_lb = Dijkstra_(mp, start_x, start_y, start_rotate);
    if (!mp_lb[end_rotate][end_x + 1][end_y + 1]) return ret; // impossible move
    std::vector<std::pair<int, MoveType>> lb =
        MovesFromMap_(std::move(mp_lb), end_x, end_y, end_rotate);
    // Get upper bound using the sequence of lb
    Map_ mp_rb = Dijkstra_(mp, start_x, start_y, start_rotate, lb);
    std::vector<std::pair<int, MoveType>> rb =
        MovesFromMap_(std::move(mp_rb), end_x, end_y, end_rotate);
    if (lb.size() != rb.size()) throw 1;
    for (size_t i = 0; i < lb.size(); i++) {
      if (lb[i].second != rb[i].second) throw 2L;
      Move now = {lb[i].first, rb[i].first, lb[i].second};
      (now.height_start == 0 ? ret.start : ret.fin).push_back(now);
    }
    ret.valid = true;
    return ret;
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

#include <cstdio>

void Print(const Tetris::Field& field) {
  for (auto& i : field) {
    for (auto& j : i) printf("%d ", (int)j);
    putchar('\n');
  }
}
void Print(const Tetris::PlaceMap& mp) {
  for (size_t i = 0; i < Tetris::kN; i++) {
    for (size_t j = 0; j < mp.size(); j++) {
      for (auto& k : mp[j][i]) printf("%d ", (int)k);
      putchar('|');
    }
    putchar('\n');
  }
}
void Print(const Tetris::Move& mv) {
  switch (mv.type) {
    case Tetris::MoveType::kL: printf("L"); break;
    case Tetris::MoveType::kR: printf("R"); break;
    case Tetris::MoveType::kA: printf("A"); break;
    case Tetris::MoveType::kB: printf("B"); break;
  }
  printf("%d-%d ", mv.height_start, mv.height_end);
}
void Print(const std::vector<Tetris::Move>& seq) {
  for (auto& i : seq) Print(i);
}
void Print(const Tetris::MoveSequence& seq) {
  Print(seq.start);
  putchar('|');
  Print(seq.fin);
  putchar('\n');
}

int main() {
  Tetris t;
  Tetris::Field field = {{
    {{0,0,0,0,0,0,0,0,0,0}},
    {{0,0,0,0,0,0,0,0,0,0}},
    {{0,0,0,1,0,0,0,0,0,0}},
    {{0,0,0,1,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}},
    {{0,0,1,1,0,0,0,0,0,0}},
    {{0,0,1,0,0,1,0,0,0,0}},
    {{0,1,0,0,0,0,0,0,0,0}},
    {{0,0,0,0,0,1,0,0,0,0}},
    {{0,1,0,0,1,1,0,0,0,0}},
  }};
  Print(field);
  Print(Tetris::GetPlacements(field, 6));
  Print(Tetris::GetMoveSequence(field, 6, 0, 5, 0, 18, 0, 1));
}

#endif
