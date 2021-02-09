#include <cmath>
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
  static constexpr bool IsAB(MoveType mv) {
    return mv == MoveType::kA || mv == MoveType::kB;
  }

  struct Move {
    int height_start, height_end;
    MoveType type;
  };
  struct MoveSequence {
    bool valid;
    std::vector<Move> moves;
  };
  struct Position {
    int rotate, x, y;
    bool operator==(const Position& pos) const {
      return rotate == pos.rotate && x == pos.x && y == pos.y;
    }
  };

 private:
  // # RNG
  std::mt19937_64 rng_;
  using RealRand_ = std::uniform_real_distribution<double>;
  using NormalRand_ = std::normal_distribution<double>;

  // # Game state
  int start_level_, level_, lines_, score_, pieces_;
  int now_piece_, next_piece_; // piece
  bool game_over_;
  Field field_;

  // # Parameters
  // Movement model
  double hz_avg_, hz_dev_;
  // The actual hz is sampled from NormalDistribution(hz_avg_, hz_dev_)
  // Note that for default DAS we can set hz_avg_ = 10, hz_dev_ = 0
  double first_tap_max_; // in milliseconds
  // The actual first-tap delay is sampled from UniformDistribution(0, first_tap_max_)
  bool das_;
  // DAS's effect:
  //  (1) Try to quicktap if the placement is infeasible
  //  (2) The next piece after microadjustment / spintucks / quicktap would have
  //      more delay (specifically, (8/3) / hz_avg_) because of unchared DAS

  // For simplicity, we assume a fixed microadjustment delay (the first
  //   microadjustment input would be done at exactly this time)
  double microadj_delay_; // in milliseconds

  // Misdrop model
  double orig_misdrop_rate_; // base misdrop rate
  double misdrop_param_time_; // in milliseconds
  double misdrop_param_pow_;
  // The actual base misdrop rate will be
  //   ((misdrop_param_time_ / x)^misdrop_param_pow_+1)*orig_misdrop_rate_
  //   where x is the dropping time of the previous piece. This simulates that
  //   it is more likely to misdrop with less thinking time.
  //   (misdrop_param_time_ indicates the dropping time that doubles the misdrop rate)
  // Misdrop multiplier for each misdrop types
  //   (the multiplier will be multiplied with the base misdrop rate to get the
  //    actual misdrop rate):
  //   1. Missing L/R tap
  static constexpr double kMissLRTapMultiplier_ = 0.5;
  //   2. Missing A/B tap
  static constexpr double kMissABTapMultiplier_ = 0.5;
  //   3. Miss tuck
  static constexpr double kMissTuckMultiplier_ = 1.0;
  //   4. Miss spin
  static constexpr double kMissSpinMultiplier_ = 2.0;
  //   5. Miss (each) microadjustment
  static constexpr double kMissMicroMultiplier_ = 3.0;
  //   6. Miss quicktap
  static constexpr double kMissQuicktapMultiplier_ = 2.0;
  //   7. The next piece after any misdrop will have higher misdrop rate
  static constexpr double kAfterMisdropMultiplier_ = 2.0;
  //   8. The next piece after any microadjustment will have higher misdrop rate
  static constexpr double kAfterMicroMultiplier_ = 1.3;
  //   9. Spins or tucks with less input window (e.g. higher level) will have
  //      higher misdrop rate. The actual multiplier is (3/input_window)^x,
  //      where x is this parameter (that is, 1 on level 18 of ordinary spins).
  //      This also give much higher misdrop probability to spintucks since it
  //      has much lower input window.
  static constexpr double kTuckOrSpinWindowMultiplierParam_ = 1.2;
  // Note: The training process would try suboptimal moves, so we don't simulate
  //   misdrops due to incorrect decision.
  // Again, for simplicity, we fix all the multipliers here.
  // I assigned these multipliers at my will, so some of them would differ a lot
  //   from human behavior. A future work can be analyzing human-played games to
  //   get a more promising misdrop model.

  // Reward model
  int target_; // in points
  static constexpr double kRewardMultiplierBeforeTarget_ = 2e-6;
  static constexpr double kRewardMultiplierAfterTarget_ = 5e-6;
  static constexpr double kTargetReward = 100;
  // The agent will get 2e-6 reward per point before reaching the target
  //   (that is, 2 per max-out), and get 100 reward immediately when reaching
  //   the target; after that, 5e-6 reward per point.
  // We use this to guide the agent toward the appropriate aggression to
  //   maximize the probability of reaching the point target.
  static constexpr double kInvalidReward = -0.3;
  // Provide a large reward deduction if the agent makes an invalid placement
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
  static constexpr double kFrameLength_ = 655171. / 39375;
  static constexpr Position kStartPosition_ = {0, 0, 5}; // (r, x, y)
  static constexpr double kDASDelayMultiplier_ = 8. / 3;

  void SpawnPiece_() {
    int next = std::discrete_distribution<int>(
        kTransitionProb_[next_piece_], kTransitionProb_[next_piece_] + kT)(rng_);
    now_piece_ = next_piece_;
    next_piece_ = next;
  }

  static bool IsGround_(int piece, const Position& pos) {
    auto& pl = kBlocks_[piece][pos.rotate];
    for (auto& i : pl) {
      if (pos.x + i.first == kN - 1) return true;
    }
    return false;
  }

  static double GetDropTime_(int piece, const Position& pos, int level,
                             bool clear) {
    return (pos.x + 1 + kBaseDelay_ +
            (IsGround_(piece, pos) ? 0 : kNotGroundDelay_) +
            (clear ? kLineClearDelay_ : 0)) *
           kFrameLength_;
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
  static Map_ Dijkstra_(const Map_& v, const Position& start) {
    const int R = v.size();
    int cx = start.x + 1, cy = start.y + 1, rotate = start.rotate;
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
  static Map_ Dijkstra_(const Map_& v, const Position& start,
                        const std::vector<std::pair<int, MoveType>>& moves) {
    const int R = v.size(), N = moves.size();
    int cx = start.x + 1, cy = start.y + 1, rotate = start.rotate;
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
      if (!ret[nd.r][nd.x][nd.y]) ret[nd.r][nd.x][nd.y] = nd.dir;
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
      const Map_& mp, const Position& pos) {
    int x = pos.x + 1, y = pos.y + 1, r = pos.rotate;
    int R = mp.size();
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

  // # Placements

  // The agent execution for each placement:
  //   1. Environment input: The placement of the current piece (A)
  //      Environment internal: simulate misdrops and compute the real placement
  //        result (S) after trying to place the piece according to B' (previous
  //        step) and A
  //   2. Environment output:
  //      - Reward: kInvalidReward [and return to step 1] <if invalid>
  //                all rewards (including score, target, infeasible, misdrop,...) <otherwise>
  //      - State: The board after placing the current piece by A (**without**
  //          misdrops) with place_stage_ = true (A is also available);
  //          the current piece is next_piece_, and the next piece is empty
  //   3. Environment input: The planned placement of the next piece (B)
  //      Environment internal: Store the planned placement as move sequence
  //        to use at the next step (B'); update now_piece_ and next_piece_
  //   4. Environment output:
  //      - Reward: kInvalidReward [and return to step 3] <if invalid>
  //                0 <otherwise>
  //      - State: The board after placing piece by S (simulated at step 1) with
  //          place_stage_ = false (B is also available);
  //          the current piece is now_piece_, and the next piece is next_piece_
  //  Note: The first step would have empty planned placement and place_stage_ = false
  double prev_drop_time_; // in milliseconds
  bool prev_misdrop_, prev_micro_, das_charged_;

  bool place_stage_;
  MoveSequence planned_seq_;
  Position real_placement_; // S in step 1
  Position planned_placement_; // A in step 1, or B in step 3
  int consecutive_invalid_;

  Map_ stored_mp_, stored_mp_lb_;
  bool mp_stored_;

  static MoveSequence GetMoveSequence_(const Map_& mp, const Map_& mp_lb,
                                       const Position& start,
                                       const Position& end) {
    std::vector<std::pair<int, MoveType>> lb = MovesFromMap_(mp_lb, end);
    // Get upper bound using the sequence of lb
    Map_ mp_ub = Dijkstra_(mp, start, lb);
    std::vector<std::pair<int, MoveType>> ub = MovesFromMap_(mp_ub, end);
    MoveSequence ret;
    for (size_t i = 0; i < lb.size(); i++) {
      ret.moves.push_back({lb[i].first, ub[i].first, lb[i].second});
    }
    ret.valid = true;
    return ret;
  }

  static bool CheckMovePossible_(const Map_& mp, const Position& pos) {
    return mp[pos.rotate][pos.x + 1][pos.y + 1];
  }

  void StoreMap_() {
    if (mp_stored_) return;
    stored_mp_ = GetMap_(field_, now_piece_);
    stored_mp_lb_ = Dijkstra_(stored_mp_, kStartPosition_);
    mp_stored_ = true;
  }

  static std::vector<int> GetInputWindow_(const MoveSequence& seq,
                                          int start_row, int frames_per_drop) {
    size_t start = 0;
    while (start < seq.moves.size() &&
           seq.moves[start].height_start == start_row) {
      start++;
    }
    if (start == seq.moves.size()) return std::vector<int>();
    size_t sz = seq.moves.size() - start;
    std::vector<int> lb(sz), ub(sz);
    int prev = -3;
    for (size_t i = 0; i < sz; i++) {
      size_t idx = i + start;
      auto& mv = seq.moves[idx];
      int lower = prev;
      if (i != 0) {
        auto& prev_type = seq.moves[idx - 1].type;
        if (mv.type == prev_type) {
          lower += 2;
        } else if (!IsAB(mv.type) && IsAB(prev_type)) {
          lower += 1;
        }
      }
      prev = lb[i] = std::max(mv.height_start * frames_per_drop, lower);
    }
    prev = 2147483647;
    for (size_t i = 0; i < sz; i++) {
      size_t idx = seq.moves.size() - i - 1;
      auto& mv = seq.moves[idx];
      int upper = prev;
      if (i != 0) {
        auto& prev_type = seq.moves[idx + 1].type;
        if (mv.type == prev_type) {
          upper -= 2;
        } else if (IsAB(mv.type) && !IsAB(prev_type)) {
          upper -= 1;
        }
      }
      prev = ub[i] = std::min((mv.height_end + 1) * frames_per_drop - 1, upper);
    }
    for (size_t i = 0; i < sz; i++) {
      ub[i] = std::max(1, ub[i] - lb[i] + 1);
    }
    return ub;
  }

  struct FrameInput {
    bool l, r, a, b; // l+r / a+b invalid here
  };

  struct FrameSequence {
    std::vector<FrameInput> seq;
    bool is_charged;
  };

  FrameSequence SequenceToFrame_(
      const MoveSequence& seq, double hz, bool misdrop, bool quicktap,
      int frames_per_drop, int start_row = 0,
      const std::vector<FrameInput>& prev_input = std::vector<FrameInput>()) {
    const double start_delay = RealRand_(0, first_tap_max_)(rng_);
    const double base_interval = 1000. / hz;
    const bool is_micro = !prev_input.empty();
    double prev_lr = -1000, prev_ab = -1000, prev = start_delay;

    // TODO: parse prev
    for (size_t i = 0; i < prev_input.size(); i++) {
      auto& input = prev_input[i];
      if (input.a || input.b) prev = prev_ab = i * kFrameLength_;
      if (input.l || input.r) prev = prev_lr = i * kFrameLength_;
    }

    size_t second_lr = 0, last_lr = 0, lr_count = 0;
    size_t last_ab = 0, ab_count = 0;
    if (das_ || misdrop) {
      for (size_t i = 0; i < seq.moves.size(); i++) {
        auto& mv = seq.moves[i];
        if (mv.height_start == start_row) {
          if (IsAB(mv.type)) {
            ++ab_count;
            last_ab = i;
          } else {
            if (++lr_count == 2) second_lr = i;
            last_lr = i;
          }
        }
      }
    }

    std::vector<FrameInput> ret;
    bool is_charged = true;
    auto Set = [&ret](size_t frame, MoveType mv) {
      if (ret.size() <= frame) ret.resize(frame + 1, FrameInput{});
      switch (mv) {
        case MoveType::kA: ret[frame].a = true; break;
        case MoveType::kB: ret[frame].b = true; break;
        case MoveType::kL: ret[frame].l = true; break;
        case MoveType::kR: ret[frame].r = true; break;
      }
    };
    auto RunLoop = [&](auto&& func) {
      for (size_t i = 0; i < seq.moves.size(); i++) {
        const auto& move = seq.moves[i];
        double lower = prev;
        if (i > 0 && IsAB(move.type) && !IsAB(seq.moves[i - 1].type)) {
          lower += kFrameLength_;
        }
        double prev_cat = IsAB(move.type) ? prev_ab : prev_lr;
        double frame_start = (move.height_start * frames_per_drop) * kFrameLength_;
        double interval = base_interval;
        bool is_quicktap = false, is_quicktap_fast = false;
        if (das_ && !is_micro) {
          // Assume that microadjustments don't use DAS, and has the same speed
          //   as charged DAS.
          if (quicktap) {
            // Assume that if tap count = 2, the quicktap has the same speed as
            //   the speed of charged DAS; otherwise, perfect (2 frame) quicktap
            //   is assumed with additional missing probability.
            if (!das_charged_ && lr_count > 2 && i == second_lr) {
              interval *= kDASDelayMultiplier_;
            }
            if (lr_count >= 2 && i == last_lr) is_quicktap = true;
            if (lr_count > 2 && i == last_lr) {
              interval = kFrameLength_ * 2;
              is_quicktap_fast = true;
            }
          } else {
            if (!das_charged_ && lr_count >= 2 && i == second_lr) {
              interval *= kDASDelayMultiplier_;
            }
          }
        }
        double time = std::max(std::max(lower, frame_start), prev_cat + interval);
        (IsAB(move.type) ? prev_ab : prev_lr) = time;
        prev = time;
        if (func(i, move, is_quicktap_fast)) {
          Set(time / kFrameLength_, move.type);
          if (!IsAB(move.type)) {
            // Quicktap / microadjustments uncharges DAS
            if (is_quicktap || is_micro && move.height_start == start_row) {
              is_charged = false;
            }
            // Tucks recharges DAS (but not spintucks that need same-height
            //   inputs)
            int last_start = move.height_start - (frames_per_drop == 1 ? 2 : 1);
            if (move.height_start != start_row &&
                (i == 0 || seq.moves[i - 1].height_start <= last_start)) {
              is_charged = true;
            }
          }
        }
      }
    };
    if (misdrop) {
      double base_misdrop_rate = orig_misdrop_rate_ * (std::pow(
            misdrop_param_time_ / prev_drop_time_, misdrop_param_pow_) + 1);
      if (prev_misdrop_) base_misdrop_rate *= kAfterMisdropMultiplier_;
      if (prev_micro_) base_misdrop_rate *= kAfterMicroMultiplier_;
      auto window = GetInputWindow_(seq, start_row, frames_per_drop);
      size_t start = seq.moves.size() - window.size();
      if (is_micro) {
        double micro_miss_rate = base_misdrop_rate * kMissMicroMultiplier_;
        RunLoop([&](int i, const Move& move, bool) {
          if (RealRand_(0, 1)(rng_) < micro_miss_rate) return false;
          if (i >= start) {
            double rate = base_misdrop_rate * std::pow(3. / window[i - start],
                kTuckOrSpinWindowMultiplierParam_);
            rate *= IsAB(move.type) ? kMissSpinMultiplier_ : kMissTuckMultiplier_;
            if (RealRand_(0, 1)(rng_) < rate) return false;
          }
          return true;
        });
      } else {
        double lr_miss_rate = base_misdrop_rate * kMissLRTapMultiplier_;
        double ab_miss_rate = base_misdrop_rate * kMissABTapMultiplier_;
        double quicktap_rate = base_misdrop_rate * kMissQuicktapMultiplier_;
        RunLoop([&](int i, const Move& move, bool is_quicktap) {
          if ((i == last_lr && RealRand_(0, 1)(rng_) < lr_miss_rate) ||
              (i == last_ab && RealRand_(0, 1)(rng_) < ab_miss_rate) ||
              (is_quicktap && RealRand_(0, 1)(rng_) < quicktap_rate)) {
            return false;
          }
          if (i >= start) {
            double rate = base_misdrop_rate * std::pow(3. / window[i - start],
                kTuckOrSpinWindowMultiplierParam_);
            rate *= IsAB(move.type) ? kMissSpinMultiplier_ : kMissTuckMultiplier_;
            if (RealRand_(0, 1)(rng_) < rate) return false;
          }
          return true;
        });
      }
    } else { // height_end is not used
      RunLoop([&](int, const Move&, bool) { return true; });
    }
    return {ret, is_charged};
  }

  static Position Simulate_(const Map_& mp, const FrameSequence& seq, int frames_per_drop, bool finish = true) {
    Position now = kStartPosition_;
    int R = mp.size();
    for (size_t i = 0; i < seq.seq.size(); i++) {
      auto& input = seq.seq[i];
      if (input.r) {
        if (mp[now.rotate][now.x + 1][now.y + 2]) now.y++;
      } else if (input.l) {
        if (mp[now.rotate][now.x + 1][now.y]) now.y--;
      }
      if (input.a || input.b) {
        int r;
        if (input.a) {
          r = now.rotate == R - 1 ? 0 : now.rotate + 1;
        } else {
          r = now.rotate == 0 ? R - 1 : now.rotate - 1;
        }
        if (mp[r][now.x + 1][now.y + 1]) now.rotate = r;
      }
      if ((i + 1) % frames_per_drop == 0) {
        if (!mp[now.rotate][now.x + 2][now.y + 1]) return now;
        now.x++;
      }
    }
    if (finish) {
      while (mp[now.rotate][now.x + 2][now.y + 1]) now.x++;
    }
    return now;
  }

  double InputPlacement_(const Position& pos) {
    if (place_stage_) { // step 3
      Field tmp_field = field_;
      PlaceField(tmp_field, now_piece_, planned_placement_);
      MoveSequence seq = GetMoveSequence(tmp_field, next_piece_, kStartPosition_, pos);
      if (!seq.valid) {
        if (++consecutive_invalid_ == 3) game_over_ = true;
        return kInvalidReward;
      }
      planned_seq_ = seq;
      planned_placement_ = pos;
      // place now_piece according to real_placement_
      // mp_stored_ = false;
      return 0;
    }
    // step 1
    StoreMap_();
    if (!CheckMovePossible_(stored_mp_lb_, pos)) {
      if (++consecutive_invalid_ == 3) game_over_ = true;
      return kInvalidReward;
    }
    if (!planned_seq_.valid) { // special case: first piece
      // Assume one won't misdrop on the first piece
      real_placement_ = pos;
      planned_placement_ = pos;
      prev_misdrop_ = false;
      prev_micro_ = false;
      das_charged_ = true; // we can make it charged anyway
      return;
    }
    Position start_pos;
    start_pos = kStartPosition_;
  }

 public:
  Tetris(uint64_t seed = 0) : rng_(seed) {
    ResetGame(18);
  }

  void ResetGame(int start_level, double hz_avg = 10, double hz_dev = 0,
                 bool das = true, double first_tap_max = 0,
                 double microadj_delay = 400, double orig_misdrop_rate = 5e-3,
                 double misdrop_param_time = 250,
                 double misdrop_param_pow = 1.5, int target = 1000000) {
    start_level_ = start_level;
    level_ = start_level;
    lines_ = 0;
    score_ = 0;
    next_piece_ = 0;
    pieces_ = 0;
    game_over_ = false;
    SpawnPiece_();
    SpawnPiece_();
    for (auto& i : field_) std::fill(i.begin(), i.end(), false);
    hz_avg_ = hz_avg;
    hz_dev_ = hz_dev;
    das_ = das;
    first_tap_max_ = first_tap_max;
    microadj_delay_ = microadj_delay;
    orig_misdrop_rate_ = orig_misdrop_rate;
    misdrop_param_time_ = misdrop_param_time;
    misdrop_param_pow_ = misdrop_param_pow;
    prev_drop_time_ = 1800;
    prev_misdrop_ = false;
    prev_micro_ = false;
    das_charged_ = false;
    target_ = target;
    place_stage_ = false;
    planned_seq_.valid = false;
    mp_stored_ = false;
  }

  static int PlaceField(Field& field, int piece, const Position& pos) {
    auto& pl = kBlocks_[piece][pos.rotate];
    for (auto& i : pl) {
      int nx = pos.x + i.first, ny = pos.y + i.second;
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

  // # For training

  // # For evaluating
  void SetNowPiece(int piece) { now_piece_ = piece; }
  void SetNextPiece(int piece) { next_piece_ = piece; }



  // # Helpers

  using PlaceMap = std::vector<std::array<std::array<bool, kM>, kN>>;

  static PlaceMap GetPlacements(const Field& field, int piece) {
    Map_ mp = Dijkstra_(GetMap_(field, piece), kStartPosition_);
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

  static MoveSequence GetMoveSequence(const Field& field, int piece,
                                      const Position& start,
                                      const Position& end) {
    MoveSequence ret{};
    if (start == end) {
      ret.valid = true;
      return ret;
    }

    Map_ mp = GetMap_(field, piece);
    Map_ mp_lb = Dijkstra_(mp, start);
    if (!CheckMovePossible_(mp_lb, end)) return ret; // impossible move
    return GetMoveSequence_(mp, mp_lb, start, end);
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
decltype(Tetris::kStartPosition_) Tetris::kStartPosition_;

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
  Print(seq.moves);
  putchar('\n');
}

int main() {
  Tetris t;
  Tetris::Field field = {{
    {{0,0,0,0,0,0,0,0,0,0}},
    {{0,0,0,0,0,0,0,0,0,0}},
    {{0,0,0,1,0,0,0,0,0,0}},
    {{0,0,1,1,1,0,0,0,0,0}},
    {{0,0,0,1,1,0,0,0,0,0}},
    {{1,0,0,0,1,0,0,0,0,0}},
    {{1,0,0,0,0,1,0,0,0,0}},
    {{0,1,0,0,0,0,1,0,0,0}},
    {{0,0,1,0,0,0,0,1,0,0}},
    {{0,0,0,1,0,0,0,0,1,0}},
    {{0,0,0,0,1,0,0,0,0,1}},
    {{0,0,0,0,0,1,1,0,0,0}},
    {{0,0,0,0,0,1,0,0,0,0}},
    {{0,0,0,1,1,0,0,0,0,0}},
    {{0,0,0,1,1,0,0,0,0,0}},
    {{0,0,1,1,0,0,0,0,0,0}},
    {{0,0,1,0,0,0,0,0,0,0}},
    {{0,1,0,0,0,1,0,0,0,0}},
    {{0,1,0,0,0,1,0,0,0,0}},
    {{0,1,0,0,1,1,0,0,0,0}},
  }};
  Print(field);
  Print(Tetris::GetPlacements(field, 0));
  Print(Tetris::GetMoveSequence(field, 0, {0, 0, 5}, {3, 18, 2}));
}

#endif
