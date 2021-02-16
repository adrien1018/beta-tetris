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

#ifdef DEBUG
#define DEBUG_METHODS
#endif

#ifdef DEBUG_METHODS
#include <cstdio>
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
    bool operator!=(const Position& pos) const { return !(*this == pos); }
  };

  using State = std::array<std::array<std::array<float, kM>, kN>, 14>;

 private:
  // # RNG
  std::mt19937_64 rng_;
  using RealRand_ = std::uniform_real_distribution<double>;
  using NormalRand_ = std::normal_distribution<double>;

  // # Game state
  int start_level_, lines_, score_, pieces_;
  int now_piece_, next_piece_; // piece
  bool game_over_;
  Field field_;

  static constexpr int GetLevel_(int start_level, int lines) {
    int first = kLinesBeforeLevelUp_[start_level];
    return lines < first ? start_level :
        start_level + 1 + (lines - first) / 10;
  }

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
  int microadj_delay_; // in frames

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
  static constexpr double kMissLRTapMultiplier_ = 0.3;
  //   2. Missing A/B tap
  static constexpr double kMissABTapMultiplier_ = 0.3;
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
  double reward_multiplier_;
  static constexpr double kTargetReward_ = 100;
  // The agent will get reward_multiplier_ reward per point before reaching the
  //   target, and get 100 reward immediately when reaching the target; after
  //   that, (reward_multiplier_ * 0.5) reward per point.
  // We use this to guide the agent toward the appropriate aggression to
  //   maximize the probability of reaching the point target.
  static constexpr double kInvalidReward_ = -0.3;
  // Provide a large reward deduction if the agent makes an invalid placement
  static constexpr double kInfeasibleReward_ = -0.00;
  // Provide a large reward deduction if the agent makes an infeasible placement
  // "Infeasible" placements are those cannot be done by +3σ tapping speeds
  //   (750-in-1 chance) and without misdrop
  static constexpr double kMisdropReward_ = -0.01;
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

  static double GetDropTime_(int piece, const Position& pos,
                             int frames_per_drop, bool clear) {
    return ((pos.x + 1) * frames_per_drop + kBaseDelay_ +
            (IsGround_(piece, pos) ? 0 : kNotGroundDelay_) +
            (clear ? kLineClearDelay_ : 0)) *
           kFrameLength_;
  }

  static double GetScore_(int lines, int level) {
    return kScoreBase_[lines] * (level + 1);
  }

  static int GetFramesPerDrop_(int level) {
    return level >= 29 ? 1 : kFramesPerDrop_[level];
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

  static bool MapEmpty_(const Map_& mp) {
    for (auto& i : mp) {
      for (auto& j : i) {
        for (auto& k : j) {
          if (k) return false;
        }
      }
    }
    return true;
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
    ret[rotate][cx][cy] = 0;
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
  struct FrameInput_ {
    bool l, r, a, b; // l+r=r / a+b=a here
  };
  struct FrameSequence_ {
    std::vector<FrameInput_> seq;
    bool is_charged;
  };

  double prev_drop_time_; // in milliseconds
  bool prev_misdrop_, prev_micro_, das_charged_;

  bool place_stage_;
  Position real_placement_; // S in step 1
  // Information about B (used as B' in step 1)
  Position planned_placement_; // also used to store A (used as state in step 2)
  MoveSequence planned_seq_;
  FrameSequence_ planned_fseq_;
  bool planned_quicktap_;

  // Game information after placed by A
  Field temp_field_;
  int temp_lines_, temp_score_;

  Map_ stored_mp_, stored_mp_lb_;
  int consecutive_invalid_;

  static MoveSequence GetMoveSequenceLb_(const Map_& mp_lb,
                                         const Position& end) {
    if (!CheckMovePossible_(mp_lb, end)) return MoveSequence{};
    std::vector<std::pair<int, MoveType>> lb = MovesFromMap_(mp_lb, end);
    MoveSequence ret;
    for (size_t i = 0; i < lb.size(); i++) {
      ret.moves.push_back({lb[i].first, 0, lb[i].second});
    }
    ret.valid = true;
    return ret;
  }

  static MoveSequence GetMoveSequence_(const Map_& mp, const Map_& mp_lb,
                                       const Position& start,
                                       const Position& end) {
    if (!CheckMovePossible_(mp_lb, end)) return MoveSequence{};
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
    return mp[pos.rotate][pos.x + 1][pos.y + 1] && !mp[pos.rotate][pos.x + 2][pos.y + 1];
  }

  void StoreMap_(bool use_temp) {
    stored_mp_ = use_temp ? GetMap_(temp_field_, next_piece_)
                          : GetMap_(field_, now_piece_);
    stored_mp_lb_ = Dijkstra_(stored_mp_, kStartPosition_);
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

  FrameSequence_ SequenceToFrame_(
      const MoveSequence& seq, double hz, bool misdrop, bool quicktap,
      int frames_per_drop, int start_row = 0,
      const FrameSequence_& prev_input = FrameSequence_{}) {
    const double start_delay = misdrop ? RealRand_(0, first_tap_max_)(rng_) : 0;
    const double base_interval = 1000. / hz;
    const bool is_micro = !prev_input.seq.empty();
    double prev_lr = -1000, prev_ab = -1000;

    for (size_t i = 0; i < prev_input.seq.size(); i++) {
      auto& input = prev_input.seq[i];
      if (input.a || input.b) prev_ab = i * kFrameLength_;
      if (input.l || input.r) prev_lr = i * kFrameLength_;
    }
    double prev = std::max(start_delay, prev_input.seq.size() * kFrameLength_);

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

    std::vector<FrameInput_> ret = prev_input.seq;
    bool is_charged = true;
    auto Set = [&ret](size_t frame, MoveType mv) {
      if (ret.size() <= frame) ret.resize(frame + 1, FrameInput_{});
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
        if (!IsAB(move.type) && move.height_start == start_row) {
          lower = start_delay;
        } else if (i > 0 && !IsAB(move.type) && IsAB(seq.moves[i - 1].type)) {
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
            if (is_quicktap || (is_micro && move.height_start == start_row)) {
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
        RunLoop([&](size_t i, const Move& move, bool) {
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
        RunLoop([&](size_t i, const Move& move, bool is_quicktap) {
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
      RunLoop([&](size_t, const Move&, bool) { return true; });
    }
    return {ret, is_charged};
  }

  static Position Simulate_(const Map_& mp, const FrameSequence_& seq, int frames_per_drop, bool finish = true) {
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

  bool SequenceEquivalent_(const MoveSequence& seq1, const MoveSequence& seq2) {
    if (seq1.moves.size() != seq2.moves.size()) return false;
    for (size_t i = 0; i < seq1.moves.size(); i++) {
      auto& mv1 = seq1.moves[i];
      auto& mv2 = seq2.moves[i];
      if (mv1.type != mv2.type ||
          (mv1.height_start == 0) != (mv2.height_start == 0)) {
        return false;
      }
    }
    return true;
  }

  double InputPlacement_(const Position& pos) {
    if (place_stage_) { // step 3
      if (!CheckMovePossible_(stored_mp_lb_, pos)) {
        if (++consecutive_invalid_ == 3) game_over_ = true;
        return kInvalidReward_;
      }
      MoveSequence seq = GetMoveSequence_(stored_mp_, stored_mp_lb_, kStartPosition_, pos);
      double reward = 0;
      planned_placement_ = pos;
      planned_seq_ = seq;
      int frames_per_drop = GetFramesPerDrop_(GetLevel_(start_level_, temp_lines_));
      planned_fseq_ = SequenceToFrame_(seq, hz_avg_ + hz_dev_ * 3, false, false,
                                       frames_per_drop);
      if (Simulate_(stored_mp_, planned_fseq_, frames_per_drop) != pos) {
        bool flag = false;
        if (das_) {
          auto fseq = SequenceToFrame_(seq, hz_avg_ + hz_dev_ * 3, false, true,
                                       frames_per_drop);
          if (Simulate_(stored_mp_, fseq, frames_per_drop) == pos) {
            planned_fseq_ = fseq;
            flag = true;
          }
          planned_quicktap_ = flag;
        }
        if (!flag) reward += kInfeasibleReward_;
      }
      place_stage_ = false;
      consecutive_invalid_ = 0;
      return reward;
    }
    // step 1
    if (!CheckMovePossible_(stored_mp_lb_, pos)) {
      if (++consecutive_invalid_ == 3) game_over_ = true;
      return kInvalidReward_;
    }
    double reward = 0;
    int level = GetLevel_(start_level_, lines_);
    int frames_per_drop = GetFramesPerDrop_(level);
    if (!planned_seq_.valid) { // special case: first piece
      // Assume one won't misdrop on the first piece
      real_placement_ = pos;
      prev_misdrop_ = false;
      prev_micro_ = false;
      das_charged_ = true; // we can make it charged anyway
    } else {
      if (!real_placement_set_) throw 1; // TODO
      FrameSequence_ fseq;
      double hz = NormalRand_(hz_avg_, hz_dev_)(rng_);
      if (hz < 8) hz = 8;
      if (hz > 30) hz = 30;

      bool flag = false;
      if (pos == planned_placement_) {
        MoveSequence cur_seq =
            GetMoveSequence_(stored_mp_, stored_mp_lb_, kStartPosition_, pos);
        if (SequenceEquivalent_(cur_seq, planned_seq_)) { // no micro
          fseq = SequenceToFrame_(cur_seq, hz, true, planned_quicktap_,
                                  frames_per_drop);
          flag = true;
        }
      }
      if (!flag) { // has micro
        if (!prev_misdrop_) {
          planned_fseq_.seq.resize(microadj_delay_);
          Position pos_before_adj =
              Simulate_(stored_mp_, planned_fseq_, frames_per_drop, false);
          Map_ tmp_mp = Dijkstra_(stored_mp_, pos_before_adj);
          MoveSequence seq = GetMoveSequenceLb_(tmp_mp, pos);
          if (seq.valid) {
            fseq = SequenceToFrame_(seq, hz_avg_ + 3 * hz_dev_, false, false,
                                    frames_per_drop, pos_before_adj.x,
                                    planned_fseq_);
            if (Simulate_(stored_mp_, fseq, frames_per_drop) != pos) {
              reward += kInfeasibleReward_;
            }
          } else {
            reward += kInfeasibleReward_;
          }
        }
        fseq = SequenceToFrame_(planned_seq_, hz, true, planned_quicktap_,
                                frames_per_drop);
        fseq.seq.resize(microadj_delay_);
        Position pos_before_adj =
            Simulate_(stored_mp_, fseq, frames_per_drop, false);
        MoveSequence seq =
            GetMoveSequence_(stored_mp_, Dijkstra_(stored_mp_, pos_before_adj),
                             pos_before_adj, pos);
        if (seq.valid) {
          fseq = SequenceToFrame_(seq, hz, true, false, frames_per_drop,
                                  pos_before_adj.x, fseq);
        }
      }
      //Print(fseq);
      real_placement_ = Simulate_(stored_mp_, fseq, frames_per_drop);
      prev_misdrop_ = real_placement_ != pos;
      if (prev_misdrop_) reward += kMisdropReward_;
      prev_micro_ = !flag;
      das_charged_ = fseq.is_charged;
    }
    planned_placement_ = pos;
    temp_field_ = field_;
    int planned_lines = PlaceField(temp_field_, now_piece_, planned_placement_);
    temp_lines_ = lines_ + planned_lines;
    temp_score_ = score_ + GetScore_(planned_lines, level);
    Field field(field_);
    int real_lines = PlaceField(field, now_piece_, real_placement_);
    int orig_score = score_;
    int score_delta = GetScore_(real_lines, level);
    int new_score = score_ + score_delta;
    prev_drop_time_ = GetDropTime_(now_piece_, real_placement_, frames_per_drop,
                                   real_lines > 0);
    pieces_++;
    if (orig_score >= target_) {
      reward += (reward_multiplier_ * 0.5) * score_delta;
    } else if (new_score < target_) {
      reward += reward_multiplier_ * score_delta;
    } else {
      reward += reward_multiplier_ * (target_ - orig_score);
      reward += (reward_multiplier_ * 0.5) * (new_score - target_);
      reward += kTargetReward_;
    }
    real_placement_set_ = false;
    place_stage_ = true;
    consecutive_invalid_ = 0;
    StoreMap_(true);
    game_over_ = MapEmpty_(stored_mp_lb_);
    return reward;
  }

  bool real_placement_set_;

  bool SetRealPlacement_(const Position& pos) {
    if (real_placement_set_) return false;
    real_placement_ = pos;
    int real_lines = PlaceField(field_, now_piece_, real_placement_);
    int level = GetLevel_(start_level_, lines_);
    lines_ += real_lines;
    score_ += GetScore_(real_lines, level);
    SpawnPiece_();
    StoreMap_(false);
    game_over_ = MapEmpty_(stored_mp_lb_);
    if (lines_ >= 350) game_over_ = true; // prevent game going indefinitely
    real_placement_set_ = true;
    return true;
  }

 public:
  Tetris(uint64_t seed = 0) : rng_(seed) {
    ResetGame(18);
  }

  void Reseed(uint64_t seed = 0) {
    rng_.seed(seed);
  }

  void ResetGame(int start_level = 18, double hz_avg = 10, double hz_dev = 0,
                 bool das = true, double first_tap_max = 30,
                 int microadj_delay = 40, double orig_misdrop_rate = 5e-3,
                 double misdrop_param_time = 400,
                 double misdrop_param_pow = 1.0, int target = 1000000,
                 double reward_multiplier = 2e-6) {
    start_level_ = start_level;
    lines_ = 0;
    score_ = 0;
    next_piece_ = 0;
    pieces_ = 0;
    game_over_ = false;
    SpawnPiece_();
    SpawnPiece_();
    consecutive_invalid_ = 0;
    for (auto& i : field_) std::fill(i.begin(), i.end(), false);
    reward_multiplier_ = reward_multiplier;
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
    real_placement_set_ = false;
    planned_quicktap_ = false;
    StoreMap_(false);
  }

  void ResetRandom(double fix_prob) {
    using IntRand = std::uniform_int_distribution<int>;
    using GammaRand = std::gamma_distribution<double>;
    auto Padded = [&](auto&& dist, double uniform_ratio, double l, double r) {
      if (RealRand_(0, 1)(rng_) < uniform_ratio) {
        return RealRand_(l, r)(rng_);
      } else {
        double val = dist(rng_);
        if (val < l) val = l;
        if (val > r) val = r;
        return val;
      }
    };
    double s_param = RealRand_(0, 1)(rng_); // strength parameter
    int start_level = IntRand(18, 19)(rng_);
    bool das = false;
    double hz_avg = Padded(NormalRand_(11 + 5 * s_param, 2), 0.2, 8, 15 + 15 * s_param);
    double hz_dev = Padded(GammaRand(4, (hz_avg / (3 + s_param * 2)) * 0.1),
                           0.4, 0, hz_avg / (3 + s_param * 2));
    if (RealRand_(0, 1)(rng_) < 0.3) {
      das = true;
      hz_dev = 0;
      if (RealRand_(0, 1)(rng_) < 0.8) hz_avg = 10;
    }
    double first_tap_max = Padded(NormalRand_(30 - s_param * 20, 4), 0.25, 0, 50 - s_param * 30);
    int microadj_delay = Padded(NormalRand_(40 - s_param * 20, 5), 0.25, 0, 60 - s_param * 20);
    double orig_misdrop_rate = RealRand_(0, 1)(rng_) < 0.1 ? 0 :
        std::exp(Padded(NormalRand_(-4.5 - s_param, 1), 0.2, -6 - s_param * 4, -3 - s_param));
    double misdrop_param_time = Padded(NormalRand_(400, 100) , 0.6, 200, 700);
    double misdrop_param_pow = RealRand_(0.7, 1.8)(rng_);
    int target = Padded(NormalRand_(1.05e+6, 1.5e+5), 0.4, 2e+5, 1.5e+6);
    double reward_multiplier = RealRand_(0, 1)(rng_) < 0.1 ? 0 :
        Padded(GammaRand(0.5, 3e-6), 0.3, 0, 2e-5);
    if (RealRand_(0, 1)(rng_) < fix_prob) {
      if (RealRand_(0, 1)(rng_) < 0.6) {
        hz_avg = das ? 10 : 14;
        hz_dev = das ? 0 : 1;
      }
      first_tap_max = 20;
      microadj_delay = 30;
      orig_misdrop_rate = std::exp(-6);
      misdrop_param_time = 400;
      misdrop_param_pow = 1;
      reward_multiplier = 2e-5;
    }
    ResetGame(start_level, hz_avg, hz_dev, das, first_tap_max, microadj_delay,
              orig_misdrop_rate, misdrop_param_time, misdrop_param_pow, target,
              reward_multiplier);
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

  bool IsOver() const { return game_over_; }
  int GetScore() const { return score_; }
  int GetLines() const { return lines_; }

  State GetState() const {
    State ret{};
    // 0: field
    // 1-4: planned_placement_ (place_stage_ = true: step 1's input)
    // 5-8: planned_placement_ (place_stage_ = false)
    // 9-12: possible placements
    // 13: misc
    if (place_stage_) {
      for (int i = 0; i < kN; i++) {
        for (int j = 0; j < kM; j++) ret[0][i][j] = 1 - temp_field_[i][j];
      }
      ret[1 + planned_placement_.rotate][planned_placement_.x]
         [planned_placement_.y] = 1;
    } else {
      for (int i = 0; i < kN; i++) {
        for (int j = 0; j < kM; j++) ret[0][i][j] = 1 - field_[i][j];
      }
      if (planned_seq_.valid) {
        ret[5 + planned_placement_.rotate][planned_placement_.x]
          [planned_placement_.y] = 1;
      }
    }
    for (size_t r = 0; r < stored_mp_lb_.size(); r++) {
      for (int i = 0; i < kN; i++) {
        for (int j = 0; j < kM; j++) {
          ret[9 + r][i][j] = stored_mp_lb_[r][i + 1][j + 1] &&
              !stored_mp_lb_[r][i + 2][j + 1];
        }
      }
    }
    // misc
    float* misc = (float*)ret[13].data();
    // 0-6: current
    misc[0 + (place_stage_ ? next_piece_ : now_piece_)] = 1;
    // 7-14: next / 7(if place_stage_)
    misc[place_stage_ ? 14 : 7 + next_piece_] = 1;
    // 15-16: start (18 or 19)
    misc[start_level_ == 19 ? 15 : 16] = 1;
    // 17: score
    misc[17] = (place_stage_ ? temp_score_ : score_) * 5e-6;
    // 18: target
    misc[18] = target_ * 5e-6;
    // 19: reward_multiplier
    misc[19] = reward_multiplier_ * 2e+5;
    // 20: level
    int lines = place_stage_ ? temp_lines_ : lines_;
    int level = GetLevel_(start_level_, lines);
    misc[20] = level * 1e-1;
    // 21: lines
    misc[21] = lines * 2e-2;
    // 22: pieces
    misc[22] = pieces_ * 2.5e-3;
    // 23-25: speed
    misc[23 + (GetFramesPerDrop_(level) - 1)] = 1;
    // 26-73: lines to next speed
    //   26: level 29+
    //   27-36: 1-10
    //   37-46: 11-30 (2)
    //   47-60: 31-100 (5)
    //   61-73: 101-230 (10)
    if (level >= 29) {
      misc[26] = 1;
    } else {
      int lines_to_next = level == 18 ? 130 - lines : 230 - lines;
      if (lines_to_next <= 10) {
        misc[27 + (lines_to_next - 1)] = 1;
      } else if (lines_to_next <= 30) {
        misc[37 + (lines_to_next - 11) / 2] = 1;
      } else if (lines_to_next <= 100) {
        misc[47 + (lines_to_next - 31) / 5] = 1;
      } else {
        misc[61 + (lines_to_next - 101) / 10] = 1;
      }
    }
    // 74-78: hz_avg, hz_dev, first_tap_max, das, microadj_delay
    misc[74] = hz_avg_ / 5;
    misc[75] = hz_dev_ / 5;
    misc[76] = first_tap_max_ * 1e-2;
    misc[77] = das_;
    misc[78] = microadj_delay_ * 1e-1;
    // 79-81: orig_misdrop_rate, misdrop_param_time, misdrop_param_pow
    misc[79] = std::max(-12., std::log(orig_misdrop_rate_ + 1e-12)) / 2;
    misc[80] = misdrop_param_time_ * 1e-3;
    misc[81] = misdrop_param_pow_;
    // 82: prev_misdrop
    misc[82] = !place_stage_ && prev_misdrop_;
    // 13 + 83 = 96 (channels)
    return ret;
  }

  double InputPlacement(const Position& pos, bool training = true) {
    if (game_over_) return 0;
    bool orig_stage = place_stage_;
    if (pos.rotate >= (int)stored_mp_lb_.size() || pos.x >= kN || pos.y >= kM) {
      return kInvalidReward_;
    }
    double ret = InputPlacement_(pos);
    if (training && orig_stage && !place_stage_) TrainingSetPlacement();
    return ret;
  }

  bool TrainingSetPlacement() {
    return SetRealPlacement_(real_placement_);
  }

  // # Training steps:
  //   init. <nothing>
  //   1. GetState
  //   2. InputPlacement
  //   3. GetState
  //   4. InputPlacement
  //   5. TrainingSetPlacement

  // # Evaluating steps:
  //   init. SetNowPiece, SetNextPiece
  //   1. GetState
  //   2. InputPlacement
  //   3. GetState
  //   4. InputPlacement
  //   -> drop result get <-
  //   5. SetPreviousPlacement
  //   6. SetNextPiece
  bool SetNowPiece(int piece) {
    if (pieces_ != 0 || place_stage_) return false;
    now_piece_ = piece;
    StoreMap_(false);
    return true;
  }
  void SetNextPiece(int piece) { next_piece_ = piece; }

  bool SetPreviousPlacement(const Position& pos) {
    return SetRealPlacement_(pos);
  }

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

#ifdef DEBUG_METHODS
private:
  static void Print(const Position& pos) {
    printf("(%d,%d,%d)", pos.rotate, pos.x, pos.y);
  }
  static void Print(const Field& field) {
    for (auto& i : field) {
      for (auto& j : i) printf("%d ", (int)j);
      putchar('\n');
    }
  }
  static void Print(const Map_& mp) {
    for (size_t i = 0; i < kN; i++) {
      for (size_t j = 0; j < mp.size(); j++) {
        for (auto& k : mp[j][i+1]) printf("%d ", (int)k);
        putchar('|');
      }
      putchar('\n');
    }
  }
  static void Print(const Move& mv) {
    switch (mv.type) {
      case MoveType::kL: printf("L"); break;
      case MoveType::kR: printf("R"); break;
      case MoveType::kA: printf("A"); break;
      case MoveType::kB: printf("B"); break;
    }
    printf("%d-%d ", mv.height_start, mv.height_end);
  }
  static void Print(const MoveSequence& seq) {
    if (!seq.valid) printf("(invalid)");
    for (auto& i : seq.moves) Print(i);
    putchar('\n');
  }
  static void Print(const FrameSequence_& seq) {
    printf("(charged:%d)", (int)seq.is_charged);
    for (auto& i : seq.seq) {
      std::string str;
      if (i.l) str += 'L';
      if (i.r) str += 'R';
      if (i.a) str += 'A';
      if (i.b) str += 'B';
      if (str.empty()) str += '-';
      printf(" %s", str.c_str());
    }
    putchar('\n');
  }

public:
  void PrintAllState() const {
    puts("Game:");
    printf("Start: %d, lines: %d, score: %d, pieces: %d\n",
        start_level_, lines_, score_, pieces_);
    constexpr char kType[] = "TJZOSLI";
    printf("Now: %c, next: %c; Game over: %d\n", kType[now_piece_],
           kType[next_piece_], (int)game_over_);
    puts("Field:");
    Print(field_);
    printf("Place stage: %d\n", (int)place_stage_);
    printf("Prev drop time: %f ms, prev misdrop: %d, prev micro: %d, das charged %d\n",
           prev_drop_time_, (int)prev_misdrop_, (int)prev_micro_,
           (int)das_charged_);
    printf("Real placement: "); Print(real_placement_);
    printf("\nPlanned placement: "); Print(planned_placement_);
    printf("\nPlanned sequence: "); Print(planned_seq_);
    printf("Planned frame sequence: "); Print(planned_fseq_);
    printf("Planned quicktap: %d\n", (int)planned_quicktap_);
    puts("Temp:");
    printf("Lines: %d, score: %d, field:\n", temp_lines_, temp_score_);
    Print(temp_field_);
    puts("Stored map lb:");
    Print(stored_mp_lb_);
    printf("Consecutive invalid: %d\n", consecutive_invalid_);
    puts("Parameters:");
    printf("Target: %d, reward multiplier: %e\n", target_, reward_multiplier_);
    printf("Hz: avg %f dev %f, first tap max: %f, das: %d\n", hz_avg_, hz_dev_,
           first_tap_max_, das_);
    printf("Microadj delay: %d, misdrop rate: %e\n", microadj_delay_,
           orig_misdrop_rate_);
    printf("Misdrop param: time %f, pow %f\n", misdrop_param_time_,
           misdrop_param_pow_);
    puts("");
    fflush(stdout);
  }

  void PrintState(bool field_only = false) const {
    auto st = GetState();
    constexpr char kType[] = "TJZOSLI";
    printf("Now: %c, next: %c; Game over: %d\n", kType[now_piece_],
           kType[next_piece_], (int)game_over_);
    puts("Field:");
    for (auto& i : st[0]) {
      for (auto& j : i) printf("%d ", (int)(1 - j));
      puts("");
    }
    if (field_only) {
      fflush(stdout);
      return;
    }
    puts("Possible:");
    for (size_t i = 0; i < kN; i++) {
      for (size_t j = 0; j < 4; j++) {
        for (auto& k : st[9+j][i]) printf("%d ", (int)k);
        putchar('|');
      }
      putchar('\n');
    }
    puts("Planned:");
    for (size_t i = 1; i <= 8; i++) {
      for (size_t j = 0; j < kN; j++) {
        for (size_t k = 0; k < kM; k++) {
          if (st[i][j][k]) printf("%d %d %d\n", (int)i, (int)j, (int)k);
        }
      }
    }
    puts("Misc:");
    float* misc = (float*)st[13].data();
    for (int i = 0; i < 7; i++) printf("%d ", (int)misc[i]);
    puts("");
    for (int i = 7; i < 15; i++) printf("%d ", (int)misc[i]);
    puts("");
    printf("%6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s\n",
        "s19", "s18", "sc", "tg", "rmul", "lev", "ln", "pcs", "sp29", "sp19", "sp18");
    for (int i = 15; i < 26; i++) printf("%6.3f ", misc[i]);
    puts("");
    constexpr int arr[] = {29,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50,55,60,65,70,75,80,85,90,95,10,11,12,13,14,15,16,17,18,19,20,21,22,23};
    for (int i = 0; i < 48; i++) printf("%2d", arr[i]);
    puts("");
    for (int i = 26; i < 74; i++) printf("%d ", (int)misc[i]);
    puts("");
    printf("%6s %6s %6s %6s %6s %6s %6s %6s %6s\n",
        "hzavg", "hzdev", "ftap", "das", "micdl", "misr", "mist", "misp", "prvmis");
    for (int i = 74; i < 83; i++) printf("%6.3f ", misc[i]);
    puts("");
    fflush(stdout);
  }
#endif
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
#define TETRIS_DEFINE_STATIC(x) decltype(Tetris::x) Tetris::x
TETRIS_DEFINE_STATIC(kTransitionProb_);
TETRIS_DEFINE_STATIC(kStartPosition_);
TETRIS_DEFINE_STATIC(kLinesBeforeLevelUp_);
TETRIS_DEFINE_STATIC(kFramesPerDrop_);
TETRIS_DEFINE_STATIC(kScoreBase_);
#undef TETRIS_DEFINE_STATIC

#ifndef DEBUG

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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

static PyObject* Tetris_ResetRandom(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"fix_prob", nullptr};
  double fix_prob = 0.9;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", (char**)kwlist, &fix_prob)) {
    return nullptr;
  }
  self->ResetRandom(fix_prob);
  Py_RETURN_NONE;
}

static PyObject* Tetris_IsOver(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong((long)self->IsOver());
}

static PyObject* Tetris_InputPlacement(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"rotate", "x", "y", "training", nullptr};
  int rotate, x, y, training = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii|p", (char**)kwlist, &rotate,
                                   &x, &y, &training)) {
    return nullptr;
  }
  double reward = self->InputPlacement({rotate, x, y}, training);
  return PyFloat_FromDouble(reward);
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
  return PyTuple_Pack(3, dim1, dim2, dim3);
}

static PyObject* Tetris_ResetGame(Tetris* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {
      "start_level", "hz_avg", "hz_dev", "das", "first_tap_max",
      "microadj_delay", "orig_misdrop_rate", "misdrop_param_time",
      "misdrop_param_pow", "target", "reward_multiplier", nullptr
  };
  int start_level = 18;
  double hz_avg = 10;
  double hz_dev = 0;
  int das = 1;
  double first_tap_max = 30;
  int microadj_delay = 40;
  double orig_misdrop_rate = 5e-3;
  double misdrop_param_time = 400;
  double misdrop_param_pow = 1.0;
  int target = 1000000;
  double reward_multiplier = 2e-6;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iddpdidddid", (char**)kwlist,
        &start_level, &hz_avg, &hz_dev, &das, &first_tap_max, &microadj_delay,
        &orig_misdrop_rate, &misdrop_param_time, &misdrop_param_pow, &target,
        &reward_multiplier)) {
    return nullptr;
  }
  self->ResetGame(start_level, hz_avg, hz_dev, das, first_tap_max,
      microadj_delay, orig_misdrop_rate, misdrop_param_time, misdrop_param_pow,
      target, reward_multiplier);
  Py_RETURN_NONE;
}

static PyObject* Tetris_GetScore(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->GetScore());
}

static PyObject* Tetris_GetLines(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->GetLines());
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

#endif

static PyMethodDef py_tetris_methods[] = {
    {"ResetRandom", (PyCFunction)Tetris_ResetRandom, METH_VARARGS | METH_KEYWORDS,
     "Reset a game using random parameters"},
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
#ifdef DEBUG_METHODS
    {"PrintState", (PyCFunction)Tetris_PrintState, METH_NOARGS,
     "Print state array"},
    {"PrintField", (PyCFunction)Tetris_PrintField, METH_NOARGS,
     "Print current field"},
    {"PrintAllState", (PyCFunction)Tetris_PrintAllState, METH_NOARGS,
     "Print all internal state"},
#endif
    {nullptr}};

static PyTypeObject py_tetris_class = {PyVarObject_HEAD_INIT(NULL, 0)};
static PyModuleDef py_tetris_module = {PyModuleDef_HEAD_INIT};

PyMODINIT_FUNC PyInit_tetris() {
  py_tetris_class.tp_name = "tetris.Tetris";
  py_tetris_class.tp_doc = "Tetris class";
  py_tetris_class.tp_basicsize = sizeof(Tetris);
  py_tetris_class.tp_itemsize = 0;
  py_tetris_class.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  py_tetris_class.tp_new = TetrisNew;
  py_tetris_class.tp_init = (initproc)TetrisInit;
  py_tetris_class.tp_dealloc = (destructor)TetrisDealloc;
  py_tetris_class.tp_methods = py_tetris_methods;

  py_tetris_module.m_name = "tetris";
  py_tetris_module.m_doc = "Tetris module";
  py_tetris_module.m_size = -1;

  import_array();

  if (PyType_Ready(&py_tetris_class) < 0) return nullptr;
  PyObject *m = PyModule_Create(&py_tetris_module);
  if (m == nullptr) return nullptr;
  Py_INCREF(&py_tetris_class);
  if (PyModule_AddObject(m, "Tetris", (PyObject*)&py_tetris_class) < 0) {
    Py_DECREF(&py_tetris_class);
    Py_DECREF(m);
    return nullptr;
  }
  return m;
}

#else

int main() {
  Tetris t;
  char buf[12];
  while (true) {
    printf("Input>");
    fflush(stdout);
    if (scanf("%s", buf) == -1) break;
    switch (buf[0]) {
      case 'p': t.PrintAllState(); break;
      case 'f': t.PrintState(true); break;
      case 's': t.PrintState(); break;
      case 'r': t.ResetRandom(0.5); break;
      case 'i': {
        int r, x, y;
        scanf("%d %d %d", &r, &x, &y);
        printf("Reward: %f\n", t.InputPlacement({r, x, y}));
        break;
      }
    }
  }
}

#endif
