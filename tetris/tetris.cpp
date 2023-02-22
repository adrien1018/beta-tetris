#include <cmath>
#include <cstdint>
#include <array>
#include <queue>
#include <random>
#include <vector>
#include <utility>
#include <stdexcept>
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

uint64_t clear_col_count[4][10] = {};
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
  struct FrameInput {
    bool l, r, a, b; // l+r=r / a+b=a here
  };
  struct FrameSequence {
    std::vector<FrameInput> seq;
  };
  struct Position {
    int rotate, x, y;
    bool operator==(const Position& pos) const {
      return rotate == pos.rotate && x == pos.x && y == pos.y;
    }
    bool operator!=(const Position& pos) const { return !(*this == pos); }
  };

  using State = std::array<std::array<std::array<float, kM>, kN>, 15>;

 private:
  // # RNG
  std::mt19937_64 rng_, piece_rng_;
  using RealRand_ = std::uniform_real_distribution<double>;
  using NormalRand_ = std::normal_distribution<double>;

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
  static constexpr int kTransitionProbDrought_[kT][kT] = {
  // T  J  Z  O  S  L  I (next)
    { 3,11,14,11,11,11, 3}, // T (current)
    {14, 3,11,11,11,11, 3}, // J
    {11,14, 3,11,11,11, 3}, // Z
    {11,11,11, 6,11,11, 3}, // O
    {11,11,11,11, 6,11, 3}, // S
    {14,11,11,11,11, 3, 3}, // L
    {10,10,10,10,12,10, 2}, // I
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

  // # Statistics
  int tetris_count_, right_tetris_count_;

  // # Parameters
  // Movement model
  double hz_avg_, hz_dev_;
  // The actual hz is sampled from NormalDistribution(hz_avg_, hz_dev_)
  static constexpr double kFirstTap_ = kFrameLength_ * 0.6; // in milliseconds

  // For simplicity, we assume a fixed microadjustment delay (the first
  //   microadjustment input would be done at exactly this time)
  int microadj_delay_; // in frames
  int start_lines_;
  bool drought_mode_; // whether it is in drought mode (TAOPOPYA APOPXPEY)

  // Reward model
  static constexpr double kRewardMultiplier_ = 1e-5; // 10 per maxout
  static constexpr double kInvalidReward_ = -0.3;
  // Provide a large reward deduction if the agent makes an invalid placement
  static constexpr double kInfeasibleReward_ = -0.01;
  // Provide a large reward deduction if the agent makes an infeasible placement
  // "Infeasible" placements are those cannot be done by +3σ tapping speeds
  //   (750-in-1 chance) and without misdrop
  static constexpr double kMisdropReward_ = -0.001;
  // Provide a small reward deduction each time the agent makes an misdrop;
  //   this can guide the agent to avoid high-risk movements
  static constexpr double kBottomGain_ = 0.1;
  // Provide a reward gain for bottom row scoring to guide the agent to not
  //   score dirty tetrises.
  // This can be decreased during training.
  double penalty_multiplier_ = 1.0;
  // Multiplier of misdrop & infeasible penalty. Set to 0 in early training to
  //   avoid misguiding the agent.
  double step_reward_ = 5e-4;
  int target_column_ = -1;
  bool target_column_lock_ = false;
  int prev_target_column_change_ = 0;

  // For random state balancing
  int current_state_steps_ = 1000;
  double steps_exp_avg_ = 40.;

  void SetTargetColumn_(bool force = false) {
    if (!force && target_column_lock_) return;
    prev_target_column_change_ = lines_;
    target_column_ = std::uniform_int_distribution<int>(-3, 14)(rng_);
    if (target_column_ > 9) target_column_ = 9;
    if (target_column_ < -1) target_column_ = -1;
  }

  void SpawnPiece_() {
    const auto probs = drought_mode_ ? kTransitionProbDrought_ :
        kTransitionProb_;
    int next = std::discrete_distribution<int>(
        probs[next_piece_], probs[next_piece_] + kT)(piece_rng_);
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

  double prev_drop_time_; // in milliseconds
  bool prev_misdrop_, prev_micro_;

  bool place_stage_;
  Position real_placement_; // S in step 1
  // Information about B (used as B' in step 1)
  Position planned_placement_; // also used to store A (used as state in step 2)
  Position prev_planned_placement_;
  MoveSequence planned_seq_;
  FrameSequence planned_fseq_, planned_real_fseq_;
  FrameSequence real_fseq_;
  double sampled_hz_;

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

  FrameSequence SequenceToFrame_(const MoveSequence& seq, double hz, int frames_per_drop,
      int start_row = 0, const FrameSequence& prev_input = FrameSequence{}) {
    const double start_delay = kFirstTap_;
    const double base_interval = 1000. / hz;
    double prev_lr = -1000, prev_ab = -1000;

    for (size_t i = 0; i < prev_input.seq.size(); i++) {
      auto& input = prev_input.seq[i];
      if (input.a || input.b) prev_ab = i * kFrameLength_;
      if (input.l || input.r) prev_lr = i * kFrameLength_;
    }
    double prev = std::max(start_delay, prev_input.seq.size() * kFrameLength_);
    const double n_start_delay = prev;

    std::vector<FrameInput> ret = prev_input.seq;
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
        if (!IsAB(move.type) && move.height_start == start_row) {
          lower = n_start_delay;
        } else if (i > 0 && !IsAB(move.type) && IsAB(seq.moves[i - 1].type)) {
          lower += kFrameLength_;
        }
        double prev_cat = IsAB(move.type) ? prev_ab : prev_lr;
        double frame_start = (move.height_start * frames_per_drop) * kFrameLength_;
        double interval = base_interval;
        double time = std::max(std::max(lower, frame_start), prev_cat + interval);
        (IsAB(move.type) ? prev_ab : prev_lr) = time;
        prev = time;
        if (func(i, move)) {
          Set(time / kFrameLength_, move.type);
        }
      }
    };
    RunLoop([&](size_t, const Move&) { return true; });
    return {ret};
  }

  Position Simulate_(const Map_& mp, const FrameSequence& seq, int frames_per_drop, bool finish = true) {
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

  void CheckLineLimit_() {
    if ((drought_mode_ || start_level_ == 29) && lines_ >= 230) game_over_ = true;
    if (hz_avg_ >= 15 && lines_ >= 330) game_over_ = true;
    if (lines_ >= 530) game_over_ = true;
  }

  // needed: StoreMap_(true) (stored_mp_, stored_mp_lb_), temp_lines_
  double SetPlannedPlacement_(const Position& pos) {
    MoveSequence seq = GetMoveSequence_(stored_mp_, stored_mp_lb_, kStartPosition_, pos);
    planned_placement_ = pos;
    planned_seq_ = seq;
    int frames_per_drop = GetFramesPerDrop_(GetLevel_(start_level_, temp_lines_));
    planned_fseq_ = SequenceToFrame_(seq, hz_avg_ + hz_dev_ * 3, frames_per_drop);
    double hz = NormalRand_(hz_avg_, hz_dev_)(rng_);
    if (hz < 8) hz = 8;
    if (hz > 30) hz = 30;
    sampled_hz_ = hz;
    double reward = 0;
    if (Simulate_(stored_mp_, planned_fseq_, frames_per_drop) != pos) {
      reward += kInfeasibleReward_ * penalty_multiplier_;
    }
    planned_real_fseq_ = SequenceToFrame_(seq, hz, frames_per_drop);
    return reward;
  }

  // reward, raw score reward
  std::pair<double, double> InputPlacement_(const Position& pos) {
    if (place_stage_) { // step 3
      if (!CheckMovePossible_(stored_mp_lb_, pos)) {
        if (++consecutive_invalid_ == 3) game_over_ = true;
        return {kInvalidReward_, 0};
      }
      prev_planned_placement_ = planned_placement_;
      double reward = SetPlannedPlacement_(pos) + step_reward_;
      place_stage_ = false;
      consecutive_invalid_ = 0;
      return {reward, 0};
    }
    // step 1
    if (!CheckMovePossible_(stored_mp_lb_, pos)) {
      if (++consecutive_invalid_ == 3) game_over_ = true;
      return {kInvalidReward_, 0};
    }
    double reward = 0;
    int level = GetLevel_(start_level_, lines_);
    int frames_per_drop = GetFramesPerDrop_(level);
    if (!planned_seq_.valid) { // special case: first piece
      // Assume one won't misdrop on the first piece
      real_placement_ = pos;
      prev_misdrop_ = false;
      prev_micro_ = false;
      MoveSequence cur_seq =
          GetMoveSequence_(stored_mp_, stored_mp_lb_, kStartPosition_, pos);
      real_fseq_ = SequenceToFrame_(cur_seq, hz_avg_, frames_per_drop);
    } else {
      if (!real_placement_set_) throw 1; // TODO
      FrameSequence fseq;
      double hz = sampled_hz_;

      bool flag = false;
      if (pos == planned_placement_) {
        MoveSequence cur_seq =
            GetMoveSequence_(stored_mp_, stored_mp_lb_, kStartPosition_, pos);
        if (SequenceEquivalent_(cur_seq, planned_seq_)) { // no micro
          fseq = SequenceToFrame_(cur_seq, hz, frames_per_drop);
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
            fseq = SequenceToFrame_(seq, hz_avg_ + 3 * hz_dev_, frames_per_drop,
                                    pos_before_adj.x, planned_fseq_);
            if (Simulate_(stored_mp_, fseq, frames_per_drop) != pos) {
              reward += kInfeasibleReward_ * penalty_multiplier_;
            }
          } else {
            reward += kInfeasibleReward_ * penalty_multiplier_;
          }
        }
        fseq = planned_real_fseq_;
        fseq.seq.resize(microadj_delay_);
        Position pos_before_adj =
            Simulate_(stored_mp_, fseq, frames_per_drop, false);
        MoveSequence seq =
            GetMoveSequence_(stored_mp_, Dijkstra_(stored_mp_, pos_before_adj),
                             pos_before_adj, pos);
        if (seq.valid) {
          fseq = SequenceToFrame_(seq, hz, frames_per_drop,
                                  pos_before_adj.x, fseq);
        }
      }
      real_placement_ = Simulate_(stored_mp_, fseq, frames_per_drop);
      prev_misdrop_ = real_placement_ != pos;
      if (prev_misdrop_) reward += kMisdropReward_ * penalty_multiplier_;
      prev_micro_ = !flag;
      real_fseq_ = std::move(fseq);
    }
    planned_placement_ = pos;
    temp_field_ = field_;
    int planned_lines = PlaceField(temp_field_, now_piece_, planned_placement_);
    temp_lines_ = lines_ + planned_lines;
    temp_score_ = score_ + GetScore_(planned_lines, level);
    Field field(field_);
    int real_lines = PlaceField(field, now_piece_, real_placement_);
    int score_delta = GetScore_(real_lines, level);
    prev_drop_time_ = GetDropTime_(now_piece_, real_placement_, frames_per_drop,
                                   real_lines > 0);
    pieces_++;
    double orig_score_reward = kRewardMultiplier_ * score_delta;
    double score_reward = orig_score_reward;
    if (real_lines >= 2 && pos.y == target_column_) {
      score_reward *= real_lines * 0.75;
    }
    if (!target_column_lock_ &&
        lines_ + real_lines - prev_target_column_change_ >= 16 &&
        std::uniform_int_distribution<int>(0, 5)(rng_) == 0) {
      SetTargetColumn_();
    }
    if (pos.x >= 18) score_reward *= 1 + kBottomGain_;
    reward += score_reward + step_reward_;
    real_placement_set_ = false;
    place_stage_ = true;
    consecutive_invalid_ = 0;
    StoreMap_(true);
    game_over_ = MapEmpty_(stored_mp_lb_);
    return {reward, orig_score_reward};
  }

  bool real_placement_set_;

  bool SetRealPlacement_(const Position& pos) {
    if (real_placement_set_) return false;
    prev_misdrop_ = pos != prev_planned_placement_;
    real_placement_ = pos;
    int real_lines = PlaceField(field_, now_piece_, real_placement_);
    int level = GetLevel_(start_level_, lines_);
#ifdef DEBUG_METHODS
    if (real_lines > 0) {
      clear_col_count[real_lines-1][pos.y]++;
    }
#endif
    if (real_lines == 4) {
      tetris_count_++;
      if (pos.y == 9) right_tetris_count_++;
    }
    lines_ += real_lines;
    score_ += GetScore_(real_lines, level);
    SpawnPiece_();
    StoreMap_(false);
    game_over_ = MapEmpty_(stored_mp_lb_);
    // prevent game from going indefinitely
    CheckLineLimit_();
    real_placement_set_ = true;
    return true;
  }

 public:
  Tetris(uint64_t seed = 0) : rng_(seed), piece_rng_(seed) {
    ResetGame(18);
  }

  void Reseed(uint64_t seed = 0) {
    rng_.seed(seed);
    piece_rng_.seed(seed);
  }

  void ResetGame(int start_level = 18, double hz_avg = 12, double hz_dev = 0,
                 int microadj_delay = 40, int start_lines = 0, bool drought_mode = false,
                 double step_points = 0, double penalty_multiplier = 1.0,
                 int target_column = -2) {
    start_level_ = start_level;
    lines_ = start_lines;
    start_lines_ = start_lines;
    score_ = 0;
    next_piece_ = 0;
    pieces_ = start_lines * 10 / 4;
    game_over_ = false;
    SpawnPiece_();
    SpawnPiece_();
    consecutive_invalid_ = 0;
    for (auto& i : field_) std::fill(i.begin(), i.end(), false);
    drought_mode_ = drought_mode;
    step_reward_ = step_points * kRewardMultiplier_ * 0.5;
    penalty_multiplier_ = penalty_multiplier;
    hz_avg_ = hz_avg;
    hz_dev_ = hz_dev;
    microadj_delay_ = microadj_delay;
    prev_drop_time_ = 1800;
    prev_misdrop_ = false;
    prev_micro_ = false;
    place_stage_ = false;
    planned_seq_.valid = false;
    real_placement_set_ = false;
    tetris_count_ = 0;
    right_tetris_count_ = 0;
    target_column_ = target_column_;
    target_column_lock_ = target_column_ != -2;
    SetTargetColumn_();
    StoreMap_(false);
  }

  void ResetRandom(float pre_trans, double penalty_multiplier, double reward_ratio) {
    int steps = pieces_ - start_lines_ * 10 / 4;
    current_state_steps_ += steps;
    steps_exp_avg_ = steps * (1./128) + steps_exp_avg_ * (1 - 1./128);
    if (current_state_steps_ < 1.25 * steps_exp_avg_) {
      // don't change mode if steps not enough to ensure correct data distribution
      ResetGame(start_level_, hz_avg_, hz_dev_, microadj_delay_, start_lines_,
                drought_mode_, int(step_reward_ / (kRewardMultiplier_ * 0.5) + 5) / 10 * 10,
                penalty_multiplier, target_column_lock_ ? target_column_ : -2);
      return;
    }
    current_state_steps_ = 0;
    using IntRand = std::uniform_int_distribution<int>;
    using RealRand = std::uniform_real_distribution<float>;
    constexpr double hz_table[] = {12, 13.5, 15, 20, 30};
    constexpr double start_level_table[] = {18, 19, 29};
    constexpr int adj_delay_table[] = {8, 16, 21, 25, 61};
    constexpr double step_points_table[] = {40, 200, 3000};
    int hz_ind = IntRand(0, 4)(rng_);
    int start_level = start_level_table[IntRand(0, 2)(rng_)];
    double hz_avg = hz_table[hz_ind];
    double hz_dev = hz_avg >= 29 || IntRand(0, 2)(rng_) ? 0 : int(hz_avg) / 6 * 0.5;
    int microadj_delay = adj_delay_table[IntRand(0, 4)(rng_)];
    double step_points = step_points_table[std::discrete_distribution<int>({1, reward_ratio, std::pow(reward_ratio, 1.5)})(rng_)];
    int start_lines = 0;
    bool drought_mode = IntRand(0, 2)(rng_) == 0;
    int target_column = (SetTargetColumn_(true), target_column_);
    if (IntRand(0, 2)(rng_) == 0) target_column = -2;
    if (IntRand(0, 1)(rng_)) {
      hz_avg = IntRand(0, 1)(rng_) ? 12 : 20;
      microadj_delay = 21;
    }
    if (RealRand(0, 1)(rng_) < pre_trans) {
      // pre-transition training
      int rnd = IntRand(0, 19)(rng_), rnd2 = IntRand(0, 3)(rng_);
      if (start_level == 18) {
        if (rnd >= 13) {
          start_lines = 205 + rnd2; // 230-25
        } else {
          start_lines = 105 + rnd2; // 130-25
        }
      } else if (start_level == 19) {
        start_lines = 205 + rnd2;
      }
    }
    /*{ // killscreen only
      hz_avg = IntRand(0,1)(rng_) ? 12 : 13.5;
      if (start_level != 29) {
        start_lines = 210 + IntRand(0, 5)(rng_);
      }
    }*/
    ResetGame(start_level, hz_avg, hz_dev, microadj_delay, start_lines,
              drought_mode, step_points, penalty_multiplier, target_column);
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
  int GetLines() const { return lines_ - start_lines_; }
  bool GetPlaceStage() const { return place_stage_; }
  std::pair<int, int> GetTetrisStat() const {
    return {tetris_count_, right_tetris_count_};
  }

  const int* GetNextPieceDistribution() const {
    const auto probs = drought_mode_ ? kTransitionProbDrought_ :
        kTransitionProb_;
    return probs[next_piece_];
  }

  State GetState() const {
    State ret{};
    // 0: field
    // 1-4: planned_placement_ (place_stage_ = true: step 1's input)
    // 5-8: planned_placement_ (place_stage_ = false)
    // 9-12: possible placements
    // 13: column
    // 14: misc
    auto& planned_placement = planned_placement_;
    if (place_stage_) {
      for (int i = 0; i < kN; i++) {
        for (int j = 0; j < kM; j++) ret[0][i][j] = 1 - temp_field_[i][j];
      }
      ret[1 + planned_placement.rotate][planned_placement.x]
         [planned_placement.y] = 1;
    } else {
      for (int i = 0; i < kN; i++) {
        for (int j = 0; j < kM; j++) ret[0][i][j] = 1 - field_[i][j];
      }
      if (planned_seq_.valid) {
        ret[5 + planned_placement.rotate][planned_placement.x]
          [planned_placement.y] = 1;
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
    if (target_column_ >= 0) {
      for (int i = 0; i < kN; i++) ret[13][i][target_column_] = 1;
    }
    int cur_piece = place_stage_ ? next_piece_ : now_piece_;
    // misc
    float* misc = (float*)ret[14].data();
    // 0-6: current
    misc[0 + cur_piece] = 1;
    // 7-14: next / 7(if place_stage_)
    misc[place_stage_ ? 14 : 7 + next_piece_] = 1;
    // 15-17: start (18/19/29)
    misc[start_level_ != 29 ? start_level_ == 18 ? 15 : 16 : 17] = 1;
    // 18: level
    int lines = place_stage_ ? temp_lines_ : lines_;
    int level = GetLevel_(start_level_, lines);
    int pieces = pieces_;
    if (hz_avg_ < 15 && lines > 280) {
      int reduce = (lines - 280) / 2 * 2;
      pieces -= reduce * 10 / 4;
      lines -= reduce;
    }
    misc[18] = level * 1e-1 - 1;
    // 19: lines
    misc[19] = lines * 1e-2 - 1;
    // 20: pieces
    misc[20] = pieces * 4e-3 - 1;
    // 21-23: speed (29/19/18)
    misc[20 + (GetFramesPerDrop_(level) - 1)] = 1;
    // 24: drought
    misc[24] = drought_mode_;
    // 25: board
    misc[25] = 1;
    // 26-39: lines to next speed / final
    //   26-35: 1-10
    //   36-38: 11-19 (3)
    //   39: 20~ (5)
    int lines_to_next = level >= 29 ? 1000 - lines :
        (level == 18 ? 130 - lines : 230 - lines);
    if (lines_to_next <= 10) {
      misc[26 + (lines_to_next - 1)] = 1;
    } else if (lines_to_next <= 19) {
      misc[36 + (lines_to_next - 11) / 3] = 1;
    } else {
      misc[39] = 1;
    }
    // 40-44: hz_avg
    if (hz_avg_ >= 25) {
      misc[40] = 1; // 30
    } else if (hz_avg_ >= 18) {
      misc[41] = 1; // 20
    } else if (hz_avg_ >= 14) {
      misc[42] = 1; // 15
    } else if (hz_avg_ >= 13) {
      misc[43] = 1; // 13.5
    } else {
      misc[44] = 1; // 12
    }
    // 45: hz_dev
    misc[45] = hz_dev_;
    // 46-50: microadj_delay
    if (microadj_delay_ >= 50) {
      misc[46] = 1; // 61
    } else if (microadj_delay_ >= 23) {
      misc[47] = 1; // 25
    } else if (microadj_delay_ >= 20) {
      misc[48] = 1; // 21
    } else if (microadj_delay_ >= 14) {
      misc[49] = 1; // 16
    } else {
      misc[50] = 1; // 8
    }
    // 51-53: step_reward_
    if (step_reward_ >= 1000 * kRewardMultiplier_ * 0.5) {
      misc[51] = 1; // 3000
    } else if (step_reward_ >= 100 * kRewardMultiplier_ * 0.5) {
      misc[52] = 1; // 200
    } else {
      misc[53] = 1; // 40
    }
    // 54: prev_misdrop
    misc[54] = !place_stage_ && prev_misdrop_;
    // 14 + 55 = 69 (channels)
    return ret;
  }

  FrameSequence GetPlannedSequence(bool truncate = true) const {
    FrameSequence fseq = game_over_ ? FrameSequence{} : planned_real_fseq_;
    if (truncate) fseq.seq.resize(microadj_delay_, FrameInput{});
    return fseq;
  }

  FrameSequence GetMicroadjSequence(bool truncate = true) const {
    FrameSequence fseq = game_over_ ? FrameSequence{} : real_fseq_;
    if (truncate && pieces_ > 1) {
      if (fseq.seq.size() <= (size_t)microadj_delay_) {
        fseq.seq.clear();
      } else {
        fseq.seq.erase(fseq.seq.begin(), fseq.seq.begin() + microadj_delay_);
      }
    }
    return fseq;
  }

  std::pair<double, double> InputPlacement(const Position& pos, bool training = true) {
    if (game_over_) return {0, 0};
    bool orig_stage = place_stage_;
    auto& npos = pos;
    if (npos.rotate >= (int)stored_mp_lb_.size() || npos.x >= kN || npos.y >= kM) {
      return {kInvalidReward_, 0};
    }
    auto ret = InputPlacement_(npos);
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
  //   5. TrainingSetPlacement (auto-called by InputPlacement)

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

  // Set a particular board; the next step will be step 1.
  // If failed (returned false), the object is left in an invalid state,
  //   and requires another successful SetState or a game reset to recover.
  bool SetState(const Field& field, int now_piece, int next_piece,
                const Position& planned_pos, int lines, int score, int pieces,
                double prev_drop_time = 1000, bool prev_misdrop = false,
                bool prev_micro = false) {
    temp_field_ = field;
    temp_lines_ = lines;
    next_piece_ = now_piece;
    StoreMap_(true);
    if (!CheckMovePossible_(stored_mp_lb_, planned_pos)) return false;
    SetPlannedPlacement_(planned_pos);
    lines_ = lines;
    score_ = score;
    pieces_ = pieces;
    now_piece_ = now_piece;
    next_piece_ = next_piece;
    consecutive_invalid_ = 0;
    field_ = field;
    prev_drop_time_ = prev_drop_time;
    prev_misdrop_ = prev_misdrop;
    prev_micro_ = prev_micro;
    place_stage_ = false;
    StoreMap_(false);
    game_over_ = MapEmpty_(stored_mp_lb_);
    CheckLineLimit_();
    // for stage setting only, though real_placement_ is not set (not used)
    real_placement_set_ = true;
    return true;
  }

  // # Helpers
  using PlaceMap = std::vector<std::array<std::array<bool, kM>, kN>>;

  PlaceMap GetPlacements(const Field& field, int piece) {
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
  static void Print(const FrameSequence& seq) {
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
    printf("Prev drop time: %f ms, prev misdrop: %d, prev micro: %d\n",
           prev_drop_time_, (int)prev_misdrop_, (int)prev_micro_);
    printf("Real placement: "); Print(real_placement_);
    printf("\nPlanned placement: "); Print(planned_placement_);
    printf("\nPlanned sequence: "); Print(planned_seq_);
    printf("Planned frame sequence: "); Print(planned_fseq_);
    puts("Temp:");
    printf("Lines: %d, score: %d, field:\n", temp_lines_, temp_score_);
    Print(temp_field_);
    puts("Stored map lb:");
    Print(stored_mp_lb_);
    printf("Consecutive invalid: %d\n", consecutive_invalid_);
    puts("Parameters:");
    printf("Hz: avg %f dev %f, micro delay: %d\n", hz_avg_, hz_dev_, microadj_delay_);
    printf("Step: %f, drought: %d\n", step_reward_, (int)drought_mode_);
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
    float* misc = (float*)st[14].data();
    for (int i = 0; i < 7; i++) printf("%d ", (int)misc[i]);
    puts("");
    for (int i = 7; i < 15; i++) printf("%d ", (int)misc[i]);
    puts("");
    printf("%6s %6s %6s %6s %6s %6s %6s %6s %6s %6s\n",
        "s18", "s19", "s29", "lev", "ln", "pcs", "sp29", "sp19", "sp18", "dro");
    for (int i = 15; i < 25; i++) printf("%6.3f ", misc[i]);
    puts("");
    constexpr int arr[] = {1,2,3,4,5,6,7,8,9,10,13,16,19,99};
    for (int i = 0; i < 14; i++) printf("%2d", arr[i]);
    puts("");
    for (int i = 26; i < 40; i++) printf("%d ", (int)misc[i]);
    printf("\nhzavg 30 20 15 13 12\n     ");
    for (int i = 40; i < 45; i++) printf("  %d", (int)misc[i]);
    printf("\nhzdev %.2f\nmicdl 61 25 20 16 8\n     ", misc[45]);
    for (int i = 46; i < 51; i++) printf("  %d", (int)misc[i]);
    printf("\npenal  0  1 10\n     ");
    for (int i = 51; i < 54; i++) printf("  %d", (int)misc[i]);
    printf("\nprvmis %.1f\n", misc[54]);
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
#ifndef _MSC_VER
#define TETRIS_DEFINE_STATIC(x) decltype(Tetris::x) Tetris::x
TETRIS_DEFINE_STATIC(kTransitionProb_);
TETRIS_DEFINE_STATIC(kTransitionProbDrought_);
// TETRIS_DEFINE_STATIC(kStartPosition_);
TETRIS_DEFINE_STATIC(kLinesBeforeLevelUp_);
TETRIS_DEFINE_STATIC(kFramesPerDrop_);
TETRIS_DEFINE_STATIC(kScoreBase_);
#undef TETRIS_DEFINE_STATIC
#endif

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
  static const char* kwlist[] = {"pre_trans", "penalty_multiplier", "reward_ratio", nullptr};
  double pre_trans = 1.0, penalty_multiplier = 0.0, reward_ratio = 0.2;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddd", (char**)kwlist,
                                   &pre_trans, &penalty_multiplier, &reward_ratio)) {
    return nullptr;
  }
  self->ResetRandom(pre_trans, penalty_multiplier, reward_ratio);
  Py_RETURN_NONE;
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
  int start_level = 18;
  double hz_avg = 12;
  double hz_dev = 0;
  int microadj_delay = 40;
  int start_lines = 0;
  int drought_mode = 0;
  double step_points = 0;
  double penalty_multiplier = 1.0;
  int target_column = -1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iddiipddi", (char**)kwlist,
        &start_level, &hz_avg, &hz_dev, &microadj_delay, &start_lines,
        &drought_mode, &step_points, &penalty_multiplier, &target_column)) {
    return nullptr;
  }
  self->ResetGame(start_level, hz_avg, hz_dev, microadj_delay, start_lines,
      drought_mode, step_points, penalty_multiplier);
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

static PyObject* Tetris_GetClearCol(Tetris* self, PyObject* Py_UNUSED(ignored)) {
  npy_intp dims[] = {4, Tetris::kM};
  PyObject* ret = PyArray_SimpleNew(2, dims, NPY_UINT64);
  memcpy(PyArray_DATA((PyArrayObject*)ret), clear_col_count, sizeof(clear_col_count));
  return ret;
}

#endif

static PyMethodDef py_tetris_methods[] = {
    {"ResetRandom", (PyCFunction)Tetris_ResetRandom,
     METH_VARARGS | METH_KEYWORDS, "Reset a game using random parameters"},
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
    {"GetClearCol", (PyCFunction)Tetris_GetClearCol, METH_NOARGS,
     "Print line clear column stats"},
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
      case 'r': t.ResetRandom(1, 0.3, 0.0); break;
      case 'i': {
        int r, x, y;
        scanf("%d %d %d", &r, &x, &y);
        double rr = t.InputPlacement({r, x, y}).first;
        printf("Reward: %f\n", rr);
        break;
      }
    }
  }
  /*
  int field[][10] = {
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {0,0,1,0,0,0,0,0,0,0},
    {0,1,1,1,0,0,0,1,0,0},
    {1,1,1,1,1,0,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,0},
    {1,1,1,1,1,1,1,1,1,0},
    {1,1,1,1,1,1,1,1,1,0},
    {1,1,1,1,1,1,1,1,0,1},
  };
  Tetris::Field r_field;
  for (int i = 0; i < 20; i++) for (int j = 0; j < 10; j++) r_field[i][j] = field[i][j];
  t.ResetGame(18, 12, 0, 21, 0, true, 0, 0, 1);
  t.SetState(r_field, 3, 5, {0, 12, 7}, 139, 401880, 362);
  t.InputPlacement({0, 13, 9});
  t.PrintAllState();
  t.InputPlacement({0, 12, 8});
  t.PrintAllState();
  t.InputPlacement({1, 12, 9});
  t.PrintAllState();
  t.InputPlacement({1, 14, 5});
  t.PrintAllState();
  */
}

#endif
