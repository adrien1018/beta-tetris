#ifndef GAME_H_
#define GAME_H_

#include <array>

#ifndef NO_PYTHON
#include "python.h"
#endif

#ifdef DEBUG_METHODS
extern uint64_t clear_col_count[4][10];
#endif

#include "params.h"
#include "rng.h"

class Tetris {
 public:
#ifndef NO_PYTHON
  PyObject_HEAD
#endif

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
  std::mt19937_64 rng_;
  PieceRNG piece_rng_;
  using RealRand_ = std::uniform_real_distribution<double>;
  using NormalRand_ = std::normal_distribution<double>;

  // # Game constants
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
  static constexpr double kBottomMultiplier_ = 1.2;
  static constexpr double kTargetColumnMultiplier_ = 2.5;
  // Provide a reward gain for bottom row scoring to guide the agent to not
  //   score dirty tetrises.
  // This can be decreased during training.
  double penalty_multiplier_ = 1.0;
  // Multiplier of misdrop & infeasible penalty. Set to 0 in early training to
  //   avoid misguiding the agent.
  int step_points_ = 100;
  int target_column_ = -1;
  bool target_column_lock_ = false;
  int prev_target_column_change_ = 0;

  void SetTargetColumn_(bool force = false);
  void SpawnPiece_();
  static bool IsGround_(int piece, const Position& pos);
  static double GetDropTime_(int piece, const Position& pos, int frames_per_drop, bool clear);
  static double GetScore_(int lines, int level);
  static int GetFramesPerDrop_(int level);

  template <class T>
  using CMap_ = std::vector<std::array<std::array<T, kM + 2>, kN + 2>>;
  using Map_ = CMap_<uint8_t>;

  static Map_ GetMap_(const Field& field, int poly);
  static bool MapEmpty_(const Map_& mp);

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
  static Map_ Dijkstra_(const Map_& v, const Position& start);

  // Lowest possible move (constrained to a specific move sequence)
  static Map_ Dijkstra_(const Map_& v, const Position& start,
                        const std::vector<std::pair<int, MoveType>>& moves);

  static std::vector<std::pair<int, MoveType>> MovesFromMap_(
      const Map_& mp, const Position& pos);

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

  static MoveSequence GetMoveSequenceLb_(const Map_& mp_lb, const Position& end);
  static MoveSequence GetMoveSequence_(const Map_& mp, const Map_& mp_lb,
                                       const Position& start, const Position& end);

  static bool CheckMovePossible_(const Map_& mp, const Position& pos);

  void StoreMap_(bool use_temp);

  static std::vector<int> GetInputWindow_(
      const MoveSequence& seq, int start_row, int frames_per_drop);

  FrameSequence SequenceToFrame_(
      const MoveSequence& seq, double hz, int frames_per_drop,
      int start_row = 0, const FrameSequence& prev_input = FrameSequence{});

  Position Simulate_(const Map_& mp, const FrameSequence& seq, int frames_per_drop, bool finish = true);

  bool SequenceEquivalent_(const MoveSequence& seq1, const MoveSequence& seq2);
  void CheckLineLimit_();

  // needed: StoreMap_(true) (stored_mp_, stored_mp_lb_), temp_lines_
  double SetPlannedPlacement_(const Position& pos);

  // reward, raw score reward
  std::pair<double, double> InputPlacement_(const Position& pos);

  bool real_placement_set_;

  bool SetRealPlacement_(const Position& pos);

 public:
  Tetris(uint64_t seed = 0) : rng_(seed), piece_rng_(seed) {
    ResetGame(GameParams());
  }

  void Reseed(uint64_t seed = 0) {
    rng_.seed(seed);
    piece_rng_.Reset(seed);
  }

  void ResetGame(const GameParams& params);

  static int PlaceField(Field& field, int piece, const Position& pos);

  bool IsOver() const { return game_over_; }
  int GetScore() const { return score_; }
  int GetLines() const { return lines_ - start_lines_; }
  bool GetPlaceStage() const { return place_stage_; }
  std::pair<int, int> GetTetrisStat() const {
    return {tetris_count_, right_tetris_count_};
  }

  const int* GetNextPieceDistribution() const {
    const auto probs = drought_mode_ ?
        PieceRNG::kTransitionProbDrought_ : PieceRNG::kTransitionProb_;
    return probs[next_piece_];
  }

  State GetState() const;

  FrameSequence GetPlannedSequence(bool truncate = true) const;
  FrameSequence GetMicroadjSequence(bool truncate = true) const;
  std::pair<double, double> InputPlacement(const Position& pos, bool training = true);
  bool TrainingSetPlacement();

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
                bool prev_micro = false);

  // # Helpers
  using PlaceMap = std::vector<std::array<std::array<bool, kM>, kN>>;

  PlaceMap GetPlacements(const Field& field, int piece);
  static MoveSequence GetMoveSequence(
      const Field& field, int piece, const Position& start, const Position& end);

#ifdef DEBUG_METHODS
private:
  static void Print(const Position& pos);
  static void Print(const Field& field);
  static void Print(const Map_& mp);
  static void Print(const Move& mv);
  static void Print(const MoveSequence& seq);
  static void Print(const FrameSequence& seq);

public:
  void PrintAllState() const;
  void PrintState(bool field_only = false) const;
#endif
};

#endif // GAME_H_
