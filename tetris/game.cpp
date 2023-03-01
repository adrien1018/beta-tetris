#include "game.h"

#include <queue>
#include <algorithm>

#ifdef DEBUG_METHODS
#include <cstdio>
#include <cstdint>
uint64_t clear_col_count[4][10] = {};
#endif

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
TETRIS_DEFINE_STATIC(kStartPosition_);
TETRIS_DEFINE_STATIC(kLinesBeforeLevelUp_);
TETRIS_DEFINE_STATIC(kFramesPerDrop_);
TETRIS_DEFINE_STATIC(kScoreBase_);
#undef TETRIS_DEFINE_STATIC
#endif

void Tetris::SetTargetColumn_(bool force) {
  if (!force && target_column_lock_) return;
  prev_target_column_change_ = lines_;
  target_column_ = RandTargetColumn(rng_);
}

void Tetris::SpawnPiece_() {
  now_piece_ = next_piece_;
  next_piece_ = piece_rng_.GeneratePiece(drought_mode_, next_piece_, pieces_);
}

bool Tetris::IsGround_(int piece, const Position& pos) {
  auto& pl = kBlocks_[piece][pos.rotate];
  for (auto& i : pl) {
    if (pos.x + i.first == kN - 1) return true;
  }
  return false;
}

double Tetris::GetDropTime_(int piece, const Position& pos, int frames_per_drop, bool clear) {
  return ((pos.x + 1) * frames_per_drop + kBaseDelay_ +
          (IsGround_(piece, pos) ? 0 : kNotGroundDelay_) +
          (clear ? kLineClearDelay_ : 0)) *
          kFrameLength_;
}

double Tetris::GetScore_(int lines, int level) {
  return kScoreBase_[lines] * (level + 1);
}

int Tetris::GetFramesPerDrop_(int level) {
  return level >= 29 ? 1 : kFramesPerDrop_[level];
}

Tetris::Map_ Tetris::GetMap_(const Field& field, int poly) {
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

bool Tetris::MapEmpty_(const Tetris::Map_& mp) {
  for (auto& i : mp) {
    for (auto& j : i) {
      for (auto& k : j) {
        if (k) return false;
      }
    }
  }
  return true;
}

// Highest possible move
Tetris::Map_ Tetris::Dijkstra_(const Tetris::Map_& v, const Position& start) {
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
Tetris::Map_ Tetris::Dijkstra_(
    const Map_& v, const Position& start,
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

std::vector<std::pair<int, Tetris::MoveType>> Tetris::MovesFromMap_(
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


// sequence

Tetris::MoveSequence Tetris::GetMoveSequenceLb_(const Map_& mp_lb, const Position& end) {
  if (!CheckMovePossible_(mp_lb, end)) return MoveSequence{};
  std::vector<std::pair<int, MoveType>> lb = MovesFromMap_(mp_lb, end);
  MoveSequence ret;
  for (size_t i = 0; i < lb.size(); i++) {
    ret.moves.push_back({lb[i].first, 0, lb[i].second});
  }
  ret.valid = true;
  return ret;
}

Tetris::MoveSequence Tetris::GetMoveSequence_(const Map_& mp, const Map_& mp_lb,
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

bool Tetris::CheckMovePossible_(const Map_& mp, const Position& pos) {
  return mp[pos.rotate][pos.x + 1][pos.y + 1] && !mp[pos.rotate][pos.x + 2][pos.y + 1];
}

void Tetris::StoreMap_(bool use_temp) {
  stored_mp_ = use_temp ? GetMap_(temp_field_, next_piece_)
                        : GetMap_(field_, now_piece_);
  stored_mp_lb_ = Dijkstra_(stored_mp_, kStartPosition_);
}

std::vector<int> Tetris::GetInputWindow_(const Tetris::MoveSequence& seq,
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

Tetris::FrameSequence Tetris::SequenceToFrame_(
    const Tetris::MoveSequence& seq, double hz, int frames_per_drop,
    int start_row, const Tetris::FrameSequence& prev_input) {
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

Tetris::Position Tetris::Simulate_(
    const Map_& mp, const Tetris::FrameSequence& seq,
    int frames_per_drop, bool finish) {
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

bool Tetris::SequenceEquivalent_(const Tetris::MoveSequence& seq1, const Tetris::MoveSequence& seq2) {
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

void Tetris::CheckLineLimit_() {
  if ((drought_mode_ || start_level_ == 29) && lines_ >= 230) game_over_ = true;
  if (hz_avg_ >= 15 && lines_ >= 330) game_over_ = true;
  if (lines_ >= 530) game_over_ = true;
}

// needed: StoreMap_(true) (stored_mp_, stored_mp_lb_), temp_lines_
double Tetris::SetPlannedPlacement_(const Position& pos) {
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
std::pair<double, double> Tetris::InputPlacement_(const Position& pos) {
  if (place_stage_) { // step 3
    if (!CheckMovePossible_(stored_mp_lb_, pos)) {
      if (++consecutive_invalid_ == 3) game_over_ = true;
      return {kInvalidReward_, 0};
    }
    prev_planned_placement_ = planned_placement_;
    double reward = SetPlannedPlacement_(pos) + step_points_ * (kRewardMultiplier_ * 0.5);
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
  if (real_lines == 4 && pos.y == target_column_) {
    score_reward *= kTargetColumnMultiplier_;
  }
  if (!target_column_lock_ &&
      lines_ + real_lines - prev_target_column_change_ >= 16 &&
      std::uniform_int_distribution<int>(0, 5)(rng_) == 0) {
    SetTargetColumn_();
  }
  if (real_lines >= 3 && pos.x >= 18) score_reward *= kBottomMultiplier_;
  reward += score_reward + step_points_ * (kRewardMultiplier_ * 0.5);
  real_placement_set_ = false;
  place_stage_ = true;
  consecutive_invalid_ = 0;
  StoreMap_(true);
  game_over_ = MapEmpty_(stored_mp_lb_);
  return {reward, orig_score_reward};
}

bool Tetris::SetRealPlacement_(const Tetris::Position& pos) {
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

void Tetris::ResetGame(const GameParams& params) {
  start_level_ = params.start_level;
  lines_ = params.start_lines;
  start_lines_ = params.start_lines;
  score_ = 0;
  next_piece_ = 0;
  pieces_ = params.start_lines * 10 / 4;
  game_over_ = false;
  SpawnPiece_();
  SpawnPiece_();
  consecutive_invalid_ = 0;
  for (auto& i : field_) std::fill(i.begin(), i.end(), false);
  drought_mode_ = params.drought_mode;
  step_points_ = params.step_points;
  penalty_multiplier_ = params.penalty_multiplier;
  hz_avg_ = params.hz_avg;
  hz_dev_ = params.hz_dev;
  microadj_delay_ = params.microadj_delay;
  prev_drop_time_ = 1800;
  prev_misdrop_ = false;
  prev_micro_ = false;
  place_stage_ = false;
  planned_seq_.valid = false;
  real_placement_set_ = false;
  tetris_count_ = 0;
  right_tetris_count_ = 0;
  target_column_ = params.target_column;
  target_column_lock_ = params.target_column != -2;
  SetTargetColumn_();
  StoreMap_(false);
}

int Tetris::PlaceField(Tetris::Field& field, int piece, const Tetris::Position& pos) {
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

Tetris::State Tetris::GetState() const {
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
  // 51-53: step_points_
  if (step_points_ >= 1000) {
    misc[51] = 1; // 3000
  } else if (step_points_ >= 100) {
    misc[52] = 1; // 200
  } else {
    misc[53] = 1; // 40
  }
  // 54: prev_misdrop
  misc[54] = !place_stage_ && prev_misdrop_;
  // 14 + 55 = 69 (channels)
  return ret;
}

Tetris::FrameSequence Tetris::GetPlannedSequence(bool truncate) const {
  FrameSequence fseq = game_over_ ? FrameSequence{} : planned_real_fseq_;
  if (truncate) fseq.seq.resize(microadj_delay_, FrameInput{});
  return fseq;
}

Tetris::FrameSequence Tetris::GetMicroadjSequence(bool truncate) const {
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

std::pair<double, double> Tetris::InputPlacement(const Position& pos, bool training) {
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

bool Tetris::TrainingSetPlacement() {
  return SetRealPlacement_(real_placement_);
}

// Set a particular board; the next step will be step 1.
// If failed (returned false), the object is left in an invalid state,
//   and requires another successful SetState or a game reset to recover.
bool Tetris::SetState(
    const Field& field, int now_piece, int next_piece,
    const Position& planned_pos, int lines, int score, int pieces,
    double prev_drop_time, bool prev_misdrop, bool prev_micro) {
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

Tetris::PlaceMap Tetris::GetPlacements(const Field& field, int piece) {
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

 Tetris::MoveSequence Tetris::GetMoveSequence(
     const Field& field, int piece, const Position& start, const Position& end) {
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
void Tetris::Print(const Position& pos) {
  printf("(%d,%d,%d)", pos.rotate, pos.x, pos.y);
}
void Tetris::Print(const Field& field) {
  for (auto& i : field) {
    for (auto& j : i) printf("%d ", (int)j);
    putchar('\n');
  }
}
void Tetris::Print(const Map_& mp) {
  for (size_t i = 0; i < kN; i++) {
    for (size_t j = 0; j < mp.size(); j++) {
      for (auto& k : mp[j][i+1]) printf("%d ", (int)k);
      putchar('|');
    }
    putchar('\n');
  }
}
void Tetris::Print(const Move& mv) {
  switch (mv.type) {
    case MoveType::kL: printf("L"); break;
    case MoveType::kR: printf("R"); break;
    case MoveType::kA: printf("A"); break;
    case MoveType::kB: printf("B"); break;
  }
  printf("%d-%d ", mv.height_start, mv.height_end);
}
void Tetris::Print(const Tetris::MoveSequence& seq) {
  if (!seq.valid) printf("(invalid)");
  for (auto& i : seq.moves) Print(i);
  putchar('\n');
}
void Tetris::Print(const Tetris::FrameSequence& seq) {
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

void Tetris::PrintAllState() const {
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
  printf("Step: %d, drought: %d\n", step_points_, (int)drought_mode_);
  puts("");
  fflush(stdout);
}

void Tetris::PrintState(bool field_only) const {
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
