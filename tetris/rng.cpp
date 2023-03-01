#include "rng.h"

int PieceRNG::GeneratePiece(bool drought_mode, int prev_piece, int piece_num) {
  // int tot = (piece_num + offset_) % 8;
  const auto probs = drought_mode ? kTransitionProbDrought_ : kTransitionProb_;
  return std::discrete_distribution<int>(
      probs[prev_piece], probs[prev_piece] + kT)(gen_);
}

#ifndef _MSC_VER
#define RNG_DEFINE_STATIC(x) decltype(PieceRNG::x) PieceRNG::x
RNG_DEFINE_STATIC(kTransitionProb_);
RNG_DEFINE_STATIC(kTransitionProbDrought_);
#undef RNG_DEFINE_STATIC
#endif
