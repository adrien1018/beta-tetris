#ifndef RNG_H_
#define RNG_H_

#include <cstdint>
#include <random>

class PieceRNG {
  static constexpr int kT = 7;
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
  std::mt19937_64 gen_;
  int offset_;

 public:
  PieceRNG(uint64_t seed = 0) { Reset(seed); }
  void Reset(uint64_t seed) {
    gen_.seed(seed);
    offset_ = std::uniform_int_distribution<int>(0, 7)(gen_);
  }
  int GeneratePiece(bool drought_mode, int prev_piece, int piece_num);

  friend class Tetris;
};

#endif // RNG_H_
