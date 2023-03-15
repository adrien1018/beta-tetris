#include "params.h"

NormalizingParams GameParams::GetNormalizingParams() const {
  NormalizingParams ret;
  if (hz_avg < 14) ret.hz_mode = 0;
  else if (hz_avg < 18) ret.hz_mode = 1;
  else if (hz_avg < 25) ret.hz_mode = 2;
  else ret.hz_mode = 3;
  ret.step_points = step_points;
  if (start_level == 29 && ret.hz_mode == 0) {
    ret.target_column_mode = 0;
  } else if (target_column < 0) {
    ret.target_column_mode = target_column;
  } else if (target_column == 0 || target_column == 9) {
    ret.target_column_mode = 0;
  } else {
    ret.target_column_mode = 1;
  }
  ret.drought_mode = drought_mode;
  return ret;
}
