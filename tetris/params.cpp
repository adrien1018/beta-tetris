#include "params.h"

NormalizingParams GameParams::GetNormalizingParams() const {
  NormalizingParams ret;
  ret.start_level = start_level;
  if (start_level == 29) {
    if (hz_avg < 14) ret.hz_mode = 0;
    else if (hz_avg < 17) ret.hz_mode = 1;
    else if (hz_avg < 22) ret.hz_mode = 2;
    else ret.hz_mode = 3;
  } else {
    ret.hz_mode = 0;
  }
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
  if (start_lines > 200) {
    ret.start_line_mode = 2;
  } else if (start_lines > 100) {
    ret.start_line_mode = 1;
  } else {
    ret.start_line_mode = 0;
  }
  ret.drought_mode = drought_mode;
  return ret;
}
