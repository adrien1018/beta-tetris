#include "params.h"

NormalizingParams GameParams::GetNormalizingParams() const {
  NormalizingParams ret;
  if (hz_avg < 14) ret.hz_mode = 0;
  else if (hz_avg < 18) ret.hz_mode = 1;
  else if (hz_avg < 25) ret.hz_mode = 2;
  else ret.hz_mode = 3;
  ret.step_points = step_points;
  ret.drought_mode = drought_mode;
  return ret;
}
