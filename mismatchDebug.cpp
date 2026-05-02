#include "CommonLib/Unit.h"
#include "CommonLib/MotionInfo.h"

static inline int floorMod16( int v )
{
  int r = v % 16;
  return r < 0 ? r + 16 : r;
}

static inline int getFrac16( int mvVal )
{
  return floorMod16( mvVal ); // result: 0 ~ 15
}






