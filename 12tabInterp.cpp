#include <vector>
#include <cstdint>
#include <algorithm>

static constexpr int INTERP12_FILTERS[16][12] =
{
  {  0,  0,   0,   0,   0, 256,   0,   0,   0,   0,  0,  0 },
  { -1,  2,  -3,   6, -14, 254,  16,  -7,   4,  -2,  1,  0 },
  { -1,  3,  -7,  12, -26, 249,  35, -15,   8,  -4,  2,  0 },
  { -2,  5,  -9,  17, -36, 241,  54, -22,  12,  -6,  3, -1 },
  { -2,  5, -11,  21, -43, 230,  75, -29,  15,  -8,  4, -1 },
  { -2,  6, -13,  24, -48, 216,  97, -36,  19, -10,  4, -1 },
  { -2,  7, -14,  25, -51, 200, 119, -42,  22, -12,  5, -1 },
  { -2,  7, -14,  26, -51, 181, 140, -46,  24, -13,  6, -2 },
  { -2,  6, -13,  25, -50, 162, 162, -50,  25, -13,  6, -2 },
  { -2,  6, -13,  24, -46, 140, 181, -51,  26, -14,  7, -2 },
  { -1,  5, -12,  22, -42, 119, 200, -51,  25, -14,  7, -2 },
  { -1,  4, -10,  19, -36,  97, 216, -48,  24, -13,  6, -2 },
  { -1,  4,  -8,  15, -29,  75, 230, -43,  21, -11,  5, -2 },
  { -1,  3,  -6,  12, -22,  54, 241, -36,  17,  -9,  5, -2 },
  {  0,  2,  -4,   8, -15,  35, 249, -26,  12,  -7,  3, -1 },
  {  0,  1,  -2,   4,  -7,  16, 254, -14,   6,  -3,  2, -1 }
};

static inline int floorDiv16(const int v)
{
  return (v >= 0) ? (v >> 4) : -(((-v) + 15) >> 4);
}

static inline int mod16Pos(const int v)
{
  const int q = floorDiv16(v);
  return v - (q << 4);
}

static inline int roundShiftSigned32(const int v, const int shift)
{
  const int offset = 1 << (shift - 1);
  return (v >= 0) ? ((v + offset) >> shift)
                  : -(((-v) + offset) >> shift);
}

void interp12PerPixelFromFlow_Int32(
    const CodingUnit&       cu,
    const std::vector<int>& flowInt16,   // h*w*2, xy order
    PelBuf&                 dstBuf)
{
  const int h = dstBuf.height;
  const int w = dstBuf.width;

  CHECK(int(flowInt16.size()) != h * w * 2, "flowInt16 size mismatch");

  const RefPicList refList = (cu.interDir == 1) ? RPL0 : RPL1;
  const int refIdx         = cu.refIdx[refList];
  const Picture* refPic    = cu.slice->getRefPic(refList, refIdx);

  CHECK(refPic == nullptr, "refPic is null");

  const ComponentID compID = COMPONENT_Y;
  const CompArea& cuArea   = cu.blocks[compID];

  // 네 branch에 맞게 필요하면 이 부분 수정
  const CPelBuf refBuf = refPic->getRecoBuf(compID);

  const int refW = refBuf.width;
  const int refH = refBuf.height;

  const ClpRng& clpRng = cu.slice->clpRng(compID);

  auto getRefPelClamped = [&](int py, int px) -> int
  {
    px = std::max(0, std::min(px, refW - 1));
    py = std::max(0, std::min(py, refH - 1));
    return int(refBuf.at(px, py));
  };

  constexpr int TAPS = 12;
  constexpr int OFF0 = -5;

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      const int flowIdx = (y * w + x) * 2;
      const int mvx16   = flowInt16[flowIdx + 0];
      const int mvy16   = flowInt16[flowIdx + 1];

      const int intDx = floorDiv16(mvx16);
      const int intDy = floorDiv16(mvy16);
      const int fracX = mod16Pos(mvx16);
      const int fracY = mod16Pos(mvy16);

      const int* fx = INTERP12_FILTERS[fracX];
      const int* fy = INTERP12_FILTERS[fracY];

      const int picX = cuArea.x + x;
      const int picY = cuArea.y + y;

      const int baseX = picX + intDx;
      const int baseY = picY + intDy;

      int tmp[12];

      for (int ry = 0; ry < TAPS; ++ry)
      {
        const int srcY = baseY + (OFF0 + ry);

        int sumH = 0;
        for (int rx = 0; rx < TAPS; ++rx)
        {
          const int srcX = baseX + (OFF0 + rx);
          const int p = getRefPelClamped(srcY, srcX);
          sumH += p * fx[rx];
        }

        // 1차 정규화: /256
        tmp[ry] = roundShiftSigned32(sumH, 8);
      }

      int sumV = 0;
      for (int ry = 0; ry < TAPS; ++ry)
      {
        sumV += tmp[ry] * fy[ry];
      }

      // 2차 정규화: /256
      int predVal = roundShiftSigned32(sumV, 8);

      predVal = ClipPel(predVal, clpRng);
      dstBuf.at(x, y) = Pel(predVal);
    }
  }
}
