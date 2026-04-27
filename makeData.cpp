struct ReconExtPatch
{
  int x0 = 0;       // patch의 picture 좌표 기준 시작 x
  int y0 = 0;       // patch의 picture 좌표 기준 시작 y
  int width = 0;    // w + 24
  int height = 0;   // h + 24

  std::vector<uint8_t>  mask;   // size = height * width, available이면 1, 아니면 0
  std::vector<uint16_t> value;  // size = height * width, 10-bit recon 값, unavailable이면 0

  inline int idx(const int y, const int x) const
  {
    return y * width + x;
  }
};






static inline bool isInsideCurrentCuLuma(const CodingUnit& cu, const int picX, const int picY)
{
  const Position pos = cu.Y().pos();
  const Size     size = cu.Y().size();

  return picX >= pos.x &&
         picX <  pos.x + size.width &&
         picY >= pos.y &&
         picY <  pos.y + size.height;
}








static inline bool isReconAvailableAtLumaPos(const CodingUnit& cu, const int picX, const int picY)
{
  const CodingStructure& cs = *cu.cs;
  const ChannelType chType = CHANNEL_TYPE_LUMA;

  // 1) picture boundary 밖이면 unavailable
  const int picW = cs.picture->lwidth();
  const int picH = cs.picture->lheight();

  if (picX < 0 || picX >= picW || picY < 0 || picY >= picH)
  {
    return false;
  }

  // 2) 현재 CU 내부는 아직 현재 CU 자신이므로 unavailable 처리
  if (isInsideCurrentCuLuma(cu, picX, picY))
  {
    return false;
  }

  // 3) 해당 위치에 이미 decode된 CU가 존재하는지 확인
  const CodingUnit* cuAtPos = cs.getCU(Position(picX, picY), chType);

  if (cuAtPos == nullptr)
  {
    return false;
  }

  // 4) 혹시 현재 CU 자신이면 unavailable
  if (cuAtPos == &cu)
  {
    return false;
  }

  return true;
}









static ReconExtPatch makeReconExtPatchLuma10bit(const CodingUnit& cu, const int margin = 12)
{
  const CodingStructure& cs = *cu.cs;

  const CompArea& yArea = cu.Y();

  const int cuX = yArea.x;
  const int cuY = yArea.y;
  const int cuW = yArea.width;
  const int cuH = yArea.height;

  ReconExtPatch out;

  out.x0 = cuX - margin;
  out.y0 = cuY - margin;
  out.width  = cuW + 2 * margin;
  out.height = cuH + 2 * margin;

  const int numSamples = out.width * out.height;

  out.mask.assign(numSamples, 0);
  out.value.assign(numSamples, 0);

  // picture 전체 recon buffer
  const CPelBuf recoBuf = cs.picture->getRecoBuf(COMPONENT_Y);

  // bitdepth가 10-bit라면 max=1023
  // 혹시 내부 Pel 값이 범위를 벗어나는 경우 방어적으로 clip
  const int bitDepth = cs.sps->getBitDepth(CHANNEL_TYPE_LUMA);
  const int maxVal   = (1 << bitDepth) - 1;

  for (int py = 0; py < out.height; ++py)
  {
    const int picY = out.y0 + py;

    for (int px = 0; px < out.width; ++px)
    {
      const int picX = out.x0 + px;
      const int idx  = out.idx(py, px);

      if (!isReconAvailableAtLumaPos(cu, picX, picY))
      {
        out.mask[idx]  = 0;
        out.value[idx] = 0;
        continue;
      }

      const Pel p = recoBuf.at(picX, picY);

      const int clipped = std::min(std::max<int>(p, 0), maxVal);

      out.mask[idx]  = 1;
      out.value[idx] = static_cast<uint16_t>(clipped);
    }
  }

  return out;
}













ReconExtPatch patch = makeReconExtPatchLuma10bit(cu, 12);

const int extH = patch.height; // cu.Y().height + 24
const int extW = patch.width;  // cu.Y().width  + 24

// contiguous 접근
for (int y = 0; y < extH; ++y)
{
  for (int x = 0; x < extW; ++x)
  {
    const int idx = y * extW + x;

    uint8_t  available = patch.mask[idx];
    uint16_t reconVal  = patch.value[idx];

    // available == 1이면 reconVal 사용
    // available == 0이면 reconVal은 0
  }
}








static std::vector<char> serializePatchValueU16LE(const ReconExtPatch& patch)
{
  const int numSamples = patch.width * patch.height;

  std::vector<char> bytes(numSamples * 2);

  for (int i = 0; i < numSamples; ++i)
  {
    const uint16_t v = patch.value[i];

    // little-endian 저장
    bytes[2 * i + 0] = static_cast<char>( v        & 0xFF);
    bytes[2 * i + 1] = static_cast<char>((v >> 8)  & 0xFF);
  }

  return bytes;
}












