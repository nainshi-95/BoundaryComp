void InterPrediction::motionCompensationLumaExt12Tap(
  CodingUnit &cu,
  PelBuf     &dstExtY,
  int         margin
)
{
  PROFILER_SCOPE(1, g_timeProfiler, P_MOTION_COMPENSATION);

  const ChromaFormat chFmt = cu.chromaFormat;

  const int cuW  = cu.lwidth();
  const int cuH  = cu.lheight();
  const int extW = cuW + 2 * margin;
  const int extH = cuH + 2 * margin;

  CHECK(dstExtY.width  != extW, "motionCompensationLumaExt12Tap: dstExtY width mismatch");
  CHECK(dstExtY.height != extH, "motionCompensationLumaExt12Tap: dstExtY height mismatch");

  CHECK(cu.affine,     "motionCompensationLumaExt12Tap does not support affine CU");
  CHECK(CU::isIBC(cu), "motionCompensationLumaExt12Tap does not support IBC CU");
  CHECK(cu.licFlag,   "motionCompensationLumaExt12Tap does not support LIC CU");

  // Use syntax-level inter direction.
  // 1: L0 only, 2: L1 only, 3: bi-pred.
  CHECK(cu.interDir != 1 && cu.interDir != 2,
        "motionCompensationLumaExt12Tap only supports uni-prediction CU");

  const RefPicList refList = cu.interDir == 1 ? RPL0 : RPL1;
  const int        refIdx  = cu.refIdx[refList];

  CHECK(refIdx < 0,
        "motionCompensationLumaExt12Tap: invalid refIdx for selected uni-prediction list");

  const Picture *refPic = cu.slice->getRefPic(refList, refIdx);
  CHECK(refPic == nullptr,
        "motionCompensationLumaExt12Tap: null reference picture");

  CHECK(refPic->isRefScaled(cu.cs->pps),
        "motionCompensationLumaExt12Tap currently does not support RPR/scaled reference");

  const ClpRng &clpRng = cu.slice->clpRng(COMP_Y);

  Mv mv = cu.mv[refList];

  const Position extCurPos = cu.lumaPos().offset(-margin, -margin);
  const Size     extCurSize(extW, extH);

  bool wrapRef = false;

  if (refPic->isWrapAroundEnabled(cu.cs->pps))
  {
    wrapRef = wrapClipMv(
      mv,
      extCurPos,
      extCurSize,
      cu.cs->sps,
      cu.cs->pps
    );
  }
  else
  {
    clipMv(
      mv,
      extCurPos,
      extCurSize,
      *cu.cs->sps,
      *cu.cs->pps
    );
  }

  const int shiftHor = MV_FRACTIONAL_BITS_INTERNAL;
  const int shiftVer = MV_FRACTIONAL_BITS_INTERNAL;

  const int xInt  = mv.getHor() >> shiftHor;
  const int yInt  = mv.getVer() >> shiftVer;
  const int xFrac = mv.getHor() & 15;
  const int yFrac = mv.getVer() & 15;

  const Position refPos = extCurPos.offset(xInt, yInt);

  CPelBuf refBuf = refPic->getRecoBuf(
    CompArea(COMP_Y, chFmt, refPos, Size(extW, extH)),
    wrapRef
  );

  const bool rndRes = true;

  const auto filterIdx =
    cu.imv == IMV_HPEL
      ? InterpolationFilter::Filter::HALFPEL_ALT
      : InterpolationFilter::Filter::DEFAULT;

  if (yFrac == 0)
  {
    m_if->filterHor(
      COMP_Y,
      refBuf.buf,
      refBuf.stride,
      dstExtY.buf,
      dstExtY.stride,
      extW,
      extH,
      xFrac,
      rndRes,
      clpRng,
      filterIdx
    );
  }
  else if (xFrac == 0)
  {
    m_if->filterVer(
      COMP_Y,
      refBuf.buf,
      refBuf.stride,
      dstExtY.buf,
      dstExtY.stride,
      extW,
      extH,
      yFrac,
      true,
      rndRes,
      clpRng,
      filterIdx
    );
  }
  else
  {
    const int filterSize   = NTAPS_LUMA;
    const int filterMargin = (filterSize >> 1) - 1;

    PelStorage tmpStorage;

    UnitArea tmpArea(
      chFmt,
      Area(0, 0, extW, extH + filterSize - 1)
    );

    tmpStorage.create(tmpArea);

    PelBuf tmpBuf = tmpStorage.getBuf(COMP_Y);

    m_if->filterHor(
      COMP_Y,
      refBuf.bufAt(0, -filterMargin),
      refBuf.stride,
      tmpBuf.buf,
      tmpBuf.stride,
      extW,
      extH + filterSize - 1,
      xFrac,
      false,
      clpRng,
      filterIdx
    );

    m_if->filterVer(
      COMP_Y,
      tmpBuf.bufAt(0, filterMargin),
      tmpBuf.stride,
      dstExtY.buf,
      dstExtY.stride,
      extW,
      extH,
      yFrac,
      false,
      rndRes,
      clpRng,
      filterIdx
    );

    tmpStorage.destroy();
  }
}
