void PU::appendBoundaryCompMergeCandidates(const CodingUnit& cu, MergeCtx& mrgCtx)
{
#if BOUNDARY_COMPENSATE
  const int orgNumMergeCand = (int)cu.cs->sps->m_maxNumMergeCand;
  const int extraBCCand     = 5;
  const int extNumMergeCand = orgNumMergeCand + extraBCCand;

  CHECK(extNumMergeCand > MRG_MAX_NUM_CANDS, "extended merge candidate count exceeds MRG_MAX_NUM_CANDS");

  int arrayAddr = mrgCtx.numValidMergeCand;
  const int numOrigCand = std::min(arrayAddr, orgNumMergeCand);

  int numAddedBCCand = 0;

  for (int baseIdx = 0; baseIdx < numOrigCand; baseIdx++)
  {
    if (numAddedBCCand >= extraBCCand || arrayAddr >= extNumMergeCand)
    {
      break;
    }

    if (!isBCApplicableCandidate(mrgCtx, baseIdx))
    {
      continue;
    }

    mrgCtx.interDirNeighbours[arrayAddr]   = mrgCtx.interDirNeighbours[baseIdx];
    mrgCtx.bcwIdx[arrayAddr]               = mrgCtx.bcwIdx[baseIdx];
    mrgCtx.LICFlags[arrayAddr]             = mrgCtx.LICFlags[baseIdx];
    mrgCtx.useAltHpelIf[arrayAddr]         = mrgCtx.useAltHpelIf[baseIdx];
    mrgCtx.mvFieldNeighbours[arrayAddr][0] = mrgCtx.mvFieldNeighbours[baseIdx][0];
    mrgCtx.mvFieldNeighbours[arrayAddr][1] = mrgCtx.mvFieldNeighbours[baseIdx][1];
    mrgCtx.BCCandFlags[arrayAddr]          = true;

    arrayAddr++;
    numAddedBCCand++;
  }

  // CABAC이 N+5까지 읽을 수 있으므로 실제 list도 항상 N+5까지 채움.
  // 부족한 자리는 non-BC zero-like 후보로 padding.
  while (arrayAddr < extNumMergeCand)
  {
    mrgCtx.interDirNeighbours[arrayAddr] = 1;
    mrgCtx.bcwIdx[arrayAddr]             = BCW_DEFAULT;
    mrgCtx.LICFlags[arrayAddr]           = false;
    mrgCtx.useAltHpelIf[arrayAddr]       = false;
    mrgCtx.BCCandFlags[arrayAddr]        = false;

    mrgCtx.mvFieldNeighbours[arrayAddr][RPL0].setMvField(Mv(0, 0), 0);
    mrgCtx.mvFieldNeighbours[arrayAddr][RPL1].setMvField(Mv(0, 0), NOT_VALID);

    if (cu.cs->slice->isInterB())
    {
      mrgCtx.interDirNeighbours[arrayAddr] = 3;
      mrgCtx.mvFieldNeighbours[arrayAddr][RPL1].setMvField(Mv(0, 0), 0);
    }

    arrayAddr++;
  }

  mrgCtx.numValidMergeCand = extNumMergeCand;
  mrgCtx.numCandToTestEnc  = extNumMergeCand;
#endif
}













static bool isBCApplicableCandidate(const MergeCtx& mrgCtx, int baseIdx)
{
  if (mrgCtx.BCCandFlags[baseIdx])
  {
    return false;
  }

  if (mrgCtx.LICFlags[baseIdx])
  {
    return false;
  }

  const uint8_t interDir = mrgCtx.interDirNeighbours[baseIdx];

  if (interDir != 1 && interDir != 2)
  {
    return false;
  }

  if (interDir == 1)
  {
    return mrgCtx.mvFieldNeighbours[baseIdx][RPL0].refIdx >= 0;
  }
  else
  {
    return mrgCtx.mvFieldNeighbours[baseIdx][RPL1].refIdx >= 0;
  }
}











PU::getInterMergeCandidates(*cu, mergeCtx, 0, -1, 1);

PU::getInterMMVDMergeCandidates(*cu, mergeCtx);

#if BOUNDARY_COMPENSATE
PU::appendBoundaryCompMergeCandidates(*cu, mergeCtx);
#endif

cu->regularMergeFlag = true;
cu->mergeFlag        = true;





























