printf("[ENC merge_idx] poc=%d addr=(%d,%d) mergeIdx=%d numCandminus1=%d regular=%d bcFlag=%d\n",
       cu.slice->getPOC(), cu.lx(), cu.ly(), (int)cu.mergeIdx, numCandminus1,
       (int)(!CU::isIBC(cu) && !cu.bmMergeFlag && !cu.oppositeLicFlag),
       (int)cu.bcFlag);



printf("[DEC merge_idx] poc=%d addr=(%d,%d) mergeIdx=%d numCandminus1=%d regular=%d\n",
       cu.slice->getPOC(), cu.lx(), cu.ly(), (int)cu.mergeIdx, numCandminus1,
       (int)(!CU::isIBC(cu) && !cu.bmMergeFlag && !cu.oppositeLicFlag));







static void appendBoundaryCompMergeCandidates(const CodingUnit& cu, MergeCtx& mrgCtx);




void PU::appendBoundaryCompMergeCandidates(const CodingUnit& cu, MergeCtx& mrgCtx)
{
  const uint32_t maxNumMergeCand    = cu.cs->sps->m_maxNumMergeCand;
  const uint32_t extraBCCand        = 5;
  const uint32_t maxNumMergeCandExt = maxNumMergeCand + extraBCCand;

  int arrayAddr = mrgCtx.numValidMergeCand;
  const int numOrigCand = arrayAddr;

  int numAddedBCCand = 0;
  for (int baseIdx = 0; baseIdx < numOrigCand; ++baseIdx)
  {
    if (numAddedBCCand >= (int)extraBCCand || arrayAddr >= (int)maxNumMergeCandExt)
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

  mrgCtx.numValidMergeCand = arrayAddr;
  mrgCtx.numCandToTestEnc  = arrayAddr;
}void PU::appendBoundaryCompMergeCandidates(const CodingUnit& cu, MergeCtx& mrgCtx)
{
  const uint32_t maxNumMergeCand    = cu.cs->sps->m_maxNumMergeCand;
  const uint32_t extraBCCand        = 5;
  const uint32_t maxNumMergeCandExt = maxNumMergeCand + extraBCCand;

  int arrayAddr = mrgCtx.numValidMergeCand;
  const int numOrigCand = arrayAddr;

  int numAddedBCCand = 0;
  for (int baseIdx = 0; baseIdx < numOrigCand; ++baseIdx)
  {
    if (numAddedBCCand >= (int)extraBCCand || arrayAddr >= (int)maxNumMergeCandExt)
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

  mrgCtx.numValidMergeCand = arrayAddr;
  mrgCtx.numCandToTestEnc  = arrayAddr;
}
