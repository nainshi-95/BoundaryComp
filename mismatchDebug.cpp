// --- DEBUG START ---
if (isEncoding()
    && cu.slice->getPOC() == 4
    && cu.lx() == 0
    && cu.ly() == 72
    && cu.lwidth() == 4
    && cu.lheight() == 8
    && cu.regularMergeFlag)
{
  printf("[ENC REG_MERGE] POC=%d x=%d y=%d w=%d h=%d "
         "mergeIdx=%d bc=%d "
         "ciip=%d mmvd=%d geo=%d aff=%d ibc=%d bm=%d oppLic=%d\n",
         cu.slice->getPOC(),
         cu.lx(), cu.ly(), cu.lwidth(), cu.lheight(),
         (int)cu.mergeIdx,
         (int)cu.boundaryCompensation,
         (int)cu.ciipFlag,
         (int)cu.mmvdMergeFlag,
         (int)cu.geoFlag,
         (int)cu.affine,
         (int)CU::isIBC(cu),
         (int)cu.bmMergeFlag,
         (int)cu.oppositeLicFlag);
}
// --- DEBUG END ---

















#if BOUNDARY_COMPENSATE
  const int orgNumMergeCand = (int)cu->cs->sps->m_maxNumMergeCand;
  const int loopNumMergeCand = bcMergeCtx ? bcMergeCtx->numCandToTestEnc : mergeCtx.numCandToTestEnc;
#else
  const int loopNumMergeCand = mergeCtx.numCandToTestEnc;
#endif

  for (uint32_t uiMergeCand = 0; uiMergeCand < (uint32_t)loopNumMergeCand; uiMergeCand++)
  {
#if BOUNDARY_COMPENSATE
    const bool useBcCtx = bcMergeCtx && ((int)uiMergeCand >= orgNumMergeCand);
    const MergeCtx &curMergeCtx = useBcCtx ? *bcMergeCtx : mergeCtx;

    // Extra 영역은 반드시 BC candidate만 허용
    if (useBcCtx && !curMergeCtx.BCCandFlags[uiMergeCand])
    {
      continue;
    }

    // 원본 영역에서는 BC candidate가 들어오면 안 됨
    if (!useBcCtx && curMergeCtx.BCCandFlags[uiMergeCand])
    {
      continue;
    }
#else
    const MergeCtx &curMergeCtx = mergeCtx;
#endif

    if (cu->oppositeLicFlag && !curMergeCtx.LICFlags[uiMergeCand] && !cu->cs->sps->m_biLicEnabledFlag &&
        curMergeCtx.interDirNeighbours[uiMergeCand] == 3)
    {
      continue;
    }

    MergeItem *regularMerge = mergeItemList.allocateNewMergeItem();
    regularMerge->importMergeInfo(curMergeCtx, uiMergeCand, MergeItem::MergeItemType::REGULAR, *cu);

#if BOUNDARY_COMPENSATE
    regularMerge->boundaryCompensation = curMergeCtx.BCCandFlags[uiMergeCand];
#endif

    auto dstBuf = regularMerge->getPredBuf(localUnitArea);

    PelUnitBuf *noMvRefineBuf = nullptr;
    PelUnitBuf *noCiipBuf     = nullptr;

#if BOUNDARY_COMPENSATE
    if (!useBcCtx)
#endif
    {
      noMvRefineBuf = mrgPredBufNoMvRefine ? (*mrgPredBufNoMvRefine)[uiMergeCand] : nullptr;
      noCiipBuf     = mrgPredBufNoCiip ? (*mrgPredBufNoCiip)[uiMergeCand] : nullptr;
    }

    generateMergePrediction(localUnitArea, regularMerge, *cu, true, true, dstBuf, false, false,
                            noMvRefineBuf, noCiipBuf, &curMergeCtx);

#if BOUNDARY_COMPENSATE
    if (useBcCtx && regularMerge->boundaryCompensationMode == BoundaryCompensateFunctions::BC_WARP_NONE)
    {
      mergeItemList.discardMergeItem(regularMerge);
      continue;
    }
#endif

    regularMerge->cost = calcLumaCost4MergePrediction(ctxStart, dstBuf, sqrtLambdaForFirstPassIntra, *cu, distParam);
    mergeItemList.insertMergeItemToList(regularMerge);
  }







