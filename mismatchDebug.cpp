static void printCuModeDebug(const char* tag, const CodingUnit& cu, int targetPoc)
{
  if (cu.slice->m_poc != targetPoc)
  {
    return;
  }

  if (cu.geoFlag)
  {
    printf("[%s_CU_GEO] POC=%d x=%d y=%d w=%d h=%d "
           "splitDir=%d geoMrg0=%d geoMrg1=%d "
           "geoMMVD0=%d geoMMVD1=%d geoMMVDIdx0=%d geoMMVDIdx1=%d "
           "bldIdx=%d bcw=%d\n",
           tag, cu.slice->m_poc, cu.lx(), cu.ly(), cu.lwidth(), cu.lheight(),
           (int)cu.geoSplitDir,
           (int)cu.geoMergeIdx[0], (int)cu.geoMergeIdx[1],
           (int)cu.geoMMVDFlag[0], (int)cu.geoMMVDFlag[1],
           (int)cu.geoMMVDIdx[0], (int)cu.geoMMVDIdx[1],
           (int)cu.geoBldIdx,
           (int)cu.bcwIdx);
  }
  else if (cu.mmvdMergeFlag || cu.mmvdSkip)
  {
    printf("[%s_CU_MMVD] POC=%d x=%d y=%d w=%d h=%d "
           "mmvdVal=%d base=%d step=%d dir=%d mmvdSkip=%d bc=%d\n",
           tag, cu.slice->m_poc, cu.lx(), cu.ly(), cu.lwidth(), cu.lheight(),
           (int)cu.mmvdMergeIdx.val,
           (int)cu.mmvdMergeIdx.pos.baseIdx,
           (int)cu.mmvdMergeIdx.pos.step,
           (int)cu.mmvdMergeIdx.pos.dir,
           (int)cu.mmvdSkip,
           (int)cu.boundaryCompensation);
  }
  else if (cu.affine)
  {
    printf("[%s_CU_AFFINE] POC=%d x=%d y=%d w=%d h=%d "
           "mergeIdx=%d affineType=%d interDir=%d ref0=%d ref1=%d bc=%d\n",
           tag, cu.slice->m_poc, cu.lx(), cu.ly(), cu.lwidth(), cu.lheight(),
           (int)cu.mergeIdx,
           (int)cu.affineType,
           (int)cu.interDir,
           (int)cu.refIdx[0],
           (int)cu.refIdx[1],
           (int)cu.boundaryCompensation);
  }
  else if (cu.ciipFlag)
  {
    printf("[%s_CU_CIIP] POC=%d x=%d y=%d w=%d h=%d "
           "mergeIdx=%d ciipMode=%d bc=%d\n",
           tag, cu.slice->m_poc, cu.lx(), cu.ly(), cu.lwidth(), cu.lheight(),
           (int)cu.mergeIdx,
           (int)cu.ciipMode,
           (int)cu.boundaryCompensation);
  }
  else if (cu.regularMergeFlag)
  {
    printf("[%s_CU_REG] POC=%d x=%d y=%d w=%d h=%d "
           "mergeIdx=%d bc=%d lic=%d bm=%d oppLic=%d bcw=%d interDir=%d ref0=%d ref1=%d\n",
           tag, cu.slice->m_poc, cu.lx(), cu.ly(), cu.lwidth(), cu.lheight(),
           (int)cu.mergeIdx,
           (int)cu.boundaryCompensation,
           (int)cu.licFlag,
           (int)cu.bmMergeFlag,
           (int)cu.oppositeLicFlag,
           (int)cu.bcwIdx,
           (int)cu.interDir,
           (int)cu.refIdx[0],
           (int)cu.refIdx[1]);
  }
  else
  {
    printf("[%s_CU_OTHER] POC=%d x=%d y=%d w=%d h=%d "
           "skip=%d merge=%d predMode=%d ibc=%d intra=%d\n",
           tag, cu.slice->m_poc, cu.lx(), cu.ly(), cu.lwidth(), cu.lheight(),
           (int)cu.skip,
           (int)cu.mergeFlag,
           (int)cu.predMode,
           (int)CU::isIBC(cu),
           (int)CU::isIntra(cu));
  }
}
