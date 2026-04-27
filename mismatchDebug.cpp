#if BOUNDARY_COMPENSATE
#define DBG_BC(cu) ((int)(cu).boundaryCompensation)
#else
#define DBG_BC(cu) 0
#endif

if (isEncoding() && cu.slice->getPOC() == 4)
{
  printf("[ENC_CU_MODE] POC=%d x=%d y=%d w=%d h=%d "
         "predMode=%d skip=%d merge=%d regular=%d mergeIdx=%d "
         "ciip=%d geo=%d affine=%d mmvd=%d mmvdSkip=%d "
         "ibc=%d bm=%d bmDir=%d oppLic=%d lic=%d bcw=%d imv=%d "
         "bc=%d\n",
         cu.slice->getPOC(),
         cu.lx(), cu.ly(), cu.lwidth(), cu.lheight(),
         (int)cu.predMode,
         (int)cu.skip,
         (int)cu.mergeFlag,
         (int)cu.regularMergeFlag,
         (int)cu.mergeIdx,
         (int)cu.ciipFlag,
         (int)cu.geoFlag,
         (int)cu.affine,
         (int)cu.mmvdMergeFlag,
         (int)cu.mmvdSkip,
         (int)CU::isIBC(cu),
         (int)cu.bmMergeFlag,
         (int)cu.bmDir,
         (int)cu.oppositeLicFlag,
         (int)cu.licFlag,
         (int)cu.bcwIdx,
         (int)cu.imv,
         DBG_BC(cu));
}
