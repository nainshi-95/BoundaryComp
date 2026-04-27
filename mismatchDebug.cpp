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
