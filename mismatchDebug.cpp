printf("[ENC merge_idx] poc=%d addr=(%d,%d) mergeIdx=%d numCandminus1=%d regular=%d bcFlag=%d\n",
       cu.slice->getPOC(), cu.lx(), cu.ly(), (int)cu.mergeIdx, numCandminus1,
       (int)(!CU::isIBC(cu) && !cu.bmMergeFlag && !cu.oppositeLicFlag),
       (int)cu.bcFlag);



printf("[DEC merge_idx] poc=%d addr=(%d,%d) mergeIdx=%d numCandminus1=%d regular=%d\n",
       cu.slice->getPOC(), cu.lx(), cu.ly(), (int)cu.mergeIdx, numCandminus1,
       (int)(!CU::isIBC(cu) && !cu.bmMergeFlag && !cu.oppositeLicFlag));
