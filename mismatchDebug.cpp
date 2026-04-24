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














import numpy as np

W, H = 1920, 1080      # 시퀀스 해상도
bitdepth = 10          # 8 or 10
frame_count = 100
enc_path = "enc_recon.yuv"
dec_path = "dec_recon.yuv"

dtype = np.uint16 if bitdepth > 8 else np.uint8
bytes_per_sample = 2 if bitdepth > 8 else 1

y_size = W * H
uv_size = (W // 2) * (H // 2)
frame_samples = y_size + 2 * uv_size
frame_bytes = frame_samples * bytes_per_sample

with open(enc_path, "rb") as fe, open(dec_path, "rb") as fd:
    for poc in range(frame_count):
        enc = np.frombuffer(fe.read(frame_bytes), dtype=dtype)
        dec = np.frombuffer(fd.read(frame_bytes), dtype=dtype)

        if enc.size != frame_samples or dec.size != frame_samples:
            print("EOF at frame", poc)
            break

        diff = enc != dec
        if np.any(diff):
            idx = np.flatnonzero(diff)[0]

            if idx < y_size:
                plane = "Y"
                y = idx // W
                x = idx % W
                enc_val = int(enc[idx])
                dec_val = int(dec[idx])
            elif idx < y_size + uv_size:
                plane = "U"
                j = idx - y_size
                y = j // (W // 2)
                x = j % (W // 2)
                enc_val = int(enc[idx])
                dec_val = int(dec[idx])
            else:
                plane = "V"
                j = idx - y_size - uv_size
                y = j // (W // 2)
                x = j % (W // 2)
                enc_val = int(enc[idx])
                dec_val = int(dec[idx])

            print(f"First mismatch: POC/frame={poc}, plane={plane}, x={x}, y={y}, enc={enc_val}, dec={dec_val}")
            break
    else:
        print("No mismatch")















