#include "CommonLib/Unit.h"
#include "CommonLib/MotionInfo.h"

static inline int floorMod16( int v )
{
  int r = v % 16;
  return r < 0 ? r + 16 : r;
}

static inline int getFrac16( int mvVal )
{
  return floorMod16( mvVal ); // result: 0 ~ 15
}














#include <fstream>
#include <string>
#include <iomanip>

#include "CommonDef.h"
#include "Buffer.h"

static inline void savePelBufAsTxt(
  const CPelBuf& pelBuf,
  const std::string& filePath
)
{
  std::ofstream ofs(filePath);
  if (!ofs.is_open())
  {
    printf("[savePelBufAsTxt] Failed to open file: %s\n", filePath.c_str());
    return;
  }

  const int w = pelBuf.width;
  const int h = pelBuf.height;

  ofs << "# width " << w << "\n";
  ofs << "# height " << h << "\n";

  for (int y = 0; y < h; ++y)
  {
    const Pel* row = pelBuf.buf + y * pelBuf.stride;

    for (int x = 0; x < w; ++x)
    {
      ofs << row[x];

      if (x + 1 < w)
      {
        ofs << " ";
      }
    }

    ofs << "\n";
  }

  ofs.close();
}




















#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <iomanip>

static inline void saveFloatVectorAsTxt(
  const std::vector<float>& data,
  const int width,
  const int height,
  const std::string& filePath
)
{
  if ((int)data.size() != width * height)
  {
    printf("[saveFloatVectorAsTxt] Size mismatch\n");
    printf("data.size() = %d, width * height = %d\n",
           (int)data.size(), width * height);
    return;
  }

  std::ofstream ofs(filePath);

  if (!ofs.is_open())
  {
    printf("[saveFloatVectorAsTxt] Failed to open file: %s\n",
           filePath.c_str());
    return;
  }

  ofs << "# width "  << width  << "\n";
  ofs << "# height " << height << "\n";

  ofs << std::fixed << std::setprecision(6);

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      const float v = data[y * width + x];

      ofs << v;

      if (x + 1 < width)
      {
        ofs << " ";
      }
    }

    ofs << "\n";
  }

  ofs.close();
}
