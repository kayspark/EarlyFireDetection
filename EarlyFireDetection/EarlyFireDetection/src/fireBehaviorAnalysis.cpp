

#include "fireBehaviorAnalysis.h"
/* Counting the foldback point at each directions */
void foldbackPoint(const std::vector<cv::Rect> &vecRect, DirectionsCount &count) {
  if (vecRect.size() <= 2)
    std::cerr << "logical errors" << std::endl;
  for (auto i = 1; i < vecRect.size() - 1; i ++) {
    const auto rn = vecRect[i - 1];
    auto it = vecRect[i];
    const auto rp = vecRect[i + 1];
    if ((rn.y - it.y) * (it.y - rp.y) < 0) {
      ++count.countUp;
    }
    if ((rn.x - it.x) * (it.x - rp.x) < 0) {
      ++count.countLeft;
    }
    if (((rn.y + rn.height) - (it.y + it.height)) * ((it.y + it.height) - (rp.y + rp.height)) < 0) {
      ++count.countDown;
    }
    if (((rn.x + rn.width) - (it.x + it.width)) * ((it.x + it.width) - (rp.x + rp.width)) < 0) {
      ++count.countRight;
    }
  }
}

/* Analysis the rect information */
bool judgeDirectionsMotion(const std::vector<cv::Rect> &vecRect, cv::Rect &rectFire) {
  DirectionsCount count;
  zeroCount(count);
  foldbackPoint(vecRect, count);
  const int thresh_foldback_cnt = 3; // 3
  /* Direction Up required to be growth and sparkle */
  if ((vecRect.front().y - vecRect.back().y) > 2 && count.countUp >= thresh_foldback_cnt) {
    /* set up the last rect to rect the frame */
    rectFire = vecRect.back();
    //std::cout << "by directions likely to be fire" << std::endl;
    return true;
  } else {
    return false;
  }
}
