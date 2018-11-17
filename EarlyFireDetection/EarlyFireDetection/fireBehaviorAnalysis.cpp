

#include "fireBehaviorAnalysis.h"
/* Counting the foldback point at each directions */
void foldbackPoint(const std::vector<CvRect> &vecRect, DirectionsCount &count) {
  std::vector<CvRect>::const_iterator itVec;

  for (itVec = ++vecRect.begin(); (itVec + 1) != vecRect.end(); ++itVec) {
    const auto rn = std::next(itVec);
    const auto rp = std::prev(itVec);
    if (((*rn).y - (*itVec).y) * ((*itVec).y - (*rp).y) < 0) {
      ++count.countUp;
    }
    if ((((*rn).x - (*itVec).x) * ((*itVec).x - (*rp).x)) < 0) {
      ++count.countLeft;
    }
    if ((((*rn).y + (*rn).height) - ((*itVec).y + (*itVec).height))
        * (((*itVec).y + (*itVec).height) - ((*rp).y + (*rp).height)) < 0) {
      ++count.countDown;
    }
    if ((((*rn).x + (*rn).width) - ((*itVec).x + (*itVec).width))
        * (((*itVec).x + (*itVec).width) - ((*rp).x + (*rp).width)) < 0) {
      ++count.countRight;
    }
  }
}

/* Analysis the rect information */
bool judgeDirectionsMotion(const std::vector<CvRect> &vecRect, CvRect &rectFire) {
  DirectionsCount count;
  zeroCount(count);
  foldbackPoint(vecRect, count);
  const int thresh_foldback_cnt = 3; // 3
  /* Direction Up required to be growth and sparkle */
  if ((vecRect.front().y - vecRect.back().y) > 2 && count.countUp >= thresh_foldback_cnt) {
    /* set up the last rect to rect the frame */
    rectFire = vecRect.back();
    std::cout << "directions likely" << std::endl;
    return true;
  } else {
    return false;
  }
}
