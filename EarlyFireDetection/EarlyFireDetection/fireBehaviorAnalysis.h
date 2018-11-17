

#ifndef FIREBEHAVIORANALYSIS_H
#define FIREBEHAVIORANALYSIS_H

#include <vector>
#include "ds.h"
#include "opencv/cv.h"

/* Counting the foldback point at each directions */
void foldbackPoint(const std::vector<CvRect> &vecRect, DirectionsCount &count);

/* Analysis the rect information */
bool judgeDirectionsMotion(const std::vector<CvRect> &vecRect, CvRect &rectFire);

#endif
