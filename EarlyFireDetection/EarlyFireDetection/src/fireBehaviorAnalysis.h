#ifndef FIREBEHAVIORANALYSIS_H
#define FIREBEHAVIORANALYSIS_H

#include "ds.h"
#include "opencv/cv.h"
#include <vector>

/* Counting the foldback point at each directions */
void foldbackPoint(const std::vector<cv::Rect> &vecRect,
                   DirectionsCount &count);

/* Analysis the rect information */
bool judgeDirectionsMotion(const std::vector<cv::Rect> &vecRect,
                           cv::Rect &rectFire);

#endif
