#ifndef OPTFLOWTOOL_H
#define OPTFLOWTOOL_H

#include "ds.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <map>
/* Optical Flow Parameters */
#define _max_corners 10000
/* Drawing Arrow for Optical Flow */
void drawArrow(cv::Mat &imgDisplay,
               const std::vector<cv::Point2f> &featuresPrev,
               const std::vector<cv::Point2f> &featuresCurr, int cornerCount,
               const std::vector<uchar> &featureFound);

/* get the feature points from contour
input:
imgDisplayCntr      : img for display contours
imgDisplayFireRegion: img for boxing the fire-like region with rectangle
contour             : after cvFindContour()
trd                 : threshold
output:
vecOFRect           : fire-like obj will be assign to this container
featuresPrev        : previous contours points
featuresCurr        : current contours points
return:
the number of contour points:
*/
int getContourFeatures(cv::Mat &img, cv::Mat &imgDisplayFireRegion,
                       std::vector<std::vector<cv::Point>> &contours,
                       std::vector<OFRect> &vecOFRect, const RectThresh &trd,
                       std::vector<cv::Point2f> &featuresPrev,
                       std::vector<cv::Vec4i> &hierachy);

/* assign feature points to fire-like obj and then push to multimap
input:
vecOFRect:      fire-like obj
status:			the feature stutas(found or not) after tracking
featuresPrev:	previous feature points
featuresCurr:   current feature points after tracking

output:
mulMapOFRect:	new candidate fire-like obj in current frame(with rectangle and
motion vector information)

*/
void assignFeaturePoints(std::multimap<int, OFRect> &mulMapOFRect,
                         std::vector<OFRect> &vecOFRect,
                         std::vector<uchar> &status,
                         std::vector<cv::Point2f> &featuresPrev,
                         std::vector<cv::Point2f> &featuresCurr);

#endif
