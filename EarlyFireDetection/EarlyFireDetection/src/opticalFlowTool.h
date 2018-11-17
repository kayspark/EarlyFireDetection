#ifndef OPTFLOWTOOL_H
#define OPTFLOWTOOL_H

#include <map>
#include "ds.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
/* Optical Flow Parameters */
#define MAX_CORNER 10000
/* Drawing Arrow for Optical Flow */
void drawArrow(cv::Mat &imgDisplay,
               const std::array<cv::Point2f, MAX_CORNER> &featuresPrev,
               const std::array<cv::Point2f, MAX_CORNER> &featuresCurr,
               int cornerCount,
               const std::array<char, MAX_CORNER> &featureFound);

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
the number of contour points
*/
int getContourFeatures(cv::Mat &img,
                       cv::Mat &imgDisplayFireRegion,
                       std::vector<std::vector<cv::Point>> &contours,
                       std::vector<OFRect> &vecOFRect,
                       const RectThrd &trd,
                       std::array<cv::Point2f, MAX_CORNER> &featuresPrev,
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
                         std::array<char, MAX_CORNER> &status,
                         std::array<cv::Point2f, MAX_CORNER> &featuresPrev,
                         std::array<cv::Point2f, MAX_CORNER> &featuresCurr);

#endif
