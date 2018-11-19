#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include "opticalFlowTool.h"

void drawArrow(cv::Mat &imgDisplay,
               const std::vector<cv::Point2f> &featuresPrev,
               const std::vector<cv::Point2f> &featuresCurr,
               int cornerCount,
               const std::vector<uchar> &featureFound) {
  static int i, lineThickness = 1;
  static cv::Scalar lineColor(100, 200, 250);
  static double angle, hypotenuse, tmpCOS, tmpSIN;
  static CvPoint p, q;
  static const double PI_DIV_4 = CV_PI / 4;

  // Draw the flow field
  for (i = 0; i < cornerCount; ++i) {
    // if the feature point wasn't be found
    if (featureFound[i] == 0) {
      continue;
    }

    p.x = static_cast<int>(featuresPrev[i].x);
    p.y = static_cast<int>(featuresPrev[i].y);
    q.x = static_cast<int>(featuresCurr[i].x);
    q.y = static_cast<int>(featuresCurr[i].y);

    angle = atan2(static_cast<double>(p.y - q.y), static_cast<double>(p.x - q.x));
    hypotenuse = sqrt(pow(p.y - q.y, 2.0) + pow(p.x - q.x, 2.0));

    q.x = static_cast<int>(p.x - 10 * hypotenuse * cos(angle));
    q.y = static_cast<int>(p.y - 10 * hypotenuse * sin(angle));

    // '|'
    cv::line(imgDisplay, p, q, lineColor, lineThickness, CV_AA, 0);

    tmpCOS = 3 * cos(angle + PI_DIV_4);
    tmpSIN = 3 * sin(angle + PI_DIV_4);

    p.x = static_cast<int>(q.x + tmpCOS);
    p.y = static_cast<int>(q.y + tmpSIN);
    // '/'
    cv::line(imgDisplay, p, q, CV_RGB(255, 0, 0), lineThickness, CV_AA, 0);

    p.x = static_cast<int>(q.x + tmpCOS);
    p.y = static_cast<int>(q.y + tmpSIN);
    // '\'
    cv::line(imgDisplay, p, q, CV_RGB(255, 0, 0), lineThickness, CV_AA, 0);
  }
}

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
                       std::vector<OFRect> &vecOFRect, const RectThrd &trd, std::vector<cv::Point2f> &featuresPrev,
                       std::vector<cv::Vec4i> &hierachy) {
  //TODO: seems need to reset
  //featuresPrev.clear();
  static unsigned int countCtrP;
  auto ContourFeaturePointCount = 0;
  /* thresholding on connected component */
  for (int index = 0; index < contours.size(); index++) {  // contours-based visiting
    /* Recting the Contour with smallest rectangle */
    cv::Rect rect_ = cv::boundingRect(contours[index]);
    /* checking the area */
    if (((rect_.width > trd.rectWidth) && (rect_.height > trd.rectHeight))
        && (fabs(cv::contourArea(contours[index])) > trd.cntrArea)) {
      /* Drawing the Contours */
      cv::drawContours(img, contours, index, cv::Scalar(250, 0, 0),  // Red
                       2,  // Vary max_level and compare results
                       8, hierachy); //line type
      cv::imshow("Fire-like Contours", img);
      /* Drawing the region */
      cv::rectangle(imgDisplayFireRegion,
                    cvPoint(rect_.x, rect_.y),
                    cv::Point((rect_.x) + (rect_.width), (rect_.y) + (rect_.height)),
                    CV_RGB(255, 10, 0),
                    2);

      /* for each contours pixel count	*/
      countCtrP = 0;

      /* access points on each contours */
      // printf(" (%d,%d)\n", p->x, p->y );

      for (int i = 0; i < contours[index].size(); i++) {
        //const auto &p : contours[index]) {
        const auto &p = contours[index][i];
        featuresPrev[i] = p;
        ++countCtrP;
        ++ContourFeaturePointCount;
      }
      /* push to tmp vector for quick access ofrect node */
      vecOFRect.push_back(ofRect(rect_, countCtrP));
    }
  }

  return ContourFeaturePointCount;
}

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
                         std::vector<cv::Point2f> &featuresCurr) {
  // visit each ofrect in vecOFRect
  for (auto &aRect : vecOFRect) {
    int i = 0;  // feature point index
    // contour points count
    for (int p = 0; p < aRect.countCtrP; ++p) {
      // if the feature point was be found
      if (status[i] == 0) {
        ++i;
        continue;
      } else {
        /* push feature to vector of ofrect */
        aRect.vecFeature.push_back(feature(featuresPrev[i], featuresCurr[i]));
        ++i;
      }
    }
    /* insert ofrect to multimap */
    mulMapOFRect.insert(std::pair<int, OFRect>(aRect.rect.x, aRect));
  }
  /* clear up container */
  vecOFRect.clear();
}

