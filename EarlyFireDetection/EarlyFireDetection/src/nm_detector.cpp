#include "nm_detector.h"
#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;

/*
cv::Ptr<cv::Tracker> createTrackerByName(cv::String name) {
  cv::Ptr<cv::Tracker> tracker;

  if (name == "KCF")
    tracker = cv::TrackerKCF::create();
  else if (name == "TLD")
    tracker = cv::TrackerTLD::create();
  else if (name == "BOOSTING")
    tracker = cv::TrackerBoosting::create();
  else if (name == "MEDIAN_FLOW")
    tracker = cv::TrackerMedianFlow::create();
  else if (name == "MIL")
    tracker = cv::TrackerMIL::create();
  else if (name == "GOTURN")
    tracker = cv::TrackerGOTURN::create();
  else if (name == "MOSSE")
    tracker = cv::TrackerMOSSE::create();
  else if (name == "CSRT")
    tracker = cv::TrackerCSRT::create();
  else
    CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

  return tracker;
}
*/
nm_detector::nm_detector(std::string &cascade)
    :
    scale(1.1),
    objects(std::vector<cv::Rect>()),
    colors(std::array<cv::Scalar, 8>() =
               {cv::Scalar(255, 0, 0), cv::Scalar(255, 128, 0), cv::Scalar(255, 255, 0),
                cv::Scalar(0, 255, 0), cv::Scalar(0, 128, 255), cv::Scalar(0, 255, 255),
                cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 255)
               }

    ) {
  if (cascade.empty())
    cascade = "./dataset/cascade.xml";
  if (!_cascade.load(cascade)) {
    std::cerr << "ERROR: Could not load classifier cascade" << std::endl;
    return;
  }

}

nm_detector::~nm_detector() {
}
// detect roi
void nm_detector::detect_objects(cv::Mat &gray, cv::Mat &display) {
  double timer = 0;
  cv::Mat smallImg;

  // cvtColor(img, gray, COLOR_BGR2GRAY);
  double fx = 1 / scale;
  resize(gray, smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR_EXACT);
  // equalizeHist(smallImg, smallImg);

  timer = (double) cv::getTickCount();
  _cascade.detectMultiScale(smallImg, objects, scale, 2,
                            0
                                //|CASCADE_FIND_BIGGEST_OBJECT
                                //|CASCADE_DO_ROUGH_SEARCH
                                | cv::CASCADE_SCALE_IMAGE,
                            Size(24, 24), Size(350, 350));

  timer = (double) getTickCount() - timer;
  std::cout << "suspicious object: " << objects.size()
            << " , time: " << timer * 1000 / getTickFrequency() << std::endl;
  int color_code = 0;
  draw_objects(display, color_code);
  // imshow( "result", img );
}
void nm_detector::draw_objects(const Mat &display,
                               int color_code) const {
  for (const auto &r : objects) {
    Scalar color = colors[color_code++ % 8];
    rectangle(display, Point(cvRound(r.x * scale), cvRound(r.y * scale)),
              Point(cvRound((r.x + r.width - 1) * scale),
                    cvRound((r.y + r.height - 1) * scale)),
              color, 3, 8, 0);
  }
}
