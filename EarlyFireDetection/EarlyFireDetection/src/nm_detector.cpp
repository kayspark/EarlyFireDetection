#include "nm_detector.h"
#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;

nm_detector::nm_detector(std::string &cascade) {
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
void nm_detector::detectAndDraw(cv::Mat &gray, cv::Mat &display, double scale) {
  double timer = 0;
  std::vector<Rect> objects;
  const static std::array<cv::Scalar, 8> colors = {
      Scalar(255, 0, 0), Scalar(255, 128, 0), Scalar(255, 255, 0),
      Scalar(0, 255, 0), Scalar(0, 128, 255), Scalar(0, 255, 255),
      Scalar(0, 0, 255), Scalar(255, 0, 255)};
  cv::Mat smallImg;

  // cvtColor(img, gray, COLOR_BGR2GRAY);
  double fx = 1 / scale;
  resize(gray, smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR_EXACT);
  // equalizeHist(smallImg, smallImg);

  timer = (double) cv::getTickCount();
  _cascade.detectMultiScale(smallImg, objects, 1.1, 2,
                            0
                                //|CASCADE_FIND_BIGGEST_OBJECT
                                //|CASCADE_DO_ROUGH_SEARCH
                                | cv::CASCADE_SCALE_IMAGE,
                            Size(24, 24), Size(350, 350));

  timer = (double) getTickCount() - timer;
  std::cout << "suspicious object: " << objects.size()
            << " , time: " << timer * 1000 / getTickFrequency() << std::endl;
  int color_code = 0;
  for (const auto &r : objects) {
    Scalar color = colors[color_code++ % 8];
    rectangle(display, Point(cvRound(r.x * scale), cvRound(r.y * scale)),
              Point(cvRound((r.x + r.width - 1) * scale),
                    cvRound((r.y + r.height - 1) * scale)),
              color, 3, 8, 0);
  }
  // imshow( "result", img );
}
