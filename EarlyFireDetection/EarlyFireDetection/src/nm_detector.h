#pragma once
#include <opencv2/opencv.hpp>

class nm_detector {
private :
  cv::CascadeClassifier _cascade;

public:
  explicit nm_detector(std::string &cascade);
  ~nm_detector();
  void detectAndDraw(cv::Mat &gray, cv::Mat &display, double scale);
};
