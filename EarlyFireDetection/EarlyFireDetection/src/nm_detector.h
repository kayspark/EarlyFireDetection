#pragma once
#include <opencv2/opencv.hpp>

class nm_detector {
private :
  cv::CascadeClassifier _cascade;
  std::vector<cv::Rect> objects;
  std::array<cv::Scalar, 8> colors;
  double scale;
public:
  explicit nm_detector(std::string &cascade);
  ~nm_detector();
  void detect_objects(cv::Mat &gray, cv::Mat &display);
  void draw_objects(const cv::Mat &display,
                    int color_code) const;
};
