#include "motionDetection.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/* Create buffer for image */
motionDetection::motionDetection(const int &frame_count, const cv::Size &frameSize)
    : frameNumber(frame_count), _count(0), _size(frameSize) {
  // create n IplImage pointer, and assign for _vec_frame
  _vec_frame.reserve((unsigned long) frameNumber);
  for (int i = 0; i < frameNumber; ++i) {
    _vec_frame.emplace_back(cv::Mat(_size, CV_8UC1));
  }
  // create memory for background model
  m_imgBackgroundModel = cv::Mat(_size, CV_8UC1);
  // create memory for Standard Deviation(threshold)
}
/* Release memory */
motionDetection::~motionDetection() {
  for (int i = 0; i < frameNumber; ++i) {
    _vec_frame[i].release();
  }
}
/* Calculate Background Model */
void motionDetection::getBackgroundModel(cv::VideoCapture &capture, cv::Mat &out) {
  // accumulate frame from video
  while (_count != frameNumber) {
    cv::Mat frame;
    if (capture.isOpened()) {
      capture >> frame;
      // convert rgb to gray
      if (frame.empty())
        continue;
      cv::cvtColor(frame, frame, CV_BGR2GRAY);
      cv::accumulate(frame, out);
      ++_count;
      if (cv::waitKey(10) >= 0) {
        break;
      }
    } else {
      break;
    }
  }
  // average the frame series as background model
}

/* Standard Deviation */
// in 32UC1, out 8UC1
void motionDetection::getStandardDeviationFrame(cv::Mat &out) {
  out.setTo(cv::Scalar::all(0));
  // Initialize
  cv::Mat tmp(_size, CV_32FC1, cv::Scalar());
  cv::Mat tmp8U(_size, CV_8UC1, cv::Scalar());
  for (int i = 0; i < frameNumber; ++i) {
    cv::absdiff(_vec_frame[i], m_imgBackgroundModel, tmp8U);
    tmp8U.convertTo(tmp, CV_32FC1);
    cv::pow(tmp, 2.0, tmp);
    out += tmp;
  }

  // variance: mTmp <= mSum / (frameNumber-1)
  // standard deviation
  out.forEach<float>([this](float &pixel, const int *position) -> void {
    pixel = sqrt(pixel / (frameNumber - 1));
  });
  // float->uchar
  out.convertTo(out, CV_8UC1);
}

/* Negative processing, convert darkest areas to lightest and lightest to
 * darkest */
void motionDetection::maskNegative(cv::Mat &img) {
  img.forEach<uint8_t>([](uint8_t &pixel, const int *position) -> void {
    pixel = static_cast<uint8_t>(pixel == 0 ? 255 : 0);
  });
}

/* th = th * coefficient */
// imgThreshhold = 32FC1, use uchar
void motionDetection::coefficientThreshold(cv::Mat &imgThreshold, const int coef) {
  imgThreshold.forEach<uchar>([&coef](uchar &pixel, const int *position) -> void {
    pixel = pixel * coef;
    pixel = static_cast<uchar>(pixel > 255 ? 255 : pixel < 0 ? 0 : pixel);
  });
}

/* one channel & uchar only => imgDiff, imgThreshold, mask
 * the mask always needed to be reflash( cvZero(mask) ) first!!
 */
void motionDetection::backgroundSubtraction(const cv::Mat &imgDiff, const cv::Mat &imgThreshold, cv::Mat &mask) {
  for (int i = 0; i < imgDiff.rows; ++i) {
    auto m = mask.ptr<uint8_t>(i);
    const auto diff = imgDiff.ptr<uchar>(i);
    const auto tr = imgThreshold.ptr<uchar>(i);
    for (int j = 0; j < imgDiff.cols; ++j) {
      // foreground(255)
      if (diff[j] > tr[j]) {
        m[j] = 255;
      }
      // else background(0)
    }
  }
}

