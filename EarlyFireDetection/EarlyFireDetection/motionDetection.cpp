#include <opencv2/imgproc.hpp>
#include "motionDetection.h"
#include <iostream>

/* Create buffer for image */
motionDetection::motionDetection(const int frame_count, const CvSize frameSize)
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
      if (cvWaitKey(10) >= 0) {
        break;
      }
    } else {
      break;
    }
  }
  // average the frame series as background model
}

/* Standard Deviation */
void motionDetection::getStandardDeviationFrame(cv::Mat &out) {
  out.setTo(cv::Scalar::all(0));
  // Initialize
  cv::Mat total(_size, CV_32FC1, cv::Scalar());
  cv::Mat tmp(_size, CV_32FC1, cv::Scalar());
  cv::Mat tmp8U(_size, CV_8UC1, cv::Scalar());
  for (int i = 0; i < frameNumber; ++i) {
    // frame[i] <= | frame[i] - Background Model |
    cv::absdiff(_vec_frame[i], m_imgBackgroundModel, tmp8U);
    //TODO: check later for correctness
    //cvAbsDiff(_vec_frame[i], m_imgBackgroundModel, mTmp8U);
    //cvConvert(mTmp8U, mTmp);j
    tmp8U.convertTo(tmp, CV_32FC1);
    // mTmp = mTmp * mTmp
    cv::pow(tmp, 2.0, tmp);
    total += tmp;
  }

  // variance: mTmp <= mSum / (frameNumber-1)
  for (int i = 0; i < _size.height; ++i) {
    for (int j = 0; j < _size.width; ++j) {
      ((float *) (tmp.data + i * tmp.step))[j] = ((float *) (out.data + i * out.step))[j] / (frameNumber - 1);
    }
  }
  // standard deviation
  cv::sqrt(tmp, out);
  // float->uchar
  out.convertTo(out, CV_8UC1);
}

/* Negative processing, convert darkest areas to lightest and lightest to
 * darkest */
void motionDetection::maskNegative(cv::Mat &img) {
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      if ((img.data + i * img.step)[j] == 0) {
        (img.data + i * img.step)[j] = 255;
      } else {
        (img.data + i * img.step)[j] = 0;
      }
    }
  }
}

/* th = th * coefficient */
void motionDetection::coefficientThreshold(cv::Mat &imgThreshold, const int coef) {

  uchar *dataThreshold = imgThreshold.data;
  auto step = static_cast<int>(imgThreshold.step / sizeof(uchar));

  for (int i = 0; i < imgThreshold.rows; ++i) {
    for (int j = 0; j < imgThreshold.cols; ++j) {
      auto tmp = static_cast<uchar>(dataThreshold[i * step + j] * coef);
      if (tmp > 255) {
        tmp = 255;
      }
      if (tmp < 0) {
        tmp = 0;
      }
      dataThreshold[i * step + j] = tmp;
    }
  }
}

/* one channel & uchar only => imgDiff, imgThreshold, mask
 * the mask always needed to be reflash( cvZero(mask) ) first!!
 */
void motionDetection::backgroundSubtraction(const cv::Mat &imgDiff, const cv::Mat &imgThreshold, cv::Mat &mask) {

  static uchar *dataMask = mask.data;
  static uchar *dataDiff = imgDiff.data;
  static uchar *dataThreshold = imgThreshold.data;

  static const auto step = static_cast<int>(mask.step / sizeof(uchar));
  for (int i = 0; i < imgDiff.rows; ++i) {
    for (int j = 0; j < imgDiff.cols; ++j) {
      // foreground(255)
      if (dataDiff[i * step + j] > dataThreshold[i * step + j]) {
        dataMask[i * step + j] = 255;
      }
      // else background(0)
    }
  }
}


