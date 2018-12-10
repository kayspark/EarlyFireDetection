#include "motionDetection.h"

/* Create buffer for image */
motionDetection::motionDetection(const int &frame_count,
                                 cv::Size frameSize)
    : _frameno(frame_count), _count(0), _size(std::move(frameSize)) {
  // create n IplImage pointer, and assign for _vec_frame
  _vec_frame.reserve((unsigned long) _frameno);
  for (int i = 0; i < _frameno; ++i) {
    _vec_frame.emplace_back(cv::Mat(_size, CV_8UC1));
  }
  // create memory for background model
  _img_background = cv::Mat(_size, CV_8UC1);
  // create memory for Standard Deviation(threshold)
}
/* Release memory */
motionDetection::~motionDetection() {
  for (int i = 0; i < _frameno; ++i) {
    _vec_frame[i].release();
  }
}
/* Calculate Background Model */
void motionDetection::getBackgroundModel(cv::VideoCapture &cap, cv::Mat &out) {
  // accumulate frame from video

  while (_count != _frameno) {
    cv::Mat frame;
    if (cap.isOpened()) {
      cap.read(frame);
      // cap >> frame;
      // convert rgb to gray
      if (frame.empty())
        continue;
      //  cv::resize(frame, frame, cv::Size(out.cols, out.rows));
      // cv::cvtColor(frame,frame, cv::COLOR_YUV2BGR_YV12,3);
      cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
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
void motionDetection::getBackgroundModel(vlc_capture &cap, cv::Mat &out) {
  // accumulate frame from video

  while (_count != _frameno) {
    cv::Mat frame;
    if (cap.isOpened()) {
      cap.read(frame);
      // cap >> frame;
      // convert rgb to gray
      if (frame.empty())
        continue;
      //  cv::resize(frame, frame, cv::Size(out.cols, out.rows));
      // cv::cvtColor(frame,frame, cv::COLOR_YUV2BGR_YV12,3);
      cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
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

  std::for_each(_vec_frame.begin(), _vec_frame.end(),
                [this, &out, &tmp, &tmp8U](const auto &frame) {
                  cv::absdiff(frame, _img_background, tmp8U);
                  tmp8U.convertTo(tmp, CV_32FC1);
                  cv::pow(tmp, 2.0, tmp);
                  out += tmp;
                });

  // variance: mTmp <= mSum / (_frameno-1)
  // standard deviation
  out.forEach<float>([this](float &pixel, const int *position) -> void {
    pixel = sqrt(pixel / (_frameno - 1));
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
void motionDetection::coefficientThreshold(cv::Mat &imgThreshold,
                                           const int coef) {
  imgThreshold.forEach<uchar>(
      [&coef](uchar &pixel, const int *position) -> void {
        pixel = static_cast<uchar>(pixel * coef);
        pixel = static_cast<uchar>(pixel > 255 ? 255 : pixel < 0 ? 0 : pixel);
      });
}

/* one channel & uchar only => imgDiff, imgThreshold, mask
 * the mask always needed to be reflash( cvZero(mask) ) first!!
 */
void motionDetection::backgroundSubtraction(const cv::Mat &imgDiff,
                                            const cv::Mat &imgThreshold,
                                            cv::Mat &mask) {
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