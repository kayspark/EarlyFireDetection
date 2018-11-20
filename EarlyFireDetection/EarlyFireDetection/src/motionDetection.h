#ifndef MOTIONDETECTION_H
#define MOTIONDETECTION_H

#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

/* Motion Detection
* purpose: For get the initialization background model and threshold(depends on standard deviation)
* support:
*          backgroundSubtraction(): | imgDiff | > threshold, get the foreground from mask(255)
*          coefficientThreshold():  coef * threshold
*          maskNegative(): 0->255; 255->0
*/
class motionDetection {
private:
  std::vector<cv::Mat> _vec_frame;
  cv::Mat _img_background;    // background model


  const int _frameno;            // the number of frame for calculate background model
  int _count;
  cv::Size _size;                      // image size


  /* avoid copy & assignment */
  motionDetection(const motionDetection &bgs);
  void operator=(const motionDetection &bgs);

public:

  /*
  * constructor
  * _frameno: the number of frame that want to be processing as background model
  * frameSize: the size o frame
  */
  motionDetection(const int &frame_count, const cv::Size &frameSize);

  /* destructor */
  ~motionDetection();

  /* Need pass capture  ptr */
  void getBackgroundModel(cv::VideoCapture &capture, cv::Mat &ret);

  void getStandardDeviationFrame(cv::Mat &ret);

  /* one channel & uchar only => imgDiff, imgThreshold, mask
  * the mask always needed to be reflash( cvZero(mask) ) first!!
  */
  void backgroundSubtraction(const cv::Mat &imgDiff, const cv::Mat &imgThreshold, cv::Mat &mask);
  /* th = th * coefficient */
  void coefficientThreshold(cv::Mat &imgThreshold, int coef);
  /* Negative processing, convert darkest areas to lightest and lightest to darkest */
  void maskNegative(cv::Mat &img);

};

#endif
