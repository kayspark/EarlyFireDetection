// opencv libarary
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
/* Self-Developed Library */
#include "ds.h"
//#include "fileStream.h"
#include "motionDetection.h"
#include "nm_detector.h"
#include "vlccap.h"
#include "fire_detector.h"

#include <list>

using namespace std;
using namespace cv;

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

auto main(int argc, char *argv[]) -> int {
  std::string tracker_algorithm = "MEDIAN_FLOW";
  std::string data_file("./dataset/cascade.xml");
  nm_detector detector(data_file);
  fire_detector fireDetector;

  cv::Ptr<Tracker> tracker = createTrackerByName(tracker_algorithm);

  if (tracker.empty()) {
    cout << "***Error in the instantiation of the tracker...***\n";
    return -1;
  }
  bool initialized_tracker = false;
  cv::Rect2d fire_area(0, 0, 0, 0);
  cv::VideoCapture capture(CAP_ANY);

  // vlc_capture capture("RV24", 800, 600);
  capture.open(argv[1]);
  if (!capture.isOpened()) {
    cout << "Cannot open video!\n" << endl;
    return 1;
  }
  // default size for analysis

  // cv::Size sizeImg(800, 600);
  // cv::Size sizeImg = capture.get_size();

  cv::Mat imgSrc; //(sizeImg, CV_8UC3); //capture.read(imgSrc);
  capture >> imgSrc;
  cv::Size sizeImg = imgSrc.size();
  /************************Motion Detection*************************/
  auto imgGray = cv::Mat(sizeImg, CV_8UC1, cv::Scalar());
  // mask motion
  auto maskMotion = cv::Mat(sizeImg, CV_8UC1);
  // for rgb image display copy from src
  auto imgRGB = cv::Mat(sizeImg, CV_8UC3);
  auto imgHSI = cv::Mat(sizeImg, CV_8UC3);
  // mask rgb
  auto maskRGB = cv::Mat(sizeImg, CV_8UC1);
  // mask hsi
  auto maskHSI = cv::Mat(sizeImg, CV_8UC1);
  auto bufHSI = cv::Mat(sizeImg, CV_64FC3);
  // Optical FLow
  auto imgCurr = cv::Mat(sizeImg, CV_8UC1);
  auto imgDisplay = cv::Mat(sizeImg, CV_8UC3);
  // Buffer for Pyramid image
  cv::Size sizePyr = cv::Size(sizeImg.width + 8, sizeImg.height / 3);
  auto pyrPrev = cv::Mat(sizePyr, CV_32FC1);
  auto pyrCurr = cv::Mat(sizePyr, CV_32FC1);

  unsigned long curr_frm = 0;

  int key = 0;

  // create motionDetection object
  motionDetection bgs(fireDetector.get_bgm_frame_count(), sizeImg);
  // get background model
  cv::Mat imgBackgroundModel(sizeImg, CV_32FC1, cv::Scalar());
  // get standard deviation
  cv::Mat imgStandardDeviation(sizeImg, CV_32FC1, cv::Scalar());
  bgs.getStandardDeviationFrame(imgStandardDeviation);
  auto img32FBackgroundModel = cv::Mat(sizeImg, CV_32FC1);
  auto img32FStandardDeviation = cv::Mat(sizeImg, CV_32FC1);
  // coefficient * Threshold
  bgs.coefficientThreshold(
      imgStandardDeviation,
      fireDetector.get_threshold_coefficient()); // cvShowImage( "Standard Deviation",
  // setup background modes with video capture
  bgs.getBackgroundModel(capture, imgBackgroundModel);
  while (key != 'x') { // exit if user presses 'x'
    // flash
    maskRGB.setTo(cv::Scalar::all(0));
    maskHSI.setTo(cv::Scalar::all(0));
    // set frame
    capture.read(imgSrc);

    if (imgSrc.empty()) {
      continue;
    }
    cv::cvtColor(imgSrc, imgGray, CV_BGR2GRAY);
    imgSrc.copyTo(imgDisplay);
    detector.detectAndDraw(imgSrc, imgDisplay, 1.1);

    if (!fire_area.empty()) {
      if (initialized_tracker) {
        if (tracker->update(imgDisplay, fire_area))
          cv::rectangle(imgDisplay, fire_area, cv::Scalar(0, 0, 255), 3);
        else {
          fire_area = cv::Rect2d(0, 0, 0, 0);
          tracker = createTrackerByName(tracker_algorithm);
          initialized_tracker = false;
        }
      } else {
        // initializes the tracker
        if (!tracker->init(imgDisplay, fire_area)) {
          cout << "***Could not initialize tracker...***\n";
          return -1;
        }
        initialized_tracker = true;
      }
    } else {
      capture.read(imgSrc);
      cv::cvtColor(imgSrc, imgCurr, CV_BGR2GRAY);
      cv::Mat imgDiff;

      imgBackgroundModel.convertTo(imgBackgroundModel, CV_8UC1);
      cv::absdiff(imgGray, imgBackgroundModel, imgDiff);
      // imgDiff > standarDeviationx
      bgs.backgroundSubtraction(imgDiff, imgStandardDeviation, maskMotion);
      // cv::imshow("maskMotion", maskMotion);
      imgDisplay.copyTo(imgRGB);
      fireDetector.checkByRGB(imgDisplay, maskMotion, maskRGB);
      // markup the fire-like region
      fireDetector.regionMarkup(imgDisplay, imgRGB, maskRGB);
      /* HSI */
      imgDisplay.copyTo(imgHSI);
      fireDetector.RGB2HSIMask(imgDisplay, bufHSI, maskRGB);
      fireDetector.checkByHSI(imgDisplay, bufHSI, maskRGB, maskHSI);
      fireDetector.regionMarkup(imgDisplay, imgHSI, maskHSI);
      maskHSI.copyTo(maskRGB);
      // flip maskMotion 0 => 255, 255 => 0
      bgs.maskNegative(maskMotion);
      /* Background update */
      // 8U -> 32F
      imgBackgroundModel.convertTo(img32FBackgroundModel, CV_32FC1);
      accumulateWeighted(imgGray, img32FBackgroundModel,
                         fireDetector.get_accumulate_weighted_alpha_bgm(), maskMotion);
      // 32F -> 8U
      img32FBackgroundModel.convertTo(imgBackgroundModel, CV_8UC1);
      /* Threshold update */
      // 8U -> 32F
      imgStandardDeviation.convertTo(img32FStandardDeviation, CV_32FC1);
      // T( x, y; t+1 ) = ( 1-alpha )T( x, y; t ) + ( alpha ) | Src( x, y; t )/
      // - B( x, y; t ) |, if the pixel is stationary
      accumulateWeighted(imgDiff, img32FStandardDeviation,
                         fireDetector.get_accumulate_weighted_alpha_threshold(), maskMotion);
      // 32F -> 8U
      img32FStandardDeviation.convertTo(imgStandardDeviation, CV_8UC1);
      /* Step4: Morphology */
      fireDetector.dilate(maskHSI);
      fireDetector.findContours(maskHSI);
      fireDetector.getContourFeatures(imgDisplay, imgDisplay);
      fireDetector.calcOpticalFlow(imgGray, imgCurr);
      fireDetector.assignFeaturePoints();
      fire_area = fireDetector.matchCentroid(imgSrc, imgDisplay, static_cast<int>(curr_frm++));
    }
    cv::imshow("Display", imgDisplay);
    key = cv::waitKey(5);
  }
  return 0;
}