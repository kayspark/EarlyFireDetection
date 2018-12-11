// opencv libarary
#include "fire_detector.h"
#include "nm_detector.h"

using namespace std;
using namespace cv;

auto main(int argc, char *argv[]) -> int {
  std::string data_file("./dataset/cascade.xml");
  std::string algorithm("CSRT");
  nm_detector detector(data_file, algorithm);
  cv::VideoCapture capture;
  cv::Size sizeImg(1024, 768);
  //vlc_capture capture("RV24", 1024, 768);
  //capture.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
  //capture.set(cv::CAP_PROP_FRAME_HEIGHT, 768);
  capture.open(argv[1]);
  if (!capture.isOpened()) {
    cout << "Cannot open video!\n" << endl;
    return 1;
  }
  // default size for analysis

  //cv::Size sizeImg = capture.get_size();

  cv::Mat imgSrc; //(sizeImg, CV_8UC3); //capture.read(imgSrc);
  capture.read(imgSrc);
  cv::resize(imgSrc, imgSrc, sizeImg, 0, 0, cv::INTER_CUBIC);
  //cv::Size sizeImg = imgSrc.size();
  fire_detector fireDetector(sizeImg);
  /************************Motion Detection*************************/
  auto imgGray = cv::Mat(sizeImg, CV_8UC1, cv::Scalar());
  // mask motion
  auto maskMotion = cv::Mat(sizeImg, CV_8UC1);
  auto imgDisplay = cv::Mat(sizeImg, CV_8UC3);
  // Buffer for Pyramid image

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
      fireDetector
          .get_threshold_coefficient()); // cvShowImage( "Standard Deviation",
  // setup background modes with video capture
  bgs.getBackgroundModel(capture, imgBackgroundModel);
  while (key != 'x') { // exit if user presses 'x'
    // set frame
    capture.read(imgSrc);
    if (imgSrc.empty()) {
      break;
    }
    cv::resize(imgSrc, imgSrc, sizeImg, 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(imgSrc, imgGray, CV_BGR2GRAY);
    imgSrc.copyTo(imgDisplay);
    //detector.detect_objects(imgSrc, imgDisplay);
    detector.update_tracker(imgSrc, imgDisplay);
    if (!fireDetector.update_tracker(imgDisplay)) {
      capture.read(imgSrc);
      if (imgSrc.empty())
        continue;
      cv::resize(imgSrc, imgSrc, sizeImg, 0, 0, cv::INTER_CUBIC);
      fireDetector.detectFire(maskMotion, bgs, imgBackgroundModel,
                              imgStandardDeviation, img32FBackgroundModel,
                              img32FStandardDeviation, imgSrc, imgGray,
                              imgDisplay);

    } else {
    }
    cv::imshow("Predator", imgDisplay);
    key = cv::waitKey(5);
  }
  return 0;
}
