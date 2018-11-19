/**
 * @ Purpose: Early Fire Detection Based On Video Sequences
 *
 *
 * @ Author: TC, Hsieh, KsPark at nepes.co.kr
 * @ Date: 2014.05.03 , 2018.11.1
 * @
 */

/* OpenCV Library */
#include <opencv/cv.hpp>

/* Self-Developed Library */
#include "ds.h"
//#include "fileStream.h"
#include "motionDetection.h"
#include "opticalFlowTool.h"
#include "colorModel.h"
#include "fireBehaviorAnalysis.h"

/* C-PlusPlus Library */

/* STL Library */
#include <list>

/* Switch */
#define ON (-1)

/* Debug Mode */
#define DEBUG_MODE (ON)

/* Background Subtraction */
#define BGS_MODE (ON)

/* Optical Flow Motion Vector */
#define OFMV_DISPLAY (ON)

/* Halting While Fire Alarm */

using namespace std;
using namespace cv;

/* Non-named namespace, global constants */
namespace {

/* Background Mode */
#if defined(BGS_MODE) && (BGS_MODE == ON)
const int BGM_FRAME_COUNT = 20;
#else
const int BGM_FRAME_COUNT = 0;
#endif

const int WIN_SIZE = 5;

/* Processing Window Size (Frame) */
const unsigned int PROCESSING_WINDOWS = 15;  // 15

/* Background Model Update Coefficients */
const auto ACCUMULATE_WEIGHTED_ALPHA_BGM = 0.1;
const auto ACCUMULATE_WEIGHTED_ALPHA_THRESHOLD = 0.05;
const int THRESHOLD_COEFFICIENT = 5;

/* Fire-like Region Threshold */
const auto RECT_WIDTH_THRESHOLD = 5;
const auto RECT_HEIGHT_THRESHOLD = 5;
const auto CONTOUR_AREA_THRESHOLD = 12;
const auto CONTOUR_POINTS_THRESHOLD = 12;

}  // namespace

/* File Path (Resource and Results ) */
namespace {

// const char* InputVideoPath = "test.mp4";
// const char *InputVideoPath = "Y:\\Downloads\\02.mp4";
// const char *InputVideoPath = "Y:\\Downloads\\04.mp4";
// const char *InputVideoPath = "Y:\\Downloads\\hanjun_C_2.0V 14.mov";
// const char *InputVideoPath = "Y:\\Downloads\\hanjun_B_2.0V 4.mov";
// const char *InputVideoPath = "Y:\\Downloads\\hanjun_22.mov";
// const char *InputVideoPath = "/Users/kspark/Downloads/hanjun_22.mov";
// const char *InputVideoPath = "/Users/kspark/Downloads/02.mp4";

}  // namespace

std::string mat2str(cv::Mat &mat) {
  std::string r;
  int type = mat.type();
  auto depth = static_cast<uchar>(type & CV_MAT_DEPTH_MASK);
  auto chans = static_cast<uchar>(1 + (type >> CV_CN_SHIFT));

  switch (depth) {
  case CV_8U:r = "8U";
    break;
  case CV_8S:r = "8S";
    break;
  case CV_16U:r = "16U";
    break;
  case CV_16S:r = "16S";
    break;
  case CV_32S:r = "32S";
    break;
  case CV_32F:r = "32F";
    break;
  case CV_64F:r = "64F";
    break;
  default:r = "User";
    break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}
// detect roi
void detectAndDraw(Mat &img, CascadeClassifier &cascade, double scale) {
  double t = 0;
  vector<Rect> objects;
  const static std::array<Scalar, 8> colors =
      {Scalar(255, 0, 0), Scalar(255, 128, 0), Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 128, 255),
       Scalar(0, 255, 255), Scalar(0, 0, 255), Scalar(255, 0, 255)};
  Mat gray, smallImg;

  cvtColor(img, gray, COLOR_BGR2GRAY);
  double fx = 1 / scale;
  resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
  equalizeHist(smallImg, smallImg);

  t = (double) getTickCount();
  cascade.detectMultiScale(smallImg, objects, 1.1, 2, 0
      //|CASCADE_FIND_BIGGEST_OBJECT
      //|CASCADE_DO_ROUGH_SEARCH
      | CASCADE_SCALE_IMAGE, Size(24, 24));

  t = (double) getTickCount() - t;
  printf("detection time = %g ms\n", t * 1000 / getTickFrequency());
  int color_code = 0;
  for (const auto &r : objects) {
    Scalar color = colors[color_code++ % 8];
    rectangle(img,
              Point(cvRound(r.x * scale), cvRound(r.y * scale)),
              Point(cvRound((r.x + r.width - 1) * scale), cvRound((r.y + r.height - 1) * scale)),
              color,
              3,
              8,
              0);
  }
  // imshow( "result", img );
}
/*
deque< vector<feature> >
----------------
|   --------   |
|   |  f1  |   |
|   |------|   |
|   |  f2  |   |   <===   frame 1
|   |------|   |
|   |   .  |   |
|   |   .  |   |
|   |   .  |   |
|   |------|   |
|   |  fi  |   |
|   |------|   |
----------------
|   --------   |
|   |  f1  |   |
|   |------|   |
|   |  f2  |   |   <===   frame 2
|   |------|   |
|   |   .  |   |
|   |   .  |   |
|   |   .  |   |
|   |------|   |
|   |  fj  |   |
|   |------|   |
----------------
.
.
.
*/

/* the contour points in each frame must more then thrdcp and more then
processingwindows/3 input: strd    : centroid of candiadate thrdcp  : threshold
of contourpoints pwindows: processing windows0 output  : true or flase（legal or
not）
*/
bool checkContourPoints(Centroid &ctrd, const int thrdcp, const unsigned int pwindows) {
  long countFrame =
      // contour points of each frame
      std::count_if(ctrd.dOFRect.begin(),
                    ctrd.dOFRect.end(),
                    [&thrdcp](const auto &itrDeq) { return (itrDeq.size() < thrdcp); });
  bool out = countFrame < pwindows / 3;
  if (out) {
    std::cout << "countours are likely" << countFrame << " , " << pwindows / 3 << std::endl;
  }
  return out;
}
/* accumulate the motin vector depends on its orientation( based on 4 directions
) input: vecFeature : Contour Features orient      : accumulate array output :
orien[4]
*/
void motionOrientationHist(std::vector<Feature> &vecFeature, vector<unsigned int> &orient) {
  // std::vector<Feature>::iterator itrVecFeature;
  /* each point of contour  */
  std::for_each(vecFeature.begin(), vecFeature.end(), [&orient](const Feature &feature) {
    /* orientation */
    if (feature.prev.x >= feature.curr.x) {
      if (feature.prev.y >= feature.curr.y) {
        ++orient[0];  // up-left
      } else {
        ++orient[2];  // down-left
      }
    } else {
      if (feature.prev.y >= feature.curr.y) {
        ++orient[1];  // up-right
      } else {
        ++orient[3];  // down-right
      }
    }
  });
}

/* calculate the energy of fire contour based on motion vector
input:
vecFeature : Contour Features
staticCount: centroid want to analysis
totalPoints: current frame

output:
staticCount: the feature counts who's energy is lower than 1.0
totalPoints: the feature counts that energy is between 1.0 ~ 100.0
return: energy
*/
double getEnergy(std::vector<Feature> &vecFeature, unsigned int &staticCount, unsigned int &totalPoints) {
  /* initialization */
  double energy = 0.0;
  /* each contour point */
  for_each(vecFeature.begin(), vecFeature.end(), [&staticCount, &energy, &totalPoints](const auto &feature) {
    /* energy */
    double tmp = pow(abs(feature.curr.x - feature.prev.x), 2) + pow(abs(feature.curr.y - feature.prev.y), 2);
    if (tmp < 1.0) {
      ++staticCount;
    } else if (tmp < 100.0) {
      energy += tmp;
      ++totalPoints;
    }
  });
  return energy;
}

/* Analysis the contour motion vector
input:
ctrd    : cadidate fire object
pwindows: processing window
return  : fire-like or not
*/
bool checkContourEnergy(Centroid &ctrd, const unsigned int pwindows) {
  unsigned int orientFrame = 0;
  // unsigned int totalPoints = 0;
  unsigned int passFrame = 0;
  unsigned int staticFrame = 0;
  std::vector<unsigned int> orient{0, 0, 0, 0};
  /* contour motion vector of each frame */
  for (auto &feature : ctrd.dOFRect) {
    /* flash */
    unsigned int staticCount = staticFrame = staticCount = 0;
    unsigned int totalPoints = 0;

    /* energy analysis */
    if (getEnergy(feature, staticCount, totalPoints) > totalPoints >> 1) {
      ++passFrame;
    }
    if (staticCount > feature.size() >> 1) {
      ++staticFrame;
    }

    /* flash */
    std::fill(begin(orient), end(orient), 0);
    // memset(&orient, 0, sizeof(unsigned int) << 2);
    /* orientation analysis */
    motionOrientationHist(feature, orient);

    if (std::count(orient.begin(), orient.end(), 0) >= 1) {
      ++orientFrame;
    }
  }

  /* by experience */
  static const unsigned int thrdPassFrame = pwindows >> 1, thrdStaticFrame = pwindows >> 2,
      thrdOrienFrame = (pwindows >> 3) + 1;

  bool out = staticFrame < thrdStaticFrame ? passFrame > thrdPassFrame && orientFrame < thrdOrienFrame : false;
  if (out)
    std::cout << "energy is likely " << std::endl;
  return out;
}

/* compare the mulMapOFRect space with listCentroid space, if matching insert to
listCentroid space as candidate fire-like obj input: mulMapOFRect:	new
candidate fire-like obj in current frame(with rectangle and motion vector
information) currentFrame:   current processing frame thrdcp      :   threshold
of contour points pwindows    :	processing windows

output:
imgDisplay  :	boxing the alarm region
listCentroid:	candidate fire-like obj those matching with mulMapOFRect's obj

*/
void matchCentroid(cv::Mat &imgCentriod,
                   cv::Mat &imgFireAlarm,
                   std::list<Centroid> &listCentroid,
                   std::multimap<int, OFRect> &mulMapOFRect,
                   int currentFrame,
                   const int thrdcp,
                   const unsigned int pwindows) {
  static cv::Rect rectFire = cvRect(0, 0, 0, 0);

  listCentroid.remove_if([&mulMapOFRect, &pwindows, &thrdcp, &imgFireAlarm, &currentFrame](Centroid &centre) {
    bool out = false;
    /* visit mulMapOFRect between range [itlow,itup) */
    for (auto &aRect : mulMapOFRect) {
      const cv::Rect &rect = (aRect).second.rect;
      /* matched */
      if (centre.centroid.y >= rect.y && (rect.x + rect.width) >= centre.centroid.x
          && (rect.y + rect.height) >= centre.centroid.y) {
        /* push rect to the matched listCentroid node */
        centre.vecRect.push_back(rect);
        /* push vecFeature to matched listCentroid node */
        centre.dOFRect.push_back((aRect).second.vecFeature);
        /* Update countFrame and judge the threshold of it */
        if (++(centre.countFrame) == pwindows) {
          /* GO TO PROCEESING DIRECTION MOTION */
          if (!judgeDirectionsMotion(centre.vecRect, rectFire))
            break;
          if (checkContourPoints(centre, thrdcp, pwindows) && checkContourEnergy(centre, pwindows)) {
            /* recting the fire region */
            cv::rectangle(imgFireAlarm,
                          cvPoint(rectFire.x, rectFire.y),
                          cvPoint((rectFire.x) + (rectFire.width), (rectFire.y) + (rectFire.height)),
                          CV_RGB(0, 100, 255),
                          3);
            cv::putText(imgFireAlarm, "Fire !!", cv::Point(rectFire.x, rectFire.y), 2, 1.2, CV_RGB(255, 0, 0));
            cout << "Alarm: " << currentFrame << endl;
            cv::imshow("Video", imgFireAlarm);
          } else {
            break;  // if not on fire go to erase it
          }
          /* mark this rect as matched */
        }
        aRect.second.match = true;
        out = true;
        // ++itCentroid;
        break;  // if matched break the inner loop
      }
      // if ended the map rect and not matched anyone go to erase it
    }  // for (multimapBRect)
    return !out;
  });
  /* push new rect to listCentroid */
  std::for_each(mulMapOFRect.begin(), mulMapOFRect.end(), [&listCentroid](const auto &rect) {
    if (!rect.second.match) {
      /* push new node to listCentroid */
      listCentroid.push_back(centroid(rect.second));
      // cout << "after rect: " << endl;
      // cout << (*itBRect).second << endl;	x
    }
  });

  // cout <<"after list count: "<< listCentroid.size() << endl;

  /* check the list node with image */
  std::for_each(listCentroid.begin(), listCentroid.end(), [&imgCentriod](const auto &centre) {
    cv::rectangle(imgCentriod,
                  cv::Point(centre.centroid.x, centre.centroid.y),
                  cv::Point((centre.centroid.x) + 2, (centre.centroid.y) + 2),
                  cv::Scalar(0, 0, 0),
                  3);
  });

  /* clear up container */
  mulMapOFRect.clear();
}

auto main(int argc, char *argv[]) -> int {
  // capture from video
  cv::VideoCapture capture(argv[1]);
  if (!capture.isOpened()) {
    cerr << "Cannot open video!\n" << endl;
    return 1;
  }
  cv::Mat imgSrc;
  capture >> imgSrc;
  // Get the fps
  const auto FPS = capture.get(CV_CAP_PROP_FPS);
  cout << "Video fps: " << FPS << endl;
  CascadeClassifier cascade;
  std::string cascadeName = "./dataset/cascade2.xml";
  if (!cascade.load(cascadeName)) {
    cerr << "ERROR: Could not load classifier cascade" << endl;
    return -1;
  }
  // set frame size
  // TODO : check later this conversion is OK
  auto sizeImg = imgSrc.size();

  // Fire-like pixels count
  // unsigned int fireLikeCount = 0;
  /************************Get Initialization BGModel & Threshold(Standard Deviation)*************************/
  // create motionDetection object
  motionDetection bgs(BGM_FRAME_COUNT, sizeImg);
  // get background model
  cv::Mat imgBackgroundModel(sizeImg, CV_32FC1, cv::Scalar());
  bgs.getBackgroundModel(capture, imgBackgroundModel);
  // get standard deviation
  cv::Mat imgStandardDeviation(sizeImg, CV_32FC1, cv::Scalar());
  bgs.getStandardDeviationFrame(imgStandardDeviation);
  auto img32FBackgroundModel = cv::Mat(sizeImg, CV_32FC1);
  auto img32FStandardDeviation = cv::Mat(sizeImg, CV_32FC1);

  /************************Motion Detection*************************/
  // gray
  cv::Mat imgGray = cv::Mat(sizeImg, CV_8UC1, cv::Scalar());

  // coefficient * Threshold
  bgs.coefficientThreshold(imgStandardDeviation, THRESHOLD_COEFFICIENT);  // cvShowImage( "Standard Deviation",
  // imgStandardDeviation );
  // cvSaveImage( "Coefficient Standard Deviation.bmp", imgStandardDeviation
  // );

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
  auto imgPrev = cv::Mat(sizeImg, CV_8UC1);
  auto imgCurr = cv::Mat(sizeImg, CV_8UC1);
  auto imgDisplay = cv::Mat(sizeImg, CV_8UC3);
  auto imgDisplay2 = cv::Mat(sizeImg, CV_8UC3);
  auto imgFireAlarm = cv::Mat(sizeImg, CV_8UC3);
  // Buffer for Pyramid image
  cv::Size sizePyr = cvSize(sizeImg.width + 8, sizeImg.height / 3);
  auto pyrPrev = cv::Mat(sizePyr, CV_32FC1);
  auto pyrCurr = cv::Mat(sizePyr, CV_32FC1);
  std::vector<cv::Point2f> featuresPrev(_max_corners);
  std::vector<cv::Point2f> featuresCurr(_max_corners);
  cv::Size sizeWin = cv::Size(WIN_SIZE, WIN_SIZE);
  auto imgEig = cv::Mat(sizeImg, CV_32FC1);
  auto imgTemp = cv::Mat(sizeImg, CV_32FC1);

  // Pyramid Lucas-_max_corners
  std::vector<uchar> featureFound(_max_corners);
  std::vector<float> featureErrors(_max_corners);

  // Go to the end of the AVI
  capture.set(CV_CAP_PROP_POS_AVI_RATIO, 1.0);

  // Now that we're at the end, read the AVI position in frames
  long NumberOfFrames = static_cast<int>(capture.get(CV_CAP_PROP_POS_FRAMES) - 1);

  // Return to the beginning
  capture.set(CV_CAP_PROP_POS_FRAMES, 0.0);

  cout << NumberOfFrames << endl;

  // notify the current frame
  unsigned long currentFrame = 0;
  // write as video
  //	CvVideoWriter* writer = cvCreateVideoWriter(OutputVideoPath, -1, FPS,
  // sizeImg, 1);

  /* Morphology */
  // create morphology mask
  auto maskMorphology = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5), cv::Point(1, 2));
  /* Contour */

  /* Rect Motion */
  std::list<Centroid> listCentroid;         // Centroid container
  std::vector<OFRect> vecOFRect;            // tmp container for ofrect
  std::multimap<int, OFRect> mulMapOFRect;  // BRect container

  RectThrd rThrd = rectThrd(RECT_WIDTH_THRESHOLD, RECT_HEIGHT_THRESHOLD, CONTOUR_AREA_THRESHOLD);
  int key = 0;
  std::vector<vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierachy;
  while (key != 'x') {  // exit if user presses 'x'
    // flash
    maskRGB.setTo(cv::Scalar::all(0));
    maskHSI.setTo(cv::Scalar::all(0));
    // set frame
    capture.set(CV_CAP_PROP_POS_FRAMES, currentFrame);
    capture >> imgSrc;

    if (imgSrc.empty()) {
      break;  // exit if unsuccessful or Reach the end of the video
    }
    detectAndDraw(imgSrc, cascade, 1.2);
    // convert rgb to gray
    cv::cvtColor(imgSrc, imgGray, CV_BGR2GRAY);

    // copy for display
    imgSrc.copyTo(imgDisplay);
    imgSrc.copyTo(imgDisplay2);
    imgSrc.copyTo(imgFireAlarm);
    capture >> imgSrc;

    if (imgSrc.empty()) {
      break;
    }

    // the second frame ( gray level )
    cv::cvtColor(imgSrc, imgCurr, CV_BGR2GRAY);
    cv::Mat imgDiff;

    imgBackgroundModel.convertTo(imgBackgroundModel, CV_8UC1);
    cv::absdiff(imgGray, imgBackgroundModel, imgDiff);  //      cvShowImage( "cvAbsDiff", imgDiff );
    // imgDiff > standarDeviationx
    bgs.backgroundSubtraction(imgDiff, imgStandardDeviation, maskMotion);  // cvShowImage( "maskMotion", maskMotion );

    // sprintf( outfile, ImgForegroundSavePath, currentFrame );
    // cvSaveImage( outfile, maskMotion );

    /* Step2: Chromatic Filtering */

    //std::string ty = mat2str(imgDisplay);
    //printf("imgDisplay: %s %dx%d \n", ty.c_str(), imgDisplay.cols, imgDisplay.rows);
    //ty = mat2str(imgRGB);
    //printf("imgRGB:  %s %dx%d \n", ty.c_str(), imgRGB.cols, imgRGB.rows);
    /* RGB */
    imgDisplay.copyTo(imgRGB);
    checkByRGB(imgDisplay, maskMotion, maskRGB);
    // markup the fire-like region
    regionMarkup(imgDisplay, imgRGB, maskRGB);

#if defined(DEBUG_MODE) && (DEBUG_MODE == ON)
    //	cvShowImage("Chromatic Filtering-RGB Model", imgRGB);
#endif

    /* HSI */
    imgDisplay.copyTo(imgHSI);
    // convert rgb to hsiso
    /*  std::string t;
      t = mat2str(imgDisplay);
      cout << "imgDisp: " << t <<endl;
      t = mat2str(bufHSI);
      cout << "HSI: " << t << endl;
      t= mat2str(maskRGB);
      cout << "maskrgb: " << t << endl;
  */
    RGB2HSIMask(imgDisplay, bufHSI, maskRGB);
    checkByHSI(imgDisplay, bufHSI, maskRGB, maskHSI);
    regionMarkup(imgDisplay, imgHSI, maskHSI);

#if defined(DEBUG_MODE) && (DEBUG_MODE == ON)
    //   cvShowImage("Chromatic Filtering- HSI Model", imgHSI);
#endif
    maskHSI.copyTo(maskRGB);
    /* Step3: Background Model & Threshold update */

    // flip maskMotion 0 => 255, 255 => 0
    bgs.maskNegative(maskMotion);

    /* Background update */

    // 8U -> 32F
    imgBackgroundModel.convertTo(img32FBackgroundModel, CV_32FC1);
    // B( x, y; t+1 ) = ( 1-alpha )B( x, y; t ) + ( alpha )Src( x, y; t ),
    // if the pixel is stationary

    accumulateWeighted(imgGray, img32FBackgroundModel, ACCUMULATE_WEIGHTED_ALPHA_BGM, maskMotion);
    // 32F -> 8U
    img32FBackgroundModel.convertTo(imgBackgroundModel, CV_8UC1);
    /* Threshold update */
    // 8U -> 32F

    imgStandardDeviation.convertTo(img32FStandardDeviation, CV_32FC1);
    // T( x, y; t+1 ) = ( 1-alpha )T( x, y; t ) + ( alpha ) | Src( x, y; t )/
    // - B( x, y; t ) |, if the pixel is stationary
    accumulateWeighted(imgDiff, img32FStandardDeviation, ACCUMULATE_WEIGHTED_ALPHA_THRESHOLD, maskMotion);
    // 32F -> 8U
    img32FStandardDeviation.convertTo(imgStandardDeviation, CV_8UC1);

    /* Step4: Morphology */
    cv::dilate(maskHSI, maskHSI, maskMorphology);

#if defined(DEBUG_MODE) && (DEBUG_MODE == ON)
    // cvShowImage("Morphology-Dilate", maskHSI);
#endif
    /* Step5: matching fire-like object */
    /* find contours */
    cv::findContours(maskHSI, contours, hierachy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    /* assign feature points and get the number of feature */
    auto ContourFeaturePointCount =
        getContourFeatures(imgDisplay2, imgDisplay, contours, vecOFRect, rThrd, featuresPrev, hierachy);
    // Pyramid L-K Optical Flowo
    //ty = mat2str(imgGray);
    //printf("imgGray: %s %dx%d \n", ty.c_str(), imgGray.cols, imgGray.rows);

    //ty = mat2str(imgCurr);
    //printf("imgCurr: %s %dx%d \n", ty.c_str(), imgCurr.cols, imgCurr.rows);
    cv::calcOpticalFlowPyrLK(imgGray, imgCurr, featuresPrev, // the feature points that needed to be found(trace)
        featuresCurr,  // the feature points that be traced
        // ContourFeaturePointCount, // the number of feature points
        featureFound,  // notify whether the feature points be traced or not
        featureErrors, sizeWin,  // searching window size
        2,        // using pyramid layer 2: will be 3 layers
        TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.3)  // iteration criteria

    );

    // Display the flow field
#if defined(OFMV_DISPLAY) && (OFMV_DISPLAY == ON)

    drawArrow(imgDisplay2, featuresPrev, featuresCurr, ContourFeaturePointCount, featureFound);
    // cvShowImage("Optical Flow", imgDisplay2);
#endif
    /* Save the OFMV image */
#if 0
    char str[30];
    sprintf(str, "MotionVector//%d.bmp", currentFrame);
    cvSaveImage(str, imgDisplay2);
#endif
    /* assign feature points to fire-like obj and then push to multimap */
    assignFeaturePoints(mulMapOFRect, vecOFRect, featureFound, featuresPrev, featuresCurr);
    /* compare the mulMapOFRect space with listCentroid space, if matching
     * insert to listCentroid space as candidate fire-like obj */
    matchCentroid(imgDisplay,
                  imgFireAlarm,
                  listCentroid,
                  mulMapOFRect,
                  static_cast<int>(currentFrame++),
                  CONTOUR_POINTS_THRESHOLD,
                  PROCESSING_WINDOWS);
    cv::imshow("Fire Alarm", imgFireAlarm);
    // cvWriteFrame(writer, imgFireAlarm);
    // cout << "< Frame >: " << currentFrame++ << endl;
    key = cv::waitKey(5);

    /* Don't run past the end of the AVI. */
    if (currentFrame == NumberOfFrames) {
      break;
    }
  }
  // release memory
  imgFireAlarm.release();
  imgTemp.release();
  imgEig.release();
  imgPrev.release();
  imgCurr.release();
  imgDisplay.release();
  imgDisplay2.release();
  pyrPrev.release();
  pyrCurr.release();
  maskMotion.release();
  maskHSI.release();
  maskRGB.release();
  imgRGB.release();
  imgHSI.release();
  bufHSI.release();
  imgGray.release();

  img32FBackgroundModel.release();
  img32FStandardDeviation.release();
  maskMorphology.release();
  cvDestroyAllWindows();
  capture.release();
  return 0;
}
