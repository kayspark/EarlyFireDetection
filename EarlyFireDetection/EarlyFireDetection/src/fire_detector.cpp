//
// Created by 박기수 on 2018-12-10.
//

#include "motionDetection.h"
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/types_c.h>
#include "fire_detector.h"

fire_detector::fire_detector(cv::Size &imgSize)
    :
    tracker(cv::TrackerMedianFlow::create()),
    initialized_tracker(false),
    detected_area(cv::Rect2d(0, 0, 0, 0)),
    BGM_FRAME_COUNT(0),
    _win_size(5),
    RECT_WIDTH_THRESHOLD(5),
    RECT_HEIGHT_THRESHOLD(5),
    CONTOUR_POINTS_THRESHOLD(12),
    CONTOUR_AREA_THRESHOLD(12),
    ACCUMULATE_WEIGHTED_ALPHA_BGM(0.1),
    ACCUMULATE_WEIGHTED_ALPHA_THRESHOLD(0.05),
    THRESHOLD_COEFFICIENT(5),
    listCentroid(std::list<Centroid>()),
    vecOFRect(std::vector<OFRect>()),
    mulMapOFRect(std::multimap<int, OFRect>()),
    maskMorphology(cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(3, 5), cv::Point(1, 2))),
    featuresPrev(std::vector<cv::Point2f>(_max_corners)),
    featuresCurr(std::vector<cv::Point2f>(_max_corners)),
    featureFound(std::vector<uchar>(_max_corners)),
    featureErrors(std::vector<float>(_max_corners)),
    contours(std::vector<std::vector<cv::Point>>()),
    hierachy(std::vector<cv::Vec4i>()),
    bufHSI(cv::Mat(imgSize, CV_64FC3)),
    imgRGB(cv::Mat(imgSize, CV_8UC3)),
    imgHSI(cv::Mat(imgSize, CV_8UC3)),
    maskRGB(cv::Mat(imgSize, CV_8UC1)),
    maskHSI(cv::Mat(imgSize, CV_8UC1)) {

  sizeWin = cv::Size(_win_size, _win_size);
  _rect_thresh = rectThrd(RECT_WIDTH_THRESHOLD, RECT_HEIGHT_THRESHOLD,
                          CONTOUR_AREA_THRESHOLD);

  /* Rect Motion */

}

//TODO: verfiy logics
bool fire_detector::update_tracker(cv::Mat &img) {
  bool ret = !detected_area.empty();
  //initialized already
  if (initialized_tracker) {
    if (tracker->update(img, detected_area))
      cv::rectangle(img, detected_area, cv::Scalar(0, 0, 255), 3);
    else { //if cannot find
      detected_area = cv::Rect2d(0, 0, 0, 0);
      tracker = cv::TrackerMedianFlow::create();
      initialized_tracker = false;
    }
  } else {
    if (ret) {// detected area
      // initializes the tracker
      if (!tracker->init(img, detected_area)) {
        std::cout << "***Could not initialize tracker...***\n";
        return ret;
      }
      initialized_tracker = true;
    }
  }

  // true if not empty
  return ret;

}
bool fire_detector::checkContourPoints(Centroid &ctrd, const int thrdcp,
                                       const unsigned int pwindows) {
  auto countFrame =
      // contour points of each frame
      std::count_if(
          ctrd.dOFRect.begin(), ctrd.dOFRect.end(),
          [&thrdcp](const auto &itrDeq) { return (itrDeq.size() < thrdcp); });
  bool out = countFrame < pwindows / 3;
  if (out) {
    std::cout << "countours are likely" << countFrame << " , " << pwindows / 3
              << std::endl;
  }
  return out;
}
/* accumulate the motin vector depends on its orientation( based on 4 directions
) input: vecFeature : Contour Features orient      : accumulate array output :
orien[4]
*/
void fire_detector::motionOrientationHist(std::vector<Feature> &vecFeature,
                                          std::vector<unsigned int> &orient) {
  // std::vector<Feature>::iterator itrVecFeature;
  /* each point of contour  */
  std::for_each(vecFeature.begin(), vecFeature.end(),
                [&orient](const Feature &feature) {
                  /* orientation */
                  if (feature.prev.x >= feature.curr.x) {
                    if (feature.prev.y >= feature.curr.y) {
                      ++orient[0]; // up-left
                    } else {
                      ++orient[2]; // down-left
                    }
                  } else {
                    if (feature.prev.y >= feature.curr.y) {
                      ++orient[1]; // up-right
                    } else {
                      ++orient[3]; // down-right
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
double fire_detector::getEnergy(std::vector<Feature> &vecFeature, unsigned int &staticCount,
                                unsigned int &totalPoints) {
  /* initialization */
  double energy = 0.0;
  /* each contour point */
  for_each(vecFeature.begin(), vecFeature.end(),
           [&staticCount, &energy, &totalPoints](const auto &feature) {
             /* energy */
             double tmp = pow(abs(feature.curr.x - feature.prev.x), 2) +
                 pow(abs(feature.curr.y - feature.prev.y), 2);
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
bool fire_detector::checkContourEnergy(Centroid &ctrd, const unsigned int pwindows) {
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
  static const unsigned int thrdPassFrame = pwindows >> 1,
      thrdStaticFrame = pwindows >> 2,
      thrdOrienFrame = (pwindows >> 3) + 1;

  bool out = staticFrame < thrdStaticFrame
             ? passFrame > thrdPassFrame && orientFrame < thrdOrienFrame
             : false;
  /*   if (out)
      std::cout << "energy is likely " << std::endl; */
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
void fire_detector::matchCentroid(cv::Mat &imgCenteroid, cv::Mat &img) {
  listCentroid.remove_if([this, &img](Centroid &centre) {
    bool out = false;
    /* visit mulMapOFRect between range [itlow,itup) */
    for (auto &aRect : mulMapOFRect) {
      const cv::Rect &rect = (aRect).second.rect;
      cv::Rect rectFire = cv::Rect(0, 0, 0, 0);
      /* matched */
      if (centre.centroid.y >= rect.y &&
          (rect.x + rect.width) >= centre.centroid.x &&
          (rect.y + rect.height) >= centre.centroid.y) {
        /* push rect to the matched listCentroid node */
        centre.vecRect.emplace_back(rect);
        /* push vecFeature to matched listCentroid node */
        centre.dOFRect.emplace_back((aRect).second.vecFeature);
        /* Update countFrame and judge the threshold of it */
        if (++(centre.countFrame) == PROCESSING_WINDOWS) {
          /* GO TO PROCEESING DIRECTION MOTION */
          if (!judgeDirectionsMotion(centre.vecRect, rectFire))
            break;
          if (checkContourPoints(centre, CONTOUR_POINTS_THRESHOLD, PROCESSING_WINDOWS) &&
              checkContourEnergy(centre, PROCESSING_WINDOWS)) {
            /* recting the fire region */
            // cv::rectangle(img, rectFire, cv::Scalar(255, 100, 0), 3);
            //        cv::putText(img, "Fire !!", rectFire.tl(), 2, 1.2,
            //                    cv::Scalar(0, 0, 255));
            std::cout << "Fire Alarm: " << std::endl;
            detected_area = rectFire;
            //            cv::imshow("Video", imgFireAlarm);
          } else {
            break; // if not on fire go to erase it
          }
          /* mark this rect as matched */
        }
        aRect.second.match = true;
        out = true;
        // ++itCentroid;
        break; // if matched break the inner loop
      }
      // if ended the map rect and not matched anyone go to erase it
    } // for (multimapBRect)
    return !out;
  });
  /* push new rect to listCentroid */
  std::for_each(mulMapOFRect.begin(), mulMapOFRect.end(),
                [this](const auto &rect) {
                  if (!rect.second.match) {
                    /* push new node to listCentroid */
                    listCentroid.emplace_back(centroid(rect.second));
                    // cout << "after rect: " << endl;
                    // cout << (*itBRect).second << endl;	x
                  }
                });
  // cout <<"after list count: "<< listCentroid.size() << endl;
  /* check the list node with image */
  std::for_each(listCentroid.begin(), listCentroid.end(),
                [&imgCenteroid](const auto &centre) {
                  cv::rectangle(imgCenteroid,
                                cv::Point(centre.centroid.x, centre.centroid.y),
                                cv::Point((centre.centroid.x) + 2,
                                          (centre.centroid.y) + 2),
                                cv::Scalar(0, 0, 0), 3);
                });
  /* clear up container */
  mulMapOFRect.clear();
}

/* get the feature points from contour
input:
imgDisplayCntr      : img for display contours
imgDisplayFireRegion: img for boxing the fire-like region with rectangle
contour             : after cvFindContour()
trd                 : threshold
output:
vecOFRect           : fire-like obj will be assign to this container
featuresPrev        : previous contours points
featuresCurr        : current contours points
return:
the number of contour points
*/
int fire_detector::getContourFeatures(cv::Mat &img, cv::Mat &display) {
  static unsigned int countCtrP;
  auto ContourFeaturePointCount = 0;
  /* thresholding on connected component */
  for (int index = 0; index < contours.size();
       index++) { // contours-based visiting
    /* Recting the Contour with smallest rectangle */
    cv::Rect rect_ = boundingRect(contours[index]);
    /* checking the area */
    if (((rect_.width > _rect_thresh.rectWidth) && (rect_.height > _rect_thresh.rectHeight)) &&
        (fabs(contourArea(contours[index])) > _rect_thresh.cntrArea)) {
      /* Drawing the Contours */
      /*    cv::drawContours(img, contours, index, cv::Scalar(250, 0, 0), // Red
                          2,            // Vary max_level and compare results
                          8, hierachy); // line type */
      /* Drawing the region */
      rectangle(display, rect_, cv::Scalar(255, 0, 0), 1);
      /* for each contours pixel count	*/
      countCtrP = 0;
      /* access points on each contours */
      // printf(" (%d,%d)\n", p->x, p->y );
      for (int i = 0; i < contours[index].size(); i++) {
        // const auto &p : contours[index]) {
        const auto &p = contours[index][i];
        featuresPrev[i] = p;
        ++countCtrP;
        ++ContourFeaturePointCount;
      }
      /* push to tmp vector for quick access ofrect node */
      vecOFRect.emplace_back(ofRect(rect_, countCtrP));
    }
  }

  return ContourFeaturePointCount;
}
/* assign feature points to fire-like obj and then push to multimap
input:
vecOFRect:      fire-like obj
status:			the feature stutas(found or not) after tracking
featuresPrev:	previous feature points
featuresCurr:   current feature points after tracking

output:
mulMapOFRect:	new candidate fire-like obj in current frame(with rectangle and
motion vector information)

*/
void fire_detector::assignFeaturePoints() {
  // visit each ofrect in vecOFRect
  for_each(vecOFRect.begin(), vecOFRect.end(),
           [this](auto &aRect) {
             // contour points count
             for (int i = 0; i < aRect.countCtrP; ++i) {
               // if the feature point was be found
               if (featureFound[i] == 0) {
                 continue;
               } else {
                 /* push feature to vector of ofrect */
                 aRect.vecFeature.emplace_back(
                     feature(featuresPrev[i], featuresCurr[i]));
               }
             }
             /* insert ofrect to multimap */
             mulMapOFRect.insert(std::pair<int, OFRect>(aRect.rect.x, aRect));
           });

  /* clear up container */
  vecOFRect.clear();
}
/* Counting the foldback point at each directions */
void fire_detector::foldbackPoint(const std::vector<cv::Rect> &vecRect,
                                  DirectionsCount &count) {
  if (vecRect.size() <= 2)
    std::cerr << "logical errors" << std::endl;
  for (auto i = 1; i < vecRect.size() - 1; i++) {
    const auto rn = vecRect[i - 1];
    auto it = vecRect[i];
    const auto rp = vecRect[i + 1];
    if ((rn.y - it.y) * (it.y - rp.y) < 0) {
      ++count.countUp;
    }
    if ((rn.x - it.x) * (it.x - rp.x) < 0) {
      ++count.countLeft;
    }
    if (((rn.y + rn.height) - (it.y + it.height)) *
        ((it.y + it.height) - (rp.y + rp.height)) <
        0) {
      ++count.countDown;
    }
    if (((rn.x + rn.width) - (it.x + it.width)) *
        ((it.x + it.width) - (rp.x + rp.width)) <
        0) {
      ++count.countRight;
    }
  }
}
/* Analysis the rect information */
bool fire_detector::judgeDirectionsMotion(const std::vector<cv::Rect> &vecRect,
                                          cv::Rect &rectFire) {
  DirectionsCount count;
  zeroCount(count);
  foldbackPoint(vecRect, count);
  const int thresh_foldback_cnt = 3; // 3
  /* Direction Up required to be growth and sparkle */
  if ((vecRect.front().y - vecRect.back().y) > 2 &&
      count.countUp >= thresh_foldback_cnt) {
    /* set up the last rect to rect the frame */
    rectFire = vecRect.back();
    // std::cout << "by directions likely to be fire" << std::endl;
    return true;
  } else {
    return false;
  }
}
void fire_detector::calcOpticalFlow(cv::Mat &gray, cv::Mat &curr) {
  cv::calcOpticalFlowPyrLK(
      gray, curr,
      featuresPrev, // the feature points that needed to be found(trace)
      featuresCurr, // the feature points that be traced
      // ContourFeaturePointCount, // the number of feature points
      featureFound, // notify whether the feature points be traced or not
      featureErrors, sizeWin, // searching window size
      2,                      // using pyramid layer 2: will be 3 layers
      cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20,
                       0.3) // iteration criteria
  );
}
void fire_detector::findContours(cv::Mat &mask) {
  cv::findContours(mask, contours, hierachy, CV_RETR_TREE,
                   CV_CHAIN_APPROX_NONE);
}

void fire_detector::dilate(cv::Mat &mask) {
  cv::dilate(mask, mask, maskMorphology);
}
/**
 *	@Purpose: check fire-like pixels by rgb model base on reference method
 *			  This function will change fire-like pixels to red
 *	@Parameter:
 *		frame: input source image
 *		mask: output mask
 */
void fire_detector::checkByRGB(const cv::Mat &imgSrc, const cv::Mat &maskMotion,
                               cv::Mat &maskRGB) {
  static const int RT = 250;
  const static uint8_t red_ = 255;
  for (int i = 0; i < imgSrc.rows; ++i) {
    const auto ptr = imgSrc.ptr<_normal_pixel>(i);
    const auto mMaskMotion = maskMotion.ptr<_short_pixel>(i);
    auto mRGB = maskRGB.ptr<_short_pixel>(i);
    for (int j = 0; j < imgSrc.cols; j++) {
      if (mMaskMotion[j] == 255 && (ptr[j].z > RT) && (ptr[j].z >= ptr[j].y) &&
          (ptr[j].y > ptr[j].x)) { // RGB color model determine rule
        mRGB[j] = red_;
      }
    }
  }
}
/**
 *   @ Function: Convert RGB to HSI
 *   H: 0~360(degree)  HUE_R = 0 , HUE_G = 120 , HUE_B = 240
 *   S&I: 0~1
 *   @Parameter:       all img require same size
 *
 *                          [depth]           [channel]
 *		     imgRGB:     CV8UC3                  3
 *		     imgHSI:     CV64FC3			     3
 *		     maskRGB:    CV8UC1      			 1
 */
void fire_detector::RGB2HSIMask(cv::Mat &imgRGB, cv::Mat &imgHSI, cv::Mat &maskRGB) {
  static const double efs_ = 0.000000000000001;               // acceptable bias
  static const double div13_ = 0.333333333333333333333333333; // 1/3
  static const double div180PI = 180 / CV_PI;                 // (180 / PI)

  static cv::Mat imgTemp;
  imgTemp.create(imgRGB.size(), CV_64FC3); // every times
  static double tmp1 = 0.0, tmp2 = 0.0, x = 0.0, theta = 0.0, tmpAdd = 0.0;
  // normalize rgb to [0,1]
  for (int i = 0; i < imgRGB.rows; ++i) {
    auto tmp = imgTemp.ptr<_long_pixel>(i);
    const auto mRGB = maskRGB.ptr<_short_pixel>(i);
    const auto img = imgRGB.ptr<_normal_pixel>(i);
    auto hsi = imgHSI.ptr<_long_pixel>(i);
    for (int j = 0; j < imgRGB.cols; j++) { // loop times = width
      if (mRGB[j] == 255) {                 // if the pixel is moving object
        tmp[j].x = img[j].x / 255.0;        // tmp[ k ] = img[ k ] / 255.0;
        tmp[j].y = img[j].y / 255.0; // tmp[ k + 1 ] = img[ k + 1 ] / 255.0;
        tmp[j].z = img[j].z / 255.0;
        // IF ( R = G = B ) , IN INTENSITY AXIS THERE IS NO SATURATRION ,AND NO
        // DEFINE HUE VALUE
        if (fabs(tmp[j].z - tmp[j].y) < efs_ &&
            fabs(tmp[j].y - tmp[j].x) < efs_) {
          hsi[j].x = -1.0; // UNDEFINE
          hsi[j].y = 0.0;
          hsi[j].z = tmp[j].x;
        } else {
          tmpAdd = tmp[j].x + tmp[j].y + tmp[j].z;
          tmp1 = tmp[j].z - tmp[j].y; // r-g
          tmp2 = tmp[j].z - tmp[j].x; // r-b
          x = 0.5 * (tmp1 + tmp2) /
              (sqrt(pow(tmp1, 2) + tmp2 * (tmp[j].y - tmp[j].x)));
          // exam
          if (x < -1.0) {
            x = -1.0;
          }
          if (x > 1.0) {
            x = 1.0;
          }
          theta = div180PI * acos(x);

          if (tmp[j].x <= tmp[j].y) {
            hsi[j].x = theta;
          } else {
            hsi[j].x = 360.0 - theta;
          }
          hsi[j].y =
              1.0 - (3.0 / tmpAdd) * (minrgb(tmp[j].x, tmp[j].y, tmp[j].z));
          hsi[j].z = div13_ * tmpAdd;
        }
      }
    }
  }
  imgTemp.release();
}
/**
 *	@Purpose: check fire-like pixels by rgb model base on reference method
 *			  This function will change fire-like pixels to red
 *	@Parameter:
 *		frame: input source image
 *		mask: output mask
 */
void fire_detector::checkByHSI(cv::Mat &imgRGB, cv::Mat &imgHSI, cv::Mat &maskRGB,
                               cv::Mat &maskHSI) {
  /* HSI threshold */
  static const int trdH = 60;
  static const double trdS = 0.003043487826087;
  static const double trdI = 0.588235294117647;
  // static const int stepImgRGB = imgRGB.step / sizeof(uchar);
  for (int i = 0; i < imgHSI.rows; ++i) {
    const auto mRGB = maskRGB.ptr<_short_pixel>(i);
    const auto img = imgRGB.ptr<_normal_pixel>(i);
    const auto hsi = imgHSI.ptr<_long_pixel>(i);
    auto mHSI = maskHSI.ptr<_short_pixel>(i);
    for (int j = 0; j < imgHSI.cols; ++j) { // stepImg = imgWidth * channel
      if (mRGB[j] == 255 && hsi[j].x <= trdH && hsi[j].x >= 0 &&
          hsi[j].z > trdI &&
          hsi[j].y >=
              (255 - img[j].z) * trdS) { // HSI color model determine rule
        mHSI[j] = static_cast<_short_pixel>(255);
      }
    }
  }
}
/**
 *	@Function: markup the intrest region based on mask
 *  @Parameter
 *		src: input image
 *		backup: output image (for display)
 *		mask: input mask
 */
void fire_detector::regionMarkup(cv::Mat &imgSrc, cv::Mat &imgBackup, cv::Mat &mask) {
  for (int i = 0; i < imgSrc.rows; ++i) {
    const auto m = mask.ptr<_short_pixel>(i);
    auto hsi = imgBackup.ptr<_normal_pixel>(i);
    for (int j = 0; j < imgSrc.cols; ++j) {
      if (255 == m[j]) {
        hsi[j].x = static_cast<uint8_t>(0);
        hsi[j].y = static_cast<uint8_t>(0);
        hsi[j].z = static_cast<uint8_t>(255);
      }
    }
  }
}
void fire_detector::detectFire(
    cv::Mat &maskMotion,
    motionDetection &bgs,
    cv::Mat &imgBackgroundModel,
    cv::Mat &imgStandardDeviation,
    cv::Mat &img32FBackgroundModel,
    cv::Mat &img32FStandardDeviation,
    cv::Mat &imgSrc,
    cv::Mat &imgGray,
    cv::Mat &imgDisplay) {
  if (imgSrc.empty())
    return;
  // flash
  maskRGB.setTo(cv::Scalar::all(0));
  maskHSI.setTo(cv::Scalar::all(0));

  // Optical FLow
  auto imgCurr = cv::Mat(imgSrc.size(), CV_8UC1);
  cvtColor(imgSrc, imgCurr, CV_BGR2GRAY);
  cv::Mat imgDiff;

  imgBackgroundModel.convertTo(imgBackgroundModel, CV_8UC1);
  absdiff(imgGray, imgBackgroundModel, imgDiff);
  // imgDiff > standarDeviationx
  bgs.backgroundSubtraction(imgDiff, imgStandardDeviation, maskMotion);
  setup_motion_model(maskMotion, imgDisplay);
  // flip maskMotion 0 => 255, 255 => 0
  bgs.maskNegative(maskMotion);
  /* Background update */
  // 8U -> 32F
  imgBackgroundModel.convertTo(img32FBackgroundModel, CV_32FC1);
  accumulateWeighted(imgGray, img32FBackgroundModel,
                     get_accumulate_weighted_alpha_bgm(),
                     maskMotion);
  // 32F -> 8U
  img32FBackgroundModel.convertTo(imgBackgroundModel, CV_8UC1);
  /* Threshold update */
  // 8U -> 32F
  imgStandardDeviation.convertTo(img32FStandardDeviation, CV_32FC1);
  // T( x, y; t+1 ) = ( 1-alpha )T( x, y; t ) + ( alpha ) | Src( x, y; t )/
// - B( x, y; t ) |, if the pixel is stationary
  accumulateWeighted(imgDiff, img32FStandardDeviation,
                     get_accumulate_weighted_alpha_threshold(),
                     maskMotion);
  // 32F -> 8U
  img32FStandardDeviation.convertTo(imgStandardDeviation, CV_8UC1);
  /* Step4: Morphology */
  dilate(maskHSI);
  findContours(maskHSI);
  getContourFeatures(imgDisplay, imgDisplay);
  calcOpticalFlow(imgGray, imgCurr);
  assignFeaturePoints();
  matchCentroid(imgSrc, imgDisplay);
}
void fire_detector::setup_motion_model(const cv::Mat &maskMotion, cv::Mat &imgDisplay) {
  imgDisplay.copyTo(imgRGB);
  checkByRGB(imgDisplay, maskMotion, maskRGB);
  // markup the fire-like region
  regionMarkup(imgDisplay, imgRGB, maskRGB);
  /* HSI */
  imgDisplay.copyTo(imgHSI);
  RGB2HSIMask(imgDisplay, bufHSI, maskRGB);
  checkByHSI(imgDisplay, bufHSI, maskRGB, maskHSI);
  regionMarkup(imgDisplay, imgHSI, maskHSI);
  maskHSI.copyTo(maskRGB);
}