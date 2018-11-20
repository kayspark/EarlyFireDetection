#include <iostream>
#include "colorModel.h"

/**
 *	@Purpose: check fire-like pixels by rgb model base on reference method
 *			  This function will change fire-like pixels to red
 *	@Parameter:
 *		frame: input source image
 *		mask: output mask
 */
void checkByRGB(const cv::Mat &imgSrc, const cv::Mat &maskMotion, cv::Mat &maskRGB) {
  static const int RT = 250;
  const static uint8_t RED = 255;
  for (int i = 0; i < imgSrc.rows; ++ i) {
    const auto ptr = imgSrc.ptr<_normal_pixel>(i);
    const auto mMaskMotion = maskMotion.ptr<_short_pixel>(i);
    auto mRGB = maskRGB.ptr<_short_pixel>(i);
    for (int j = 0; j < imgSrc.cols; j ++) {
      if (mMaskMotion[j] == 255 && (ptr[j].z > RT) && (ptr[j].z >= ptr[j].y)
          && (ptr[j].y > ptr[j].x)) {  // RGB color model determine rule
        mRGB[j] = RED;
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
 *		     imgRGB:     IPL_DEPTH_8U			 3
 *		     imgHSI:     IPL_DEPTH_64F			 3
 *		     maskRGB:    IPL_DEPTH_8U			 1
 */
void RGB2HSIMask(cv::Mat &imgRGB, cv::Mat &imgHSI, cv::Mat &maskRGB) {
  static const double EFS = 0.000000000000001;                // acceptable bias
  static const double DIV13 = 0.333333333333333333333333333;  // 1/3
  static const double DIV180PI = 180 / CV_PI;                 // (180 / PI)

  // Temp buffer for H S I spectrum
  static cv::Mat imgTemp;
  imgTemp.create(imgRGB.size(), CV_64FC3);  // every times
  static double tmp1 = 0.0, tmp2 = 0.0, x = 0.0, theta = 0.0, tmpAdd = 0.0;
  // normalize rgb to [0,1]
  for (int i = 0; i < imgRGB.rows; ++ i) {
    auto tmp = imgTemp.ptr<_long_pixel>(i);
    const auto mRGB = maskRGB.ptr<_short_pixel>(i);
    const auto img = imgRGB.ptr<_normal_pixel>(i);
    auto hsi = imgHSI.ptr<_long_pixel>(i);
    for (int j = 0; j < imgRGB.cols; j ++) {  // loop times = width
      if (mRGB[j] == 255) {  // if the pixel is moving object
        tmp[j].x = img[j].x / 255.0;  // tmp[ k ] = img[ k ] / 255.0;
        tmp[j].y = img[j].y / 255.0;  // tmp[ k + 1 ] = img[ k + 1 ] / 255.0;
        tmp[j].z = img[j].z / 255.0;

        // IF ( R = G = B ) , IN INTENSITY AXIS THERE IS NO SATURATRION ,AND NO
        // DEFINE HUE VALUE
        if (fabs(tmp[j].z - tmp[j].y) < EFS && fabs(tmp[j].y - tmp[j].x) < EFS) {
          hsi[j].x = - 1.0;  // UNDEFINE
          hsi[j].y = 0.0;
          hsi[j].z = tmp[j].x;
        } else {
          tmpAdd = tmp[j].x + tmp[j].y + tmp[j].z;
          tmp1 = tmp[j].z - tmp[j].y;  // r-g
          tmp2 = tmp[j].z - tmp[j].x;      // r-b
          x = 0.5 * (tmp1 + tmp2) / (sqrt(pow(tmp1, 2) + tmp2 * (tmp[j].y - tmp[j].x)));
          // exam
          if (x < - 1.0) {
            x = - 1.0;
          }
          if (x > 1.0) {
            x = 1.0;
          }
          theta = DIV180PI * acos(x);

          if (tmp[j].x <= tmp[j].y) {
            hsi[j].x = theta;
          } else {
            hsi[j].x = 360.0 - theta;
          }
          hsi[j].y = 1.0 - (3.0 / tmpAdd) * (minrgb(tmp[j].x, tmp[j].y, tmp[j].z));
          hsi[j].z = DIV13 * tmpAdd;
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
void checkByHSI(cv::Mat &imgRGB, cv::Mat &imgHSI, cv::Mat &maskRGB, cv::Mat &maskHSI) {
  /* HSI threshold */
  static const int trdH = 60;
  static const double trdS = 0.003043487826087;
  static const double trdI = 0.588235294117647;
  // static const int stepImgRGB = imgRGB.step / sizeof(uchar);
  for (int i = 0; i < imgHSI.rows; ++ i) {
    const auto mRGB = maskRGB.ptr<_short_pixel>(i);
    const auto img = imgRGB.ptr<_normal_pixel>(i);
    const auto hsi = imgHSI.ptr<_long_pixel>(i);
    auto mHSI = maskHSI.ptr<_short_pixel>(i);
    for (int j = 0; j < imgHSI.cols; ++ j) {  // stepImg = imgWidth * channel
      if (mRGB[j] == 255 && hsi[j].x <= trdH && hsi[j].x >= 0 && hsi[j].z > trdI
          && hsi[j].y >= (255 - img[j].z) * trdS) {  // HSI color model determine rule
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
void regionMarkup(cv::Mat &imgSrc, cv::Mat &imgBackup, cv::Mat &mask) {

  for (int i = 0; i < imgSrc.rows; ++ i) {
    const auto m = mask.ptr<_short_pixel>(i);
    auto hsi = imgBackup.ptr<_normal_pixel>(i);
    for (int j = 0; j < imgSrc.cols; ++ j) {
      if (255 == m[j]) {
        hsi[j].x = static_cast<uint8_t>(0);
        hsi[j].y = static_cast<uint8_t>(0);
        hsi[j].z = static_cast<uint8_t>(255);
      }
    }
  }
}
