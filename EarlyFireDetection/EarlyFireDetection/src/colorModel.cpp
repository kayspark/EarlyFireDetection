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
  static const int step = static_cast<const int>(imgSrc.step / sizeof(uchar));
  static auto dataSrc = imgSrc.data;
  static const int RT = 250;

  // mask
  static const int stepMask = static_cast<const int>(maskRGB.step / sizeof(uchar));
  static auto dataMask = maskRGB.data;
  static auto dataMaskMotion = maskMotion.data;

  static int i = 0, j = 0, k = 0, idx = 0;
  for (i = 0; i < imgSrc.rows; ++ i) {
    for (j = 0, k = 0; j < step; j += 3, ++ k) {
      idx = i * step + j;
      if (dataMaskMotion[i * stepMask + k] == 255 && dataSrc[idx + 2] > RT && dataSrc[idx + 2] >= dataSrc[idx + 1]
          && dataSrc[idx + 1] > dataSrc[idx]) {  // RGB color model determine rule
        dataMask[i * stepMask + k] = static_cast<uchar>(255);
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
  static cv::Mat imgTemp(imgRGB.size(), CV_64FC3, cv::Scalar());  // every times

  static auto step = static_cast<int>(imgTemp.step / sizeof(double));

  static auto dataTmp = reinterpret_cast<double *>(imgTemp.data);  // for easy access tmp data
  static auto dataHSI = reinterpret_cast<double *>(imgHSI.data);  // for easy access hsi data
  static auto dataRGB = imgRGB.data;  // for easy access rgb data

  // mask
  static const int stepMaskRGB = static_cast<const int>(maskRGB.step / sizeof(uchar));
  static auto dataMaskRGB = maskRGB.data;

  // idx i, j, idx
  static int i = 0, j = 0, k = 0, idx = 0;
  static double tmp1 = 0.0, tmp2 = 0.0, x = 0.0, theta = 0.0, tmpAdd = 0.0;

  // normalize rgb to [0,1]
  for (i = 0; i < imgRGB.rows; ++ i) {
    for (j = 0, k = 0; j < step; j += 3, ++ k) {  // loop times = width
      if (dataMaskRGB[i * stepMaskRGB + k] == 255) {  // if the pixel is moving object
        idx = i * step + j;
        dataTmp[idx] = dataRGB[idx] / 255.0;  // dataTmp[ idx ] = dataRGB[ idx ] / 255.0;
        dataTmp[idx + 1] = dataRGB[idx + 1] / 255.0;  // dataTmp[ idx + 1 ] = dataRGB[ idx + 1 ] / 255.0;
        dataTmp[idx + 2] = dataRGB[idx + 2] / 255.0;

        // IF ( R = G = B ) , IN INTENSITY AXIS THERE IS NO SATURATRION ,AND NO
        // DEFINE HUE VALUE
        if (fabs(dataTmp[idx + 2] - dataTmp[idx + 1]) < EFS && fabs(dataTmp[idx + 1] - dataTmp[idx]) < EFS) {
          dataHSI[idx] = - 1.0;  // UNDEFINE
          dataHSI[idx + 1] = 0.0;
          dataHSI[idx + 2] = dataTmp[idx];
        } else {
          tmpAdd = dataTmp[idx] + dataTmp[idx + 1] + dataTmp[idx + 2];
          tmp1 = dataTmp[idx + 2] - dataTmp[idx + 1];  // r-g
          tmp2 = dataTmp[idx + 2] - dataTmp[idx];      // r-b
          x = 0.5 * (tmp1 + tmp2) / (sqrt(pow(tmp1, 2) + tmp2 * (dataTmp[idx + 1] - dataTmp[idx])));

          // exam
          if (x < - 1.0) {
            x = - 1.0;
          }
          if (x > 1.0) {
            x = 1.0;
          }
          theta = DIV180PI * acos(x);

          if (dataTmp[idx] <= dataTmp[idx + 1]) {
            dataHSI[idx] = theta;
          } else {
            dataHSI[idx] = 360.0 - theta;
          }
          dataHSI[idx + 1] = 1.0 - (3.0 / tmpAdd) * (minrgb(dataTmp[idx], dataTmp[idx + 1], dataTmp[idx + 2]));
          dataHSI[idx + 2] = DIV13 * tmpAdd;
        }
      }
    }
  }


  /*
   // normalize rgb to [0,1]
   for (i = 0; i < imgRGB.rows; ++i) {
     for (j = 0, k = 0; j < step; j += 3, ++k) {  // loop times = width
       if (255 != dataMaskRGB[i * stepMaskRGB + k])
         continue;  // if the pixel is moving object
       idx = i * step + j;
       dataTmp[idx] = dataRGB[idx] / 255.0;  // dataTmp[ idx ] = dataRGB[ idx ] / 255.0;
       dataTmp[idx+1] = dataRGB[idx+1] / 255.0;  // dataTmp[ idx + 1 ] = dataRGB[ idx + 1 ] / 255.0;
       dataTmp[idx+2] = dataRGB[idx+2 ] / 255.0;
     }
   }
 for (i = 0; i  < imgRGB.rows; ++i) {
     auto * rgb = imgRGB.ptr<_normal_pixel>(0, i);
     auto * mask = maskRGB.ptr<_normal_pixel>(0, i);
     auto* temp = imgTemp.ptr<dPixel>(0,i);
     auto* hsi = imgHSI.ptr<dPixel>(0,i);
     const _normal_pixel* ptr_end = rgb + imgRGB.cols;
     for (; rgb != ptr_end; ++rgb, ++mask, ++temp, ++hsi) {
        if (255 != mask->x)
          continue;

   }
   */

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
  static const int stepImg = static_cast<const int>(imgHSI.step / sizeof(double));
  // static const int stepImgRGB = imgRGB.step / sizeof(uchar);

  static auto dataHSI = reinterpret_cast<double *>(imgHSI.data);
  static auto dataSrc = imgRGB.data;

  /* HSI threshold */
  static const int trdH = 60;
  static const double trdS = 0.003043487826087;
  static const double trdI = 0.588235294117647;

  // static const int stepImgRGB = imgRGB.step / sizeof(uchar);
  // mask
  static const int stepMask = static_cast<const int>(maskRGB.step / sizeof(uchar));

  static auto dataMaskRGB = maskRGB.data;
  static auto dataMaskHSI = maskHSI.data;

  static int i = 0, j = 0, k = 0, idx = 0;
  for (i = 0; i < imgHSI.rows; ++ i) {
    for (j = 0, k = 0; j < stepImg; j += 3, ++ k) {  // stepImg = imgWidth * channel
      idx = i * stepImg + j;
      if (dataMaskRGB[i * stepMask + k] == 255 && dataHSI[idx] <= trdH && dataHSI[idx] >= 0 && dataHSI[idx + 2] > trdI
          && dataHSI[idx + 1] >= (255 - dataSrc[idx + 2]) * trdS) {  // HSI color model determine rule
        dataMaskHSI[i * stepMask + k] = static_cast<uchar>(255);
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

// mask
  static const int stepMask = static_cast<const int>(mask.step / sizeof(uchar));
  static uchar *dataMask = NULL;
  dataMask = mask.data;

  for (int i = 0; i < imgSrc.rows; ++ i) {
    auto ptr = imgSrc.ptr<_normal_pixel>(0, i);
    auto hsi = imgBackup.ptr<_normal_pixel>(0, i);
    const _normal_pixel *ptr_end = ptr + imgSrc.cols;
    for (int k = 0; ptr != ptr_end; ++ ptr, ++ k, ++ hsi) {
      if (255 == dataMask[i * stepMask + k]) {
        hsi->x = static_cast<uint8_t>(0);
        hsi->y = static_cast<uint8_t>(0);
        hsi->z = static_cast<uint8_t>(255);
      }
    }
  }
}
