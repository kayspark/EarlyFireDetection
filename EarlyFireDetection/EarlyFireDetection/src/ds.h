#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

typedef struct RectThrd {
  int rectWidth;
  int rectHeight;
  int cntrArea;
} RectThresh;

/* for initialize rectThrd node */
CV_INLINE RectThresh rectThrd(const int rectWidth, const int rectHeight,
                              const int cntrArea) {
  RectThresh rt;

  rt.rectWidth = rectWidth;
  rt.rectHeight = rectHeight;
  rt.cntrArea = cntrArea;

  return rt;
}

/* Optical Flow feature points */
typedef struct Feature {
  cv::Point prev;
  cv::Point curr;
} Feature;

/* for initialize Feature node */
CV_INLINE Feature feature(const cv::Point &prev, const cv::Point &curr) {
  Feature fr;

  fr.prev = prev;
  fr.curr = curr;

  return fr;
}

/* rect node(rect space) */
typedef struct OFRect {
  bool match{};    // determine whether the rect is match or not
  int countCtrP{}; // the pixel count of contour
  // the pixel count of contour which is only be detected
  cv::Rect rect;                   // rect
  std::vector<Feature> vecFeature; // optical flow feature points
} OFRect;

/* for initialize ofrect node */
CV_INLINE OFRect ofRect(const cv::Rect &rect, const int countCtrP) {
  OFRect ofr;

  ofr.match = false;
  ofr.countCtrP = countCtrP;
  ofr.rect = rect;

  return ofr;
}

/* marker node */
typedef struct Centroid {
  int countFrame{};   // how many frame the centroid keeping in the region
  cv::Point centroid; // first detected centroid
  std::vector<cv::Rect> vecRect;            // rect information
  std::deque<std::vector<Feature>> dOFRect; // optical flow feature points
} Centroid;

/* for initailize the new centroid node */
CV_INLINE Centroid centroid(const OFRect &ofRect) {
  Centroid centroid1;

  centroid1.countFrame = 1; // first node
  centroid1.centroid =
      cv::Point(ofRect.rect.x + (ofRect.rect.width >> 1),
                ofRect.rect.y + (ofRect.rect.height >> 1)); // centroid position
  centroid1.vecRect.emplace_back(ofRect.rect); // push rect information
  centroid1.dOFRect.emplace_back(
      ofRect.vecFeature); // push contour optical flow feature(after optical
  // flow)

  return centroid1;
}

typedef struct DirectionsCount {
  int countUp;
  int countDown;
  int countLeft;
  int countRight;
} DirectionsCount;

CV_INLINE void zeroCount(DirectionsCount &count) {
  count.countDown = count.countLeft = count.countRight = count.countUp = 0;
}
