#ifndef VLCCAP_H
#define VLCCAP_H

#include "vlc/vlc.h"
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>

class vlc_capture {
public:
  vlc_capture(const std::string &chroma, int width, int height);
  ~vlc_capture();

  void open(const char *url);
  void release();
  bool isOpened();

  bool read(cv::Mat &outFrame);
  cv::Size get_size();

private:
  vlc_capture();
  unsigned format(char *chroma, unsigned *width, unsigned *height,
                  unsigned *pitches, unsigned *lines);
  void *lock(void **p_pixels);
  void unlock(void *id, void *const *p_pixels);

  static unsigned formater(void **data, char *chroma, unsigned *width,
                           unsigned *height, unsigned *pitches,
                           unsigned *lines);
  static void *locker(void *data, void **p_pixels);
  static void unlocker(void *data, void *id, void *const *p_pixels);

private:
  std::mutex _mutex;
  std::string _url;
  std::string _chroma;
  cv::Mat _source_frame;
  cv::Mat _rgb;
  bool _is_open;
  std::atomic<bool> _has_frame;
  libvlc_instance_t *_vlc_instance = nullptr;
  libvlc_media_player_t *_media_player = nullptr;
  cv::Size _size;
};

#endif // VLCCAP_H
