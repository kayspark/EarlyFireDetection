#include "vlccap.h"
#include <cstring>
#include <thread>

vlc_capture::vlc_capture()
    : _is_open(false), _has_frame(false), _vlc_instance(nullptr),
      _media_player(nullptr) {}

vlc_capture::vlc_capture(const std::string &chroma, int width, int height)
    : vlc_capture() {
  _chroma = chroma;
  _size.width = width;
  _size.height = height;
}

vlc_capture::~vlc_capture() {
  try {
    release();
  } catch (...) {
    // TODO log
  }
}

void vlc_capture::open(const char *url) {
  release();
  _url = url;
  std::string arg_width = " --vmem-width=" + std::to_string(_size.width);
  std::string arg_height = " --vmem-height=" + std::to_string(_size.height);
  std::string arg_chroma = " --vmem-chroma=" + _chroma;
  const char *args[] = {"-I",
                        "dummy",
                        "--ignore-config",
                        "--no-audio",
                        "--network-caching=1000",
                        "--clock-jitter=0",
                        "--clock-synchro=0",
                        "--verbose=2",
                        arg_width.c_str(),
                        arg_height.c_str(),
                        arg_chroma.c_str()};
  _vlc_instance = libvlc_new(sizeof(args) / sizeof(args[0]), args);

  if (_vlc_instance) {
    libvlc_media_t *media = nullptr;
    media = libvlc_media_new_location(_vlc_instance, _url.c_str());
    if (media) {
      _media_player = libvlc_media_player_new_from_media(media);
      if (_media_player) {
        libvlc_media_release(media);
        libvlc_video_set_callbacks(_media_player, locker, unlocker, nullptr,
                                   this);
        libvlc_video_set_format_callbacks(_media_player, formater, nullptr);

        int resp = libvlc_media_player_play(_media_player);
        if (resp == 0) {
          _is_open = true;
        } else {
          release();
        }
      }
    }
  } else {
    std::cout << "why??" << std::endl;
  }
}

void vlc_capture::release() {
  if (_vlc_instance) {
    libvlc_media_player_stop(_media_player);
    libvlc_release(_vlc_instance);
    libvlc_media_player_release(_media_player);
    _vlc_instance = nullptr;
    _media_player = nullptr;
  }
  _has_frame = false;
  _is_open = false;
}

bool vlc_capture::isOpened() {
  if (!_is_open)
    return false;

  libvlc_state_t state = libvlc_media_player_get_state(_media_player);
  return (state != libvlc_Paused && state != libvlc_Stopped &&
      state != libvlc_Ended && state != libvlc_Error);
}

bool vlc_capture::read(cv::Mat &outFrame) {
  while (!_has_frame) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!isOpened())
      return false; // connection closed
  }

  {
    std::lock_guard<std::mutex> guard(_mutex);
    if ("J420" == _chroma)
      cv::cvtColor(_source_frame, _rgb, cv::COLOR_YUV420p2BGR);
    outFrame = _rgb.clone();
    _has_frame = false;
  }
  return true;
}

cv::Size vlc_capture::get_size() { return _size; }

unsigned vlc_capture::format(char *chroma, unsigned *width, unsigned *height,
                             unsigned *pitches, unsigned *lines) {
  // TODO: Allow overriding of native size?
  std::lock_guard<std::mutex> guard(_mutex);
  std::cout << "vlc_capture::format - " << chroma << " - " << *width << "/"
            << *height << std::endl;
  *width = _size.width;
  *height = _size.height;
  memcpy(chroma, _chroma.c_str(), 4);
  if ("J420" == _chroma) { // seems like rtsp server choose this,
    pitches[0] = (*width);
    lines[0] = (*height) * 1.5;
    _source_frame.create((*height) * 1.5, *width, CV_8UC1);
    _rgb.create(*height, *width, CV_8UC3);
  } else { // will be RV24 in general
    pitches[0] = (*width) * 24 / 8;
    lines[0] = *height;
    _rgb.create(*height, *width, CV_8UC3);
  }
  return 1;
}
// whatever the choosen size was, need to alloc space for the whole input image
// here !

void *vlc_capture::lock(void **p_pixels) {
  // cout << "vlc_capture::lock" << endl;
  _mutex.lock();
  *p_pixels = (unsigned char *) _rgb.data;
  return nullptr;
}

void vlc_capture::unlock(void *id, void *const *p_pixels) {
  _has_frame = true;
  _mutex.unlock();
}

unsigned vlc_capture::formater(void **data, char *chroma, unsigned *width,
                               unsigned *height, unsigned *pitches,
                               unsigned *lines) {
  return static_cast<vlc_capture *>(*data)->format(chroma, width, height,
                                                   pitches, lines);
}

void *vlc_capture::locker(void *data, void **p_pixels) {
  return static_cast<vlc_capture *>(data)->lock(p_pixels);
}

void vlc_capture::unlocker(void *data, void *id, void *const *p_pixels) {
  static_cast<vlc_capture *>(data)->unlock(id, p_pixels);
}