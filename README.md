The purpose of this project is to prevent disasters with AI technologies.


But before applying AI technology, I need a solid baseline to be a good starting point. The paper and codes from TCHsieh is great enough.

However, it is based on old opencv libaray, which is troublesome especially for cv::Videoio. For example, from 3.4.3, CvVideoCaputreFromAVI is not working anymore. RTP and RTSP support from latest opencv is another good execuse for this fork.

To use this,
- CMake latest version
- C++ compiler. currently I am testing on llvm 7 and MS visual studio 2017 15.9XX
- opencv 3.4.3 or higher is recommended
- vlc sdk : can be downloaded from here https://nightlies.videolan.org/build/
- boost test is optional

For fire detection wise, tried to keep below paper model which has below process
-  background substraction 
-  color model 
-  contour shape detection
-  optical flow analysis 
-  motion analysis

Implementation changes by myself are
-  modernizing : most of clang-tidy advices are adopted.
-  simplifying with standard stl algorithms and range based process.
-  opencv libraries are upgraded and tested with 4.0

works to be done
- main.cpp need to be refactored to be object oriented way.
- parallel processing will be revisited with opencv or with cpp-taksflow

 
For efficiency wise,
-  tracking algorithms are introduced.
-  once fire is detected, just track it with median flow tracker in opencv tracker algorithms. 

For usability on generic purpose wise,
- cascade classifier will be replaced with
  : motion detection & multi tracking algorithms.
  : in between motion detection and tracking, hardwware accelerated SVM will be introduced.


And I will apply improvements frequently. if you needs some help, let me know with below email.

wfms123 at gmail dot com


                                
              VISION-BASED FIRE DETECTION USING VIDEO SEQUENCES

Abstract

Fire, if improperly used, could pose great threats to peoples’ security, life, and property. Motivated by the requirement to detect fire at its early stage, we aimed to develop an automatic system for vision-based fire detection using video sequences. Our system included four major steps, namely image preprocessing, foreground region analysis, fire dynamic behavior analysis, and fire flow energy analysis. Overall, our system could achieve the detection rates of over 91% in either indoor or outdoor environments. In addition, our system could achieve the system response time within 1 second (average delay of ~25 frames) once the fire occurred. In summary, our system could be used in surveillance systems, leading to prevent damage to peoples’ security, life, and property.

