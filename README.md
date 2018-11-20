
The purpose of this project is to prevent disasters with AI technologies.

I am in charge of neuromorphic hardware bsed AI application development in nepes.
Please refer http://www.theneuromorphic.com


But before applying AI technology, I need a solid baseline to be a good starting point. The paper and codes from TCHsieh is great enough.

However, it is based on old opencv libaray, which is troublesome especially for cv::Videoio. For example, from 3.4.3, CvVideoCaputreFromAVI is not working anymore. RTP and RTSP support from latest opencv is another good execuse for this fork.

To use this,
- CMake latest version
- C++ compiler. currently I am testing on llvm 7 , apple latest clang,and MS visual studio 2017 15.9XX
- opencv 3.4.3 or higher is recommended
- boost test is optional

And I will apply improvements frequently. if you needs some help, let me know with below email 

kspark at nepes dot co dot kr
or wfms123 at gmail dot com

------  below ----------

                                
              VISION-BASED FIRE DETECTION USING VIDEO SEQUENCES

Abstract

Fire, if improperly used, could pose great threats to peoples’ security, life, and property. Motivated by the requirement to detect fire at its early stage, we aimed to develop an automatic system for vision-based fire detection using video sequences. Our system included four major steps, namely image preprocessing, foreground region analysis, fire dynamic behavior analysis, and fire flow energy analysis. Overall, our system could achieve the detection rates of over 91% in either indoor or outdoor environments. In addition, our system could achieve the system response time within 1 second (average delay of ~25 frames) once the fire occurred. In summary, our system could be used in surveillance systems, leading to prevent damage to peoples’ security, life, and property.

