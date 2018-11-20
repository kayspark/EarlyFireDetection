//
// Created by 박기수 on 05/11/2018.
//

#define BOOST_TEST_MODULE bitwise_test
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE(bitwise_test) {
  int i = 1;
  BOOST_TEST(i);
  BOOST_TEST((sizeof(unsigned int) << 2) == 16);
}