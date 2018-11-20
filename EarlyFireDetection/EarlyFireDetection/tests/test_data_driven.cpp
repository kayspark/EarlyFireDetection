//
// Created by 박기수 on 05/11/2018.
//

#define BOOST_TEST_MODULE test_datadriven
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/included/unit_test.hpp>
#include <sstream>

namespace bdata = boost::unit_test::data;

// Dataset generating a Fibonacci sequence
class fibonacci_dataset {
public:
  // Samples type is int
  using sample = int;
  enum { arity = 1 };

  struct iterator {

    iterator() : a(1), b(1) {}

    int operator*() const { return b; }
    void operator++() {
      a = a + b;
      std::swap(a, b);
    }

  private:
    int a;
    int b; // b is the output
  };

  fibonacci_dataset() {}

  // size is infinite
  bdata::size_t size() const { return bdata::BOOST_TEST_DS_INFINITE_SIZE; }

  // iterator
  iterator begin() const { return iterator(); }
};

namespace boost {
namespace unit_test {
namespace data {
namespace monomorphic {
// registering fibonacci_dataset as a proper dataset
template<> struct is_dataset<fibonacci_dataset> : boost::mpl::true_ {};
} // namespace monomorphic
} // namespace data
} // namespace unit_test
} // namespace boost

// Creating a test-driven dataset
BOOST_DATA_TEST_CASE(test_datadriven,
                     fibonacci_dataset() ^ bdata::make({1, 2, 3, 5, 8, 13, 21, 35, 56}),
                     fib_sample,
                     exp) {
  BOOST_TEST(fib_sample == exp);
}