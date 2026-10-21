PHP_ARG_ENABLE([mlnative],
  [whether to enable mlnative support],
  [AS_HELP_STRING([--enable-mlnative], [Enable native machine learning classes (ML\\*)])],
  [no])

if test "$PHP_MLNATIVE" != "no"; then
  PHP_REQUIRE_CXX()

  MLNATIVE_SOURCES="
    src/mlnative.cpp
    src/php_bridge.cpp
    src/classes/linear_regression.cpp
    src/classes/logistic_regression.cpp
    src/classes/knn.cpp
    src/classes/gaussian_nb.cpp
    src/classes/kmeans.cpp
    src/classes/standard_scaler.cpp
    src/classes/metrics.cpp
    src/core/linear_regression.cpp
    src/core/logistic_regression.cpp
    src/core/knn.cpp
    src/core/gaussian_nb.cpp
    src/core/kmeans.cpp
    src/core/standard_scaler.cpp
  "

  PHP_NEW_EXTENSION(mlnative, $MLNATIVE_SOURCES, $ext_shared,, [-std=c++17 -DZEND_ENABLE_STATIC_TSRMLS_CACHE=1], cxx)
  PHP_ADD_INCLUDE([$ext_srcdir/src])
  PHP_ADD_BUILD_DIR([$ext_builddir/src])
  PHP_ADD_BUILD_DIR([$ext_builddir/src/classes])
  PHP_ADD_BUILD_DIR([$ext_builddir/src/core])
  PHP_ADD_LIBRARY(stdc++, 1, MLNATIVE_SHARED_LIBADD)
  PHP_SUBST(MLNATIVE_SHARED_LIBADD)
fi
