#ifndef PHP_MLNATIVE_H
#define PHP_MLNATIVE_H

extern zend_module_entry mlnative_module_entry;
#define phpext_mlnative_ptr &mlnative_module_entry

#define PHP_MLNATIVE_VERSION "0.1.3"

#if defined(ZTS) && defined(COMPILE_DL_MLNATIVE)
ZEND_TSRMLS_CACHE_EXTERN()
#endif

#endif /* PHP_MLNATIVE_H */
