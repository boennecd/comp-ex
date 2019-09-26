#ifndef TMP_WRAP_H
#define TMP_WRAP_H
/* from https://github.com/kaskr/adcomp/wiki/Development.
 * TODO: not needed when also using Rcpp? No this will declare a second
 * `R_CallMethodDef` static object */
/* #define TMB_LIB_INIT R_init_GSMVAEx */
#include <TMB.hpp>

/* this file does not matter as there is no header gaurd in TMB.hpp so
 * we can only include it in one file (right?)... */

#endif
