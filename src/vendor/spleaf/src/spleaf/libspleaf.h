// Copyright 2019-2024 Jean-Baptiste Delisle
// Licensed under the EUPL-1.2 or later

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void spleaf_cholesky(
    // Shape
    int64_t n, int64_t r, int64_t *offsetrow, int64_t *b,
    // Input
    double *A, double *U, double *V, double *phi, double *F,
    // Output
    double *D, double *W, double *G,
    // Temporary
    double *S, double *Z);

void spleaf_dotL(
    // Shape
    int64_t n, int64_t r, int64_t *offsetrow, int64_t *b,
    // Input
    double *U, double *W, double *phi, double *G, double *x,
    // Output
    double *y,
    // Temporary
    double *f);

void spleaf_solveL(
    // Shape
    int64_t n, int64_t r, int64_t *offsetrow, int64_t *b,
    // Input
    double *U, double *W, double *phi, double *G, double *y,
    // Output
    double *x,
    // Temporary
    double *f);

void spleaf_dotLT(
    // Shape
    int64_t n, int64_t r, int64_t *offsetrow, int64_t *b,
    // Input
    double *U, double *W, double *phi, double *G, double *x,
    // Output
    double *y,
    // Temporary
    double *g);

void spleaf_solveLT(
    // Shape
    int64_t n, int64_t r, int64_t *offsetrow, int64_t *b,
    // Input
    double *U, double *W, double *phi, double *G, double *y,
    // Output
    double *x,
    // Temporary
    double *g);

void spleaf_cholesky_back(
    // Shape
    int64_t n, int64_t r, int64_t *offsetrow, int64_t *b,
    // Input
    double *D, double *U, double *W, double *phi, double *G, double *grad_D,
    double *grad_Ucho, double *grad_W, double *grad_phicho, double *grad_G,
    // Output
    double *grad_A, double *grad_U, double *grad_V, double *grad_phi,
    double *grad_F,
    // Temporary
    double *S, double *Z);

void spleaf_dotL_back(
    // Shape
    int64_t n, int64_t r, int64_t *offsetrow, int64_t *b,
    // Input
    double *U, double *W, double *phi, double *G, double *x, double *grad_y,
    // Output
    double *grad_U, double *grad_W, double *grad_phi, double *grad_G,
    double *grad_x,
    // Temporary
    double *f);

void spleaf_solveL_back(
    // Shape
    int64_t n, int64_t r, int64_t *offsetrow, int64_t *b,
    // Input
    double *U, double *W, double *phi, double *G, double *x, double *grad_x,
    // Output
    double *grad_U, double *grad_W, double *grad_phi, double *grad_G,
    double *grad_y,
    // Temporary
    double *f);

void spleaf_dotLT_back(
    // Shape
    int64_t n, int64_t r, int64_t *offsetrow, int64_t *b,
    // Input
    double *U, double *W, double *phi, double *G, double *x, double *grad_y,
    // Output
    double *grad_U, double *grad_W, double *grad_phi, double *grad_G,
    double *grad_x,
    // Temporary
    double *g);

void spleaf_solveLT_back(
    // Shape
    int64_t n, int64_t r, int64_t *offsetrow, int64_t *b,
    // Input
    double *U, double *W, double *phi, double *G, double *x, double *grad_x,
    // Output
    double *grad_U, double *grad_W, double *grad_phi, double *grad_G,
    double *grad_y,
    // Temporary
    double *g);

void spleaf_expandsep(
    // Shape
    int64_t n, int64_t r, int64_t rsi, int64_t *sepindex,
    // Input
    double *U, double *V, double *phi,
    // Output
    double *K);

void spleaf_expandsepmixt(
    // Shape
    int64_t n1, int64_t n2, int64_t r, int64_t rsi, int64_t *sepindex,
    // Input
    double *U1, double *V1, double *phi1, double *U2, double *V2,
    int64_t *ref2left, double *phi2left, double *phi2right,
    // Output
    double *Km);

void spleaf_expandantisep(
    // Shape
    int64_t n, int64_t r, int64_t rsi, int64_t *sepindex,
    // Input
    double *U, double *V, double *phi,
    // Output
    double *K);

void spleaf_dotsep(
    // Shape
    int64_t n, int64_t r, int64_t rsi, int64_t *sepindex,
    // Input
    double *U, double *V, double *phi, double *x,
    // Output
    double *y);

void spleaf_dotsepmixt(
    // Shape
    int64_t n1, int64_t n2, int64_t r, int64_t rsi, int64_t *sepindex,
    // Input
    double *U1, double *V1, double *phi1, double *U2, double *V2,
    int64_t *ref2left, double *phi2left, double *phi2right, double *x,
    // Output
    double *y);

void spleaf_dotantisep(
    // Shape
    int64_t n, int64_t r, int64_t rsi, int64_t *sepindex,
    // Input
    double *U, double *V, double *phi, double *x,
    // Output
    double *y);
