// Bessel library: A C++ library with routines to evaluate Bessel functions of real or complex arguments (https://github.com/jodesarro/bessel-library)
// MIT License Copyright (c) 2021 Jhonas Olivati de Sarro (https://github.com/jodesarro/bessel-library/blob/main/LICENSE)

// See Refs. [[1–3](#references)] for more information concerning Bessel functions and their computation.

//-v------------------------------------------------------------------------
// CHANGELOG
// 0.1.3 Jun 09, 2024 (current version)
//  - Inclusion of Hankel functions cyl_h1 and cyl_h2 for integer or real orders and real or complex arguments.
//  - Inclusion of Airy functions airy_ai and airy_bi for real or complex arguments.
//  - Functions mod_i and mod_k were consistently renamed to cyl_i and cyl_k.
//  - The flags now print the number of components set to zero due to underflow.
//  - Routines zairy_, zbesh_, zbesj_, zbesy_, zbesi_, zbesk_ and zbiry_, based on version 930101 of D. E. Amos routines (https://doi.org/10.1145/7921.214331, ACM domain)
//      were changed (reverted) to be based on slatec ([3], public domain) versions to avoid copyright conflicts between ACM and MIT licenses and permissions. The
//      versions 0.1.1 and 0.1.2 of this code, and github commits related to them, shall be deleted and must be disconsidered and discarded by all users.
//  - Revision and reorganization of all slatec functions.
//  - Creation of functions d1mach and i1mach to make easier to compare with original slatec versions. 
// 0.1.2 Jun 06, 2024
//  - Inclusion of modified Bessel functions mod_i and mod_k for integer or real orders and real or complex arguments.
// 0.1.1 May 27, 2024
//  - Routines zairy_, zbesh_, zbesj_, and zbesy_, updated to the version 930101 of D. E. Amos routines (https://doi.org/10.1145/7921.214331).
//  - Inclusion of routines zbesi_, zbesk_, zbiry_ accordingly to version 930101 of D. E. Amos routines (https://doi.org/10.1145/7921.214331).
//  - Inclusion of C++ callable functions to overload cyl_j and cyl_y for real arguments.
//  - Static declarations removed for thread safety.
// 0.1.0 May 26, 2024
//  - Routines for cyl_j based on Ref. [2] were replaced by D. E. Amos Fortran 77 routines of SLATEC library [3].
//  - D. E. Amos routines zairy_.f, zbesh_.f, zbesj_.f, zbesy_.f, and all their dependencies, were converted to C using f2c (Availabe at: https://www.netlib.org/f2c/. Accessed: May 25, 2024).
//  - Replacement of all functions d1mach amd i1mach by C macros of float.h.
//  - Corrections of the translated f2c version and elimination of external dependencies.
//  - Reorganization of the whole code to be easily callable from C++.
//  - Inclusion of cylindrical Bessel functions of the second kind (or Neumann functions) cyl_y.
//  - Calculation of negative orders for cyl_j and cyl_y through Eq. (5.5.4) of Ref. [2].
//  - Now, cyl, Bessel functions of the first and second kinds, cyl_j and cyl_y, are available also for real (positive or negative) orders.
//  - Inclusion of cyl_j and cyl_y that returns an array of an int sequence of orders.
//  - Inclusion of parameters to print flag messages, and to return scaled versions of cyl_j and cyl_y.
//  - Inclusion of namespace bessel::slatec to call all slatec routines.
// 0.0.0 until May 12, 2024 
//  - Routines for cylindrical Bessel functions of the first kind and int order written based on Ref. [2].
// CHANGELOG
//-^------------------------------------------------------------------------


//-v------------------------------------------------------------------------
// REFERENCES
// [1] M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions With Formulas,
//      Graphs, and Mathematical Tables. Washington, D. C.: National Bureau of Standards, 1972.
// [2] S. Zhang and J. Jin, Computation of Special Functions. New York: Wiley, 1996.
// [3] SLATEC Common Mathematical Library, Version 4.1, July 1993. Comprehensive software library containing
//      over 1400 general purpose mathematical and statistical routines written in Fortran 77. Available
//      at https://www.netlib.org/slatec/ (Accessed: May 25, 2024).
// REFERENCES
//-^------------------------------------------------------------------------

#pragma once

//-v------------------------------------------------------------------------
// FORTRAN TRANSLATED TO C CODE

//-v------------------------------------------------------------------------
// C LIBRARIES
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // For M_PI, and other constant macros.
#endif
#include <math.h>   // For functions such as sin(), cos(), abs(), ... .
#include <float.h>  // For macro constants such as DBL_MIN, DBL_MAX_EXP, ... .
// C LIBRARIES
//-^------------------------------------------------------------------------


namespace bessel::slatec
{

//-v------------------------------------------------------------------------
// MAIN D. E. AMOS (SLATEC) ROUTINES DECLARATIONS
int zbesj_(double *zr, double *zi, double *fnu, int *kode, int *n, double *cyr, double *cyi, int *nz, int *ierr);
int zbesy_(double *zr, double *zi, double *fnu, int *kode, int *n, double *cyr, double *cyi, int *nz, double *cwrkr, double *cwrki, int *ierr);
int zbesh_(double *zr, double *zi, double *fnu, int *kode, int *m, int *n, double *cyr, double *cyi, int *nz, int *ierr);
int zbesi_(double *zr, double *zi, double *fnu, int *kode, int *n, double *cyr, double *cyi, int *nz, int *ierr);
int zbesk_(double *zr, double *zi, double *fnu, int *kode, int *n, double *cyr, double *cyi, int *nz, int *ierr);
int zairy_(double *zr, double *zi, int *id, int *kode, double *air, double *aii, int *nz, int *ierr);
int zbiry_(double *zr, double *zi, int *id, int *kode, double *bir, double *bii, int *ierr);
// MAIN D. E. AMOS (SLATEC) ROUTINES DECLARATIONS
//-^------------------------------------------------------------------------


//-v------------------------------------------------------------------------
// DEPENDENCY AMOS/SLATEC ROUTINES DECLARATIONS
double zabs_(double *zr, double *zi);
int zexp_(double *ar, double *ai, double *br, double *bi);
int zdiv_(double *ar, double *ai, double *br, double *bi, double *cr, double *ci);
int zsqrt_(double *ar, double *ai, double *br, double *bi);
int zlog_(double *ar, double *ai, double *br, double *bi, int *ierr);
int zs1s2_(double *zrr, double *zri, double *s1r, double *s1i, double *s2r, double *s2i, int *nz, double *ascle, double *alim, int *iuf);
int zasyi_(double *zr, double *zi, double *fnu, int *kode, int *n, double *yr, double *yi, int *nz, double *rl, double *tol, double *elim, double *alim);
int zacai_(double *zr, double *zi, double *fnu,	int *kode, int *mr, int *n, double *yr, double *yi, int *nz, double *rl, double *tol, double *elim,	double *alim);
int zuni1_(double *zr, double *zi, double *fnu, int *kode, int *n, double *yr, double *yi, int *nz, int *nlast, double *fnul, double *tol, double *elim, double *alim);
int zuni2_(double *zr, double *zi, double *fnu, int *kode, int *n, double *yr, double *yi, int *nz, int *nlast, double *fnul, double *tol, double *elim, double *alim);
int zbuni_(double *zr, double *zi, double *fnu, int *kode, int *n, double *yr, double *yi, int *nz, int *nui, int *nlast, double *fnul, double *tol, double *elim, double *alim);
int zmlri_(double *zr, double *zi, double *fnu, int *kode, int *n, double *yr, double *yi, int *nz, double *tol);
int zmlt_(double *ar, double *ai, double *br, double *bi, double *cr, double *ci);
double dgamln_(double *z__, int *ierr);
int zseri_(double *zr, double *zi, double *fnu, int *kode, int *n, double *yr, double *yi, int *nz, double *tol, double *elim, double *alim);
int zunik_(double *zrr, double *zri, double *fnu, int *ikflg, int *ipmtr, double *tol, int *init, double *phir, double *phii, double *zeta1r, double *zeta1i, double *zeta2r, double *zeta2i, double *sumr, double *sumi, double *cwrkr, double *cwrki);
int zunhj_(double *zr, double *zi, double *fnu, int *ipmtr, double *tol, double *phir, double *phii, double *argr, double *argi, double *zeta1r, double *zeta1i, double *zeta2r, double *zeta2i, double *asumr, double *asumi, double *bsumr, double *bsumi);
int zuchk_(double *yr, double *yi, int *nz, double *ascle, double *tol);
int zuoik_(double *zr, double *zi, double *fnu, int *kode, int *ikflg, int *n, double *yr, double *yi, int *nuf, double *tol, double *elim, double *alim);
int zbknu_(double *zr, double *zi, double *fnu, int *kode, int *n, double *yr, double *yi, int *nz, double *tol, double *elim, double *alim);
int zrati_(double *zr, double *zi, double *fnu, int *n, double *cyr, double *cyi, double *tol);
int zwrsk_(double *zrr, double *zri, double *fnu, int *kode, int *n, double *yr, double *yi, int *nz, double *cwr, double *cwi, double *tol, double *elim, double *alim);
int zbinu_(double *zr, double *zi, double *fnu, int *kode, int *n, double *cyr, double *cyi, int *nz, double *rl, double *fnul, double *tol, double *elim, double *alim);
int zshch_(double *zr, double *zi, double *cshr,	double *cshi, double *cchr, double *cchi);
int zkscl_(double *zrr, double *zri, double *fnu, int *n, double *yr, double *yi, int *nz, double *rzr, double *rzi, double *ascle, double *tol, double *elim);
int zacon_(double *zr, double *zi, double *fnu, int *kode, int *mr, int *n, double *yr, double *yi, int *nz, double *rl, double *fnul, double *tol, double *elim, double *alim);
int zbunk_(double *zr, double *zi, double *fnu, int *kode, int *mr, int *n, double *yr, double *yi, int *nz, double *tol, double *elim, double *alim);
int zunk1_(double *zr, double *zi, double *fnu, int *kode, int *mr, int *n, double *yr, double *yi, int *nz, double *tol, double *elim, double *alim);
int zunk2_(double *zr, double *zi, double *fnu, int *kode, int *mr, int *n, double *yr, double *yi, int *nz, double *tol, double *elim, double *alim);
// DEPENDENCY AMOS/SLATEC ROUTINES DECLARATIONS
//-^------------------------------------------------------------------------


//-v------------------------------------------------------------------------
// TABLE OF GLOBAL CONSTANT VALUES
static int c__0 = 0;
static int c__1 = 1;
static int c__2 = 2;
static int c__4 = 4;
static int c__5 = 5;
static int c__9 = 9;
static int c__14 = 14;
static int c__15 = 15;
static int c__16 = 16;
static double c_b10 = .5;
static double c_b11 = 0.;
// TABLE OF GLOBAL CONSTANT VALUES
//-^------------------------------------------------------------------------



//-v------------------------------------------------------------------------
// DEPENDENCY C ROUTINES
inline double max(double x, double y) { return((x) > (y) ? x : y); }
inline double min(double x, double y) { return((x) < (y) ? x : y); }
inline double d_sign(double *x, double *y) { return ((*y >= 0.) ? fabs(*x) : -fabs(*x)); }
inline double pow_dd(double *x, double *y) { return pow(*x,*y); }
// DEPENDENCY C ROUTINES
//-^------------------------------------------------------------------------

} //namespace bessel::slatec

// FORTRAN TRANSLATED TO C CODE
//-^------------------------------------------------------------------------


//-v------------------------------------------------------------------------
// C++ CODE

//-v------------------------------------------------------------------------
// C++ LIBRARIES
#include <complex>  // For complex variables. 
#include <iostream> // For std::cerr, std::endl, ... .
#include <climits>  // For INT_MAX
// C++ LIBRARIES
//-^------------------------------------------------------------------------

namespace bessel {
	void _flag_zbesj_(const int _ierr, const int _nz);

	template<typename T1, typename T2>
	std::complex<T2> cyl_i( const T1 _nu, const std::complex<T2> _z, bool _scaled = false, bool _flags = false );

	template<typename T1, typename T2>
	T2 cyl_i( const T1 _nu, const T2 _z, bool _scaled = false, bool _flags = false );

	double cyl_i( const int _nu, const double _z, bool _scaled = false, bool _flags = false );

} //namespace bessel