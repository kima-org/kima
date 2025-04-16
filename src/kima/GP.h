#pragma once

#include <cmath>
#include <vector>
#include <iomanip>
#include <iostream>
#include <array>
#include <numeric>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "bessel-library.hpp"
#include "utils.h"

extern "C"
{
    #include "libspleaf.h"
}

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>
// #include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;
using namespace nb::literals;

// Eigen types
using namespace Eigen;
typedef Eigen::Matrix<long, -1, 1> VectorXl; // vector of long integers
// in order to interface with C spleaf functions, matrices must be 
// stored in row-major order, and by default Eigen uses column-major
typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> MatrixXd_RM;


// kernel types
enum KernelType { qp, per, spleaf_exp, spleaf_matern32, spleaf_sho, spleaf_mep, spleaf_es, spleaf_esp };

/* Sample from a Gaussian process prior at times t*/
VectorXd sample(const Eigen::MatrixXd &K, double white_noise_variance=1.25e-12);

/* The "standard" quasi-periodic kernel, see R&W2006 */
Eigen::MatrixXd QP(std::vector<double> &t, double eta1, double eta2, double eta3, double eta4);

/* The periodic kernel, oringinally created by MacKay, see R&W2006 */
Eigen::MatrixXd PER(std::vector<double> &t, double eta1, double eta3, double eta4);


// SPLEAF

class spleaf_ExponentialKernel {
    public:
        static constexpr size_t r = 1; // rank of the kernel
        size_t offset = 0;
        double sig; // variance
        double la; // inverse lengthscale
        // delegating constructor
        spleaf_ExponentialKernel(const VectorXd &t, std::array<double, 2> params) : spleaf_ExponentialKernel(t, params, 0) {}
        // constructor with offset
        spleaf_ExponentialKernel(const VectorXd &t, std::array<double, 2> params, size_t offset)
            : sig(params[0]), la(params[1]), offset(offset) {};
        // compute the kernel and derivative
        void operator()(const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi);
        void deriv(const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV);
};

class spleaf_Matern32Kernel {
    public:
        static constexpr size_t r = 2; // rank of the kernel
        size_t offset = 0;
        double sig; // standard deviation
        double rho; // lengthscale
        // local variables
        double a, la, t0;
        VectorXd dt0, x, _1mx;
        // delegating constructor
        spleaf_Matern32Kernel(const VectorXd &t, std::array<double, 2> params) : spleaf_Matern32Kernel(t, params, 0) {};
        // constructor with offset
        spleaf_Matern32Kernel(const VectorXd &t, std::array<double, 2> params, size_t offset);
        void operator()(const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi);
        void deriv(const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV);
};

class spleaf_SHOKernel {
    public:
        static constexpr size_t r = 2; // rank of the kernel
        size_t offset = 0;
        double sig; // standard deviation
        double P0; // (un-damped) period
        double Q; // quality factor
        double eps = 1e-5; // regularization parameter 
        // local variables
        double a, la, b, nu; 
        double a1, la1, a2, la2;
        double sqQ;
        VectorXd _nut, _cnut, _snut;
        // delegating constructor
        spleaf_SHOKernel(const VectorXd &t, std::array<double, 3> params) : spleaf_SHOKernel(t, params, 0) {};
        // constructor with offset
        spleaf_SHOKernel(const VectorXd &t, std::array<double, 3> params, size_t offset);
        void operator()(const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi);
        void deriv(const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV);
};

class spleaf_QuasiPeriodicKernel {
    public:
        static constexpr size_t r = 2; // rank of the kernel
        size_t offset = 0;
        double a; // variance of the cos term
        double b; // variance of the sin term
        double la; // decay rate
        double nu; // angular frequency
        // local variables
        VectorXd _nut, _cnut, _snut;
        // delegating constructor
        spleaf_QuasiPeriodicKernel(const VectorXd &t, std::array<double, 4> params) : spleaf_QuasiPeriodicKernel(t, params, 0) {};
        // constructor with offset
        spleaf_QuasiPeriodicKernel(const VectorXd &t, std::array<double, 4> params, size_t offset);
        void operator()(const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi);
        void deriv(const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV);
};

class spleaf_MEPKernel {
    public:
        static constexpr size_t r = 6; // rank of the kernel
        size_t offset = 0;
        double sig; // standard deviation
        double rho; // lengthscale
        double P; // period
        double eta; // scale of oscillations
        // local variables
        double sig0, a1, a2, b1, b2, la, nu;
        // VectorXd _nut, _cnut, _snut;
        // delegating constructor
        spleaf_MEPKernel(const VectorXd &t, std::array<double, 4> params) : spleaf_MEPKernel(t, params, 0) {};
        // constructor with offset
        spleaf_MEPKernel(const VectorXd &t, std::array<double, 4> params, size_t offset);
        void operator()(const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi);
        void deriv(const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV);
};

class spleaf_ESKernel {
    public:
        static constexpr size_t r = 3; // rank of the kernel
        size_t offset = 0;
        double sig; // standard deviation
        double rho; // lengthscale
        double coef_la = 1.0907260149419182;
        double mu = 1.326644517327145;
        // local variables
        double coef_b, coef_a0, coef_a, a0, a, b, la, nu;
        // delegating constructor
        spleaf_ESKernel(const VectorXd &t, std::array<double, 2> params) : spleaf_ESKernel(t, params, 0) {};
        // constructor with offset
        spleaf_ESKernel(const VectorXd &t, std::array<double, 2> params, size_t offset);
        void operator()(const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi);
        void deriv(const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV);
};


// WARNING: not a valid kernel by itself, only used in spleaf_ESPKernel
class _spleaf_ESP_PKernel { 
    public:
        static constexpr size_t nharm = 3;
        static constexpr size_t r = 1 + 2 * nharm; // rank of the kernel
        size_t offset = 0;
        double P;
        double eta;
        // local variables
        std::array<double, nharm + 1> a;
        double eta2, f, deno, nu;
        // delegating constructor
        _spleaf_ESP_PKernel(const VectorXd &t, std::array<double, 2> params) : _spleaf_ESP_PKernel(t, params, 0) {};
        // constructor with offset
        _spleaf_ESP_PKernel(const VectorXd &t, std::array<double, 2> params, size_t offset);
        void operator()(const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi);
        void deriv(const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV);
};


class spleaf_ESPKernel {
    public:
        static constexpr size_t nharm = 3;
        static constexpr size_t r = 3 + 6 * nharm; // rank of the kernel
        size_t offset = 0;
        double sig; // standard deviation
        double rho; // lengthscale
        double P; // period
        double eta; // scale of oscillations
        // local variables
        VectorXd _A1, _A2;
        MatrixXd_RM _U1, _U2, _V1, _V2, _phi1, _phi2;
        // delegating constructor
        spleaf_ESPKernel(const VectorXd &t, std::array<double, 4> params) : spleaf_ESPKernel(t, params, 0) {};
        // constructor with offset
        spleaf_ESPKernel(const VectorXd &t, std::array<double, 4> params, size_t offset);
        void operator()(const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi);
        void deriv(const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV);
};



double logdet(VectorXd &D);

double chi2(VectorXd &y, size_t n, size_t r, VectorXl &offsetrow, VectorXl &b,
            MatrixXd_RM &U, MatrixXd_RM &W, MatrixXd_RM &phi, VectorXd &G, VectorXd &_f_solveL, VectorXd &D);

double loglike(VectorXd &y, size_t n, size_t r, VectorXl &offsetrow, VectorXl &b,
               MatrixXd_RM &U, MatrixXd_RM &W, MatrixXd_RM &phi, VectorXd &G, VectorXd &_f_solveL, VectorXd &D);


template< class Kernel, size_t nparams >
double spleaf_loglike(VectorXd &y, const VectorXd &t, const VectorXd &Adiag, 
                      const VectorXd &dt, size_t N,
                      std::array<double, nparams> params)
{
    size_t r = Kernel::r;

    VectorXl b = VectorXl::Zero(N);
    VectorXl offsetrow(N);

    b = b.array().max(0);

    long sum = 0;
    for (size_t i = 0; i < N; i++)
    {
        offsetrow[i] = sum;
        sum += b[i] - 1;
    }

    VectorXd A = VectorXd::Zero(N);
    A.array() += Adiag.array();

    MatrixXd_RM U = MatrixXd_RM(N, r);
    MatrixXd_RM V = MatrixXd_RM(N, r);
    MatrixXd_RM phi = MatrixXd_RM(N - 1, r);

    VectorXd F = VectorXd(b.sum());

    VectorXd D = VectorXd(N);
    MatrixXd_RM W = MatrixXd_RM(N, r);
    VectorXd G = VectorXd(b.sum());
    VectorXd _S = VectorXd(N * r * r);
    VectorXd _Z = VectorXd((F.size() + N) * r);
    VectorXd _f_solveL = VectorXd(N * r);
    // // Kernel derivative
    // MatrixXd_RM _dU = MatrixXd_RM(N, r);
    // MatrixXd_RM _dV = MatrixXd_RM(N, r);
    // VectorXd _B = VectorXd(N);

    Kernel k(t, params);
    k(t, dt, A, U, V, phi);


    spleaf_cholesky(N, r, offsetrow.data(), b.data(),
                    A.data(), U.data(), V.data(), phi.data(), F.data(),
                    D.data(), W.data(), G.data(), _S.data(), _Z.data());

    // for (auto param : params)
    // {
    //     std::cout << param << " ";
    // }
    // std::cout << N << " " << r << std::endl;
    // std::cout << std::endl << std::endl;
    // std::cout << "A:" << A.transpose() << std::endl;
    // std::cout << "U:" << U.transpose() << std::endl;
    // std::cout << "V:" << V.transpose() << std::endl;
    // std::cout << "phi:" << phi.transpose() << std::endl;
    // std::cout << std::endl << std::endl;

    // std::cout << "offsetrow:" << offsetrow.transpose() << std::endl;
    // std::cout << "b:" << b.transpose() << std::endl;
    // std::cout << "W:" << W.transpose() << std::endl;
    // std::cout << std::endl << std::endl;
    // std::cout << "D:" << D.transpose() << std::endl;

    double logL = loglike(y, N, r, offsetrow, b, U, W, phi, G, _f_solveL, D);
    // std::cout << "logL:" << logL << std::endl;
    return logL;
}


template< class Kernel, size_t nparams >
double spleaf_loglike_multiseries(VectorXd &y, const VectorXd &t, const VectorXd &Adiag, 
                                  const VectorXd &dt, const std::vector<Eigen::ArrayXi> &series_index,
                                  std::array<double, nparams> params,
                                  std::vector<double> alpha, std::vector<double> beta)
{
    size_t N = t.size();
    size_t nseries = series_index.size();
    size_t r = Kernel::r;

    // std::cout << N << " " << r << std::endl;

    VectorXl b = VectorXl::Zero(N);
    VectorXl offsetrow(N);

    b = b.array().max(0);

    long sum = 0;
    for (size_t i = 0; i < N; i++)
    {
        offsetrow[i] = sum;
        sum += b[i] - 1;
    }

    VectorXd A = VectorXd::Zero(N);
    VectorXd _A = VectorXd::Zero(N);
    A.array() += Adiag.array();

    MatrixXd_RM U = MatrixXd_RM(N, r);
    MatrixXd_RM V = MatrixXd_RM(N, r);
    MatrixXd_RM phi = MatrixXd_RM(N - 1, r);

    VectorXd F = VectorXd(b.sum());

    VectorXd D = VectorXd(N);
    MatrixXd_RM W = MatrixXd_RM(N, r);
    VectorXd G = VectorXd(b.sum());
    VectorXd _S = VectorXd(N * r * r);
    VectorXd _Z = VectorXd((F.size() + N) * r);
    VectorXd _f_solveL = VectorXd(N * r);
    // Kernel derivative
    MatrixXd_RM _dU = MatrixXd_RM(N, r);
    MatrixXd_RM _dV = MatrixXd_RM(N, r);
    VectorXd _B = VectorXd(N);

    // for (auto param : params)
    // {
    //     std::cout << param << " ";
    // }

    Kernel kernel(t, params);
    kernel(t, dt, _A, U, V, phi);

    kernel.deriv(t, dt, _dU, _dV);
    _B = (_dU.array() * _dV.array()).rowwise().sum();

    int k = 0;
    for (auto &ik : series_index)
    {
        A(ik).array() += alpha[k] * alpha[k] * _A(ik).array();
        U(ik, Eigen::placeholders::all).array() *= alpha[k];
        V(ik, Eigen::placeholders::all).array() *= alpha[k];

        A(ik).array() += beta[k] * beta[k] * _B(ik).array();
        U(ik, Eigen::placeholders::all).array() += beta[k] * _dU(ik, Eigen::placeholders::all).array();
        V(ik, Eigen::placeholders::all).array() += beta[k] * _dV(ik, Eigen::placeholders::all).array();
        
        k++;
    }

    spleaf_cholesky(N, r, offsetrow.data(), b.data(),
                    A.data(), U.data(), V.data(), phi.data(), F.data(),
                    D.data(), W.data(), G.data(), _S.data(), _Z.data());

    // std::cout << std::endl << std::endl;
    // std::cout << "A:" << A.transpose() << std::endl << std::endl;
    // std::cout << "dU:" << _dU.transpose() << std::endl << std::endl;
    // std::cout << "dV:" << _dV.transpose() << std::endl << std::endl;
    // std::cout << "B:" << _B.transpose() << std::endl << std::endl;
    // std::cout << "U:" << U.transpose() << std::endl << std::endl;
    // std::cout << "V:" << V.transpose() << std::endl << std::endl;
    // std::cout << "phi:" << phi.transpose() << std::endl;
    // std::cout << std::endl << std::endl;

    // std::cout << "offsetrow:" << offsetrow.transpose() << std::endl;
    // std::cout << "b:" << b.transpose() << std::endl;
    // std::cout << "W:" << W.transpose() << std::endl;
    // std::cout << std::endl << std::endl;
    // std::cout << "D:" << D.transpose() << std::endl;

    double logL = loglike(y, N, r, offsetrow, b, U, W, phi, G, _f_solveL, D);
    // std::cout << "logL:" << logL << std::endl;
    return logL;
}
