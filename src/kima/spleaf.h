#pragma once

#include <numeric>
#include <vector>

#include <iostream>
using namespace std;

extern "C"
{
    #include "libspleaf.h"
}

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
typedef Eigen::Matrix<long, -1, 1> VectorXl;
typedef Eigen::Matrix<double, -1, 1> VectorXd;
typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> MatrixXd;

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;
using namespace nb::literals;


template<typename Base, typename T>
inline bool instanceof(const T *ptr) {
   return dynamic_cast<const Base*>(ptr) != nullptr;
}


class Term;

class Cov
{
    friend class Term;
    friend class Noise;
    friend class Kernel;
    friend class Error;
    friend class Jitter;
    friend class ExponentialKernel;
    friend class QuasiperiodicKernel;
    friend class SumKernel;
    friend class Matern32Kernel;
    friend class Matern52Kernel;
    friend class USHOKernel;
    friend class OSHOKernel;
    friend class SHOKernel;
    friend class MultiSeriesKernel;

    protected:
        size_t n;
        size_t r;
        VectorXd t, dt;
        VectorXl b, offsetrow;
        VectorXd A, D;
        VectorXd _grad_A, _grad_B;
        MatrixXd U, V, W, phi;
        MatrixXd _dU, _dV, _grad_U, _grad_V, _grad_phi;
        MatrixXd _grad_dU, _grad_dV;
        VectorXd F, G;
        VectorXd _S, _Z;
        VectorXd _f_solveL;
        VectorXd _B;

    public:
        Cov() {};
        Cov(const VectorXd &t, Term& term);
        Cov(const VectorXd &t, Term& err, Term& gp);
        Cov(const VectorXd &t, Term& err, Term& jit, Term& gp);
        Cov(const VectorXd &t, const VectorXd &dt, size_t r); // to construct a _FakeCov
        
        // template <class... Args> 
        // Cov(const VectorXd &t, Args... terms);

        VectorXd solveL(VectorXd &y)
        {
            VectorXd _x_solveL = VectorXd(y.size());
            spleaf_solveL(n, r, offsetrow.data(), b.data(),
                          U.data(), W.data(), phi.data(), G.data(),
                          y.data(), _x_solveL.data(), _f_solveL.data());
            return _x_solveL;
        }

        double logdet()
        {
            return D.array().log().sum();
        }

        double chi2(VectorXd &y)
        {
            VectorXd x = solveL(y);
            return (x.array().square() / D.array()).sum();
        }

        double loglike(VectorXd &y)
        {
            return -0.5 * (chi2(y) + logdet() + n * log(2.0 * M_PI));
        }

};

class Term
{
    friend class MultiSeriesKernel;

    protected:
        bool _linked = false;
        Cov* _cov;

    public:
        size_t _b = 0;
        size_t _r = 0;
        size_t _offset = 0;

        Term(size_t b, size_t r) : _b(b), _r(r) {};

        virtual void _link(Cov &cov) {
            _cov = &cov;
            _linked = true;
        }
        virtual void _link(Cov &cov, size_t offset) {
            _cov = &cov;
            _linked = true;
            _offset = offset;
        }

        virtual void _compute() {};
        virtual void _deriv(bool calc_d2) {};
};

class Noise : public Term {

    public:
        Noise(size_t b) : Term(b, 0) {};

        template <typename... args>
        void set_param(args...) {};
};

class Kernel : public Term {

    public:
        Kernel(size_t r) : Term(0, r) {};

        template <typename... args>
        void set_param(args...) {};
};



class Error : public Noise
{
    VectorXd _sig;
    
    public:
        Error(VectorXd& sig) : _sig(sig), Noise(0) {};

        void _compute() override final {
            this->_cov->A.array() += _sig.array().square();
        };
};

class Jitter : public Noise
{
    double _sig;
    
    public:
        Jitter(double sig) : _sig(sig), Noise(0) {};

        void set_param(double sig) { _sig=sig; };

        void _compute() override final {
            this->_cov->A.array() += _sig * _sig;
        };
};


// class InstrumentJitter : public Noise
// class CalibrationError : public Noise
// class CalibrationJitter : public Noise

// ...

class ExponentialKernel : public Kernel
{
    double _a, _la;

    public:
        ExponentialKernel(double a, double la) : _a(a), _la(la), Kernel(1) {};

        void set_param(double a, double la)
        {
            _a = a;
            _la = la;
        }

        void _compute() override final {
            _cov->A.array() += _a;
            _cov->U.col(_offset).setConstant(_a);
            _cov->V.col(_offset).setConstant(1.0);
            _cov->phi.col(_offset) = (-_la * _cov->dt.array()).exp();
        }

        void _deriv(bool calc_d2) override final {
            _cov->_dU.col(_offset).setConstant(- _la * _a);
            if (calc_d2) {
                _cov->_dV.col(_offset).setConstant(_la);
            }
        }
};

class QuasiperiodicKernel : public Kernel
{
    double _a, _b, _la, _nu;
    VectorXd _nut, _cnut, _snut;

    public:
        QuasiperiodicKernel(double a, double b, double la, double nu) 
            : _a(a), _b(b), _la(la), _nu(nu), Kernel(2) {};

        void set_param(double a, double b, double la, double nu)
        {
            _a = a;
            _b = b;
            _la = la;
            _nu = nu;
        }

        void _compute() override final {
            _cov->A.array() += _a;
            _nut = _nu * _cov->t.array();
            _cnut = _nut.array().cos();
            _snut = _nut.array().sin();
            _cov->U.col(_offset) = _a * _cnut.array() + _b * _snut.array();
            _cov->U.col(_offset + 1) = _a * _snut.array() - _b * _cnut.array();
            _cov->V.col(_offset) = _cnut;
            _cov->V.col(_offset + 1) = _snut;
            VectorXd _e = (-_la * _cov->dt.array()).exp();
            _cov->phi.col(_offset) = _e;
            _cov->phi.col(_offset + 1) = _e;
        }

        void _deriv(bool calc_d2) override final {
            double da = - _la * _a + _nu * _b;
            double db = - _la * _b - _nu * _a;
            _cov->_dU.col(_offset) = da * _cnut.array() + db * _snut.array();
            _cov->_dU.col(_offset + 1) = da * _snut.array() - db * _cnut.array();
            if (calc_d2) {
                _cov->_dV.col(_offset) = _la * _cnut.array() - _nu * _snut.array();
                _cov->_dV.col(_offset + 1) = _la * _snut.array() + _nu * _cnut.array();
            }
        }
};

class SumKernel : public Kernel
{
    friend class OSHOKernel;

    Kernel* _k1;
    Kernel* _k2;

    public:
        SumKernel() : Kernel(0) {}
        SumKernel(Kernel& k1, Kernel& k2) : Kernel(k1._r + k2._r) {
            _k1 = &k1;
            _k2 = &k2;
        }

        void _set_k1_k2(Kernel& k1, Kernel& k2) {
            _k1 = &k1;
            _k2 = &k2;
            _r = k1._r + k2._r;
        }

        void _link(Cov &cov, size_t offset) override {
            _cov = &cov;
            _linked = true;
            _offset = offset;
            size_t off = offset;
            _k1->_link(cov, off);
            off += _k1->_r;
            _k2->_link(cov, off);
        }

        void _compute() override {
            _k1->_compute();
            _k2->_compute();
        }

        void _deriv(bool calc_d2) override {
            _k1->_deriv(calc_d2);
            _k2->_deriv(calc_d2);
        }
};

class Matern32Kernel : public Kernel
{
    double _sig, _rho;
    double _t0;
    VectorXd _dt0;
    // temps
    double _a, _la, _la2;
    VectorXd _x, _1mx;

    public:
        Matern32Kernel(double sig, double rho) : _sig(sig), _rho(rho), Kernel(2) {};

        void set_param(double sig, double rho)
        {
            _sig = sig;
            _rho = rho;
        }

        void _link(Cov &cov, size_t offset) override final
        {
            _cov = &cov;
            _linked = true;
            _offset = offset;
            _t0 = 0.5 * (_cov->t[0] + _cov->t[_cov->n - 1]);
            _dt0 = _cov->t.array() - _t0;
        }

        void _compute() override final {
            _a = _sig * _sig;
            _la = sqrt(3.0) / _rho;
            _la2 = _la * _la;
            _x = _la * _dt0.array();
            _1mx = 1.0 - _x.array();
            
            _cov->A.array() += _a;
            _cov->U.col(_offset) = _a * _x.array();
            _cov->U.col(_offset + 1).setConstant(_a);
            _cov->V.col(_offset).setConstant(1.0);
            _cov->V.col(_offset + 1) = _1mx;
            _cov->phi.col(_offset) = (-_la * _cov->dt.array()).exp();
            _cov->phi.col(_offset + 1) = (-_la * _cov->dt.array()).exp();
        }

        void _deriv(bool calc_d2) override final {
            _cov->_dU.col(_offset) = _la * _a * _1mx.array();
            _cov->_dU.col(_offset + 1).setConstant(- _la * _a);
            if (calc_d2) {
                _cov->_dV.col(_offset).setConstant(_la);
                _cov->_dV.col(_offset + 1) = - _la * _x.array();
            }
        }
};

class Matern52Kernel : public Kernel
{
    double _sig, _rho;
    double _t0;
    VectorXd _dt0;
    // temps
    double _a, _la, _la2;
    VectorXd _x, _x2_3, _1mx;

    public:
        Matern52Kernel(double sig, double rho) : _sig(sig), _rho(rho), Kernel(3) {};

        void set_param(double sig, double rho)
        {
            _sig = sig;
            _rho = rho;
        }

        void _link(Cov &cov, size_t offset) override final
        {
            _cov = &cov;
            _linked = true;
            _offset = offset;
            _t0 = 0.5 * (_cov->t[0] + _cov->t[_cov->n - 1]);
            _dt0 = _cov->t.array() - _t0;
        }

        void _compute() override final {
            _a = _sig * _sig;
            _la = sqrt(5.0) / _rho;
            _la2 = _la * _la;
            _x = _la * _dt0.array();
            _x2_3 = _x.array().square() / 3.0;
            _1mx = 1.0 - _x.array();
            
            _cov->A.array() += _a;
            _cov->U.col(_offset) = _a * (_x.array() + _x2_3.array());
            _cov->U.col(_offset + 1).setConstant(_a);
            _cov->U.col(_offset + 2) = _a * _x.array();
            _cov->V.col(_offset).setConstant(1.0);
            _cov->V.col(_offset + 1) = _1mx.array() + _x2_3.array();
            _cov->V.col(_offset + 2) = -(2.0 / 3.0) * _x.array();
            VectorXd _e = (-_la * _cov->dt.array()).exp();
            _cov->phi.col(_offset) = _e;
            _cov->phi.col(_offset + 1) = _e;
            _cov->phi.col(_offset + 2) = _e;
        }

        void _deriv(bool calc_d2) override final {
            _cov->_dU.col(_offset) = _la * _a * (1.0 - _x.array() / 3.0 - _x2_3.array());
            _cov->_dU.col(_offset + 1).setConstant(- _la * _a);
            _cov->_dU.col(_offset + 2) = _la * _a * _1mx.array();
            if (calc_d2) {
                _cov->_dV.col(_offset).setConstant(_la);
                _cov->_dV.col(_offset + 1) = - _la * _x.array() / 3.0 * _1mx.array();
                _cov->_dV.col(_offset + 2) = - 2.0 / 3.0 * _la * (1.0 + _x.array());
            }
        }
};


class USHOKernel : public Kernel
{
    double _sig, _P0, _Q;
    double _a, _b, _la, _nu;
    double _eps = 1e-5;
    double _sqQ;
    VectorXd _nut, _cnut, _snut;

    public:
        USHOKernel(double sig, double P0, double Q) : _sig(sig), _P0(P0), _Q(Q), Kernel(2) {
            _sqQ = sqrt(max(4 * _Q*_Q - 1, _eps));
            _set_coefs();
        };

        void _set_coefs()
        {
            _a = _sig * _sig;
            _b = _a / _sqQ;
            _la = M_PI / (_P0 * _Q);
            _nu = _la * _sqQ;
        }

        void set_param(double sig, double P0, double Q)
        {
            _sig = sig;
            _P0 = P0;
            _Q = Q;
            _set_coefs();
        }

        void _compute() override final {
            _cov->A.array() += _a;
            _nut = _nu * _cov->t.array();
            _cnut = _nut.array().cos();
            _snut = _nut.array().sin();
            _cov->U.col(_offset) = _a * _cnut.array() + _b * _snut.array();
            _cov->U.col(_offset + 1) = _a * _snut.array() - _b * _cnut.array();
            _cov->V.col(_offset) = _cnut;
            _cov->V.col(_offset + 1) = _snut;
            VectorXd _e = (-_la * _cov->dt.array()).exp();
            _cov->phi.col(_offset) = _e;
            _cov->phi.col(_offset + 1) = _e;
        }

        void _deriv(bool calc_d2) override final {
            double da = - _la * _a + _nu * _b;
            double db = - _la * _b - _nu * _a;
            _cov->_dU.col(_offset) = da * _cnut.array() + db * _snut.array();
            _cov->_dU.col(_offset + 1) = da * _snut.array() - db * _cnut.array();
            if (calc_d2) {
                _cov->_dV.col(_offset) = _la * _cnut.array() - _nu * _snut.array();
                _cov->_dV.col(_offset + 1) = _la * _snut.array() + _nu * _cnut.array();
            }
        }
};

class OSHOKernel : public Kernel
{
    double _sig, _P0, _Q;
    double _sqQ;
    double _a1, _la1, _a2, _la2;
    double _eps = 1e-5;
    ExponentialKernel* _k1;
    ExponentialKernel* _k2;

    public:
        OSHOKernel(double sig, double P0, double Q) : _sig(sig), _P0(P0), _Q(Q), Kernel(2)
        {
            _set_coefs();
            _k1 = new ExponentialKernel(_a1, _la1);
            _k2 = new ExponentialKernel(_a2, _la2);
        }

        void _set_coefs()
        {
            _sqQ = sqrt(max(1 - 4 * _Q*_Q, _eps));
            double _a = _sig * _sig;
            double _la = M_PI / (_P0 * _Q);
            _a1 = _a * (1 + 1 / _sqQ) / 2;
            _la1 = _la * (1 - _sqQ);
            _a2 = _a * (1 - 1 / _sqQ) / 2;
            _la2 = _la * (1 + _sqQ);
        }

        void set_param(double sig, double P0, double Q)
        {
            _sig = sig;
            _P0 = P0;
            _Q = Q;
            _set_coefs();
            _k1->set_param(_a1, _la1);
            _k2->set_param(_a2, _la2);
        }

        void _link(Cov &cov, size_t offset) override final {
            _cov = &cov;
            _linked = true;
            _offset = offset;
            _k1->_link(cov, offset);
            _k2->_link(cov, offset + _k1->_r);
        }

        void _compute() override {
            _k1->_compute();
            _k2->_compute();
        }

        void _deriv(bool calc_d2) override {
            _k1->_deriv(calc_d2);
            _k2->_deriv(calc_d2);
        }
};

class SHOKernel : public Kernel
{
    double _sig, _P0, _Q;
    double _sqQ;
    double _a1, _la1, _a2, _la2;
    double _eps = 1e-5;
    USHOKernel* _usho;
    OSHOKernel* _osho;

    public:
        SHOKernel(double sig, double P0, double Q) : _sig(sig), _P0(P0), _Q(Q), Kernel(2)
        {
            _usho = new USHOKernel(sig, P0, Q);
            _osho = new OSHOKernel(sig, P0, Q);
        }

        void set_param(double sig, double P0, double Q)
        {
            _sig = sig;
            _P0 = P0;
            _Q = Q;
            if (_Q > 0.5) {
                _usho->set_param(sig, P0, Q);
            } else {
                _osho->set_param(sig, P0, Q);
            }

        }

        void _link(Cov &cov, size_t offset) override {
            _cov = &cov;
            _linked = true;
            _offset = offset;
            _usho->_link(*_cov, offset);
            _osho->_link(*_cov, offset);
        }

        void _compute() override {
            if (_Q > 0.5) {
                _usho->_compute();
            } else {
                _osho->_compute();
            }
        }

        void _deriv(bool calc_d2) override {
            if (_Q > 0.5) {
                _usho->_deriv(calc_d2);
            } else {
                _osho->_deriv(calc_d2);
            }
        }
};

// 
class MultiSeriesKernel : public Kernel
{
    Term* _kernel;
    Cov fake_cov;
    std::vector<Eigen::ArrayXi> _series_index;
    size_t _nseries;
    VectorXd _alpha, _beta;

    public:
        MultiSeriesKernel(Term &kernel, const std::vector<Eigen::ArrayXi> &series_index, VectorXd &alpha, VectorXd &beta) : Kernel(kernel._r)
        {
            _kernel = &kernel;
            _series_index = series_index;
            _nseries = series_index.size();
            _alpha = alpha;
            _beta = beta;
        }

        void _link(Cov &cov, size_t offset) override final
        {
            _cov = &cov;
            _linked = true;
            _offset = offset;
            fake_cov = Cov(cov.t, cov.dt, _r);
            _kernel->_link(fake_cov, 0);
        }

        void _compute() override final {
            _kernel->_cov->A.setConstant(0.0);
            _kernel->_compute();
            _kernel->_deriv(true);
            _kernel->_cov->_B = (_kernel->_cov->_dU.array() * _kernel->_cov->_dV.array()).rowwise().sum();
            // cov(GP, GP), cov(GP, dGP), cov(dGP, GP), cov(dGP, dGP)
            int k = 0;
            for (auto &ik : _series_index)
            {
                _cov->A(ik).array() += _alpha[k]*_alpha[k] * _kernel->_cov->A(ik).array();
                _cov->A(ik).array() += _beta[k]*_beta[k] * _kernel->_cov->_B(ik).array();
                for (size_t j = _offset; j < _offset + _r; j++)
                {
                    _cov->U.col(j)(ik) = _alpha[k] * _kernel->_cov->U.col(j)(ik);
                    _cov->V.col(j)(ik) = _alpha[k] * _kernel->_cov->V.col(j)(ik);
                    _cov->U.col(j)(ik) += _beta[k] * _kernel->_cov->_dU.col(j)(ik);
                    _cov->V.col(j)(ik) += _beta[k] * _kernel->_cov->_dV.col(j)(ik);
                }
                k++;
            }

            for (size_t j = _offset; j < _offset + _r; j++) {
                _cov->phi.col(j) = _kernel->_cov->phi.col(j);
            }
        }
};