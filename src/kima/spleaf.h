#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <vector>
#include <memory>
#include <string>

#include <iostream>
using namespace std;

extern "C"
{
    #include "libspleaf.h"
}

#include "Distributions/ContinuousDistribution.h"
#include "RNG.h"
using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
typedef Eigen::Matrix<long, -1, 1> VectorXl;
typedef Eigen::Matrix<double, -1, 1> VectorXd;
typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> MatrixXd;

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;
using namespace nb::literals;
#include "nb_shared.h"


template<typename Base, typename T>
inline bool instanceof(const T *ptr) {
   return dynamic_cast<const Base*>(ptr) != nullptr;
}


class Cov;


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

        template <typename... args>
        void set_param(args...) {}

        virtual void _link(Cov &cov) {
            _cov = &cov;
            _linked = true;
        }

        virtual void _link(Cov &cov, size_t offset) {
            _cov = &cov;
            _linked = true;
            _offset = offset;
        }

        virtual bool is_linked() { return _linked; }

        virtual void _compute() {};
        virtual void _deriv(bool calc_d2) {};
        virtual bool _priors_defined() { return false; };
        virtual void _perturb(DNest4::RNG& rng) {};
        virtual void _generate(DNest4::RNG& rng) {};
        virtual void _print(std::ostream& out) const {};
        virtual void _description(std::string& desc, std::string sep) const {};
        virtual std::string to_string() const { return "Term"; };
};


class Noise : public Term {

    public:
        Noise(size_t b) : Term(b, 0) {};
};

class Kernel : public Term {

    public:
        Kernel(size_t r) : Term(0, r) {};
};





class Cov
{
    friend class Term;
    friend class Noise;
    friend class Kernel;
    // friend class Error;
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
        VectorXd t, yerr;

        size_t n;
        size_t r;
        VectorXd dt;
        VectorXl b, offsetrow;
        VectorXd A, D;
        // VectorXd _grad_A, _grad_B;
        MatrixXd U, V, W, phi;
        MatrixXd _dU, _dV;
        // MatrixXd _grad_U, _grad_V, _grad_phi;
        // MatrixXd _grad_dU, _grad_dV;
        VectorXd F, G;
        VectorXd _S, _Z;
        VectorXd _f_solveL;
        VectorXd _B;

    public:
        std::vector<Term*> terms {};

    public:
        Cov() {};
        Cov(const VectorXd &t, const VectorXd &yerr, Term& gp);
        // Cov(const VectorXd &t, Term& err, Term& jit, Term& gp);
        // template <class... Args> 
        // Cov(const VectorXd &t, Args... terms);


        void compute_cholesky() {
            assert( (n == A.size()) && "A has the right dimensions");
            spleaf_cholesky(n, r, offsetrow.data(), b.data(),
                            A.data(), U.data(), V.data(), phi.data(), F.data(),
                            D.data(), W.data(), G.data(), _S.data(), _Z.data());
        }

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

        void generate(DNest4::RNG& rng) {
            for (Term* term : terms)
                term->_generate(rng);
        }

        void perturb(DNest4::RNG& rng) {
            cout << "perturbing" << endl;
            for (Term* term : terms)
                term->_perturb(rng);
        }

        void print(std::ostream& out) const {
            for (Term* term : terms)
                term->_print(out);
        }

        void description(std::string& desc, std::string sep) const {
            for (Term* term : terms)
                term->_description(desc, sep);
        }

        std::string to_string() const { return "Cov"; }
};


class FakeCov : public Cov
{
    public:
        FakeCov() {};
        FakeCov(const VectorXd &t, const VectorXd &dt, size_t r);
};


// class Error : public Noise
// {
//     const VectorXd _sig;

//     public:
//         Error(const VectorXd& sig) : Noise(0), _sig(sig) {}

//         void set_param(double sig) {}

//         void _compute() override {
//             _cov->A.array() += _sig.array().square();
//         };

//         bool _priors_defined() { return true; }

//         void _generate(DNest4::RNG& rng) override {
//             cout << "called _generate of Error" << endl;
//         }

//         std::string to_string() const override {
//             return "Error";
//         }
// };

class Jitter : public Noise
{
    double _sig;

    public:
        distribution _sig_prior;
    
    public:
        Jitter(double sig) : Noise(0), _sig(sig) {};

        void set_param(double sig) { _sig = sig; };

        void _compute() override {
            _cov->A.array() += _sig * _sig;
        }

        void _generate(DNest4::RNG& rng) override {
            _sig = _sig_prior->generate(rng);
        }

        void _perturb(DNest4::RNG& rng) override {
            _sig_prior->perturb(_sig, rng);
        }

        void _print(std::ostream& out) const override {
            out << _sig << " ";
        };

        void _description(std::string& desc, std::string sep) const override {
            desc += "sig" + sep;
        };

        std::string to_string() const override {
            return "Jitter(sig=" + std::to_string(_sig) + ")";
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
        distribution _sig_prior, _rho_prior;

    public:
        Matern52Kernel(double sig, double rho) : Kernel(3), _sig(sig), _rho(rho) {};

        void set_param(double sig, double rho)
        {
            _sig = sig;
            _rho = rho;
        }

        void _link(Cov &cov, size_t offset) override
        {
            _cov = &cov;
            _linked = true;
            _offset = offset;
            _t0 = 0.5 * (_cov->t[0] + _cov->t[_cov->n - 1]);
            _dt0 = _cov->t.array() - _t0;
        }

        void _compute() override {
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

        void _deriv(bool calc_d2) override {
            _cov->_dU.col(_offset) = _la * _a * (1.0 - _x.array() / 3.0 - _x2_3.array());
            _cov->_dU.col(_offset + 1).setConstant(- _la * _a);
            _cov->_dU.col(_offset + 2) = _la * _a * _1mx.array();
            if (calc_d2) {
                _cov->_dV.col(_offset).setConstant(_la);
                _cov->_dV.col(_offset + 1) = - _la * _x.array() / 3.0 * _1mx.array();
                _cov->_dV.col(_offset + 2) = - 2.0 / 3.0 * _la * (1.0 + _x.array());
            }
        }

        bool _priors_defined() {
            if (_sig_prior && _rho_prior)
                return true;
            return false;
        };

        void _generate(DNest4::RNG& rng) override {
            // cout << "called _generate of Matern52Kernel" << endl; 
            assert( (_sig_prior) && "Matern52Kernel::sig_prior has been assigned");
            assert( (_rho_prior) && "Matern52Kernel::rho_prior has been assigned");
            _sig = _sig_prior->generate(rng);
            _rho = _rho_prior->generate(rng);
        }

        void _perturb(DNest4::RNG& rng) override {
            _sig_prior->perturb(_sig, rng);
            _rho_prior->perturb(_rho, rng);
        }

        void _print(std::ostream& out) const override {
            out << _sig << " " << _rho << " ";
        };

        void _description(std::string& desc, std::string sep) const override {
            desc += "sig" + sep + "rho" + sep;
        };

        std::string to_string() const override {
            return "Matern52Kernel(sig=" + std::to_string(_sig) + ", rho=" + std::to_string(_rho) + ")";
        }
};


class MultiSeriesKernel : public Kernel
{
    Term* _kernel;
    std::vector<Eigen::ArrayXi> _series_index;
    VectorXd _alpha, _beta;
    size_t _nseries;
    FakeCov fake_cov;

    public:
        MultiSeriesKernel(Term &kernel, const std::vector<Eigen::ArrayXi> &series_index, VectorXd &alpha, VectorXd &beta) : Kernel(kernel._r)
        {
            _kernel = &kernel;
            _series_index = series_index;
            _alpha = alpha;
            _beta = beta;
            _nseries = series_index.size();
        }

        void _link(Cov &cov, size_t offset) override
        {
            _cov = &cov;
            _linked = true;
            _offset = offset;
            fake_cov = FakeCov(cov.t, cov.dt, _r);
            _kernel->_link(fake_cov, 0);
        }

        void _compute() override {
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

        bool _priors_defined() { return _kernel->_priors_defined(); };

        void _generate(DNest4::RNG& rng) override {
            // cout << "called _generate of MultiSeriesKernel" << endl; 
            _kernel->_generate(rng);
        }

        void _perturb(DNest4::RNG& rng) override {
            _kernel->_perturb(rng);
        }

        void _print(std::ostream& out) const override {
            _kernel->_print(out);
        }

        std::string to_string() const override {
            return "MultiSeriesKernel(" + _kernel->to_string() + ")";
        }
};
