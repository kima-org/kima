#pragma once

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

    public:
        Cov* _cov;
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

        void _relink(Cov &cov) { _cov = &cov; }

        // void _reset_cov();

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
        Noise(): Term(0, 0) {};
        Noise(size_t b) : Term(b, 0) {};
};

class Kernel : public Term {

    public:
        Kernel(): Term(0, 0) {};
        Kernel(size_t r) : Term(0, r) {};
};





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
        VectorXd t, yerr;

        size_t n;
        size_t r;
        VectorXd dt;
        VectorXl b, offsetrow;  // long
        VectorXd A, D;
        MatrixXd U, V, W, phi;
        MatrixXd _dU, _dV;
        VectorXd F, G;
        VectorXd _S, _Z;
        VectorXd _f_solveL;
        VectorXd _B;

    public:
        std::vector<Term*> terms {};

    public:
        Cov() {};
        Cov(const VectorXd &t, const VectorXd &yerr, size_t _b, size_t _r);
        // Cov(const VectorXd &t, Term& err, Term& jit, Term& gp);
        // template <class... Args> 
        // Cov(const VectorXd &t, Args... terms);

        void link(Term& gp)
        {
            gp._link(*this, 0);
            gp._compute();
            compute_cholesky();
            terms.push_back(&gp);
        }

        void compute_cholesky() {
            assert( (n == A.size()) && "Cov:compute_cholesky - A has the right dimensions");
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
            assert( (n == y.size()) && "Cov:chi2 - y has the right dimensions");
            VectorXd x = solveL(y);
            return (x.array().square() / D.array()).sum();
        }

        double loglike(VectorXd &y)
        {
            assert( (n == y.size()) && "Cov:loglike - y has the right dimensions");
            return -0.5 * (chi2(y) + logdet() + n * log(2.0 * M_PI));
        }

        void reset()
        {
            A.setConstant(0.0);
            A.array() += yerr.array().square();
            F.setConstant(0.0);
            for (Term* term : terms)
                term->_relink(*this);
        }

        void generate(DNest4::RNG& rng) {
            assert( (terms.size() >= 1) && "Cov.terms has size >= 1");
            assert( (n == A.size()) && "Cov:generate - A has the right dimensions");

            for (Term* term : terms)
                term->_generate(rng);
            
            reset();

            // recompute
            for (Term* term : terms)
                term->_compute();

            // redo cholesky
            compute_cholesky();
        }

        void perturb(DNest4::RNG& rng) {
            for (Term* term : terms)
                term->_perturb(rng);
            
            reset();

            // recompute
            for (Term* term : terms)
                term->_compute();

            // redo cholesky
            compute_cholesky();
        }

        void print(std::ostream& out) const {
            for (Term* term : terms)
                term->_print(out);
        }

        void description(std::string& desc, std::string sep) const {
            for (Term* term : terms)
                term->_description(desc, sep);
        }

        std::string to_string() const {
            std::string out = "Cov(";
            for (Term* term : terms)
                out += term->to_string() + ", ";
            out += ")";
            return out;
        }
};


class FakeCov : public Cov
{
    public:
        FakeCov() {};
        FakeCov(const VectorXd &t, const VectorXd &dt, size_t r);
};


class Error : public Noise
{
    const VectorXd _sig;

    public:
        Error(const VectorXd& sig) : Noise(0), _sig(sig) {}

        void set_param(double sig) {}

        void _compute() override {
            _cov->A.array() += _sig.array().square();
        };

        bool _priors_defined() { return true; }

        // _generate does nothing, not overridden

        // _perturb does nothing, not overridden

        // _print does nothing, not overridden

        // _description does nothing, not overridden

        std::string to_string() const override {
            return "Error";
        }
};

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

class ExponentialKernel : public Kernel
{
    double _a, _la;

    public:
        distribution _a_prior, _la_prior;

    public:
        ExponentialKernel(double a, double la) : Kernel(1), _a(a), _la(la) {};

        void set_param(double a, double la)
        {
            _a = a;
            _la = la;
        }

        // _link not overridden

        void _compute() override {
            _cov->A.array() += _a;
            _cov->U.col(_offset).setConstant(_a);
            _cov->V.col(_offset).setConstant(1.0);
            _cov->phi.col(_offset) = (-_la * _cov->dt.array()).exp();
        }

        void _deriv(bool calc_d2) override {
            _cov->_dU.col(_offset).setConstant(- _la * _a);
            if (calc_d2) {
                _cov->_dV.col(_offset).setConstant(_la);
            }
        }

        bool _priors_defined() {
            if (_a_prior && _la_prior)
                return true;
            return false;
        };

        void _generate(DNest4::RNG& rng) override {
            assert( (_a_prior) && "ExponentialKernel::a_prior has been assigned");
            assert( (_la_prior) && "ExponentialKernel::la_prior has been assigned");
            _a = _a_prior->generate(rng);
            _la = _la_prior->generate(rng);
        }

        void _perturb(DNest4::RNG& rng) override {
            _a_prior->perturb(_a, rng);
            _la_prior->perturb(_la, rng);
        }

        void _print(std::ostream& out) const override {
            out << _a << " " << _la << " ";
        };

        void _description(std::string& desc, std::string sep) const override {
            desc += "a" + sep + "la" + sep;
        };

        std::string to_string() const override {
            return "ExponentialKernel(a=" + std::to_string(_a) + ", la=" + std::to_string(_la) + ")";
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
        distribution _sig_prior, _rho_prior;

    public:
        Matern32Kernel(double sig, double rho) : Kernel(2), _sig(sig), _rho(rho) {};

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

        void _deriv(bool calc_d2) override {
            _cov->_dU.col(_offset) = _la * _a * _1mx.array();
            _cov->_dU.col(_offset + 1).setConstant(- _la * _a);
            if (calc_d2) {
                _cov->_dV.col(_offset).setConstant(_la);
                _cov->_dV.col(_offset + 1) = - _la * _x.array();
            }
        }

        bool _priors_defined() {
            if (_sig_prior && _rho_prior)
                return true;
            return false;
        };

        void _generate(DNest4::RNG& rng) override {
            assert( (_sig_prior) && "Matern32Kernel::sig_prior has been assigned");
            assert( (_rho_prior) && "Matern32Kernel::rho_prior has been assigned");
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
            return "Matern32Kernel(sig=" + std::to_string(_sig) + ", rho=" + std::to_string(_rho) + ")";
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
            assert(_linked && "Matern52Kernel Term is linked");
            assert( (_cov->A.size() == _cov->t.size()) && "A has correct dimensions");
            assert( (_cov->U.rows() == _cov->t.size()) && "U has correct dimensions");
            assert( (_cov->V.rows() == _cov->t.size()) && "V has correct dimensions");
            assert( (_cov->U.cols() == _r) && "U has correct dimensions");
            assert( (_cov->V.cols() == _r) && "V has correct dimensions");

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

class USHOKernel : public Kernel
{
    double _sig, _P0, _Q;
    double _a, _b, _la, _nu;
    double _eps = 1e-5;
    double _sqQ;
    VectorXd _nut, _cnut, _snut;

    public:
        distribution _sig_prior, _P0_prior, _Q_prior;

    public:
        USHOKernel(double sig, double P0, double Q) 
            : Kernel(2), _sig(sig), _P0(P0), _Q(Q)
        {
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

        // _link not overridden

        void _compute() override {
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

        void _deriv(bool calc_d2) override {
            double da = - _la * _a + _nu * _b;
            double db = - _la * _b - _nu * _a;
            _cov->_dU.col(_offset) = da * _cnut.array() + db * _snut.array();
            _cov->_dU.col(_offset + 1) = da * _snut.array() - db * _cnut.array();
            if (calc_d2) {
                _cov->_dV.col(_offset) = _la * _cnut.array() - _nu * _snut.array();
                _cov->_dV.col(_offset + 1) = _la * _snut.array() + _nu * _cnut.array();
            }
        }

        bool _priors_defined() {
            if (_sig_prior && _P0_prior && _Q_prior)
                return true;
            return false;
        };

        void _generate(DNest4::RNG& rng) override {
            assert( (_sig_prior) && "USHOKernel::sig_prior has been assigned");
            assert( (_P0_prior) && "USHOKernel::P0_prior has been assigned");
            assert( (_Q_prior) && "USHOKernel::Q_prior has been assigned");
            _sig = _sig_prior->generate(rng);
            _P0 = _P0_prior->generate(rng);
            _Q = _Q_prior->generate(rng);
        }

        void _perturb(DNest4::RNG& rng) override {
            _sig_prior->perturb(_sig, rng);
            _P0_prior->perturb(_P0, rng);
            _Q_prior->perturb(_Q, rng);
        }

        void _print(std::ostream& out) const override {
            out << _sig << " " << _P0 << " " << _Q << " ";
        };

        void _description(std::string& desc, std::string sep) const override {
            desc += "sig" + sep + "P0" + sep + "Q" + sep;
        };

        std::string to_string() const override {
            return "USHOKernel(sig=" + std::to_string(_sig) + ", P0=" + std::to_string(_P0) + ", Q=" + std::to_string(_Q) + ")";
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
        distribution _sig_prior, _P0_prior, _Q_prior;

    public:
        OSHOKernel(double sig, double P0, double Q) : Kernel(2), _sig(sig), _P0(P0), _Q(Q)
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

        void _link(Cov &cov, size_t offset) override {
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

        bool _priors_defined() {
            if (_sig_prior && _P0_prior && _Q_prior)
                return true;
            return false;
        };

        void _generate(DNest4::RNG& rng) override {
            assert( (_sig_prior) && "OSHOKernel::sig_prior has been assigned");
            assert( (_P0_prior) && "OSHOKernel::P0_prior has been assigned");
            assert( (_Q_prior) && "OSHOKernel::Q_prior has been assigned");
            _sig = _sig_prior->generate(rng);
            _P0 = _P0_prior->generate(rng);
            _Q = _Q_prior->generate(rng);
            _set_coefs();
            _k1->set_param(_a1, _la1);
            _k2->set_param(_a2, _la2);
        }

        void _perturb(DNest4::RNG& rng) override {
            _sig_prior->perturb(_sig, rng);
            _P0_prior->perturb(_P0, rng);
            _Q_prior->perturb(_Q, rng);
            _set_coefs();
            _k1->set_param(_a1, _la1);
            _k2->set_param(_a2, _la2);
        }

        void _print(std::ostream& out) const override {
            out << _sig << " " << _P0 << " " << _Q << " ";
        };

        void _description(std::string& desc, std::string sep) const override {
            desc += "sig" + sep + "P0" + sep + "Q" + sep;
        };

        std::string to_string() const override {
            return "OSHOKernel(sig=" + std::to_string(_sig) + ", P0=" + std::to_string(_P0) + ", Q=" + std::to_string(_Q) + ")";
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
        distribution _sig_prior, _P0_prior, _Q_prior;

    public:
        SHOKernel(double sig, double P0, double Q) : Kernel(2), _sig(sig), _P0(P0), _Q(Q)
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

        bool _priors_defined() {
            if (_sig_prior && _P0_prior && _Q_prior)
                return true;
            return false;
        };

        void _generate(DNest4::RNG& rng) override {
            assert( (_sig_prior) && "SHOKernel::sig_prior has been assigned");
            assert( (_P0_prior) && "SHOKernel::P0_prior has been assigned");
            assert( (_Q_prior) && "SHOKernel::Q_prior has been assigned");
            _sig = _sig_prior->generate(rng);
            _P0 = _P0_prior->generate(rng);
            _Q = _Q_prior->generate(rng);
            if (_Q > 0.5) {
                _usho->set_param(_sig, _P0, _Q);
            } else {
                _osho->set_param(_sig, _P0, _Q);
            }
        }

        void _perturb(DNest4::RNG& rng) override {
            _sig_prior->perturb(_sig, rng);
            _P0_prior->perturb(_P0, rng);
            _Q_prior->perturb(_Q, rng);
            if (_Q > 0.5) {
                _usho->set_param(_sig, _P0, _Q);
            } else {
                _osho->set_param(_sig, _P0, _Q);
            }
        }

        void _print(std::ostream& out) const override {
            out << _sig << " " << _P0 << " " << _Q << " ";
        };

        void _description(std::string& desc, std::string sep) const override {
            desc += "sig" + sep + "P0" + sep + "Q" + sep;
        };

        std::string to_string() const override {
            return "SHOKernel(sig=" + std::to_string(_sig) + ", P0=" + std::to_string(_P0) + ", Q=" + std::to_string(_Q) + ")";
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
        MultiSeriesKernel() : Kernel(0) {};
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
            _kernel->_generate(rng);
        }

        void _perturb(DNest4::RNG& rng) override {
            _kernel->_perturb(rng);
        }

        void _print(std::ostream& out) const override {
            _kernel->_print(out);
        }

        void _description(std::string& desc, std::string sep) const override {
            _kernel->_description(desc, sep);
        };

        std::string to_string() const override {
            return "MultiSeriesKernel(" + _kernel->to_string() + ")";
        }
};
