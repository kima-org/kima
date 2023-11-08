#include "spleaf.h"


Cov::Cov(const VectorXd &t, const VectorXd &yerr, size_t _b, size_t _r)
    : t(t), yerr(yerr)
{
    assert( (t.size() == yerr.size()) && "t and yerr have the same size");

    n = t.size();
    r = 0;

    dt = t.segment(1, n-1).array() - t.segment(0, n-1).array();

    b = VectorXl::Zero(n);
    offsetrow = VectorXl(n);

    b = b.array().max(_b);
    r += _r;

    // n and r are set, can do all allocations
    A = VectorXd::Zero(n);
    U = MatrixXd(n, r);
    V = MatrixXd(n, r);
    phi = MatrixXd(n - 1, r);

    F = VectorXd(b.sum());

    D = VectorXd(n);
    W = MatrixXd(n, r);
    G = VectorXd(b.sum());
    _S = VectorXd(n * r * r);
    _Z = VectorXd((F.size() + n) * r);
    _f_solveL = VectorXd(n * r);
    // Kernel derivative
    _dU = MatrixXd(n, r);
    _dV = MatrixXd(n, r);
    _B = VectorXd(n);

    // allocations done, compute stuff and link the terms

    long sum = 0;
    for (size_t i = 0; i < n; i++)
    {
        offsetrow[i] = sum;
        sum += b[i] - 1;
    }
    // std::partial_sum((b.array() - 1).begin(), (b.array() - 1).end(), 
    //                 offsetrow.begin());
    // offsetrow.array() += 1;

    A.array() += yerr.array().square();
    
}

// Cov::Cov(const VectorXd &t, Term& err, Term& gp) : t(t)
// {
//     n = t.size();
//     r = 0;

//     dt = t.segment(1, n-1).array() - t.segment(0, n-1).array();

//     b = VectorXl::Zero(n);
//     offsetrow = VectorXl(n);

//     if (!instanceof<Noise>(&err)) {
//         cout << "something is wrong..." << endl;
//     }
//     if (!instanceof<Kernel>(&gp)) {
//         cout << "something is wrong..." << endl;
//     }

//     err._link(*this);
//     b = b.array().max(err._b);

//     gp._link(*this, r);
//     r += gp._r;

//     A = VectorXd::Zero(n);
//     U = MatrixXd(n, r);
//     V = MatrixXd(n, r);
//     phi = MatrixXd(n - 1, r);

//     long sum = 0;
//     for (size_t i = 0; i < n; i++)
//     {
//         offsetrow[i] = sum;
//         sum += b[i] - 1;
//     }
//     // std::partial_sum((b.array() - 1).begin(), (b.array() - 1).end(), 
//     //                  offsetrow.begin());
//     // offsetrow.array() += 1;

//     F = VectorXd(b.sum());

//     err._compute();
//     gp._compute();

//     D = VectorXd(n);
//     W = MatrixXd(n, r);
//     G = VectorXd(b.sum());
//     _S = VectorXd(n * r * r);
//     _Z = VectorXd((F.size() + n) * r);

//     compute_cholesky();

//     _f_solveL = VectorXd(n * r);
//     // Kernel derivative
//     _dU = MatrixXd(n, r);
//     _dV = MatrixXd(n, r);

//     terms.push_back(&err);
//     terms.push_back(&gp);
// }


FakeCov::FakeCov(const VectorXd &_t, const VectorXd &_dt, size_t r)
{
    t = _t;
    dt = _dt;
    n = t.size();
    r = r;

    b = VectorXl::Zero(n);
    offsetrow = VectorXl(n);

    A = VectorXd::Zero(n);
    U = MatrixXd(n, r);
    V = MatrixXd(n, r);
    phi = MatrixXd(n - 1, r);
    F = VectorXd(b.sum());

    D = VectorXd(n);
    W = MatrixXd(n, r);
    G = VectorXd(b.sum());
    _S = VectorXd(n * r * r);
    _Z = VectorXd((F.size() + n) * r);

    _dU = MatrixXd(n, r);
    _dV = MatrixXd(n, r);
    _B = VectorXd(n);
}


// template <class... Args> 
// Cov::Cov(const VectorXd &t, Args... terms) : t(t)
// {
//     cout << "called crazy constructor" << endl;
//     for(const auto t : {terms...}) {
//         if(instanceof<Noise>(&t)) {
//             cout << "term is a Noise" << endl;
//         } else if (instanceof<Kernel>(&t)) {
//             cout << "term is a Kernel" << endl;
//         } else {
//             cout << "something is wrong..." << endl;
//         }
//     }
// };


// void Term::_reset_cov() {
//     _cov->A.setConstant(0.0);
//     _cov->F.setConstant(0.0);
// }



class Publicist : public Cov { // helper type for exposing protected functions
public:
    // inherited with different access modifier
    using Cov::terms;
    // 
    using Cov::dt;
    using Cov::A;
    using Cov::D;
    using Cov::phi;
    using Cov::U;
    using Cov::V;
    using Cov::_dU;
    using Cov::_dV;
};


NB_MODULE(spleaf, m) {
    nb::class_<Cov>(m, "Cov")
        .def(nb::init<const VectorXd&, const VectorXd&, size_t, size_t>())
        .def("__repr__", &Cov::to_string)
        // .def("__init__", [](Cov *c, const VectorXd& t, nb::kwargs kwargs){
        //     auto terms = kwargs.values();
        //     // nb::print(terms);
        //     new (c) Cov(t, *terms);
        // })
        .def_ro("terms", &Publicist::terms)
        .def_ro("dt", &Publicist::dt)
        .def_ro("A", &Publicist::A)
        .def_ro("D", &Publicist::D)
        .def_ro("phi", &Publicist::phi)
        .def_ro("U", &Publicist::U)
        .def_ro("V", &Publicist::V)
        .def_ro("_dU", &Publicist::_dU)
        .def_ro("_dV", &Publicist::_dV)
        // 
        .def("logdet", &Cov::logdet,
             "Compute the (natural) logarithm of the determinant of the matrix")
        .def("chi2", &Cov::chi2, "y"_a,
             "Compute $y^T C^{-1} y$ for a vector of residuals $y$")
        .def("loglike", &Cov::loglike,
             "Compute the (natural) logarithm of the likelihood for a vector of residuals $y$")
        //
        .def("perturb", &Cov::perturb)
        .def("generate", &Cov::generate);

    nb::class_<Term>(m, "Term", "Generic class for covariance terms")
        .def("is_linked", &Term::is_linked);

    nb::class_<Noise, Term>(m, "Noise", "Generic class for covariance noise terms");
    nb::class_<Kernel, Term>(m, "Kernel", "Generic class for covariance kernel (Gaussian process) terms");
    
    // noises
    nb::class_<Error, Noise>(m, "Error")
        .def(nb::init<VectorXd&>())
        .def("__repr__", &Error::to_string);

    nb::class_<Jitter, Noise>(m, "Jitter")
        .def(nb::init<double>())
        .def("__repr__", &Jitter::to_string)
        .def("set_param", &Jitter::set_param)
        .def_prop_rw("sig_prior",
            [](Jitter &k) { return k._sig_prior; },
            [](Jitter &k, distribution &d) { k._sig_prior = d; },
            "Prior for the amplitude");

    // kernels
    nb::class_<ExponentialKernel, Kernel>(m, "ExponentialKernel")
        .def(nb::init<double, double>())
        .def("__repr__", &ExponentialKernel::to_string)
        .def("set_param", &ExponentialKernel::set_param)
        .def_prop_rw("a_prior",
            [](ExponentialKernel &k) { return k._a_prior; },
            [](ExponentialKernel &k, distribution &d) { k._a_prior = d; },
            "Prior for the amplitude")
        .def_prop_rw("la_prior",
            [](ExponentialKernel &k) { return k._la_prior; },
            [](ExponentialKernel &k, distribution &d) { k._la_prior = d; },
            "Prior for the scale");

    // nb::class_<QuasiperiodicKernel, Kernel>(m, "QuasiperiodicKernel")
    //     .def(nb::init<double, double, double, double>())
    //     .def("__repr__", &QuasiperiodicKernel::to_string)
    //     .def("set_param", &QuasiperiodicKernel::set_param);

    nb::class_<Matern32Kernel, Kernel>(m, "Matern32Kernel")
        .def(nb::init<double, double>())
        .def("__repr__", &Matern32Kernel::to_string)
        .def("set_param", &Matern32Kernel::set_param)
        .def_prop_rw("sig_prior",
            [](Matern32Kernel &k) { return k._sig_prior; },
            [](Matern32Kernel &k, distribution &d) { k._sig_prior = d; },
            "Prior for the amplitude")
        .def_prop_rw("rho_prior",
            [](Matern32Kernel &k) { return k._rho_prior; },
            [](Matern32Kernel &k, distribution &d) { k._rho_prior = d; },
            "Prior for the scale");;

    nb::class_<Matern52Kernel, Kernel>(m, "Matern52Kernel")
        .def(nb::init<double, double>())
        .def("__repr__", &Matern52Kernel::to_string)
        .def("set_param", &Matern52Kernel::set_param)
        .def_prop_rw("sig_prior",
            [](Matern52Kernel &k) { return k._sig_prior; },
            [](Matern52Kernel &k, distribution &d) { k._sig_prior = d; },
            "Prior for the amplitude")
        .def_prop_rw("rho_prior",
            [](Matern52Kernel &k) { return k._rho_prior; },
            [](Matern52Kernel &k, distribution &d) { k._rho_prior = d; },
            "Prior for the scale");

    nb::class_<USHOKernel, Kernel>(m, "USHOKernel")
        .def(nb::init<double, double, double>())
        .def("__repr__", &USHOKernel::to_string)
        .def("set_param", &USHOKernel::set_param)
        .def_prop_rw("sig_prior",
            [](USHOKernel &k) { return k._sig_prior; },
            [](USHOKernel &k, distribution &d) { k._sig_prior = d; },
            "Prior for the amplitude")
        .def_prop_rw("P0_prior",
            [](USHOKernel &k) { return k._P0_prior; },
            [](USHOKernel &k, distribution &d) { k._P0_prior = d; },
            "Prior for period")
        .def_prop_rw("Q_prior",
            [](USHOKernel &k) { return k._Q_prior; },
            [](USHOKernel &k, distribution &d) { k._Q_prior = d; },
            "Prior for Q");
    
    nb::class_<OSHOKernel, Kernel>(m, "OSHOKernel")
        .def(nb::init<double, double, double>())
        .def("__repr__", &OSHOKernel::to_string)
        .def("set_param", &OSHOKernel::set_param)
        .def_prop_rw("sig_prior",
            [](OSHOKernel &k) { return k._sig_prior; },
            [](OSHOKernel &k, distribution &d) { k._sig_prior = d; },
            "Prior for the amplitude")
        .def_prop_rw("P0_prior",
            [](OSHOKernel &k) { return k._P0_prior; },
            [](OSHOKernel &k, distribution &d) { k._P0_prior = d; },
            "Prior for period")
        .def_prop_rw("Q_prior",
            [](OSHOKernel &k) { return k._Q_prior; },
            [](OSHOKernel &k, distribution &d) { k._Q_prior = d; },
            "Prior for Q");

    nb::class_<SHOKernel, Kernel>(m, "SHOKernel")
        .def(nb::init<double, double, double>())
        .def("__repr__", &SHOKernel::to_string)
        .def("set_param", &SHOKernel::set_param)
        .def_prop_rw("sig_prior",
            [](SHOKernel &k) { return k._sig_prior; },
            [](SHOKernel &k, distribution &d) { k._sig_prior = d; },
            "Prior for the amplitude")
        .def_prop_rw("P0_prior",
            [](SHOKernel &k) { return k._P0_prior; },
            [](SHOKernel &k, distribution &d) { k._P0_prior = d; },
            "Prior for period")
        .def_prop_rw("Q_prior",
            [](SHOKernel &k) { return k._Q_prior; },
            [](SHOKernel &k, distribution &d) { k._Q_prior = d; },
            "Prior for Q");

    // special kernels
    nb::class_<MultiSeriesKernel, Kernel>(m, "MultiSeriesKernel")
        .def(nb::init<Term&, const std::vector<Eigen::ArrayXi>&, VectorXd&, VectorXd&>())
        .def("__repr__", &MultiSeriesKernel::to_string);
}