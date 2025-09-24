#include "GP.h"

VectorXd sample(const Eigen::MatrixXd &K, double white_noise_variance)
{
    static std::mt19937 gen{ std::random_device{}() };
    static std::normal_distribution<> dist;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(K.rows(), K.cols()) * white_noise_variance;
    Eigen::LLT<Eigen::MatrixXd> cholesky = (K + I).llt();
    MatrixXd L = cholesky.matrixL();
    return L * Eigen::VectorXd{ K.cols() }.unaryExpr([&](auto x) { return dist(gen); });
}


Eigen::MatrixXd QP(std::vector<double> &t, double eta1, double eta2, double eta3, double eta4)
{
    size_t N = t.size();
    Eigen::MatrixXd C(N, N);
    double a = eta1 * eta1;
    double p = M_PI / eta3;
    double b = -2.0 / (eta4 * eta4);

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = i; j < N; j++)
        {
            double r = t[i] - t[j];
            double s1 = r / eta2;
            double s2 = sin(p * r);
            C(i, j) = a * exp(-0.5*s1*s1 + b*s2*s2);
            C(j, i) = C(i, j);
        }
    }
    return C;
}


Eigen::MatrixXd PER(std::vector<double> &t, double eta1, double eta3, double eta4)
{
    size_t N = t.size();
    Eigen::MatrixXd C(N, N);
    double a = eta1 * eta1;
    double p = M_PI / eta3;
    double b = -2.0 / (eta4 * eta4);

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = i; j < N; j++)
        {
            double r = t[i] - t[j];
            double s = sin(p * r);
            C(i, j) = a * exp(b * s * s);
            C(j, i) = C(i, j);
        }
    }
    return C;
}


void spleaf_ExponentialKernel::operator()(
    const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi)
{
    A.array() += sig;
    U.col(offset).setConstant(sig);
    V.col(offset).setConstant(1.0);
    phi.col(offset) = (-la * dt.array()).exp();
}

void spleaf_ExponentialKernel::deriv(
    const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV)
{
    double a = sig;
    dU.col(offset).setConstant(-la * a);
    dV.col(offset).setConstant(la);
}


spleaf_Matern32Kernel::spleaf_Matern32Kernel(const VectorXd &t, std::array<double, 2> params, size_t offset)
: sig(params[0]), rho(params[1]), offset(offset)
{
    size_t n = t.size();
    a = sig * sig;
    la = sqrt(3.0) / rho;
    t0 = 0.5 * (t[0] + t[n - 1]);
    dt0 = t.array() - t0;
    x = la * dt0.array();
    _1mx = 1.0 - x.array();
};

void spleaf_Matern32Kernel::operator()(
    const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi)
{
    A.array() += a;
    U.col(offset) = a * x.array();
    V.col(offset).setConstant(1.0);
    U.col(offset + 1).setConstant(a);
    V.col(offset + 1) = _1mx;
    phi.col(offset) = (-la * dt.array()).exp();
    phi.col(offset + 1) = (-la * dt.array()).exp();
}

void spleaf_Matern32Kernel::deriv(
    const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV)
{
    dU.col(offset) = la * a * _1mx.array();
    dU.col(offset + 1).setConstant(-la * a);
    dV.col(offset).setConstant(la);
    dV.col(offset + 1) = -la * x.array();
}


// constructor
spleaf_SHOKernel::spleaf_SHOKernel(const VectorXd &t, std::array<double, 3> params, size_t offset) 
: sig(params[0]), P0(params[1]), Q(params[2]), offset(offset)
{
    if (Q > 0.5) // USHO
    {
        sqQ = sqrt(std::max(4*Q*Q - 1.0, eps));
        a = sig * sig;
        la = M_PI / (P0 * Q);
        b = a / sqQ;
        nu = la * sqQ;
    }
    else // OSHO
    {
        sqQ = sqrt(std::max(1.0 - 4*Q*Q, eps));
        a = sig * sig;
        la = M_PI / (P0 * Q);
        a1 = a * (1.0 + 1.0 / sqQ) / 2.0;
        la1 = la * (1.0 - sqQ);
        a2 = a * (1.0 - 1.0 / sqQ) / 2.0;
        la2 = la * (1.0 + sqQ);
        nu = 0.0; // no-op
    }
    _nut = nu * t.array();
    _cnut = _nut.array().cos();
    _snut = _nut.array().sin();
}

void spleaf_SHOKernel::operator()(
    const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi)
{
    if (Q > 0.5)
    {
        A.array() += a;
        U.col(offset) = a * _cnut.array() + b * _snut.array();
        V.col(offset) = _cnut;
        U.col(offset + 1) = a * _snut.array() - b * _cnut.array();
        V.col(offset + 1) = _snut;
        VectorXd _e = (-la * dt.array()).exp();
        phi.col(offset) = _e;
        phi.col(offset + 1) = _e;
    }
    else
    {
        auto exp1 = spleaf_ExponentialKernel(t, {a1, la1}, 0);
        auto exp2 = spleaf_ExponentialKernel(t, {a2, la2}, 1);
        exp1(t, dt, A, U, V, phi);
        exp2(t, dt, A, U, V, phi);
    }
}

void spleaf_SHOKernel::deriv(
    const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV)
{
    if (Q > 0.5)
    {
        double da = - la * a + nu * b;
        double db = - la * b - nu * a;
        dU.col(offset) = da * _cnut.array() + db * _snut.array();
        dV.col(offset) = la * _cnut.array() - nu * _snut.array();
        dU.col(offset + 1) = da * _snut.array() - db * _cnut.array();
        dV.col(offset + 1) = la * _snut.array() + nu * _cnut.array();
    }
    else
    {
        auto exp1 = spleaf_ExponentialKernel(t, {a1, la1}, 0);
        auto exp2 = spleaf_ExponentialKernel(t, {a2, la2}, 1);
        exp1.deriv(t, dt, dU, dV);
        exp2.deriv(t, dt, dU, dV);
    }
}

spleaf_QuasiPeriodicKernel::spleaf_QuasiPeriodicKernel(const VectorXd &t, std::array<double, 4> params, size_t offset)
 : a(params[0]), b(params[1]), la(params[2]), nu(params[3]), offset(offset)
{
    _nut = nu * t.array();
    _cnut = _nut.array().cos();
    _snut = _nut.array().sin();
};

void spleaf_QuasiPeriodicKernel::operator()(
    const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi)
{
    A.array() += a;
    U.col(offset) = a * _cnut.array() + b * _snut.array();
    V.col(offset) = _cnut;
    U.col(offset + 1) = a * _snut.array() - b * _cnut.array();
    V.col(offset + 1) = _snut;
    VectorXd _e = (-la * dt.array()).exp();
    phi.col(offset) = _e;
    phi.col(offset + 1) = _e;
}

void spleaf_QuasiPeriodicKernel::deriv(const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV)
{
    double da = - la * a + nu * b;
    double db = - la * b - nu * a;
    dU.col(offset) = da * _cnut.array() + db * _snut.array();
    dV.col(offset) = la * _cnut.array() - nu * _snut.array();
    dU.col(offset + 1) = da * _snut.array() - db * _cnut.array();
    dV.col(offset + 1) = la * _snut.array() + nu * _cnut.array();
}


spleaf_MEPKernel::spleaf_MEPKernel(const VectorXd &t, std::array<double, 4> params, size_t offset)
: sig(params[0]), rho(params[1]), P(params[2]), eta(params[3]), offset(offset)
{
    la = 1.0 / rho;
    double _var = sig * sig;
    double _eta2 = eta * eta;
    double _f = 1 / (4 * _eta2);
    double _f2 = _f * _f;
    double _f2_4 = _f2 / 4;
    double _deno = 1 + _f + _f2_4;
    double a0 = _var / _deno;
    sig0 = sqrt(a0);
    a1 = _f * a0;
    a2 = _f2_4 * a0;
    nu = 2 * M_PI / P;
    double la_nu = la / nu;
    b1 = a1 * la_nu;
    b2 = a2 * la_nu / 2.0;
}

void spleaf_MEPKernel::operator()(
    const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi)
{
    size_t r = 0;
    auto mat = spleaf_Matern32Kernel(t, {sig0, rho}, r);
    r += mat.r;
    auto qp1 = spleaf_QuasiPeriodicKernel(t, {a1, b1, la, nu}, r);
    r += qp1.r;
    auto qp2 = spleaf_QuasiPeriodicKernel(t, {a2, b2, la, 2 * nu}, r);

    mat(t, dt, A, U, V, phi);
    qp1(t, dt, A, U, V, phi);
    qp2(t, dt, A, U, V, phi);
}

void spleaf_MEPKernel::deriv(const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV)
{
    size_t r = 0;
    auto mat = spleaf_Matern32Kernel(t, {sig0, rho});
    r += mat.r;
    auto qp1 = spleaf_QuasiPeriodicKernel(t, {a1, b1, la, nu}, r);
    r += qp1.r;
    auto qp2 = spleaf_QuasiPeriodicKernel(t, {a2, b2, la, 2 * nu}, r);

    // std::cout << "dU: " << dU.rows() << "x" << dU.cols() << std::endl;
    // std::cout << "dV: " << dV.rows() << "x" << dV.cols() << std::endl;
    // std::cout << r << std::endl;

    // std::cout << "dU: " << dU.transpose() << std::endl << std::endl;
    mat.deriv(t, dt, dU, dV);
    // std::cout << "dU: " << dU.transpose() << std::endl << std::endl;
    qp1.deriv(t, dt, dU, dV);
    // std::cout << "dU: " << dU.transpose() << std::endl << std::endl;
    qp2.deriv(t, dt, dU, dV);
    // std::cout << "dU: " << dU.transpose() << std::endl << std::endl;
}


spleaf_ESKernel::spleaf_ESKernel(const VectorXd &t, std::array<double, 2> params, size_t offset)
: sig(params[0]), rho(params[1]), offset(offset)
{
    coef_b = 1.0 / mu;
    coef_a0 = 2.0 / 3.0 * (1.0 + coef_b * coef_b);
    coef_a = 1.0 - coef_a0;

    la = coef_la / rho;
    nu = mu * la;
    double var = sig * sig;
    a0 = coef_a0 * var;
    a = coef_a * var;
    b = coef_b * var;
};

void spleaf_ESKernel::operator()(
    const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi)
{
    size_t r = 0;
    auto exp = spleaf_ExponentialKernel(t, {a0, la}, r);
    r += exp.r;
    auto qp = spleaf_QuasiPeriodicKernel(t, {a, b, la, nu}, r);
    
    exp(t, dt, A, U, V, phi);
    qp(t, dt, A, U, V, phi);
}

void spleaf_ESKernel::deriv(
    const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV)
{
    size_t r = 0;
    auto exp = spleaf_ExponentialKernel(t, {a0, la}, r);
    r += exp.r;
    auto qp = spleaf_QuasiPeriodicKernel(t, {a, b, la, nu}, r);
    
    exp.deriv(t, dt, dU, dV);
    qp.deriv(t, dt, dU, dV);
}


_spleaf_ESP_PKernel::_spleaf_ESP_PKernel(const VectorXd &t, std::array<double, 2> params, size_t offset)
: P(params[0]), eta(params[1]), offset(offset)
{
    eta2 = eta * eta;
    f = 1.0 / (4 * eta2);

    // for (int i = 0; i <= nharm; i++) {
    //     double _i1 = std::cyl_bessel_i(i, f) * exp(-abs(f));
    //     double _i2 = bessel::cyl_i(i, f, true);
    //     if (!approx_equal(_i1, _i2, 1e-14)) {
    //         std::cout << std::setprecision(20);
    //         std::cout << std::endl;
    //         std::cout << f << std::endl;
    //         std::cout << _i1 << std::endl;
    //         std::cout << _i2 << std::endl;
    //         throw std::domain_error("cyl_bessel_i and cyl_i differ.");
    //     }
    // }

#ifdef __APPLE__
    for (int i = 0; i <= nharm; i++)
        a[i] = bessel::cyl_i(i, f, true);
#else
    for (int i = 0; i <= nharm; i++)
        a[i] = std::cyl_bessel_i(i, f) * exp(-abs(f));
#endif

    a[0] /= 2.0;

    deno = std::accumulate(a.begin(), a.end(), 0.0);

    for (int i = 0; i <= nharm; i++)
        a[i] /= deno;

    nu = 2 * M_PI / P;
};

void _spleaf_ESP_PKernel::operator()(
    const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi)
{
    size_t r = 0;
    auto exp = spleaf_ExponentialKernel(t, {a[0], 0.0}, r);
    exp(t, dt, A, U, V, phi);
    r += exp.r;
    for (int i = 1; i <= nharm; i++)
    {
        auto qp = spleaf_QuasiPeriodicKernel(t, {a[i], 0.0, 0.0, i * nu}, r);
        qp(t, dt, A, U, V, phi);
        r += qp.r;
    }
}

void _spleaf_ESP_PKernel::deriv(
    const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV)
{
    size_t r = 0;
    auto exp = spleaf_ExponentialKernel(t, {a[0], 0.0}, r);
    exp.deriv(t, dt, dU, dV);
    r += exp.r;
    for (int i = 1; i <= nharm; i++)
    {
        auto qp = spleaf_QuasiPeriodicKernel(t, {a[i], 0.0, 0.0, i * nu}, r);
        qp.deriv(t, dt, dU, dV);
        r += qp.r;
    }
}


spleaf_ESPKernel::spleaf_ESPKernel(const VectorXd &t, std::array<double, 4> params, size_t offset) 
 : sig(params[0]), rho(params[1]), P(params[2]), eta(params[3]), offset(offset)
{
    size_t r = 0;
    size_t es_r = spleaf_ESKernel::r;
    size_t esp_p_r = _spleaf_ESP_PKernel::r;

    size_t N = t.size();
    _A1 = VectorXd::Zero(N);
    _A2 = VectorXd::Zero(N);
    _U1 = MatrixXd_RM(N, es_r);
    _U2 = MatrixXd_RM(N, esp_p_r);
    _V1 = MatrixXd_RM(N, es_r);
    _V2 = MatrixXd_RM(N, esp_p_r);
    _phi1 = MatrixXd_RM(N, es_r);
    _phi2 = MatrixXd_RM(N, esp_p_r);
}

void spleaf_ESPKernel::operator()(
    const VectorXd &t, const VectorXd &dt, VectorXd &A, MatrixXd_RM &U, MatrixXd_RM &V, MatrixXd_RM &phi)
{
    size_t r = 0;
    spleaf_ESKernel es = spleaf_ESKernel(t, {sig, rho}, r);
    r += es.r;
    _spleaf_ESP_PKernel esp_p = _spleaf_ESP_PKernel(t, {P, eta}, r);

    es(t, dt, _A1, _U1, _V1, _phi1);
    esp_p(t, dt, _A2, _U2, _V2, _phi2);

    A.array() += _A1.array() * _A2.array();

    for (size_t i = 0; i < es.r; i++)
    {
        for (size_t j = 0; j < esp_p.r; j++) {
            U.col(offset + i*esp_p.r + j) = _U1.col(i).array() * _U2.col(j).array();
            V.col(offset + i*esp_p.r + j) = _V1.col(i).array() * _V2.col(j).array();
            phi.col(offset + i*esp_p.r + j) = _phi1.col(i).array() * _phi2.col(j).array();
        }
    }
}

void spleaf_ESPKernel::deriv(
    const VectorXd &t, const VectorXd &dt, MatrixXd_RM &dU, MatrixXd_RM &dV)
{
    size_t r = 0;
    spleaf_ESKernel es = spleaf_ESKernel(t, {sig, rho}, r);
    r += es.r;
    _spleaf_ESP_PKernel esp_p = _spleaf_ESP_PKernel(t, {P, eta}, r);

    size_t N = t.size();
    MatrixXd_RM _dU1 = MatrixXd_RM(N, es.r), _dU2 = MatrixXd_RM(N, esp_p.r);
    MatrixXd_RM _dV1 = MatrixXd_RM(N, es.r), _dV2 = MatrixXd_RM(N, esp_p.r);

    es.deriv(t, dt, _dU1, _dV1);
    esp_p.deriv(t, dt, _dU2, _dV2);

    for (size_t i = 0; i < es.r; i++)
    {
        for (size_t j = 0; j < esp_p.r; j++) {
            dU.col(offset + i*esp_p.r + j) = _dU1.col(i).array() * _U2.col(j).array() + _U1.col(i).array() * _dU2.col(j).array();
            dV.col(offset + i*esp_p.r + j) = _dV1.col(i).array() * _V2.col(j).array() + _V1.col(i).array() * _dV2.col(j).array();
        }
    }
}



double logdet(VectorXd &D)
{
    return D.array().log().sum();
}

double chi2(VectorXd &y, size_t n, size_t r, VectorXl &offsetrow, VectorXl &b,
            MatrixXd_RM &U, MatrixXd_RM &W, MatrixXd_RM &phi, VectorXd &G, VectorXd &_f_solveL, VectorXd &D)
{
    // solveL:
    VectorXd x = VectorXd(y.size());
    spleaf_solveL(n, r, offsetrow.data(), b.data(),
                  U.data(), W.data(), phi.data(), G.data(),
                  y.data(), x.data(), _f_solveL.data());
    // chi2:
    return (x.array().square() / D.array()).sum();
}

double loglike(VectorXd &y, size_t n, size_t r, VectorXl &offsetrow, VectorXl &b,
               MatrixXd_RM &U, MatrixXd_RM &W, MatrixXd_RM &phi, VectorXd &G, VectorXd &_f_solveL, VectorXd &D)
{
    double c = chi2(y, n, r, offsetrow, b, U, W, phi, G, _f_solveL, D);
    return -0.5 * (c + logdet(D) + n * log(2.0 * M_PI));
}



// Macro to define a kernel binding
#define KERNEL_BIND(CPPNAME, PYNAME, NPARAMS)                                                                                    \
    nb::class_<CPPNAME>(m, PYNAME)                                                                                               \
        .def(nb::init<const VectorXd&, std::array<double, NPARAMS>>())                                                           \
        .def_ro_static("r", &CPPNAME::r)                                                                                         \
        .def("__call__", [](CPPNAME &k, const VectorXd &t, const VectorXd &dt,                                                   \
                            nb::DRef<VectorXd> A, nb::DRef<MatrixXd_RM> U, nb::DRef<MatrixXd_RM> V, nb::DRef<MatrixXd_RM> phi)   \
            {                                                                                                                    \
                VectorXd cA = A;                                                                                                 \
                MatrixXd_RM cU = U, cV = V, cphi=phi;                                                                            \
                k(t, dt, cA, cU, cV, cphi);                                                                                      \
                A = cA; U = cU; V = cV; phi = cphi;                                                                              \
            }                                                                                                                    \
        )                                                                                                                        \
        .def("__call__", [](CPPNAME &k, const VectorXd &t)   \
            {                                                                                                                    \
                size_t N = t.size();                                                                                             \
                VectorXd dt = t.segment(1, N-1).array() - t.segment(0, N-1).array();                                             \
                VectorXd A = VectorXd::Zero(N);                                                                                  \
                MatrixXd_RM U = MatrixXd_RM(N, k.r);                                                                             \
                MatrixXd_RM V = MatrixXd_RM(N, k.r);                                                                             \
                MatrixXd_RM phi = MatrixXd_RM(N - 1, k.r);                                                                       \
                k(t, dt, A, U, V, phi);                                                                                          \
                return std::make_tuple(A, U, V, phi);                                                                            \
            }                                                                                                                    \
        )                                                                                                                        \
        .def("deriv", [](CPPNAME &k, const VectorXd &t, const VectorXd &dt,                                                      \
                            nb::DRef<MatrixXd_RM> dU, nb::DRef<MatrixXd_RM> dV)                                                  \
            {                                                                                                                    \
                MatrixXd_RM cdU = dU, cdV = dV;                                                                                  \
                k.deriv(t, dt, cdU, cdV);                                                                                        \
                dU = cdU, dV = cdV;                                                                                              \
            }                                                                                                                    \
        );


NB_MODULE(GP, m) {
    m.def("sample", &sample, "K"_a, "white_noise_variance"_a = 1.25e-12,
          "Draw samples from the GP prior distribution given a kernel matrix K.");
    m.def("QP", &QP, "t"_a, "η1"_a, "η2"_a, "η3"_a, "η4"_a,
          "Quasi-periodic kernel ");
    m.def("PER", &PER);

    nb::enum_<KernelType>(m, "KernelType")
        .value("qp", KernelType::qp)
        .value("per", KernelType::per)
        .value("spleaf_exp", KernelType::spleaf_exp)
        .value("spleaf_matern32", KernelType::spleaf_matern32)
        .value("spleaf_sho", KernelType::spleaf_sho)
        .value("spleaf_mep", KernelType::spleaf_mep)
        .value("spleaf_es", KernelType::spleaf_es)
        .value("spleaf_esp", KernelType::spleaf_esp)
        .export_values();

    KERNEL_BIND(spleaf_ExponentialKernel, "Exponential", 2);
    KERNEL_BIND(spleaf_Matern32Kernel, "Matern32", 2);
    KERNEL_BIND(spleaf_SHOKernel, "SHO", 3);
    KERNEL_BIND(spleaf_MEPKernel, "MEP", 4);
    KERNEL_BIND(spleaf_ESKernel, "ES", 2);

    KERNEL_BIND(_spleaf_ESP_PKernel, "_ESP_P", 2);
    KERNEL_BIND(spleaf_ESPKernel, "ESP", 4);
}