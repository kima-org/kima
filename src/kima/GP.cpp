#include "GP.h"

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


NB_MODULE(GP, m) {
    m.def("QP", &QP);
    m.def("PER", &PER);
}