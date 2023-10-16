#include "transits.h"

inline double ellpic_bulirsch(double n, double k)
{
    double kc = sqrt(1.0 - k * k);
    double p = sqrt(n + 1.0);
    double m0 = 1.0;
    double c = 1.0;
    double d = 1.0 / p;
    double e = kc;
    double f, g;

    int nit = 0;

    while(nit < 10000)
    {
        f = c;
        c = d / p + c;
        g = e / p;
        d = 2.0 * (f * g + d);
        p = g + p;
        g = m0;
        m0 = kc + m0;
        if (fabs(1.0 - kc / g) > 1.0e-8)
        {
            kc = 2.0 * sqrt(e);
            e = kc * m0;
        }
        else
        {
            return 0.5 * M_PI * (c * m0 + d) / (m0 * (m0 + p));
        }
        nit++;
    }
    return 0;
}


inline double ellec(double k)
{
    double m1, a1, a2, a3, a4, b1, b2, b3, b4, ee1, ee2, ellec;
    m1 = 1.0 - k * k;
    a1 = 0.44325141463;
    a2 = 0.06260601220;
    a3 = 0.04757383546;
    a4 = 0.01736506451;
    b1 = 0.24998368310;
    b2 = 0.09200180037;
    b3 = 0.04069697526;
    b4 = 0.00526449639;
    ee1 = 1.0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)));
    ee2 = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * log(1.0 / m1);
    ellec = ee1 + ee2;
    return ellec;
}

inline double ellk(double k)
{
    double a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, ellk,  ek1, ek2, m1;
    m1 = 1.0 - k * k;
    a0 = 1.38629436112;
    a1 = 0.09666344259;
    a2 = 0.03590092383;
    a3 = 0.03742563713;
    a4 = 0.01451196212;
    b0 = 0.5;
    b1 = 0.12498593597;
    b2 = 0.06880248576;
    b3 = 0.03328355346;
    b4 = 0.00441787012;
    ek1 = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)));
    ek2 = (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * log(m1);
    ellk = ek1 - ek2;
    return ellk;
}


std::vector<double> rsky(const std::vector<double>& time, double tc, double P, 
                         double a, double inc, double ecc, double omega)
{
    std::vector<double> r(time.size());

    // mean motion, once per orbit
    double n = 2.0 * M_PI / P;

    double f, E, M, tp;
    double sinE, cosE;
    double d;
    for (size_t i = 0; i < time.size(); i++)
    {
        f = M_PI / 2.0 - omega;
        E = 2.0 * atan(sqrt((1.0 - ecc)/(1.0 + ecc)) * tan(f / 2.0));
        M = E - ecc*sin(E);
        tp = tc - P * M / 2.0 / M_PI;
        M = n*(time[i] - tp);
        E = brandt::solver(M, ecc, &sinE, &cosE);
        f = 2.0 * atan(sqrt((1.0 + ecc) / (1.0 - ecc)) * tan(E / 2.0));
        // separation of centers
        d = a * (1.0 - ecc * ecc) / (1.0 + ecc * cos(f)) * sqrt(1.0 - sin(omega + f) * sin(omega + f) * sin(inc) * sin(inc));
        r[i] = d;
    }
    return r;
}

std::vector<double> quadratic_ld(const std::vector<double>& ds,
                                 double c1, double c2, double p)
{
    std::vector<double> flux(ds.size());
    double omega = 1.0 - c1 / 3.0 - c2 / 6.0;
    double tol = 1.0e-14;

    for (size_t i = 0; i < ds.size(); i++)
    {
        double kap0 = 0.0, kap1 = 0.0;
        double lambdad, lambdae, etad;
        double d = ds[i];

        // allow for negative impact parameters
        d = fabs(d);

        // check the corner cases
        if (fabs(p - d) < tol)
            d = p;
        
        if (fabs(p - 1.0 - d) < tol)
            d = p - 1.0;
        
        if (fabs(1.0 - p - d) < tol)
            d = 1.0 - p;
        
        if (d < tol)
            d = 0.0;

        double x1 = pow((p - d), 2.0);
        double x2 = pow((p + d), 2.0);
        double x3 = p*p - d*d;

        // source is unocculted
        if (d >= 1.0 + p)
        {
            flux[i] = 1.0;
            continue;
        }

        // source is completely occulted
        if (p >= 1.0 && d <= p - 1.0)
        {
            lambdad = 0.0;
            etad = 0.5;
            lambdae = 1.0;
            flux[i] = 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * (lambdad + 2.0 / 3.0) + c2 * etad) / omega;
            continue;
        }

        // source is partly occulted and occulting object crosses the limb
        if (d >= fabs(1.0 - p) && d <= 1.0 + p)
        {
            kap1 = acos(MIN((1.0 - p * p + d * d) / 2.0 / d, 1.0));
            kap0 = acos(MIN((p * p + d * d - 1.0) / 2.0 / p / d, 1.0));
            lambdae = p * p * kap0 + kap1;
            lambdae = (lambdae - 0.50 * sqrt(MAX(4.0 * d * d - pow((1.0 + d * d - p * p), 2.0), 0.0))) / M_PI;
        }

        // edge of the occulting star lies at the origin
        if (d == p)
        {
            if (d < 0.5)
            {
                double q = 2.0 * p;
                double Kk = ellk(q);
                double Ek = ellec(q);
                lambdad = 1.0 / 3.0 + 2.0 / 9.0 / M_PI * (4.0 * (2.0 * p * p - 1.0) * Ek + (1.0 - 4.0 * p * p) * Kk);
                etad = p * p / 2.0 * (p * p + 2.0 * d * d);
                flux[i] = 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad + c2 * etad) / omega;
                continue;
            }
            else if (d > 0.5)
            {
                double q = 0.5/p;
                double Kk = ellk(q);
                double Ek = ellec(q);
                lambdad = 1.0 / 3.0 + 16.0 * p / 9.0 / M_PI * (2.0 * p * p - 1.0) * Ek - (32.0 * pow(p, 4.0) - 20.0 * p * p + 3.0) / 9.0 / M_PI / p * Kk;
                etad = 1.0 / 2.0 / M_PI * (kap1 + p * p * (p * p + 2.0 * d * d) * kap0 - (1.0 + 5.0 * p * p + d * d) / 4.0 * sqrt((1.0 - x1) * (x2 - 1.0)));
            }
            else
            {
                lambdad = 1.0 / 3.0 - 4.0 / M_PI / 9.0;
                etad = 3.0 / 32.0;
                flux[i] = 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad + c2 * etad) / omega;
                continue;
            }

            flux[i] = 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad + c2 * etad) / omega;
            continue;
        }

        // occulting star partly occults the source and crosses the limb
        if ((d > 0.5 + fabs(p  - 0.5) && d < 1.0 + p) || (p > 0.5 && d > fabs(1.0 - p) && d < p))
        {
            double q = sqrt((1.0 - x1) / 4.0 / d / p);
            double Kk = ellk(q);
            double Ek = ellec(q);
            double n = 1.0 / x1 - 1.0;
            double Pk = ellpic_bulirsch(n, q);
            lambdad = 1.0 / 9.0 / M_PI / sqrt(p * d) * (((1.0 - x2) * (2.0 * x2 + x1 - 3.0) - 3.0 * x3 * (x2 - 2.0)) * Kk + 4.0 * p * d * (d * d + 7.0 * p * p - 4.0) * Ek - 3.0 * x3 / x1 * Pk);
            if (d < p)
                lambdad += 2.0 / 3.0;
            etad = 1.0 / 2.0 / M_PI * (kap1 + p * p * (p * p + 2.0 * d * d) * kap0 - (1.0 + 5.0 * p * p + d * d) / 4.0 * sqrt((1.0 - x1) * (x2 - 1.0)));
            flux[i] = 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad + c2 * etad) / omega;
            continue;
        }

        // occulting star transits the source
        if (p <= 1.0 && d <= (1.0 - p))
        {
            etad = p * p / 2.0 * (p * p + 2.0 * d * d);
            lambdae = p * p;

            double q = sqrt((x2 - x1) / (1.0 - x1));
            double Kk = ellk(q);
            double Ek = ellec(q);
            double n = x2 / x1 - 1.0;
            double Pk = ellpic_bulirsch(n, q);

            lambdad = 2.0 / 9.0 / M_PI / sqrt(1.0 - x1) * ((1.0 - 5.0 * d * d + p * p + x3 * x3) * Kk + (1.0 - x1) * (d * d + 7.0 * p * p - 4.0) * Ek - 3.0 * x3 / x1 * Pk);

            // edge of planet hits edge of star
            if (fabs(p + d - 1.0) <= tol)
            {
                lambdad = 2.0 / 3.0 / M_PI * acos(1.0 - 2.0 * p) - 4.0 / 9.0 / M_PI *
                                                                       sqrt(p * (1.0 - p)) * (3.0 + 2.0 * p - 8.0 * p * p);
            }
            if (d < p)
                lambdad += 2.0 / 3.0;
        }
        flux[i] = 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad + c2 * etad) / omega;
    }
    return flux;
}


NB_MODULE(transits, m) {
    m.def("ellpic_bulirsch", &ellpic_bulirsch);
    m.def("ellec", &ellec);
    m.def("ellk", &ellk);
    m.def("rsky", &rsky);
    m.def("quadratic_ld", &quadratic_ld);
}