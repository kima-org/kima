#include "postkepler.h"

using namespace brandt;

const double c_light = 299792458;//m/s
const double G = 6.6743 * pow(10.0,-11.0);
const double Msun = 1.988409870698051 * pow(10.0,30.0);
const double days = 24*3600;
const double AU = 149597870700;
const double sectoyr = 3600*24*365.25;
const double G_ast = G * pow(AU,-3.0) * Msun * pow(sectoyr,2.0);

namespace postKep
{
    double period_correction(double p_obs, double wdot)
    {
        //convert observational period (in days) to anomalistic period (in days) using wdot (in arcsec per year)
        double P_anom = p_obs*(1+wdot*M_PI/3600/180/365.25*p_obs/TWO_PI);
    
        return P_anom;
    }
    
    double change_omega(double w, double wdot, double ti, double Tp)
    {
        //linear change in omega and convert from arcsec/year to radians/day
        double dw = wdot*(ti-Tp)*M_PI/3600/180/365.25;
    
        return w + dw;
    }
    
    inline double semiamp(double M0, double M1, double P, double ecc)
    {
        // given M0 and M1 in solar mass and P in days, returns in m/s
        double m1 = M1 * 1047.5655; //in jupiter mass
        double m01 = (M0 + M1); //in solar mass
        
        return 28.4329*pow((1-pow(ecc,2.0)),-0.5)*m1*pow(m01,-2.0/3)*pow(P/365,-1.0/3);
    }
    
    inline double f_M(double K,double M0, double M1, double P, double ecc)
    {
        double f = semiamp(M0, M1, P, ecc) - K;
    
        return f;
    }

    inline double f_dash_M(double K,double M0, double M1, double P, double ecc)
    {
        // given M0 and M1 in solar mass and P in days, returns in m/s
        double MjMs = 1047.5655;
        double m1 = M1 * MjMs; //in jupiter mass
        double m01 = (M0 + M1); //in solar mass
    
        double f_dash_1 = 28.4329*pow((1-pow(ecc,2.0)),-0.5)*MjMs*pow(m01,-2.0/3)*pow(P/365,-1.0/3);
        double f_dash_2 = -2*28.4329*pow((1-pow(ecc,2.0)),-0.5)*m1*pow(m01,-5.0/3)*pow(P/365,-1.0/3)/3;
    
        return f_dash_1 - f_dash_2;
    }

    inline double get_K2_v2(double K1, double M, double P, double ecc)
    {
        double M_est = (K1/28.4329)*pow((1-pow(ecc,2.0)),0.5)*pow(M,2.0/3)*pow((P/365),1.0/3)/1047.5655;
        double k = semiamp(M, M_est, P, ecc);
        while(abs(k-K1)>50)
        {
            M_est = M_est - f_M(K1, M, M_est, P, ecc)/f_dash_M(K1, M, M_est, P, ecc);
            k = semiamp(M, M_est, P, ecc);
        }
        double K2 = K1*M/M_est;
    
        return K2;

    }

    inline double get_K2_v1(double K1, double M, double P, double ecc)
    {
        double M_est = (K1/28.4329)*pow((1-pow(ecc,2.0)),0.5)*pow(M,2.0/3)*pow((P/365),1.0/3)/1047.5655;
        double a = M_est/3;
        double b = M_est*3;
        double c = (a+b)/2;
        double eps = semiamp(M,b,P,ecc) - semiamp(M,a,P,ecc);
    
        while (abs(eps) > 50) {
            c = (b+a)/2;
            double x = K1 - semiamp(M,c,P,ecc);
            if (x < 0) {
                b = c;
            } else {
                a = c;
            }
            eps = semiamp(M,b,P,ecc) - semiamp(M,a,P,ecc);
        }
        double M2 = c;
        double K2 =K1*M/M_est;
        
        return K2;
    }
    
    inline double light_travel_time(double K1, double sinf, double cosf, double w, double ecc)
    {
        
        double delta_LT = pow(K1,2.0)*pow(sinf*cos(w) + cosf*sin(w), 2.0)*(1+ecc*cosf)/c_light;
    
        return delta_LT;
    }
    
    inline double transverse_doppler(double K1, double sinf, double cosf, double ecc, double cosi)
    {
        double sin2i = 1.0 - cosi*cosi;
        double delta_TD = pow(K1,2.0)*(1 + ecc*cosf - (1-pow(ecc,2.0))/2)/(c_light*sin2i);
    
        return delta_TD;
    }
    
    inline double gravitational_redshift(double K1, double K2, double sinf, double cosf, double ecc, double cosi)
    {
        double sin2i = 1.0 - cosi*cosi;
        double delta_GR = K1*(K1+K2)*(1+ecc*cosf)/(c_light*sin2i);
    
        return delta_GR;
    }
    
    inline double v_tide(double R1, double M1, double M2, double P, double sinf, double cosf, double w, double cosi)
    {
        double phi_0 = M_PI/2 - w;
        double sin2i = 1.0 - cosi*cosi;
        
        return 1184*M2/(M1*(M1+M2))*pow(R1,4.0)*pow(P,-3.0)*sin2i * 2*(sinf*cos(phi_0) - sin(phi_0)*cosf)*(cosf*cos(phi_0) + sinf*sin(phi_0));
    }
    
    double post_Newtonian(double K1, double sinf, double cosf, double ecc, double w, double P, double cosi, double M1, double M2, double R1, bool GR, bool Tid)
    {
        double K2;
        double v = 0.0;
        //if M2 is specified use it, if not, numerically calculate it
        if (GR || Tid)
        {
            if (M2 > 0.01)
            {
                K2 = K1*M1/M2;
            }
            else
            {
                K2 = get_K2_v2(K1, M1, P, ecc); 
                M2 = K1*M1/K2;
            }
        }
        if (Tid)
        {
            if (R1 <0.01)
            {
                R1 = pow(M1,0.8);
            }
            double delta_v_tide = v_tide(R1,M1,M2,P,sinf,cosf,w,cosi);
            v = v + delta_v_tide;
        }
        if (GR)
        {
            double delta_LT = light_travel_time(K1, sinf,cosf, w, ecc);
            double delta_TD = transverse_doppler(K1, sinf,cosf, ecc,cosi);
            double delta_GR = gravitational_redshift(K1, K2, sinf,cosf, ecc,cosi);
            v = v + delta_LT + delta_TD + delta_GR;
        }
        
    
        return v;
    }
    
    std::tuple <double,double> post_Newtonian_sb2(double K1,double K2, double sinf, double cosf, double ecc, double w, double P, double cosi, double q, double R1, double R2, bool GR, bool Tid)
    {

        double v1 = 0.0;
        double v2 = 0.0;
        double sin2i = 1.0 - cosi*cosi;
        //calculate masses
        double M1 = P * days * pow(K2,3.0) * pow((1 - pow(ecc,2.0)),1.5) * pow((1 + q),2.0) / pow(sin2i,1.5) / (TWO_PI * G)/Msun;
        double M2 = M1 * q;
        if (Tid)
        {
            if (R1 <0.01)
            {
                R1 = pow(M1,0.8);
            }
            if (R2 <0.01)
            {
                R2 = pow(M2,0.8);
            }
            double delta_v_tide1 = v_tide(R1,M1,M2,P,sinf,cosf,w,cosi);
            v1 = v1 + delta_v_tide1;
            double delta_v_tide2 = v_tide(R2,M2,M1,P,sinf,cosf,w-M_PI,cosi);
            v2 = v2 + delta_v_tide2;
        }
        if (GR)
        {
            double delta_LT = light_travel_time(K1, sinf,cosf, w, ecc);
            double delta_TD = transverse_doppler(K1, sinf,cosf, ecc,cosi);
            double delta_GR = gravitational_redshift(K1, K2, sinf,cosf, ecc,cosi);
            v1 = v1 + delta_LT + delta_TD + delta_GR;
            
            double delta_LT2 = light_travel_time(K2, sinf,cosf, w - M_PI, ecc);
            double delta_TD2 = transverse_doppler(K2, sinf,cosf, ecc,cosi);
            double delta_GR2 = gravitational_redshift(K2, K1, sinf,cosf, ecc,cosi);
            v2 = v2 + delta_LT2 + delta_TD2 + delta_GR2;
        }
        
    
        return { v1, v2 };
    }
    
    //
    std::vector<double> keplerian_prec(const std::vector<double> &t, const double &P,
                                  const double &K, const double &ecc,
                                  const double &w, const double &wdot, const double &M0,
                                  const double &M0_epoch, const double &cosi, const double &M1, const double &M2, 
                                  const double &R1, bool GR, bool Tid)
    {
        // allocate RVs
        std::vector<double> rv(t.size());
        
        
        // mean motion, once per orbit
        double n = 2. * M_PI / P;

        // brandt solver calculations, once per orbit
        double bounds[13];
        double EA_tab[6 * 13];
        brandt::get_bounds(bounds, EA_tab, ecc);

        // std::cout << std::endl;
        for (size_t i = 0; i < t.size(); i++)
        {
            double Tp = M0_epoch-(P*M0)/(TWO_PI);
            double w_t = change_omega(w, wdot, t[i], Tp);
            // sin and cos of argument of periastron
            double sinw, cosw;
            sincos(w_t, &sinw, &cosw);
            
            double sinEf, cosEf;
            double M = n * (t[i] - M0_epoch) + M0;
            brandt::solver_fixed_ecc(bounds, EA_tab, M, ecc, &sinEf, &cosEf);
            brandt::to_f(ecc, 1-ecc, &sinEf, &cosEf);
            
            double vrad = K * (cosw * (cosEf + ecc) - sinw * sinEf);
                
            double v_correction = postKep::post_Newtonian(K, sinEf,cosEf, ecc, w_t, P, cosi, M1, M2, R1, GR, Tid);
            
            rv[i] = vrad + v_correction;
      }

      return rv;
    }
    
    std::tuple<std::vector<double>, std::vector<double>> keplerian_prec_sb2(const std::vector<double> &t, const double &P,
                                  const double &K, const double &q, const double &ecc,
                                  const double &w, const double &wdot, const double &M0,
                                  const double &M0_epoch, const double &cosi, 
                                  const double &R1, const double &R2, bool GR, bool Tid)
    {
        // allocate RVs
        std::vector<double> rv1(t.size());
        std::vector<double> rv2(t.size());
        
        
        // mean motion, once per orbit
        double n = 2. * M_PI / P;

        // brandt solver calculations, once per orbit
        double bounds[13];
        double EA_tab[6 * 13];
        brandt::get_bounds(bounds, EA_tab, ecc);

        // std::cout << std::endl;
        for (size_t i = 0; i < t.size(); i++)
        {
            double Tp = M0_epoch-(P*M0)/(TWO_PI);
            double w_t = change_omega(w, wdot, t[i], Tp);
            // sin and cos of argument of periastron
            double sinw, cosw;
            sincos(w_t, &sinw, &cosw);
            
            double sinEf, cosEf;
            double M = n * (t[i] - M0_epoch) + M0;
            brandt::solver_fixed_ecc(bounds, EA_tab, M, ecc, &sinEf, &cosEf);
            brandt::to_f(ecc, 1-ecc, &sinEf, &cosEf);
            
            double vrad1 = K * (cosw * (cosEf + ecc) - sinw * sinEf);
            double vrad2 = K/q * (-cosw * (cosEf + ecc) + sinw * sinEf);
            
            
            auto [v_correction1,v_correction2] = postKep::post_Newtonian_sb2(K, K/q, sinEf,cosEf, ecc, w_t, P, cosi, q, R1, R2, GR, Tid);
           
            rv1[i] = vrad1 + v_correction1;
            rv2[i] = vrad2 + v_correction2;
      }

      return {rv1, rv2};
    }
}

namespace MassConv
{
    double SemiAmp(double P, double ecc, double M0, double M1, double cosi)
    {
        double sini = pow((1.0-pow(cosi,2.0)),0.5);
        double K = pow(TWO_PI*G/(P*24*3600),1.0/3) * M1 * Msun * sini * pow((M0+M1)*Msun,-2.0/3) * pow((1-pow(ecc,2.0)),-0.5);
        return K;
    }
    double SemiPhotPl(double P, double M0, double M1, double plx)
    {
        double a0 = plx * pow((P/365.25),2.0/3) * pow((M0 + M1),1.0/3) * (M1/(M0+M1));
        return a0;
    }
    double SemiPhotSt(double P, double M0, double M1, double plx, double eps)
    {
        double a0 = plx * pow((P/365.25),2.0/3) * pow((M0 + M1),1.0/3) * (M1/(M0+M1) - eps/(1+eps));
        return a0;
    }
    double SemiPhotfromK(double P, double K, double ecc, double cosi, double plx)
    {
        double sini = pow((1.0-pow(cosi,2.0)),0.5);
        double a0 = plx * P/365.25 * K/AU*sectoyr * pow(1 - pow(ecc,2.0),0.5) /(TWO_PI * sini);
        return a0;
    }
    double KfromSemiPhot(double P, double a0, double ecc, double cosi, double plx)
    {
        double sini = pow((1.0-pow(cosi,2.0)),0.5);
        if (P==0.0)
        {
            return 0.0;
        } 
        else
        {
            double K = a0  * (TWO_PI * sini) * pow(1 - pow(ecc,2.0),-0.5) / (plx*P/365.25)* AU/sectoyr ;
            return K;
        }
        
    }
}

auto KEPLERIAN_PREC_DOC = R"D(
Calculate the Keplerian curve of the orbit of a dark companion around a star at times `t` with post-Keplerian additions. Suited to the orbit of a close binary star.

Args:
    t (array):
        Times at which to calculate the Keplerian function
    P (float):
        Orbital period [days]
    K (float):
        Semi-amplitude
    ecc (float):
        Orbital eccentricity
    w (float):
        Argument of pericentre [rad]
    wdot (float):
        Pericentre precession rate [arcsecs/year]
    M0 (float):
        Mean anomaly at the epoch [rad]
    M0_epoch (float):
        Reference epoch for the mean anomaly (M=0 at this time) [days]
    cosi (float):
        Cosine of the inclination angle of the orbit (=0 for an edge-on orbit)
    M1 (float):
        Mass of primary star [Msun]
    M2 (float):
        Mass of secondary star [Msun]
    R1 (float):
        Radius of primary star [Rsun] If not specfied and tidal correction included the relation R = M^0.8 will be used.
    GR (bool):
        Whether to include the radial velocity corrections from General Relativity (Transverse Doppler, Light Travel-Time, and Graviatational Redshift)
    Tid (bool):
        Whether to include the radial velocity correction from Tides (only suitable for circular orbits)

Returns:
    v (array):
        Keplerian function with potential corrections evaluated at input times `t`
)D";

auto KEPLERIAN_PREC_SB2_DOC = R"D(
Calculate the Keplerian curve a double-lined binary star at times `t` with post-Keplerian additions. Suited to the orbit of a close double-lined binary stars. 
Masses of the individual stars do not need to be provided as they can be calculated, but knowledge of the inclination of the system is important to be able to make use of the corrections.

Args:
    t (array):
        Times at which to calculate the Keplerian function
    P (float):
        Orbital period [days]
    K (float):
        Semi-amplitude
    q (float):
        Binary mass ratio (defined as M2/M1)
    ecc (float):
        Orbital eccentricity
    w (float):
        Argument of pericentre [rad]
    wdot (float):
        Pericentre precession rate [arcsecs/year]
    M0 (float):
        Mean anomaly at the epoch [rad]
    M0_epoch (float):
        Reference epoch for the mean anomaly (M=0 at this time) [days]
    cosi (float):
        Cosine of the inclination angle of the orbit (=0 for an edge-on orbit)
    R1 (float):
        Radius of primary star [Rsun] If not specfied and tidal correction included the relation R = M^0.8 will be used.
    R2 (float):
        Radius of secondary star [Rsun] If not specfied and tidal correction included the relation R = M^0.8 will be used.
    GR (bool):
        Whether to include the radial velocity corrections from General Relativity (Transverse Doppler, Light Travel-Time, and Graviatational Redshift)
    Tid (bool):
        Whether to include the radial velocity correction from Tides (only suitable for circular orbits)

Returns:
    rv1 (array):
        Keplerian function for the radial velocity of the primary star in the binary orbit, with potential corrections, evaluated at input times `t`
    rv2 (array):
        Keplerian function for the radial velocity of the secondary star in the binary orbit, with potential corrections, evaluated at input times `t`
)D";


NB_MODULE(postkepler, m) {
    // m.def("post_keplerian", &postKep::keplerian_prec,
    //       "t"_a, "P"_a, "K"_a, "ecc"_a, "w"_a, "wdot"_a, "M0"_a, "M0_epoch"_a,"cosi"_a, "M1"_a, "M2"_a, "R1"_a, "GR"_a, "Tid"_a);
    m.def("period_correction", [](double Pobs, double wdot) {return postKep::period_correction(Pobs, wdot); }, "Pobs"_a, "wdot"_a);

    m.def("post_keplerian", [](const std::vector<double> &t, const double &P,
                                  const double &K, const double &ecc,
                                  const double &w, const double &wdot, const double &M0,
                                  const double &M0_epoch, const double &cosi, const double &M1, const double &M2, 
                                  const double &R1, bool GR, bool Tid)
    {
        size_t size = t.size();
        struct Temp { std::vector<double> v; };
        Temp *temp = new Temp();
        temp->v = postKep::keplerian_prec(t, P, K, ecc, w, wdot, M0, M0_epoch, cosi, M1, M2, R1, GR, Tid);
        nb::capsule owner(temp, [](void *p) noexcept { delete (Temp *) p; });
        return nb::ndarray<nb::numpy, double>(temp->v.data(), {size}, owner);
    }, "t"_a, "P"_a, "K"_a, "ecc"_a, "w"_a, "wdot"_a, "M0"_a, "M0_epoch"_a,"cosi"_a, "M1"_a, "M2"_a, "R1"_a, "GR"_a, "Tid"_a, KEPLERIAN_PREC_DOC);

    // m.def("post_keplerian_sb2", &postKep::keplerian_prec_sb2,
    //       "t"_a, "P"_a, "K"_a, "q"_a, "ecc"_a, "w"_a, "wdot"_a, "M0"_a, "M0_epoch"_a,"cosi"_a, "R1"_a, "R2"_a, "GR"_a, "Tid"_a);
    
    m.def("post_keplerian_sb2", [](const std::vector<double> &t, const double &P,
                                  const double &K, const double &q, const double &ecc,
                                  const double &w, const double &wdot, const double &M0,
                                  const double &M0_epoch, const double &cosi, 
                                  const double &R1, const double &R2, bool GR, bool Tid)
    {
        size_t size = t.size();
        auto[rv1,rv2] = postKep::keplerian_prec_sb2(t, P, K, q, ecc, w, wdot, M0, M0_epoch, cosi, R1, R2, GR, Tid);
        struct Temp { 
            std::vector<double> rv1; 
            std::vector<double> rv2;
        };
        Temp *temp = new Temp();
        temp->rv1 = rv1;
        temp->rv2 = rv2;
        nb::capsule owner(temp, [](void *p) noexcept { delete (Temp *) p; });
        return std::make_tuple(
            nb::ndarray<nb::numpy, double>(temp->rv1.data(), {size}, owner),
            nb::ndarray<nb::numpy, double>(temp->rv2.data(), {size}, owner)
        );
    }, "t"_a, "P"_a, "K"_a, "q"_a, "ecc"_a, "w"_a, "wdot"_a, "M0"_a, "M0_epoch"_a,"cosi"_a, "R1"_a, "R2"_a, "GR"_a, "Tid"_a, KEPLERIAN_PREC_SB2_DOC);

    m.def("a0fromK", [](double P, double K, double ecc, double cosi, double plx){ return MassConv::SemiPhotfromK(P, K, ecc, cosi, plx); }, "P"_a, "K"_a, "ecc"_a, "cosi"_a, "plx"_a);
    
    m.def("Kfroma0", [](double P, double a0, double ecc, double cosi, double plx){ return MassConv::KfromSemiPhot(P, a0, ecc, cosi, plx); }, "P"_a, "a0"_a, "ecc"_a, "cosi"_a, "plx"_a);

}