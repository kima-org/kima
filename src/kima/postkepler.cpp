#include "postkepler.h"

using namespace brandt;

const double TWO_PI = M_PI * 2;
const double c_light = 299792458;//m/s
const double G = 6.6743 * pow(10.0,-11.0);
const double Msun = 1.988409870698051 * pow(10.0,30.0);

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
    
    inline double light_travel_time(double K1, double f, double w, double ecc)
    {
        double delta_LT = pow(K1,2.0)*pow(sin(f + w), 2.0)*(1+ecc*cos(f))/c_light;
    
        return delta_LT;
    }
    
    inline double transverse_doppler(double K1, double f, double ecc, double cosi=0)
    {
        //here assume inclination of 90 (eclipsing) so sin(i) = 1
        double sini = 1.0 - cosi*cosi;
        double delta_TD = pow(K1,2.0)*(1 + ecc*cos(f) - (1-pow(ecc,2.0))/2)/(c_light*pow(sini,2.0));
    
        return delta_TD;
    }
    
    inline double gravitational_redshift(double K1, double K2, double f, double ecc, double cosi=0)
    {
        //again assume inclination of 90 (eclipsing) so sin(i) = 1
        double sini = 1.0 - cosi*cosi;
        double delta_GR = K1*(K1+K2)*(1+ecc*cos(f))/(c_light*pow(sini,2.0));
    
        return delta_GR;
    }
    
    inline double v_tide(double R1, double M1, double M2, double P, double f, double w)
    {
        double phi_0 = M_PI/2 - w;
        double MjMs = 1047.5655;
        M2 = M2*MjMs;
        
        return 1.13*M2/(M1*(M1+M2/MjMs))*pow(R1,4.0)*pow(P,-3.0)*sin(2*(f-phi_0));
    }
    
    double post_Newtonian(double K1, double f, double ecc, double w, double P, double M1, double M2, double R1, bool GR, bool Tid)
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
            double delta_v_tide = v_tide(R1,M1,M2,P,f,w);
            v = v + delta_v_tide;
        }
        if (GR)
        {
            double delta_LT = light_travel_time(K1, f, w, ecc);
            double delta_TD = transverse_doppler(K1, f, ecc);
            double delta_GR = gravitational_redshift(K1, K2, f, ecc);
            v = v + delta_LT + delta_TD + delta_GR;
        }
        
    
        return v;
    }
    
    //
    std::vector<double> keplerian_prec(const std::vector<double> &t, const double &P,
                                  const double &K, const double &ecc,
                                  const double &w, const double &wdot, const double &M0,
                                  const double &M0_epoch)
    {
        // allocate RVs
        std::vector<double> rv(t.size());
        
        
        // mean motion, once per orbit
        double n = 2. * M_PI / P;
        

        // ecentricity factor for g, once per orbit
        double g_e = sqrt((1 + ecc) / (1 - ecc));

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
            
            double sinE, cosE;
            double M = n * (t[i] - M0_epoch) + M0;
            brandt::solver_fixed_ecc(bounds, EA_tab, M, ecc, &sinE, &cosE);
            double g = g_e * ((1 - cosE) / sinE);
            double g2 = g * g;
            
            rv[i] = K * (cosw * ((1 - g2) / (1 + g2) + ecc) - sinw * ((2 * g) / (1 + g2)));
            
//             double f ;
            
//             double v_correction = postKep::post_Newtonian(K, f, ecc, w_t, P, double M1, double M2, double R1, bool GR, bool Tid);
      }

      return rv;
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
}

