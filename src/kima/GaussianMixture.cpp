#include "GaussianMixture.h"

namespace DNest4
{


GaussianMixture::GaussianMixture(std::vector<double> means, std::vector<double> sigmas, 
                                 std::vector<double> weights,
                                 double lower, double upper)
: means(means), sigmas(sigmas), weights(weights), lower(lower), upper(upper)
{
    double min_mean = *std::min_element(means.begin(), means.end());
    double max_mean = *std::max_element(means.begin(), means.end());
    double max_sigma = *std::max_element(sigmas.begin(), sigmas.end());
    min_support_approx = min_mean - 10 * max_sigma;
    max_support_approx = max_mean + 10 * max_sigma;

    for (size_t i = 0; i < means.size(); i++)
    {
        gaussians.push_back(TruncatedGaussian(means[i], sigmas[i], lower, upper));
    }
}

double GaussianMixture::cdf(double x) const
{
    double sum = 0.0;
    for (size_t i = 0; i < gaussians.size(); i++)
    {
        sum += weights[i] * gaussians[i].cdf(x);
    }
    return sum;
}

double GaussianMixture::cdf_inverse(double p) const
{
    if( (p < 0.0) || (p > 1.0) )
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
    // approximate value for large probability
    if (p > cdf(max_support_approx)) return max_support_approx;
    // approximate value for small probability
    if (p < cdf(min_support_approx)) return min_support_approx;
    return brenth([p, this](double x){return cdf(x) - p;}, min_support_approx, max_support_approx);
}

double GaussianMixture::log_pdf(double x) const
{
    double sum = 0.0;
    for (size_t i = 0; i < gaussians.size(); i++)
    {
        sum += weights[i] * exp(gaussians[i].log_pdf(x));
    }
    return log(sum);
}

} // namespace DNest4