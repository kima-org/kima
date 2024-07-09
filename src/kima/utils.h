#pragma once

#include <ctime>
#include <mutex>

inline std::tm localtime_xp(std::time_t timer)
{
    std::tm bt {};
#if defined(__unix__)
    localtime_r(&timer, &bt);
#elif defined(_MSC_VER)
    localtime_s(&bt, &timer);
#else
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    bt = *std::localtime(&timer);
#endif
    return bt;
}

// default = "YYYY-MM-DD HH:MM:SS"
inline std::string timestamp(const std::string& fmt = "%F %T")
{
    auto bt = localtime_xp(std::time(0));
    char buf[64];
    return {buf, std::strftime(buf, sizeof(buf), fmt.c_str(), &bt)};
}


// this creates an alias for std::make_shared
/**
 * @brief Assign a prior distribution.
 * 
 * This function defines, initializes, and assigns a prior distribution.
 * Possible distributions are ...
 * 
 * For example:
 * 
 * @code{.cpp}
 *          Cprior = make_prior<Uniform>(0, 1);
 * @endcode
 * 
 * @tparam T     ContinuousDistribution
 * @param args   Arguments for constructor of distribution
*/
template <class T, class... Args>
std::shared_ptr<T> make_prior(Args&&... args)
{
    return std::make_shared<T>(args...);
}