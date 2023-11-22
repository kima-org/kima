#pragma once

#if defined(_WIN32)
# define KIMA_EXPORT
# define KIMA_IMPORT
//#  define KIMA_EXPORT __declspec(dllexport)
//#  define KIMA_IMPORT __declspec(dllimport)
#else
#  define KIMA_EXPORT __attribute__ ((visibility("default")))
#  define KIMA_IMPORT __attribute__ ((visibility("default")))
#endif

#if defined(KIMA_BUILD)
#  define KIMA_API KIMA_EXPORT
#else
#  define KIMA_API KIMA_IMPORT
#endif