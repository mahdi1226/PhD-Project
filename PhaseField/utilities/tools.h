// ============================================================================
// utilities/tools.h - Utility Functions
//
// Small helper functions for code organization.
// ============================================================================
#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#include <ctime>

/**
 * @brief Generate timestamped folder path
 *
 * For HPC batch jobs where multiple runs need unique output directories.
 *
 * @param base Base folder path
 * @return base/run-YYYY-MM-DD-HH-MM-SS
 */
inline std::string timestamped_folder(const std::string& base)
{
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", std::localtime(&t));
    return base + "/run-" + buf;
}

#endif // TOOLS_H