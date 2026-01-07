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
 * @brief Generate timestamped folder path with optional run name
 *
 * For HPC batch jobs where multiple runs need unique output directories.
 *
 * @param base Base folder path
 * @param run_name Optional run identifier (e.g., "rosen-r5-amr")
 * @return base/run_name-YYYY-MM-DD-HH-MM-SS or base/run-YYYY-MM-DD-HH-MM-SS
 */
inline std::string timestamped_folder(const std::string& base,
                                       const std::string& run_name = "")
{
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", std::localtime(&t));

    if (run_name.empty())
        return base + "/" + buf;
    else
        return base + "/" + buf + "-" + run_name;  // Date first!
}

#endif // TOOLS_H