// ============================================================================
// output/logger.h - Logging Utilities
// ============================================================================
#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <iostream>

/**
 * @brief Simple logging utility for console output
 *
 * Provides colored output for different message types:
 *   - info: Normal information (no color)
 *   - success: Successful completion (green)
 *   - warning: Warnings (yellow)
 *   - error: Errors (red)
 */
class Logger
{
public:
    /// Log info message
    static void info(const std::string& message)
    {
        std::cout << "[INFO] " << message << std::endl;
    }
    
    /// Log success message (green)
    static void success(const std::string& message)
    {
        std::cout << "\033[32m[SUCCESS] " << message << "\033[0m" << std::endl;
    }
    
    /// Log warning message (yellow)
    static void warning(const std::string& message)
    {
        std::cout << "\033[33m[WARNING] " << message << "\033[0m" << std::endl;
    }
    
    /// Log error message (red)
    static void error(const std::string& message)
    {
        std::cerr << "\033[31m[ERROR] " << message << "\033[0m" << std::endl;
    }
};

#endif // LOGGER_H
