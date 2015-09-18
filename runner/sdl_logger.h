#pragma once

#include <ostream>
#include <string>
#include <SDL.h>

namespace utils {
    void logSDLError(std::ostream& os, const std::string& msg);
}