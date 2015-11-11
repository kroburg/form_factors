// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
//
// form_factors is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// form_factors is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <iostream>

#define EPS 0.000001

#define TRACE(msg) \
    std::cout << "TRACE " << __DATE__ << __TIME__ << " " << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl
/*
#ifdef _DEBUG
#define TRACE(msg) \
    std::cout << "TRACE " << __DATE__ << __TIME__ << " " << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl
#else
#define TRACE(msg)
#endif
*/

#define LOG(msg) \
    std::cout << "INFO " << __DATE__ << __TIME__ << " " << msg << std::endl

#define ERROR(msg) \
    std::cerr << "ERROR " << __DATE__ << __TIME__ << " " << __FILE__ << "(" << __LINE__ << "): " << msg << std::endl