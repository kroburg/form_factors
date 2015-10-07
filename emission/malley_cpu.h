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

/**
 * This module contains CPU single-threaded implementation of emission calculator.
 * Calculator is capable of work with CPU or GPU ray caster implementation.
 */

#pragma once

#include "../emission/system.h"

namespace malley_cpu
{ 
  /**
 *  @brief Factory method to create CPU emission calculator.
 *  @detail Implement emission calculator using Monte-Carlo algorithm.
 *  Use Mersenne's twister random numbers generator from C++ std.
 *  Use Malley algorithm for emited rays generation.
 *  Emited rays count per face (side) depends on face weight.
 *  Emit rays from both sides of face.
 *  */
  emission::system_t* system_create();
}
