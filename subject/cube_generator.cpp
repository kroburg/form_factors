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

#include "generator.h"

#include "../math/operations.h"
#include <random>
#include <stdlib.h>
#include <assert.h>

namespace cube_generator
{
  struct generator_t : subject::generator_t
  {
    /// @note Distribution taken from http://mathworld.wolfram.com/SpherePointPicking.html
    math::vec3 SurfacePoint()
    {
      float a = XDistribution(XGenerator);
      float b = YDistribution(YGenerator);
      switch (SideDistribution(SideGenerator))
      {
      case 0:
        return math::make_vec3(a, b, 0);
        break;
      case 1:
        return math::make_vec3(a, b, 1);
        break;
        break;
      case 2:
        return math::make_vec3(0, a, b);
        break;
      case 3:
        return math::make_vec3(1, a, b);
        break;
      case 4:
        return math::make_vec3(a, 0, b);
        break;
      case 5:
        return math::make_vec3(a, 1, b);
        break;
      default:
        assert(!"Invalid generator value");
        return math::make_vec3(0, 0, 0);
        break;
      }
    }

    math::vec3 VolumePoint()
    {
      return math::make_vec3(XDistribution(XGenerator), YDistribution(YGenerator), ZDistribution(ZGenerator));
    }

    std::mt19937 XGenerator;
    std::mt19937 YGenerator;
    std::mt19937 ZGenerator;
    std::mt19937 SideGenerator;
    std::uniform_real_distribution<float> XDistribution;
    std::uniform_real_distribution<float> YDistribution;
    std::uniform_real_distribution<float> ZDistribution;
    std::uniform_int_distribution<int> SideDistribution;
  };

  int init(generator_t* generator)
  {
    generator->XGenerator = std::mt19937(0);
    generator->YGenerator = std::mt19937(1);
    generator->ZGenerator = std::mt19937(2);
    generator->SideGenerator = std::mt19937(3);
    generator->XDistribution = std::uniform_real_distribution<float>(0, 1);
    generator->YDistribution = std::uniform_real_distribution<float>(0, 1);
    generator->ZDistribution = std::uniform_real_distribution<float>(0, 1);
    generator->SideDistribution = std::uniform_int_distribution<int>(0, 5);
    return 0;
  }

  void shutdown(generator_t* generator)
  {
  }

  int surface_point(generator_t* generator, int count, math::vec3* result)
  {
    for (int i = 0; i != count; ++i)
      result[i] = generator->SurfacePoint();
    return 0;
  }

  int volume_point(generator_t* generator, int count, math::vec3* result)
  {
    for (int i = 0; i != count; ++i)
      result[i] = generator->VolumePoint();
    return 0;
  }

  const subject::generator_methods_t methods =
  {
    (int(*)(subject::generator_t* generator))&init,
    (void(*)(subject::generator_t* generator))&shutdown,
    (int(*)(subject::generator_t* generator, int count, math::vec3* result))&surface_point,
    (int(*)(subject::generator_t* generator, int count, math::vec3* result))&volume_point
  };

  subject::generator_t* generator_create()
  {
    generator_t* g = (generator_t*)malloc(sizeof(generator_t));
    g->methods = &methods;
    return g;
  }
}

namespace subject
{
  generator_t* generator_create_cube()
  {
    generator_t* g = cube_generator::generator_create();
    g->methods->init(g);
    return g;
  }
}
