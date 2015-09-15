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

#include "spherical_generator.h"

#include "../math/operations.h"
#include <random>
#include <stdlib.h>

namespace spherical_generator
{
  struct generator_t : subject::generator_t
  {
    /// @note Distribution taken from http://mathworld.wolfram.com/SpherePointPicking.html
    math::vec3 SurfacePoint()
    {
      float theta = ThetaDistribution(ThetaGenerator);
      float u = UDistribution(UGenerator);

      float c = sqrtf(1 - u * u);

      float x = c * cos(theta);
      float y = c * sin(theta);
      float z = u;

      return math::make_vec3(x, y, z);
    }

    math::vec3 VolumePoint()
    {
      /// @todo Check if cubic root for radius gives uniform sphere volume distribution.
      return SurfacePoint() * powf(RDistribution(RGenerator), 0.3333f);
    }

    std::mt19937 ThetaGenerator;
    std::mt19937 UGenerator;
    std::mt19937 RGenerator;
    std::uniform_real_distribution<float> ThetaDistribution;
    std::uniform_real_distribution<float> UDistribution;
    std::uniform_real_distribution<float> RDistribution;
  };

  int init(generator_t* generator)
  {
    generator->ThetaGenerator = std::mt19937(0);
    generator->UGenerator = std::mt19937(1);
    generator->RGenerator = std::mt19937(2);
    generator->ThetaDistribution = std::uniform_real_distribution<float>(0, float(M_PI * 2.));
    generator->UDistribution = std::uniform_real_distribution<float>(-1, 1);
    generator->RDistribution = std::uniform_real_distribution<float>(0, 1);
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
  generator_t* generator_create_spherical()
  {
    generator_t* g = spherical_generator::generator_create();
    g->methods->init(g);
    return g;
  }

  int generator_init(generator_t* g)
  {
    return g->methods->init(g);
  }

  void generator_free(generator_t* g)
  {
    g->methods->shutdown;
    free(g);
  }

  int generator_surface_point(generator_t* g, int count, math::vec3* result)
  {
    return g->methods->surface_point(g, count, result);
  }

  int generator_volume_point(generator_t* g, int count, math::vec3* result)
  {
    return g->methods->volume_point(g, count, result);
  }
}
