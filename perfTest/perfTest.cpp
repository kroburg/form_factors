
#include <stdio.h>
#include <tchar.h>
#include <random>
#include <cmath>

#include "../ray_caster/system.h"
#include "../cpuRayCaster/vec_math.h"

#include <helper_timer.h>

using namespace ray_caster;

#define M_PI            3.14159265358979323846264338327950288
#define M_2PI           6.28318530717958647692528676655900576

class ShperePointsGenerator
{
public:
  ShperePointsGenerator()
    : ThetaGenerator(0)
    , UGenerator(1)
    , RGenerator(2)
    , ThetaDistribution(0, float(M_2PI))
    , UDistribution(0, 1)
    , RDistribution(-10, 10)
  {
  }

  ray_caster::vec3 Point()
  {
    // http://mathworld.wolfram.com/SpherePointPicking.html

    float theta = ThetaDistribution(ThetaGenerator);
    float u = UDistribution(UGenerator);

    float c = sqrtf(1 - u * u);

    float x = c * cos(theta);
    float y = c * sin(theta);
    float z = u;

    return make_vec3(x, y, z);
  }

  ray_caster::vec3 Position()
  {
    return Point() * RDistribution(RGenerator);
  }

private:
  std::mt19937 ThetaGenerator;
  std::mt19937 UGenerator;
  std::mt19937 RGenerator;
  std::uniform_real_distribution<float> ThetaDistribution;
  std::uniform_real_distribution<float> UDistribution;
  std::uniform_real_distribution<float> RDistribution;
};

scene_t* MakeConfettiScene(int n_faces, ShperePointsGenerator& generator)
{
  face_t* faces = (face_t*)malloc(n_faces * sizeof(face_t));
  scene_t* scene = (scene_t*)malloc(sizeof(scene_t));
  *scene = { n_faces, faces };
  for (int i = 0; i != n_faces; ++i)
  {
    face_t& f = faces[i];
    f.points[0] = generator.Point();
    f.points[1] = generator.Point();
    f.points[2] = generator.Point();
    vec3 positon = generator.Position();
    f.points[0] += positon;
    f.points[1] += positon;
    f.points[2] += positon;
  }  
  
  return scene;
}

task_t* MakeCollapsingRays(int n_rays, ShperePointsGenerator& generator)
{
  ray_t* rays = (ray_t*)malloc(n_rays * sizeof(ray_t));  
  face_t** hit_face = (face_t**)malloc(n_rays * sizeof(face_t*));
  vec3* hit_point = (vec3*)malloc(n_rays * sizeof(vec3));
  task_t* task = (task_t*)malloc(sizeof(task_t));
  *task = { n_rays, rays, hit_face, hit_point };
  for (int i = 0; i != n_rays; ++i)
  { 
    vec3 direction = generator.Point() * 11.f;
    vec3 origin = direction * 1.1f;
    
    rays[i] = { origin, direction };
  }

  return task;
}

int _tmain(int argc, _TCHAR* argv[])
{
  int n_faces = 1000;
  int n_rays = 10000;

  ShperePointsGenerator generator;
  system_t* system = system_create(RAY_CASTER_SYSTEM_CUDA);
  printf("Generating confetti scene...\n");
  scene_t* scene = MakeConfettiScene(n_faces, generator);
  printf("Generating collapsing rays...\n");
  task_t* task = MakeCollapsingRays(n_rays, generator);

  StopWatchInterface *hTimer;
  sdkCreateTimer(&hTimer);
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  printf("Casting scene...\n");
  system_set_scene(system, scene);
  system_prepare(system);
  system_cast(system, task);

  sdkStopTimer(&hTimer);
  double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer);

  printf("Done in %fs.\n", gpuTime);
  
  system_free(system);
  scene_free(scene);
  free(task->ray);
  free(task->hit_face);
  free(task->hit_point);
  free(task);

	return 0;
}

