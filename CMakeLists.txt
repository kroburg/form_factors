cmake_minimum_required(VERSION 3.0.2)
project (LALA)

add_subdirectory("emission")
add_subdirectory("math")
add_subdirectory("ray_caster")
add_subdirectory("cuda_ray_caster")
add_subdirectory("form_factors")
add_subdirectory("cpuFactorsCalculator")
add_subdirectory("import_export")
add_subdirectory("test")
add_subdirectory("thermal_equation")
add_subdirectory("thermal_solution")
add_subdirectory("subject")
add_subdirectory("controller")
add_subdirectory("perfTest")
