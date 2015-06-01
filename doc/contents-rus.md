# Form factors.

### API

Структурно система состоит из калькулятора форм-факторов и рейкастера. Калькулятор отвечает за генерацию тасков для рейкастера (лучи с граней), инитит рейкастер, запускает задачу и далее рассчитывает форм-факторы нпо методу Монте-Карло. Рейкастер можно использовать и без калькулятора, создав ему сцену и таски. Система расширяема, существуют базовые типы (C-style полиморфизм с кастомной таблицей виртуальных методов) для калькулятора ([form_factors/](../form_factors/)) и рейкастера ([ray_caster/](../ray_caster/)). Сами реализации имеют чистый C-интерфейс.

#### Пример расчета форм-факторов:

```cpp
#include <iostream>

// Хедер с фабричным методом для создания калькулятора.
#include "../form_factors/system.h"

// Хедер с фабричным методом для создания рейкастера и разными константами.
#include "../form_factors/system.h"

// Функционал для загрузки obj-файлов.
#include "../import_export/obj_import.h"

int main(int argc, char* argv[])
{
	int r = 0;

	// Создаем кастер.
	ray_caster::system_t* caster = ray_caster::system_create(RAY_CASTER_SYSTEM_CUDA);
	if (!caster) {
		std::cerr << "Failed to create ray caster" << std::endl;
		return 1;
	}

	// И калькулятор.
	form_factors::system_t* calculator = form_factors::system_create(FORM_FACTORS_CPU, caster);
	if (!calculator) {
		std::cerr << "Failed to create form factors calculator" << std::endl;
		return 1;
	}

	// Загружаем сцену.
	form_factors::scene_t* scene = 0;
	if ((r = obj_import::import_obj(input, &scene)) != OBJ_IMPORT_OK) {
		std::cerr << "Failed to load scene " << input << std::endl;
		return r;
	}

	// Создаем таск на 1М лучей.
	form_factors::task_t* task = form_factors::task_create(scene, n_rays);
	if (!task) {
		std::cerr << "Failed to create calculator task" << std::endl;
		return 1;
	}

	// Задаем сцену для калькулятора (тот сам засеттит сцену для кастера).
	if ((r = form_factors::system_set_scene(calculator, scene)) != FORM_FACTORS_OK) {
		std::cerr << "Failed to set scene." << std::endl;
		return r;
	}

	// Проверяем что все ОК перед запуском.
	if ((r = form_factors::system_prepare(calculator)) != FORM_FACTORS_OK) {
		std::cerr << "Failed to prepare scene." << std::endl;
		return r;
	}

	// Считаем.
	if ((r = form_factors::system_calculate(calculator, task)) != FORM_FACTORS_OK) {
		std::cerr << "Failed to calculate form factors." << std::endl;
		return r;
	}

	// Освобождаем ресурсы.
	form_factors::task_free(task);
	form_factors::scene_free(scene);
	ray_caster::system_free(caster);
	form_factors::system_free(calculator);

	return 0;
}

```

### Текущий статус.

* Реализован рейтрейсер на CPU и GPU (Cuda).
* Реализован расчет форм-фактор без учета свойств материала.
* Реализована загрузка моделей (сцен) для расчета из obj- и csv-файлов.
* Реализован визуалайзер obj-файлов на OpenGL и Qt.
* Проект собирается по Windows, Linux и OSX.
* Проект сопровождается как авто-, так и перфоманс-тестами.
* Оформление проекта для Visual Studio и CMake.

### Содержание репозитория.

* [build/](../build/) - Папка с собранными бинарниками для контроллера, юнит- и перфоманс тестов (см. далее).
* [controller/](../controller/) - Контроллер - консольная утилита (собственно, результат всей работы) для рассчета форм-факторов для obj-файлов (Wavefront format). Рассчет сохраняется в csv-файл. Контроллер запускается на GPU или CPU.
* [cpuFactorsCalculator/](../cpuFactorsCalculator) - Модуль для расчета форм-факторов на CPU. Имплементация *system_t* из [form_factors/](../form_factors/).
* [cpuRayCaster/](../cpuRayCaster/) - Реализация рейкастера на CPU. Пока однопоточная. Реализует *system_t* из [ray_caster/](../ray_caster/).
* [cudaRayCaster/](../cudaRayCaster/) - Реализация рейкастера на GPU. Реализует *system_t* из [ray_caster/](../ray_caster/).
* [doc/](../doc/) - Документация.
* [ext/](../ext/) - Полезные сторонние либы и хедеры, например [gtest](https://code.google.com/p/googletest/).
* [form_factors/](../form_factors/) - Фабричный метод и по совместительству базовый функционал (тип *system_t*) для реализации калькуляторов форм-факторов.
* [import_export/](../import_export/) - Функционал загрузки сцен из csv- и obj-файлов.
* [math/](../math/) - Математические примитивы и операции с ними. Треугольники и проверка пересечения.
* [models/](../models/) - Модельки в obj-формате.
* [perfTest/](../perfTest/) - Перфоманс-тест для рейтрейсера (сценка-конфетти) на CPU и GPU, а так же для калькулятора форм-факторов на CPU.
* [ray_caster/](../ray_caster/) - Фабричный метод и по совместительству базовый функционал (тип system_t) для реализации рейкастеров.
* [results/](../results/) - Папка с результатами.
* [test/](../test/) - Автотесты и юнит-тесты на [gtest](https://code.google.com/p/googletest/).
* [qt_shell/](../qt_shell/) - графическая консоль

### Текущий прогресс:

* Объединяем все проекты в один.
* Поддержка CMake.
* Дописываем тесты (табличные, для вариантов с параллельными и перпендикулярными плоскостями)
* Пишем документацию.

### Особенности сборки.

Собираем под x86.

Для линукса и сборки визуалайзера необходим [CMake](http://www.cmake.org/download/).

Необходим [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads?sid=788784). *Подумать о возможности сборки без Cuda, для расчетов на CPU.*

Для визуалайзера необходим [Qt5](http://www.qt.io/download/). Если собирается под Windows - качать x86-битные библиотеки под установленную Visual Studio.

Весь репозиторий нужно клонировать в {Cuda_Samples_Folder}/0_Simple папку. Связано с тем, что рейтресер Cuda использует примитивы от сэмплов NVidia.

Проект для визуалайзера собирается только через CMake. Необходимо выполнить (bash или Powershell/cmd в Windows):
```sh
cmake .
```
После этого либо *make* на линуксе, на Windows будут сгенерированы файлы солюшена для Visual Studio. На Windows CMake может сразу не обнаружить где лежит Qt - но он выдаст подробный мануал как это сделать.

### TODO:

* Избавиться от зависимостей от сэмплов Cuda.
