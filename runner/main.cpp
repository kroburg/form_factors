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


#include "proj_defs.h"
#include "AppContainer.h"
#include "ModelRenderer.h"
#include "../import_export/obj_import.h"
#include "../subject/system.h"
#include "TaskParser.h"
#include "Timeline.h"

#include <iostream>
#include <thread>
#include <fstream>

FILE _iob[] = { *stdin, *stdout, *stderr };
extern "C" FILE * __cdecl __iob_func(void) { return _iob; }

void consumeStdin(std::istream& in) {
    float curTime = 0.0f;
    auto onend =   [](int s, int t) {
        if (s == 0) {
            LOG("Parsing input done for " << t << " frame(s).");
        } else {
            ERROR("Parsing input failed.");
        }
        SDL_Event ev;
        ev.type = SDL_USEREVENT;
        ev.user.code = ModelRenderer::EV_PARSEDONE;
        ev.user.data1 = (void *)s;
        ev.user.data2 = NULL;
        SDL_PushEvent(&ev);
    };
    auto onframe = [](int cf, int tf, float step, vector<float>& temps) {
        LOG("Frame " << cf << " done");
        SDL_Event ev;
        ev.type = SDL_USEREVENT;
        ev.user.code = ModelRenderer::EV_NEWFRAME;
        ev.user.data1 = new Timeline::frame_t(step, temps);
        ev.user.data2 = NULL;
        SDL_PushEvent(&ev);
    };
    TaskParser parser(onend, onframe);
    string line;
    TRACE("Consuming from stdin started.");
    while (getline(in, line)) {
        if (parser.onLine(line) != 0) {
            break;
        }
    }
    TRACE("Consuming from stdin finished.");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        ERROR("Usage: runner <scene file (.obj)> <frame source [- for stdin] (.task)");
        return 1;
    }
    subject::scene_t* scene = NULL;
    if (obj_import::scene(argv[1], &scene) != OBJ_IMPORT_OK) {
        ERROR("Can not load file " << argv[1]);
        return 1;
    }
    TRACE("Scene loaded (" << scene->n_faces << " face(s), " << scene->n_meshes << " mesh(es)");

    AppContainer* container = new ModelRenderer("Model renderer", scene);

    int result = 0;
    if ((result = container->init(640, 480, true)) != 0) {
        ERROR("Can not init container");
        delete container;
        return result;
    }

    std::ifstream* frameFile = 0;
    std::istream* frameInput = &std::cin;
    if (argc > 2)
    {
      frameFile = new std::ifstream(argv[2]);
       if (!frameFile->is_open())
      {
        ERROR("Can not open frames file " << argv[2]);
        return 1;
      }
       frameInput = frameFile;
    }
    std::thread t(std::bind(consumeStdin, std::ref(*frameInput)));

    container->run();
    t.detach();

    delete container;
    return result;
}