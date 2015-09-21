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
#include <SDL.h>
#include "proj_defs.h"
#include "sdl_logger.h"

class AppContainer {
private:
    float countsPerSecond;
    bool running;
    Uint64 ticks;
    float prevUpdate;

protected:
    int width, height;
    SDL_Window *window;
    SDL_Renderer *renderer;
    const char* windowName;

    unsigned int rMask, bMask, gMask, aMask;
    int bpp;

public:
    AppContainer(const char* name);
    virtual ~AppContainer();

    int init(int width, int height, bool onlyOpenGl = false);
    int run();
    void stop();
    float getFPS();

    virtual int afterInit();
    virtual void onTick(float update);
    virtual void onEvent(SDL_Event& event);
    virtual void onRender();
    virtual void onResize(int newWidth, int newHeight);
};