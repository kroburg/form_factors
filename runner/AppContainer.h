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