#include <iostream>
#include "proj_defs.h"
#include "AppContainer.h"
#include "ModelRenderer.h"

int main(int argc, char** argv) {
    AppContainer* container = new ModelRenderer("Model renderer");

    int result = 0;
    if ((result = container->init(640, 480, true)) != 0) {
        ERROR("Can not init container");
        delete container;
        return result;
    }

    container->run();

    delete container;
    return result;
}