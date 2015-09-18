#include <iostream>
#include <conio.h>
#include "proj_defs.h"
#include "AppContainer.h"
#include "ModelRenderer.h"
#include "../import_export/obj_import.h"
#include "../subject/system.h"

int main(int argc, char** argv) {
    if (argc <= 1) {
        ERROR("Please set 1st argument as model file (*.obj).");
        return 1;
    }
    subject::scene_t* scene = NULL;
    if (obj_import::scene(argv[1], &scene) != OBJ_IMPORT_OK) {
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

    container->run();

    delete container;
    return result;
}