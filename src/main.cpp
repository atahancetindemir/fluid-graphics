#include "renderer.hpp"
#include "viewer.hpp"

int main() {
    Renderer renderer(window_width, window_height, window_title);
    Viewer viewer(renderer);

    while (!renderer.should_close())
        viewer.step_frame();

    return 0;
}
