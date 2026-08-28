#ifndef VIEWER_HPP
#define VIEWER_HPP

#include "renderer.hpp"

#include <vector>

extern "C" {
    #include "types.h"
    #include "scenarios.h"
}

// Window configuration for the viewer's renderer.
constexpr int window_width = 800;
constexpr int window_height = 800;
constexpr const char* window_title = "Fluid Simulation";

// Everything the control panel edits: what to draw, which scenario and solver to
// run, and the two one-shot requests the panel raises for the next frame.
struct ViewerControls {
    ViewMode mode = ViewMode::SMOKE;
    ScenarioType scenario_type = KARMAN_VORTEX;
    int substeps_per_frame = 5;
    int arrow_stride = 8;
    float arrow_scale = 0.04f;
    bool auto_omega = true;

    bool request_restart = false;
    bool request_reload = false;

    // Indices into the solver and preconditioner option tables in viewer.cpp.
    // These stay menu selections; they are translated to solver types at the table.
    int solver_index = 0;
    int preconditioner_index = 1;
};

/**
 * @brief Binds the solver to the renderer. Owns the simulation context, drives the
 * control panel, and turns simulation fields into the buffers the renderer draws.
 * The renderer stays unaware of FluidContext; this class is the only place the two
 * sides meet.
 */
class Viewer {
public:
    // Creates the simulation context and starts the default scenario.
    explicit Viewer(Renderer& renderer);
    ~Viewer();

    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    // Advances the simulation and draws one frame.
    void step_frame();

private:
    // Advances the physics by the configured number of substeps.
    void advance_simulation();

    // Draws the ImGui panel and services the restart/reload it may request.
    void draw_controls();

    // Uploads the field the current view mode needs, then draws.
    void present();

    // Refreshes the FPS readout in the window title, at most twice a second.
    void update_title(double delta_time);

    // Clears the simulation and starts the scenario over. Loads the scenario
    // defaults when load_defaults is set, otherwise keeps the edited parameters.
    void restart(bool load_defaults);

    Renderer& renderer;

    FluidContext* ctx = nullptr;
    ScenarioParams params{};
    Scenario scenario{};
    ViewerControls controls{};

    // Scratch buffers for the views that need derived fields, sized once with the grid.
    std::vector<float> velocity_magnitudes;
    std::vector<float> u_centered;
    std::vector<float> v_centered;

    double last_time = 0.0;
    double title_timer = 0.0;
};

#endif // VIEWER_HPP
