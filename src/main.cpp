#include "graphics.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include <omp.h>

extern "C"
{
    #include "types.h"
    #include "core.h"
    #include "scenarios.h"
    #include "preconditioners.h"
}

constexpr int window_width = 800;
constexpr int window_height = 800;
constexpr const char* window_title = "Fluid Simulation";

constexpr size_t default_grid_size = 256;
constexpr float default_dt = 0.016f;
constexpr float default_dx = 0.1f;
constexpr float default_density = 1.0f;
constexpr float default_viscosity = 0.001f;
constexpr int default_poisson_iter = 100;
constexpr float default_threshold = 1e-4f;

constexpr int min_poisson_iter = 10;

constexpr double title_update_interval = 0.5;

struct runtime_controls
{
    visual_mode mode = visual_mode::smoke;
    ScenarioType scenario_type = KARMAN_VORTEX;
    int substeps_per_frame = 5;
    int arrow_stride = 8;
    float arrow_scale = 0.04f;
    bool auto_omega = true;

    bool request_restart = false;
    bool request_reload = false;

    // 0 = PCG, 1 = RBGS, 2 = SOR
    int solver_index = 0;
    // 0 = Identity, 1 = Jacobi, 2 = Multigrid (placeholder)
    int preconditioner_index = 1;
};

[[nodiscard]] PressureSolver pick_solver(const int index)
{
    switch (index)
    {
        case 1: return solve_pressure_rbgs;
        case 2: return solve_pressure_sor;
        default: return solve_pressure_pcg;
    }
}

[[nodiscard]] PrecondType pick_precond(const int index)
{
    switch (index)
    {
        case 0: return PRECOND_IDENTITY;
        case 2: return PRECOND_MULTIGRID;
        default: return PRECOND_JACOBI;
    }
}

// Omega only enters the Gauss-Seidel style sweeps; PCG never reads it.
[[nodiscard]] bool solver_uses_omega(const int index)
{
    return index != 0;
}

void scan_min_max(const float* data, const size_t count, float& out_min, float& out_max)
{
    out_min = std::numeric_limits<float>::infinity();
    out_max = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < count; ++i)
    {
        const float value = data[i];
        if (value < out_min) out_min = value;
        if (value > out_max) out_max = value;
    }
    // A degenerate range would make the colour mapping divide by ~zero, so fall back to [0, 1].
    if (!std::isfinite(out_min) || !std::isfinite(out_max) || out_max - out_min < 1e-6f)
    {
        out_min = 0.0f;
        out_max = 1.0f;
    }
}


// Clears the simulation and starts the scenario over.
void begin_run(FluidContext* fluid_context, ScenarioParams& params, const runtime_controls& controls,
               Scenario& scenario, const bool load_defaults)
{
    if (load_defaults)
        scenario = load_scenario(controls.scenario_type, fluid_context, &params);
    else
        scenario_update_derived(controls.scenario_type, fluid_context, &params);

    fluid_reset_context(fluid_context);
    fluid_setup_physics(fluid_context, params, pick_solver(controls.solver_index),
                        pick_precond(controls.preconditioner_index));
    scenario.init(fluid_context, params);
}

void draw_control_panel(runtime_controls& controls, FluidContext* fluid_context, ScenarioParams& params)
{
    ImGui::Begin("Controls");

    const ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("%.1f FPS  (%.2f ms/frame)", io.Framerate, 1000.0f / io.Framerate);
    ImGui::Text("Reynolds: %.2f", fluid_context->reynolds);
    ImGui::Text("OpenMP threads: %d", omp_get_max_threads());
    ImGui::Separator();

    ImGui::Text("Visualization");
    int mode = (int)controls.mode;
    ImGui::RadioButton("Smoke", &mode, (int)visual_mode::smoke);
    ImGui::RadioButton("Pressure", &mode, (int)visual_mode::pressure);
    ImGui::RadioButton("Velocity Mag", &mode, (int)visual_mode::velocity_magnitude);
    ImGui::RadioButton("Vectors Only", &mode, (int)visual_mode::velocity_vectors_only);
    ImGui::RadioButton("Field + Vectors", &mode, (int)visual_mode::field_plus_vectors);
    controls.mode = (visual_mode)mode;
    ImGui::Separator();

    static const char* scenario_items[] = { "Lid-Driven", "Karman Vortex", "Airfoil", "Urban City" };
    int scenario_index = (int)controls.scenario_type;
    if (ImGui::Combo("Scenario", &scenario_index, scenario_items, IM_ARRAYSIZE(scenario_items)))
    {
        controls.scenario_type = (ScenarioType)scenario_index;
        controls.request_reload = true;
    }
    ImGui::Separator();

    ImGui::Text("Solver");
    static const char* solver_items[] = { "PCG", "RBGS", "SOR" };
    static const char* precond_items[] = { "Identity", "Jacobi", "Multigrid" };
    if (ImGui::Combo("Pressure Solver", &controls.solver_index, solver_items, IM_ARRAYSIZE(solver_items)))
        fluid_context->pressure_solver = pick_solver(controls.solver_index);
    if (ImGui::Combo("Preconditioner", &controls.preconditioner_index, precond_items, IM_ARRAYSIZE(precond_items)))
        fluid_set_preconditioner(fluid_context, pick_precond(controls.preconditioner_index));
    ImGui::Separator();

    ImGui::Text("Parameters");
    ImGui::SliderFloat("Inlet Velocity", &params.inlet_velocity, 0.0f, 5.0f);
    ImGui::SliderFloat("Viscosity", &fluid_context->visc, 0.0001f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("dt", &fluid_context->dt, 0.001f, 0.05f, "%.4f");

    const bool omega_used = solver_uses_omega(controls.solver_index);
    ImGui::BeginDisabled(!omega_used);
    ImGui::Checkbox("Auto Omega", &controls.auto_omega);
    if (controls.auto_omega)
        ImGui::Text("Omega: %.4f (auto)", fluid_context->omega);
    else
        ImGui::SliderFloat("Omega", &fluid_context->omega, 1.0f, 1.99f);
    ImGui::EndDisabled();
    if (!omega_used)
        ImGui::TextDisabled("PCG does not use omega.");

    int poisson_iter = (int)fluid_context->poisson_iter;
    if (ImGui::SliderInt("Poisson Iter", &poisson_iter, min_poisson_iter, 2000))
        fluid_context->poisson_iter = (size_t)poisson_iter;
    ImGui::SliderInt("Substeps/Frame", &controls.substeps_per_frame, 1, 50);
    ImGui::SliderInt("Arrow Stride", &controls.arrow_stride, 2, 32);
    ImGui::SliderFloat("Arrow Scale", &controls.arrow_scale, 0.001f, 0.2f, "%.4f", ImGuiSliderFlags_Logarithmic);
    ImGui::Separator();

    if (controls.scenario_type == KARMAN_VORTEX)
    {
        ImGui::Text("Karman Obstacle");
        ImGui::SliderFloat("Obstacle X", &params.obstacle_x, 1.0f, (float)fluid_context->x - 2.0f);
        ImGui::SliderFloat("Obstacle Y", &params.obstacle_y, 1.0f, (float)fluid_context->y - 2.0f);
        int radius = (int)params.obstacle_radius;
        if (ImGui::SliderInt("Obstacle Radius", &radius, 2, (int)(fluid_context->y / 4)))
            params.obstacle_radius = (size_t)radius;
        if (ImGui::Button("Rebuild Solids"))
            controls.request_restart = true;
        ImGui::Separator();
    }

    if (ImGui::Button("Reset"))
        controls.request_restart = true;

    ImGui::End();
}

int main()
{
    graphics engine(window_width, window_height, window_title);

    FluidContext* fluid_context = fluid_create_context(
        default_grid_size, default_grid_size,
        default_dt, default_dx,
        default_density, default_viscosity,
        default_poisson_iter, default_threshold);

    runtime_controls controls;
    ScenarioParams params;
    Scenario scenario;
    begin_run(fluid_context, params, controls, scenario, true);

    std::vector<float> velocity_magnitudes((size_t)fluid_context->num_cells, 0.0f);

    double last_time = glfwGetTime();
    double title_timer = 0.0;

    while (!engine.should_close())
    {
        const double now = glfwGetTime();
        const double delta_time = now - last_time;
        last_time = now;

        if (controls.auto_omega)
            fluid_context->omega = fluid_optimal_omega(fluid_context);

        // Viscosity and inlet velocity are live, so the reported Reynolds has to follow them.
        fluid_update_reynolds(fluid_context, params);

        for (int s = 0; s < controls.substeps_per_frame; ++s)
            fluid_step(fluid_context, params, scenario);

        engine.begin_ui();
        draw_control_panel(controls, fluid_context, params);

        if (controls.request_reload || controls.request_restart)
        {
            begin_run(fluid_context, params, controls, scenario, controls.request_reload);
            controls.request_reload = false;
            controls.request_restart = false;
        }

        const int width = (int)fluid_context->x;
        const int height = (int)fluid_context->y;

        switch (controls.mode)
        {
            case visual_mode::smoke:
                engine.update_field(fluid_context->smoke, width, height, 0.0f, 1.0f);
                break;
            case visual_mode::pressure:
            {
                float pressure_min;
                float pressure_max;
                scan_min_max(fluid_context->p, fluid_context->num_cells, pressure_min, pressure_max);
                engine.update_field(fluid_context->p, width, height, pressure_min, pressure_max);
                break;
            }
            case visual_mode::velocity_magnitude:
            case visual_mode::field_plus_vectors:
            {
                float velocity_min;
                float velocity_max;
                fluid_velocity_magnitude(fluid_context, velocity_magnitudes.data());
                scan_min_max(velocity_magnitudes.data(), fluid_context->num_cells, velocity_min, velocity_max);
                engine.update_field(velocity_magnitudes.data(), width, height, velocity_min, velocity_max);
                break;
            }
            case visual_mode::velocity_vectors_only:
                break;
        }

        if (controls.mode == visual_mode::velocity_vectors_only || controls.mode == visual_mode::field_plus_vectors)
            engine.update_arrows(fluid_context->u, fluid_context->v, width, height, controls.arrow_stride, controls.arrow_scale);

        engine.draw(controls.mode);

        title_timer += delta_time;
        if (title_timer > title_update_interval)
        {
            char title_buffer[80];
            std::snprintf(title_buffer, sizeof(title_buffer), "Fluid Simulation - %.1f FPS", 1.0 / (delta_time > 0.0 ? delta_time : 1.0));
            glfwSetWindowTitle(engine.window(), title_buffer);
            title_timer = 0.0;
        }
    }

    fluid_destroy_context(fluid_context);
    return 0;
}
