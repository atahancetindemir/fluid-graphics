#include "graphics_engine.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <cmath>
#include <cstdio>
#include <cstring>
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

constexpr float pi = 3.14159265358979323846f;

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

constexpr double title_update_interval = 0.5;

struct runtime_controls
{
    visual_mode mode = visual_mode::smoke;
    ScenarioType scenario_type = KARMAN_VORTEX;
    int substeps_per_frame = 5;
    int arrow_stride = 8;
    float arrow_scale = 0.04f;
    bool auto_omega = true;
    bool request_reset = false;
    bool request_rebuild_scenario = false;

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
    if (!std::isfinite(out_min) || !std::isfinite(out_max))
    {
        out_min = 0.0f;
        out_max = 1.0f;
    }
}

void compute_velocity_magnitude(const FluidContext* fluid_context, float* out, float& out_min, float& out_max)
{
    out_min = std::numeric_limits<float>::infinity();
    out_max = -std::numeric_limits<float>::infinity();

    const size_t height = fluid_context->y;
    const size_t height_plus_one = height + 1;

    for (size_t i = 0; i < fluid_context->x; ++i)
    {
        for (size_t j = 0; j < height; ++j)
        {
            const float u_center = 0.5f * (fluid_context->u[i * height + j] + fluid_context->u[(i + 1) * height + j]);
            const float v_center = 0.5f * (fluid_context->v[i * height_plus_one + j] + fluid_context->v[i * height_plus_one + (j + 1)]);
            const float magnitude = std::sqrt(u_center * u_center + v_center * v_center);
            out[i * height + j] = magnitude;
            if (magnitude < out_min) out_min = magnitude;
            if (magnitude > out_max) out_max = magnitude;
        }
    }
    if (!std::isfinite(out_min) || !std::isfinite(out_max) || out_max <= out_min)
    {
        out_min = 0.0f;
        out_max = 1.0f;
    }
}

[[nodiscard]] Scenario reload_scenario(FluidContext* fluid_context, ScenarioParams& params, const ScenarioType type,
                                       const PressureSolver solver, const PrecondType precond)
{
    const size_t u_count = (fluid_context->x + 1) * fluid_context->y;
    const size_t v_count = fluid_context->x * (fluid_context->y + 1);
    const size_t cells = fluid_context->num_cells;

    std::memset(fluid_context->u, 0, u_count * sizeof(float));
    std::memset(fluid_context->v, 0, v_count * sizeof(float));
    std::memset(fluid_context->p, 0, cells * sizeof(float));
    std::memset(fluid_context->div, 0, cells * sizeof(float));
    std::memset(fluid_context->smoke, 0, cells * sizeof(float));
    std::memset(fluid_context->solid, 0, cells * sizeof(uint8_t));
    std::memset(fluid_context->u_prev, 0, u_count * sizeof(float));
    std::memset(fluid_context->v_prev, 0, v_count * sizeof(float));
    std::memset(fluid_context->smoke_prev, 0, cells * sizeof(float));
    std::memset(fluid_context->cg_r, 0, cells * sizeof(float));
    std::memset(fluid_context->cg_d, 0, cells * sizeof(float));
    std::memset(fluid_context->cg_q, 0, cells * sizeof(float));
    std::memset(fluid_context->cg_z, 0, cells * sizeof(float));

    Scenario scenario = load_scenario(type, fluid_context, &params);
    fluid_setup_physics(fluid_context, params, solver, precond);
    scenario.init(fluid_context, params);
    return scenario;
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
        controls.request_rebuild_scenario = true;
    }
    ImGui::Separator();

    ImGui::Text("Solver");
    static const char* solver_items[] = { "PCG", "RBGS", "SOR" };
    static const char* precond_items[] = { "Identity", "Jacobi", "Multigrid" };
    if (ImGui::Combo("Pressure Solver", &controls.solver_index, solver_items, IM_ARRAYSIZE(solver_items)))
        fluid_context->pressure_solver = pick_solver(controls.solver_index);
    if (ImGui::Combo("Preconditioner", &controls.preconditioner_index, precond_items, IM_ARRAYSIZE(precond_items)))
    {
        // Setting the function pointer directly mirrors what fluid_setup_physics does internally.
        switch (pick_precond(controls.preconditioner_index))
        {
            case PRECOND_IDENTITY: fluid_context->precondition = precondition_identity; break;
            case PRECOND_JACOBI: fluid_context->precondition = precondition_jacobi; break;
            case PRECOND_MULTIGRID: fluid_context->precondition = precondition_multigrid; break;
        }
    }
    ImGui::Separator();

    ImGui::Text("Parameters");
    ImGui::SliderFloat("Inlet Velocity", &params.inlet_velocity, 0.0f, 5.0f);
    ImGui::SliderFloat("Viscosity", &fluid_context->visc, 0.0001f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("dt", &fluid_context->dt, 0.001f, 0.05f, "%.4f");
    ImGui::Checkbox("Auto Omega", &controls.auto_omega);
    if (controls.auto_omega)
        ImGui::Text("Omega: %.4f (auto)", fluid_context->omega);
    else
        ImGui::SliderFloat("Omega", &fluid_context->omega, 1.0f, 1.99f);
    ImGui::SliderInt("Poisson Iter", &fluid_context->poisson_iter, 1, 2000);
    ImGui::SliderFloat("Threshold", &fluid_context->threshold, 1e-7f, 1e-3f, "%.7f", ImGuiSliderFlags_Logarithmic);
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
            controls.request_rebuild_scenario = true;
        ImGui::Separator();
    }

    if (ImGui::Button("Reset"))
        controls.request_reset = true;

    ImGui::End();
}

int main()
{
    graphics_engine engine(window_width, window_height, window_title);

    FluidContext* fluid_context = fluid_create_context(
        default_grid_size, default_grid_size,
        default_dt, default_dx,
        default_density, default_viscosity,
        default_poisson_iter, default_threshold);

    runtime_controls controls;
    ScenarioParams params;
    Scenario scenario = reload_scenario(fluid_context, params, controls.scenario_type,
                                        pick_solver(controls.solver_index),
                                        pick_precond(controls.preconditioner_index));

    std::vector<float> velocity_magnitudes((size_t)fluid_context->num_cells, 0.0f);

    double last_time = glfwGetTime();
    double title_timer = 0.0;

    while (!engine.should_close())
    {
        const double now = glfwGetTime();
        const double delta_time = now - last_time;
        last_time = now;

        // Auto-recompute omega each frame if enabled (the user may have changed grid via rebuild).
        if (controls.auto_omega)
            fluid_context->omega = 2.0f / (1.0f + std::sin(pi / (float)fluid_context->x));

        for (int s = 0; s < controls.substeps_per_frame; ++s)
            fluid_step(fluid_context, params, scenario);

        engine.begin_ui();
        draw_control_panel(controls, fluid_context, params);

        if (controls.request_reset || controls.request_rebuild_scenario)
        {
            scenario = reload_scenario(fluid_context, params, controls.scenario_type,
                                       pick_solver(controls.solver_index),
                                       pick_precond(controls.preconditioner_index));
            controls.request_reset = false;
            controls.request_rebuild_scenario = false;
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
                if (pressure_max - pressure_min < 1e-6f) { pressure_min = 0.0f; pressure_max = 1.0f; }
                engine.update_field(fluid_context->p, width, height, pressure_min, pressure_max);
                break;
            }
            case visual_mode::velocity_magnitude:
            case visual_mode::field_plus_vectors:
            {
                float velocity_min;
                float velocity_max;
                compute_velocity_magnitude(fluid_context, velocity_magnitudes.data(), velocity_min, velocity_max);
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
