#include "viewer.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <omp.h>

extern "C" {
    #include "core.h"
    #include "preconditioners.h"
}

constexpr size_t default_grid_size = 256;
constexpr float default_dt = 0.016f;
constexpr float default_dx = 0.1f;
constexpr float default_density = 1.0f;
constexpr float default_viscosity = 0.001f;
constexpr int default_poisson_iter = 100;
constexpr float default_threshold = 1e-5f;

constexpr int min_poisson_iter = 10;

constexpr double title_update_interval = 0.5;

// Each menu below is driven by one table, so a label, the value it selects and the
// properties that depend on it are declared together and cannot drift apart.

struct ScenarioOption {
    const char* label;
    ScenarioType type;
};

struct SolverOption {
    const char* label;
    PressureSolver solve;
    // Omega only enters the Gauss-Seidel style sweeps; PCG never reads it.
    bool uses_omega;
};

struct PrecondOption {
    const char* label;
    PrecondType type;
};

constexpr ScenarioOption scenario_options[] = {
    { "Lid-Driven",    LID_DRIVEN    },
    { "Karman Vortex", KARMAN_VORTEX },
    { "Airfoil",       AIRFOIL       },
    { "Urban City",    URBAN_CITY    },
};

constexpr SolverOption solver_options[] = {
    { "PCG",  solve_pressure_pcg,  false },
    { "RBGS", solve_pressure_rbgs, true  },
    { "SOR",  solve_pressure_sor,  true  },
};

constexpr PrecondOption precond_options[] = {
    { "Identity",  PRECOND_IDENTITY  },
    { "Jacobi",    PRECOND_JACOBI    },
    { "Multigrid", PRECOND_MULTIGRID },
};

// Renders a combo straight off an option table. Returns true when the selection changed.
template <typename Option, size_t N>
static bool option_combo(const char* label, int& index, const Option (&options)[N]) {
    bool changed = false;
    if (ImGui::BeginCombo(label, options[index].label)) {
        for (int i = 0; i < (int)N; ++i) {
            const bool selected = (index == i);
            if (ImGui::Selectable(options[i].label, selected)) {
                index = i;
                changed = true;
            }
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return changed;
}

// Finds the value range a field should be colour mapped against.
static void scan_range(const float* data, const size_t count, float& out_min, float& out_max) {
    out_min = std::numeric_limits<float>::infinity();
    out_max = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < count; ++i) {
        const float value = data[i];
        if (value < out_min) out_min = value;
        if (value > out_max) out_max = value;
    }
    // A degenerate range would make the colour mapping divide by ~zero, so fall back to [0, 1].
    if (!std::isfinite(out_min) || !std::isfinite(out_max) || out_max - out_min < 1e-6f) {
        out_min = 0.0f;
        out_max = 1.0f;
    }
}

Viewer::Viewer(Renderer& renderer) : renderer(renderer) {
    ctx = fluid_create_context(
        default_grid_size, default_grid_size,
        default_dt, default_dx,
        default_density, default_viscosity,
        default_poisson_iter, default_threshold);

    restart(true);

    velocity_magnitudes.assign((size_t)ctx->num_cells, 0.0f);
    u_centered.assign((size_t)ctx->num_cells, 0.0f);
    v_centered.assign((size_t)ctx->num_cells, 0.0f);
    last_time = glfwGetTime();
}

Viewer::~Viewer() {
    fluid_destroy_context(ctx);
}

void Viewer::restart(const bool load_defaults) {
    if (load_defaults)
        scenario = load_scenario(controls.scenario_type, ctx, &params);
    else
        scenario_update_derived(controls.scenario_type, ctx, &params);

    fluid_reset_context(ctx);
    fluid_setup_physics(ctx, params, solver_options[controls.solver_index].solve,
                        precond_options[controls.preconditioner_index].type);
    scenario.init(ctx, params);
}

void Viewer::step_frame() {
    const double now = glfwGetTime();
    const double delta_time = now - last_time;
    last_time = now;

    advance_simulation();
    draw_controls();
    present();
    update_title(delta_time);
}

void Viewer::advance_simulation() {
    if (controls.auto_omega)
        ctx->omega = fluid_optimal_omega(ctx);

    // Viscosity and inlet velocity are live, so the reported Reynolds has to follow them.
    fluid_update_reynolds(ctx, params);

    for (int s = 0; s < controls.substeps_per_frame; ++s)
        fluid_step(ctx, params, scenario);
}

void Viewer::draw_controls() {
    renderer.begin_ui();

    ImGui::Begin("Controls");

    const ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("%.1f FPS  (%.2f ms/frame)", io.Framerate, 1000.0f / io.Framerate);
    ImGui::Text("Reynolds: %.2f", ctx->reynolds);
    ImGui::Text("OpenMP threads: %d", omp_get_max_threads());
    ImGui::Separator();

    ImGui::Text("Visualization");
    int mode = (int)controls.mode;
    ImGui::RadioButton("Smoke", &mode, (int)ViewMode::SMOKE);
    ImGui::RadioButton("Pressure", &mode, (int)ViewMode::PRESSURE);
    ImGui::RadioButton("Velocity Mag", &mode, (int)ViewMode::VELOCITY_MAGNITUDE);
    ImGui::RadioButton("Vectors Only", &mode, (int)ViewMode::VECTORS);
    ImGui::RadioButton("Field + Vectors", &mode, (int)ViewMode::FIELD_AND_VECTORS);
    controls.mode = (ViewMode)mode;
    ImGui::Separator();

    // The scenario is held as a domain type rather than a menu index, because the
    // solver API and the obstacle panel below both read it.
    int scenario_index = 0;
    for (int i = 0; i < IM_ARRAYSIZE(scenario_options); ++i) {
        if (scenario_options[i].type == controls.scenario_type)
            scenario_index = i;
    }
    if (option_combo("Scenario", scenario_index, scenario_options)) {
        controls.scenario_type = scenario_options[scenario_index].type;
        controls.request_reload = true;
    }
    ImGui::Separator();

    ImGui::Text("Solver");
    if (option_combo("Pressure Solver", controls.solver_index, solver_options))
        ctx->pressure_solver = solver_options[controls.solver_index].solve;
    if (option_combo("Preconditioner", controls.preconditioner_index, precond_options))
        fluid_set_preconditioner(ctx, precond_options[controls.preconditioner_index].type);
    ImGui::Separator();

    ImGui::Text("Parameters");
    ImGui::SliderFloat("Inlet Velocity", &params.inlet_velocity, 0.0f, 5.0f);
    ImGui::SliderFloat("Viscosity", &ctx->visc, 0.0001f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("dt", &ctx->dt, 0.001f, 0.05f, "%.4f");

    const bool omega_used = solver_options[controls.solver_index].uses_omega;
    ImGui::BeginDisabled(!omega_used);
    ImGui::Checkbox("Auto Omega", &controls.auto_omega);
    if (controls.auto_omega)
        ImGui::Text("Omega: %.4f (auto)", ctx->omega);
    else
        ImGui::SliderFloat("Omega", &ctx->omega, 1.0f, 1.99f);
    ImGui::EndDisabled();
    if (!omega_used)
        ImGui::TextDisabled("PCG does not use omega.");

    int poisson_iter = (int)ctx->poisson_iter;
    if (ImGui::SliderInt("Poisson Iter", &poisson_iter, min_poisson_iter, 2000))
        ctx->poisson_iter = (size_t)poisson_iter;
    ImGui::SliderInt("Substeps/Frame", &controls.substeps_per_frame, 1, 50);
    ImGui::SliderInt("Arrow Stride", &controls.arrow_stride, 2, 32);
    ImGui::SliderFloat("Arrow Scale", &controls.arrow_scale, 0.001f, 0.2f, "%.4f", ImGuiSliderFlags_Logarithmic);
    ImGui::Separator();

    if (controls.scenario_type == KARMAN_VORTEX) {
        ImGui::Text("Karman Obstacle");
        ImGui::SliderFloat("Obstacle X", &params.obstacle_x, 1.0f, (float)ctx->x - 2.0f);
        ImGui::SliderFloat("Obstacle Y", &params.obstacle_y, 1.0f, (float)ctx->y - 2.0f);
        int radius = (int)params.obstacle_radius;
        if (ImGui::SliderInt("Obstacle Radius", &radius, 2, (int)(ctx->y / 4)))
            params.obstacle_radius = (size_t)radius;
        if (ImGui::Button("Rebuild Solids"))
            controls.request_restart = true;
        ImGui::Separator();
    }

    if (ImGui::Button("Reset"))
        controls.request_restart = true;

    ImGui::End();

    if (controls.request_reload || controls.request_restart) {
        restart(controls.request_reload);
        controls.request_reload = false;
        controls.request_restart = false;
    }
}

void Viewer::present() {
    const int width = (int)ctx->x;
    const int height = (int)ctx->y;

    switch (controls.mode) {
        case ViewMode::SMOKE:
            renderer.update_field(ctx->smoke, width, height, 0.0f, 1.0f);
            break;
        case ViewMode::PRESSURE: {
            float pressure_min;
            float pressure_max;
            scan_range(ctx->p, ctx->num_cells, pressure_min, pressure_max);
            renderer.update_field(ctx->p, width, height, pressure_min, pressure_max);
            break;
        }
        case ViewMode::VELOCITY_MAGNITUDE:
        case ViewMode::FIELD_AND_VECTORS: {
            float velocity_min;
            float velocity_max;
            fluid_velocity_magnitude(ctx, velocity_magnitudes.data());
            scan_range(velocity_magnitudes.data(), ctx->num_cells, velocity_min, velocity_max);
            renderer.update_field(velocity_magnitudes.data(), width, height, velocity_min, velocity_max);
            break;
        }
        case ViewMode::VECTORS:
            break;
    }

    if (controls.mode == ViewMode::VECTORS || controls.mode == ViewMode::FIELD_AND_VECTORS) {
        fluid_velocity_centered(ctx, u_centered.data(), v_centered.data());
        renderer.update_arrows(u_centered.data(), v_centered.data(), width, height,
                               controls.arrow_stride, controls.arrow_scale);
    }

    renderer.draw(controls.mode);
}

void Viewer::update_title(const double delta_time) {
    title_timer += delta_time;
    if (title_timer > title_update_interval) {
        char title_buffer[80];
        std::snprintf(title_buffer, sizeof(title_buffer), "%s - %.1f FPS",
                      window_title, 1.0 / (delta_time > 0.0 ? delta_time : 1.0));
        glfwSetWindowTitle(renderer.window(), title_buffer);
        title_timer = 0.0;
    }
}
