#include "viewer.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <omp.h>

extern "C" {
    #include "core.h"
    #include "preconditioners.h"
}

constexpr size_t default_grid_size = 128;
constexpr float default_dt = 0.016f;
constexpr float default_dx = 0.1f;
constexpr float default_density = 1.0f;
constexpr float default_viscosity = 0.001f;
constexpr SolveTolerance default_tolerance = { 1e-4f, 1e-9f, 1000 };

constexpr float info_overlay_alpha = 0.75f;
constexpr ImU32 colorbar_text_color = IM_COL32(255, 255, 255, 255);
constexpr ImVec4 converged_color(0.45f, 0.85f, 0.45f, 1.0f);
constexpr ImVec4 capped_color(1.0f, 0.65f, 0.25f, 1.0f);
constexpr ImVec4 stalled_color(1.0f, 0.35f, 0.35f, 1.0f);

// Screen layout in pixels: the field and its colour bar centered on the left, and the panel and info
// overlay in a column along the right edge.
constexpr int layout_margin = 12;
constexpr int layout_gap = 10;
constexpr int colorbar_width = 90;
constexpr int colorbar_bar_width = 20;
constexpr int colorbar_slices = 64;
constexpr int colorbar_ticks = 5;
constexpr int ui_column_width = 240;
constexpr float ui_label_width = 110.0f;

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
    // The preconditioner only enters PCG.
    bool uses_preconditioner;
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
    { "PCG",  solve_pressure_pcg,  false, true  },
    { "RBGS", solve_pressure_rbgs, true,  false },
    { "SOR",  solve_pressure_sor,  true,  false },
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

// Field and colour bar centered left of the UI column, at the largest whole number of pixels per cell
static FieldRect layout_field(const ImVec2& display, const int width, const int height) {
    const int region_width = (int)display.x - 2 * layout_margin - layout_gap - ui_column_width;
    const int region_height = (int)display.y - 2 * layout_margin;
    const int scale = std::max(1, std::min((region_width - layout_gap - colorbar_width) / width, region_height / height));
    const int block_width = scale * width + layout_gap + colorbar_width;
    return { layout_margin + (region_width - block_width) / 2, layout_margin + (region_height - scale * height) / 2,
             scale * width, scale * height };
}

// Same polynomial as turbo() in assets/shaders/field.frag
static ImU32 turbo_color(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    const float value2 = value * value;
    const float value3 = value2 * value;
    const float value4 = value3 * value;
    const float value5 = value4 * value;

    const float r = 0.13572138f + 4.61539260f * value - 42.66032258f * value2 + 132.13108234f * value3 - 152.94239396f * value4 + 59.28637943f * value5;
    const float g = 0.09140261f + 2.19418839f * value + 4.84296658f * value2 - 14.18503333f * value3 + 4.27729857f * value4 + 2.82956604f * value5;
    const float b = 0.10667330f + 12.64194608f * value - 60.58204836f * value2 + 110.36276771f * value3 - 89.90310912f * value4 + 27.34824973f * value5;
    return ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, 1.0f));
}

// Vertical colour bar right of the field: title on top, max at the top of the bar, min at the bottom
static void draw_colorbar(const FieldRect& field, const float range_min, const float range_max, const char* title) {
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    const float font_size = ImGui::GetFontSize();
    const float left = (float)(field.x + field.width + layout_gap);
    const float right = left + (float)colorbar_bar_width;
    const float top = (float)field.y + font_size + 10.0f;
    const float bottom = (float)(field.y + field.height);

    draw_list->AddText(ImVec2(left, (float)field.y), colorbar_text_color, title);

    const float slice_height = (bottom - top) / (float)colorbar_slices;
    for (int s = 0; s < colorbar_slices; ++s) {
        const float y0 = top + (float)s * slice_height;
        const ImU32 upper = turbo_color(1.0f - (float)s / (float)colorbar_slices);
        const ImU32 lower = turbo_color(1.0f - (float)(s + 1) / (float)colorbar_slices);
        draw_list->AddRectFilledMultiColor(ImVec2(left, y0), ImVec2(right, y0 + slice_height), upper, upper, lower, lower);
    }

    for (int t = 0; t < colorbar_ticks; ++t) {
        const float fraction = (float)t / (float)(colorbar_ticks - 1);
        const float y = top + fraction * (bottom - top);
        char label[16];
        snprintf(label, sizeof(label), "%.3g", range_max - fraction * (range_max - range_min));
        draw_list->AddText(ImVec2(right + 6.0f, y - 0.5f * font_size), colorbar_text_color, label);
    }
}

Viewer::Viewer(Renderer& renderer) : renderer(renderer) {
    ctx = fluid_create_context(
        default_grid_size, default_grid_size,
        default_dt, default_dx,
        default_density, default_viscosity,
        default_tolerance);

    restart(true);

    velocity_magnitudes.assign((size_t)ctx->num_cells, 0.0f);
    u_centered.assign((size_t)ctx->num_cells, 0.0f);
    v_centered.assign((size_t)ctx->num_cells, 0.0f);
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
    fluid_setup_physics(ctx, solver_options[controls.solver_index].solve,
                        precond_options[controls.preconditioner_index].type);
    scenario.init(ctx, params);
}

void Viewer::step_frame() {
    advance_simulation();

    renderer.begin_ui();
    // Esc quits unless a widget is using the keyboard
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) && !ImGui::GetIO().WantCaptureKeyboard)
        renderer.close();

    // The overlay reports the solve advance_simulation just ran, so it goes before the panel
    // gets a chance to restart the simulation.
    draw_info_overlay();
    draw_controls();
    present();
}

void Viewer::advance_simulation() {
    if (controls.auto_omega)
        ctx->omega = fluid_optimal_omega(ctx);

    for (int s = 0; s < controls.substeps_per_frame; ++s)
        fluid_step(ctx, params, scenario);
}

void Viewer::draw_info_overlay() {
    const ImGuiIO& io = ImGui::GetIO();
    // Pinned to the bottom-right corner at the column width; the height fits the contents.
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - layout_margin, io.DisplaySize.y - layout_margin),
                            ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowSize(ImVec2((float)ui_column_width, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(info_overlay_alpha);
    ImGui::Begin("Info", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs |
                 ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing);

    // Monospaced font: padded labels line up the values
    ImGui::Text("%-10s %d x %d", "Grid", (int)ctx->x, (int)ctx->y);
    // Viscosity and inlet velocity are live, so the reported Reynolds is computed every frame.
    ImGui::Text("%-10s %.2f", "Re", fluid_reynolds(ctx, params));
    ImGui::Text("%-10s %.4f", "dt", ctx->dt);
    ImGui::Text("%-10s %llu", "Step", (unsigned long long)ctx->step);
    ImGui::Text("%-10s %.3f s", "Time", ctx->time);
    ImGui::Text("%-10s %.1f (%.2f ms/frame)", "FPS", io.Framerate, 1000.0f / io.Framerate);
    ImGui::Text("%-10s %d", "Threads", omp_get_max_threads());
    ImGui::Separator();

    const SolverOption& solver = solver_options[controls.solver_index];
    if (solver.uses_preconditioner)
        ImGui::Text("%-10s %s (%s)", "p solve", solver.label, precond_options[controls.preconditioner_index].label);
    else
        ImGui::Text("%-10s %s", "p solve", solver.label);

    if (ctx->step == 0) {
        ImGui::TextDisabled("%-10s %s", "Iter", "no solve yet");
        ImGui::End();
        return;
    }

    const SolveStats& stats = ctx->stats;
    // Fixed width keeps the status in place
    char iter_text[32];
    snprintf(iter_text, sizeof(iter_text), "%d/%d", (int)stats.iterations, (int)ctx->tol.max_iter);
    ImGui::Text("%-10s %-9s", "Iter", iter_text);
    ImGui::SameLine();
    const ImVec4 status_color = stats.status == SOLVE_CONVERGED ? converged_color
                              : stats.status == SOLVE_CAPPED    ? capped_color
                                                                : stalled_color;
    ImGui::TextColored(status_color, "%-9s", fluid_solve_status_name(stats.status));
    if (solver.uses_preconditioner)
        ImGui::TextDisabled("%-10s %d", "Restarts", (int)stats.replacements);

    ImGui::Text("%-10s %.2e", "dp", stats.delta_p);

    // Relative to the largest right-hand side term, the residual does not depend on dt or dx.
    if (stats.rhs > 0.0f) {
        ImGui::Text("%-10s %.2e / %.2e", "Residual", stats.residual / stats.rhs, ctx->tol.rtol);
        ImGui::Text("%-10s %.2e", "Initial", stats.initial_residual / stats.rhs);
    } else {
        ImGui::Text("%-10s %.2e abs", "Residual", stats.residual);
        ImGui::Text("%-10s %.2e abs", "Initial", stats.initial_residual);
    }

    ImGui::Text("%-10s %.2f ms", "Solve", stats.solve_ms);
    ImGui::Text("%-10s rms %.2e", "Continuity", ctx->continuity_rms);
    ImGui::Text("%-10s max %.2e", "", ctx->continuity_max);

    // Net outflow per side as % of total inflow
    const MassBalance& mass = ctx->mass;
    if (mass.inflow > 0.0f) {
        ImGui::Text("%-10s imbalance %+.2e", "Mass", (mass.inflow - mass.outflow) / mass.inflow);
        ImGui::Text("%-10s L %+4.0f%%  R %+4.0f%%", "Net out",
                    100.0f * mass.left / mass.inflow, 100.0f * mass.right / mass.inflow);
        ImGui::Text("%-10s B %+4.0f%%  T %+4.0f%%", "",
                    100.0f * mass.bottom / mass.inflow, 100.0f * mass.top / mass.inflow);
    } else {
        ImGui::Text("%-10s closed", "Mass");
    }

    ImGui::End();
}

void Viewer::draw_controls() {
    // Pinned to the top-right corner at the column width; the height fits the contents.
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - layout_margin, (float)layout_margin), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2((float)ui_column_width, 0.0f), ImGuiCond_Always);
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);
    // Widgets fill the width left of a fixed label column
    ImGui::PushItemWidth(-ui_label_width);

    if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
        int mode = (int)controls.mode;
        ImGui::RadioButton("Smoke", &mode, (int)ViewMode::SMOKE);
        ImGui::RadioButton("Pressure", &mode, (int)ViewMode::PRESSURE);
        ImGui::RadioButton("Velocity Mag", &mode, (int)ViewMode::VELOCITY_MAGNITUDE);
        ImGui::RadioButton("Vectors Only", &mode, (int)ViewMode::VECTORS);
        ImGui::RadioButton("Field + Vectors", &mode, (int)ViewMode::FIELD_AND_VECTORS);
        controls.mode = (ViewMode)mode;

        const bool vectors_shown = (controls.mode == ViewMode::VECTORS ||
                                    controls.mode == ViewMode::FIELD_AND_VECTORS);
        ImGui::BeginDisabled(!vectors_shown);
        ImGui::SliderInt("Arrow Stride", &controls.arrow_stride, 2, 32, "%d", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("Arrow Scale", &controls.arrow_scale, 0.001f, 0.2f, "%.4f",
                           ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        ImGui::EndDisabled();
        ImGui::SliderInt("Substeps/Frame", &controls.substeps_per_frame, 1, 50, "%d", ImGuiSliderFlags_AlwaysClamp);
    }

    if (ImGui::CollapsingHeader("Scenario##header", ImGuiTreeNodeFlags_DefaultOpen)) {
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
    }

    if (ImGui::CollapsingHeader("Solver", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (option_combo("Pressure Solver", controls.solver_index, solver_options))
            ctx->pressure_solver = solver_options[controls.solver_index].solve;

        // Read after the combo, so a solver picked this frame already sets what is enabled below.
        const SolverOption& solver = solver_options[controls.solver_index];
        ImGui::BeginDisabled(!solver.uses_preconditioner);
        if (option_combo("Preconditioner", controls.preconditioner_index, precond_options))
            fluid_set_preconditioner(ctx, precond_options[controls.preconditioner_index].type);
        ImGui::EndDisabled();

        ImGui::BeginDisabled(!solver.uses_omega);
        ImGui::Checkbox("Auto Omega", &controls.auto_omega);
        if (controls.auto_omega)
            ImGui::Text("Omega: %.4f (auto)", ctx->omega);
        else
            ImGui::SliderFloat("Omega", &ctx->omega, 1.0f, 1.99f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::EndDisabled();
        if (!solver.uses_omega)
            ImGui::TextDisabled("%s does not use omega.", solver.label);
    }

    if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Inlet Velocity", &params.inlet_velocity, 0.0f, 5.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("Viscosity", &ctx->visc, 0.0001f, 0.1f, "%.5f",
                           ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("dt", &ctx->dt, 0.001f, 0.05f, "%.4f", ImGuiSliderFlags_AlwaysClamp);
    }

    if (controls.scenario_type == KARMAN_VORTEX &&
        ImGui::CollapsingHeader("Karman Obstacle", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Obstacle X", &params.obstacle_x, 1.0f, (float)ctx->x - 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("Obstacle Y", &params.obstacle_y, 1.0f, (float)ctx->y - 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        int radius = (int)params.obstacle_radius;
        if (ImGui::SliderInt("Obstacle Radius", &radius, 2, (int)(ctx->y / 4), "%d", ImGuiSliderFlags_AlwaysClamp))
            params.obstacle_radius = (size_t)radius;
        if (ImGui::Button("Rebuild Solids"))
            controls.request_restart = true;
    }

    ImGui::Separator();
    if (ImGui::Button("Reset"))
        controls.request_restart = true;

    ImGui::PopItemWidth();
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
    const FieldRect field = layout_field(ImGui::GetIO().DisplaySize, width, height);

    float range_min = 0.0f;
    float range_max = 1.0f;
    const char* quantity = nullptr;
    switch (controls.mode) {
        case ViewMode::SMOKE:
            quantity = "Smoke";
            renderer.update_field(ctx->smoke, width, height, range_min, range_max);
            break;
        case ViewMode::PRESSURE:
            quantity = "p [Pa]";
            scan_range(ctx->p, ctx->num_cells, range_min, range_max);
            renderer.update_field(ctx->p, width, height, range_min, range_max);
            break;
        case ViewMode::VELOCITY_MAGNITUDE:
        case ViewMode::FIELD_AND_VECTORS:
            quantity = "|u| [m/s]";
            fluid_velocity_magnitude(ctx, velocity_magnitudes.data());
            scan_range(velocity_magnitudes.data(), ctx->num_cells, range_min, range_max);
            renderer.update_field(velocity_magnitudes.data(), width, height, range_min, range_max);
            break;
        case ViewMode::VECTORS:
            break;
    }
    if (quantity != nullptr)
        draw_colorbar(field, range_min, range_max, quantity);

    if (controls.mode == ViewMode::VECTORS || controls.mode == ViewMode::FIELD_AND_VECTORS) {
        fluid_velocity_centered(ctx, u_centered.data(), v_centered.data());
        renderer.update_arrows(u_centered.data(), v_centered.data(), width, height,
                               controls.arrow_stride, controls.arrow_scale);
    }

    renderer.draw(controls.mode, field);
}
