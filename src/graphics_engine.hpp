#pragma once

struct GLFWwindow;

enum class visual_mode
{
    smoke,
    pressure,
    velocity_magnitude,
    velocity_vectors_only,
    field_plus_vectors,
};

class graphics_engine
{
public:
    graphics_engine(int width, int height, const char* title);
    ~graphics_engine();

    graphics_engine(const graphics_engine&) = delete;
    graphics_engine& operator=(const graphics_engine&) = delete;

    [[nodiscard]] bool should_close() const;
    [[nodiscard]] GLFWwindow* window() const { return glfw_window; }

    void update_field(const float* data, int width, int height, float range_min, float range_max);
    void update_arrows(const float* u, const float* v, int width, int height, int stride, float scale);
    void begin_ui() const;
    void draw(visual_mode mode) const;

private:
    [[nodiscard]] static unsigned int compile_shader(const char* vert_path, const char* frag_path);
    void create_quad();
    void create_arrow_buffer();
    void init_imgui() const;
    void shutdown_imgui() const;
    void create_field_texture(int width, int height);

    GLFWwindow* glfw_window = nullptr;

    unsigned int field_shader = 0;
    unsigned int quad_vao = 0;
    unsigned int quad_vbo = 0;
    unsigned int quad_ibo = 0;
    unsigned int field_texture = 0;
    int field_width = 0;
    int field_height = 0;
    float range_min = 0.0f;
    float range_max = 1.0f;

    unsigned int arrow_shader = 0;
    unsigned int arrow_vao = 0;
    unsigned int arrow_vbo = 0;
    int arrow_vert_count = 0;
    int arrow_vbo_capacity = 0;
};
