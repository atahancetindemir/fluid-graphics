#ifndef RENDERER_HPP
#define RENDERER_HPP

struct GLFWwindow;

enum class ViewMode {
    SMOKE,
    PRESSURE,
    VELOCITY_MAGNITUDE,
    VECTORS,
    FIELD_AND_VECTORS,
};

class Renderer {
public:
    Renderer(int width, int height, const char* title);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    [[nodiscard]] bool should_close() const;
    [[nodiscard]] GLFWwindow* window() const { return glfw_window; }

    void update_field(const float* data, int width, int height, float range_min, float range_max);
    // Takes cell-centered velocity (num_cells floats each), not the staggered field.
    void update_arrows(const float* u_centered, const float* v_centered, int width, int height, int stride, float scale);
    void begin_ui() const;
    void draw(ViewMode mode) const;

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

#endif // RENDERER_HPP
