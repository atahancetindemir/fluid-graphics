#version 460 core

in vec2 v_tex_coord;
out vec4 frag_color;

uniform sampler2D u_field;
uniform float u_range_min;
uniform float u_range_max;

// Turbo colormap polynomial approximation (Google, 2019).
vec3 turbo(float value)
{
    value = clamp(value, 0.0, 1.0);
    float value2 = value * value;
    float value3 = value2 * value;
    float value4 = value3 * value;
    float value5 = value4 * value;

    float r = 0.13572138 + 4.61539260 * value - 42.66032258 * value2 + 132.13108234 * value3 - 152.94239396 * value4 + 59.28637943 * value5;
    float g = 0.09140261 + 2.19418839 * value + 4.84296658 * value2 - 14.18503333 * value3 + 4.27729857 * value4 + 2.82956604 * value5;
    float b = 0.10667330 + 12.64194608 * value - 60.58204836 * value2 + 110.36276771 * value3 - 89.90310912 * value4 + 27.34824973 * value5;
    return clamp(vec3(r, g, b), 0.0, 1.0);
}

void main()
{
    // Swap UV: simulation array layout is smoke[i*y+j], so OpenGL's S axis
    // (row-fast) corresponds to sim j, T to sim i. Sampling with .yx makes
    // screen-X map to sim i (the inlet/wind axis) and screen-Y to sim j.
    float raw = texture(u_field, v_tex_coord.yx).r;

    const float min_range = 1e-6;
    float range = u_range_max - u_range_min;
    float normalized = range > min_range ? (raw - u_range_min) / range : 0.0;

    frag_color = vec4(turbo(normalized), 1.0);
}
