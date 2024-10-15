#version 330

in vec3 frag_position;
in vec2 frag_texcoord;
in vec3 frag_normal;

uniform vec3 Kd;
uniform vec3 Ks;
uniform vec3 Ka;
uniform float ns;

uniform vec4 light_position_1;
uniform vec4 light_position_2;
uniform vec3 view_position;

// uniform sampler2D sampler_tex;
uniform samplerCube cubemap;

out vec4 outColor;

void main()
{
    vec3 light_position_1 = vec3(light_position_1);
    vec3 light_position_2 = vec3(light_position_2);

    vec3 Ld_1 = vec3(0, 0.6, 1);
    vec3 Ls_1 = vec3(0, 0.6, 1);
    vec3 La_1 = vec3(0, 0.6, 1);

    vec3 Ld_2 = vec3(1, 0.6, 0);
    vec3 Ls_2 = vec3(1, 0.6, 0);
    vec3 La_2 = vec3(1, 0.6, 0);

    // ambient
    vec3 ambient_1 = Ka * La_1;

    vec3 ambient_2 = Ka * La_2;


    // componente difuso

    float constantAttenuation = 0.001;
    float linearAttenuation = 0.01;
    float quadraticAttenuation = 0.001; 

    vec3 normal = normalize(frag_normal);

    vec3 to_light_1 = light_position_1 - frag_position;
    vec3 light_dir_1 = normalize(to_light_1);
    float diff_1 = max(dot(normal, light_dir_1), 0.0);
    vec3 diffuse_1 = Kd * Ld_1 * diff_1;

    vec3 to_light_2 = light_position_2 - frag_position;
    vec3 light_dir_2 = normalize(to_light_2);
    float diff_2 = max(dot(normal, light_dir_2), 0.0);
    vec3 diffuse_2 = Kd * Ld_2 * diff_2;


    // componente especular
    vec3 view_dir = normalize(view_position - frag_position);

    vec3 reflect_dir_1 = reflect(-light_dir_1, normal);
    float spec_1 = pow(max(dot(view_dir, reflect_dir_1), 0.0), ns);
    vec3 specular_1 = Ks * Ls_1 * spec_1;

    vec3 reflect_dir_2 = reflect(-light_dir_2, normal);
    float spec_2 = pow(max(dot(view_dir, reflect_dir_2), 0.0), ns);
    vec3 specular_2 = Ks * Ls_2 * spec_2;


    // atenuacion
    float dist_to_light_1 = length(to_light_1);
    float attenuation_1 = constantAttenuation 
        + linearAttenuation * dist_to_light_1
        + quadraticAttenuation * dist_to_light_1 * dist_to_light_1;

    float dist_to_light_2 = length(to_light_2);
    float attenuation_2 = constantAttenuation
        + linearAttenuation * dist_to_light_2
        + quadraticAttenuation * dist_to_light_2 * dist_to_light_2;
    

    vec4 frag_og_color = texture(cubemap, frag_position);

    vec3 result_1 = (ambient_1 * ((diffuse_1 + specular_1) / attenuation_1)) * frag_og_color.rgb;
    vec3 result_2 = (ambient_2 * ((diffuse_2 + specular_2) / attenuation_2)) * frag_og_color.rgb;

    vec3 result = result_1 + result_2;
    
    outColor = vec4(1, 1, 1, 1.0);
}