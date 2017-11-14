#version 330 core

precision mediump float;

in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;

out vec3 vnormal;
out vec3 vcolor;

uniform vec3 viewdir;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	vec4 pos = projection * view * vec4(in_position,1.0);
	vnormal = in_normal;
	vcolor = in_color;
	gl_Position = pos;
}
