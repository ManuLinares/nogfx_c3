#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct Vertex { vec4 position; vec4 color; };
layout(buffer_reference, scalar) readonly buffer Vertices { Vertex values[]; };
layout(set = 0, binding = 0, scalar) uniform Root { Vertices vertices; } root;

layout(location = 0) out vec4 color;

void main()
{
	Vertex vertex = root.vertices.values[gl_VertexIndex];
	gl_Position = vertex.position;
	color = vertex.color;
}
