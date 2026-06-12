#version 300 es

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 mvp;
uniform mat4 matModel;

out vec3 fragPosition;
out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;

void main()
{
    fragPosition = vec3(matModel * vec4(vertexPosition, 1.0f));
    fragTexCoord = vertexTexCoord;
    // Default vertex colors from OBJ are (0,0,0,1) which causes pitch black meshes. 
    // We ignore vertexColor and rely on material colDiffuse (tint)
    fragColor = vec4(1.0);
    fragNormal = normalize(mat3(matModel) * vertexNormal);

    gl_Position = mvp * vec4(vertexPosition, 1.0f);
}
