import ctypes

from OpenGL.GL import *
import numpy as np


def create_vertex_array(vertices):
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    
    # Pass postition data
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)

    # Enable attribute
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    
    glBindVertexArray(0)

    return vao, vbo


def create_shader_part(shader_type, file):
    try:
        with open(file) as f:
            shader_source = f.read()
        
        shader = glCreateShader(shader_type)

        if shader == 0:
            print("Couldn't create openGL shader object")
            return None
        
        glShaderSource(shader, shader_source)
        glCompileShader(shader)

        success = glGetShaderiv(shader, GL_COMPILE_STATUS, None)
        
        if not success:
            info_log = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            
            print(f"Failed to compile shader {file}")
            print(f"Info log: {info_log}")

            return None
        
        return shader

    except FileNotFoundError:
        print(f"Shader source {file} not found")
        return None
    except OSError:
        print(f"Couldn't open {file}")
        return None


def create_shader_program(vert_shader_source, frag_shader_source):
    vert_shader = create_shader_part(GL_VERTEX_SHADER, vert_shader_source)
    frag_shader = create_shader_part(GL_FRAGMENT_SHADER, frag_shader_source)

    if vert_shader is None or frag_shader is None:
        return None
    
    program = glCreateProgram()

    if program == 0:
        print("Couldn't create openGL shader program object")
        return None

    glAttachShader(program, vert_shader)
    glAttachShader(program, frag_shader)

    glLinkProgram(program)

    # Delete shaders
    glDeleteShader(vert_shader)
    glDeleteShader(frag_shader)

    if glGetProgramiv(program, GL_LINK_STATUS, None) == GL_FALSE:
        print("Couldn't link program")
        return None
    
    return program


def create_texture():
    texture = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    return texture


# Based on https://pbr-book.org/3ed-2018/Geometry_and_Transformations/Vectors
def coordinate_system(normal):
    """
    Takes the normal vector and returns two tangent vectors that form an orthogonal system with it
    """
    if abs(normal[0]) > abs(normal[1]):
        tangent1 = np.array([-normal[2], 0, normal[0]]) / np.sqrt(normal[0] ** 2 + normal[2] ** 2)
    else:
        tangent1 = np.array([0, normal[2], -normal[1]]) / np.sqrt(normal[1] ** 2 + normal[2] ** 2)
    
    tangent2 = np.cross(normal, tangent1)
    
    return tangent1, tangent2


class ScreenTexture:
    def __init__(self, resolution, vert_shader_source, frag_shader_source):
        self.resolution = resolution

        self.vao, self.vbo = self.setup_verts()
        self.program = create_shader_program(vert_shader_source, frag_shader_source)
        self.texture = create_texture()

        if self.vao is None or self.program is None or self.texture is None:
            exit(1)

    def setup_verts(self):
        # Used both as vertices' positions and texture coordinates
        verts = np.array(
            [
                [1, 1], [0, 1], [0, 0],
                [0, 0], [1, 0], [1, 1]
            ], dtype=np.float32
        )

        return create_vertex_array(verts)

    def update_texture(self, tex_data):
        self.tex_data = tex_data

        glBindTexture(GL_TEXTURE_2D, self.texture)
        w, h = self.resolution
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, tex_data)

    def render(self):
        glBindVertexArray(self.vao)
        
        glUseProgram(self.program)

        w, h = self.resolution
        glUniform2i(glGetUniformLocation(self.program, "res"), w, h)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(glGetUniformLocation(self.program, "tex"), 0)

        glDrawArrays(GL_TRIANGLES, 0, 6)

        glUseProgram(0)
        glBindVertexArray(0)


class BrushRenderer:
    def __init__(self, camera, brush, vert_shader_source, frag_shader_source):
        self.camera = camera
        self.brush = brush
        self.vao, self.vbo = self.setup_verts()
        self.program = create_shader_program(vert_shader_source, frag_shader_source)

        if self.vao is None or self.program is None:
            exit(1)

    def setup_verts(self):
        verts = np.array(
            [
                [ 1,  1], [ 1, -1], [-1, -1],
                [-1, -1], [-1,  1], [ 1,  1]
            ], dtype=np.float32
        )

        return create_vertex_array(verts)
    
    def render(self):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glBindVertexArray(self.vao)
        
        glUseProgram(self.program)
        
        glUniform1f(
            glGetUniformLocation(self.program, 'radius'),
            self.brush.radius
        )
        
        resolution = self.camera.resolution
        x_half_extent = np.tan(self.camera.fov / 2)
        x_half_extent -= x_half_extent / resolution[0]
        y_half_extent = resolution[1] / resolution[0] * x_half_extent
        glUniform2f(glGetUniformLocation(self.program, 'f'), 1 / x_half_extent,  1 / y_half_extent)
        
        inter_point = self.brush.inter_point.squeeze().cpu().numpy()
        glUniform3fv(glGetUniformLocation(self.program, 'inter'), 1, inter_point)
        
        cam_pos = self.camera.position.cpu().numpy()
        glUniform3fv(glGetUniformLocation(self.program, 'cam_pos'), 1, cam_pos)

        view_mat = np.linalg.inv(self.camera.orientation_matrix.cpu().numpy())
        glUniformMatrix3fv(glGetUniformLocation(self.program, 'view'), 1, True, view_mat)
        
        t1, t2 = coordinate_system(self.brush.inter_normal.squeeze().cpu().numpy())
        glUniform3fv(glGetUniformLocation(self.program, 't1'), 1, t1)
        glUniform3fv(glGetUniformLocation(self.program, 't2'), 1, t2)

        glDrawArrays(GL_TRIANGLES, 0, 6)

        glUseProgram(0)
        glBindVertexArray(0)
