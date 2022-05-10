import os
import ctypes

from OpenGL.GL import *
import numpy as np


class ScreenTexture:
    def __init__(self, resolution, shader_dir):
        self.resolution = resolution

        self.vao = self.vbo = self.program = self.texture = 0
        self.vao, self.vbo = self.setup_verts()
        self.program = self.create_program(shader_dir)
        self.texture = self.create_texture()
        self.tex_data = None

    def setup_verts(self):
        # Used both as vertices' positions and texture coordinates
        verts = np.array([
            [1, 1], [1, 0], [0, 0],
            [0, 0], [0, 1], [1, 1]
        ], dtype=np.float32)

        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)

        glBindVertexArray(vao)
        
        # Pass postition data
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts, GL_STATIC_DRAW)

        # Enable attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        glBindVertexArray(0)

        return vao, vbo
    
    def compile_shader(self, shader_type, file):
        try:
            with open(file) as f:
                shader_source = f.read()
            
            shader = glCreateShader(shader_type)

            if shader == 0:
                print("Couldn't create shader")
                self.cleanup_exit()
            
            glShaderSource(shader, shader_source)
            glCompileShader(shader)

            success = glGetShaderiv(shader, GL_COMPILE_STATUS, None)
            
            if not success:
                info_log = glGetShaderInfoLog(shader)
                glDeleteShader(shader)
                
                print(f"Failed to compile shader {file}")
                print(f"Info log: {info_log}")

                self.cleanup_exit()
            
            return shader

        except FileNotFoundError:
            print(f"Shader source {file} not found")
            self.cleanup_exit()
        except OSError:
            print(f"Couldn't open {file}")
            self.cleanup_exit()
    
    def create_program(self, shader_dir):
        program = glCreateProgram()

        if program == 0:
            print("Couldn't create shader program")
            self.cleanup_exit()
        
        vert_shader = self.compile_shader(GL_VERTEX_SHADER, os.path.join(shader_dir, "vert.vs"))
        frag_shader = self.compile_shader(GL_FRAGMENT_SHADER, os.path.join(shader_dir, "frag.fs"))
        
        glAttachShader(program, vert_shader)
        glAttachShader(program, frag_shader)

        glLinkProgram(program)

        # Delete shaders
        glDeleteShader(vert_shader)
        glDeleteShader(frag_shader)

        if glGetProgramiv(program, GL_LINK_STATUS, None) == GL_FALSE:
            print("Couldn't link program")
            self.cleanup_exit()
        
        return program

    def create_texture(self):
        texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        return texture

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

    def cleanup(self):
        if self.vao != 0:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo != 0:
            glDeleteBuffers(1, [self.vbo])
        if self.program != 0:
            glDeleteProgram(self.program)
        if self.texture != 0:
            glDeleteTextures(1, [self.texture])
    
    def cleanup_exit(self, exit_code=1):
        self.cleanup()
        exit(exit_code)
