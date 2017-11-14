// Copyright 2016 University of Minnesota
// 
// SHADER Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other materials
//    provided with the distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF MINNESOTA, DULUTH OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
// IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Adapted from r3dux (http://r3dux.org).

#ifndef SHADER_HPP
#define SHADER_HPP 1
#define DEBUG_ON 1
#include <fstream>
#include <sstream>
#include <unordered_map> // requires C++11

//
//	Shader utility class for managing vert/frag shaders.
//	Does not currently handle geometry shaders.
//
//	Example use:
//	Shader myshader;
//	< OpenGL context creation >
//	myshader.init_from_files( "myshader.vert", "myshader.frag" );
//	while( rendering ){
//		< OpenGL view stuff >
//		myshader.enable();
//		glUniform3f( myshader.uniform("color"), 1.f, 0.f, 0.f );
//		< Draw stuff >
//		myshader.disable();
//	}
//
namespace mcl {
	class Shader {
	public:
		Shader() : program_id(0) {}
		~Shader() { glDeleteProgram(program_id); }
		// Init the shader from files (must create OpenGL context first!)
		inline void init_from_files(std::string vertex_file, std::string frag_file);
		// Init the shader from strings (must create OpenGL context first!)
		inline void init_from_strings(std::string vertex_source, std::string frag_source) { init(vertex_source, frag_source); }
		// Be sure to initialize the shader before enabling it
		inline void enable();
		// Not really needed, but nice for readability
		inline void disable() { glUseProgram(0); }
		// Returns the bound location of a named attribute
		inline GLuint attribute(const std::string name);
		// Returns the bound location of a named uniform
		inline GLuint uniform(const std::string name);

		GLuint program_id;
	private:
		GLuint vertex_id;
		GLuint fragment_id;
		std::unordered_map<std::string, GLuint> attributes;
		std::unordered_map<std::string, GLuint> uniforms;
		// Initialize the shader, called by init_from_*
		inline void init(std::string vertex_source, std::string frag_source);
		// Compiles the shader, called by init
		inline GLuint compile(std::string shaderSource, GLenum type);
	}; // end of shader
	   //
	   //  Implementation
	   //
	GLuint Shader::compile(std::string source, GLenum type) {
		// Generate a shader id
		// Note: Shader id will be non-zero if successfully created.
		GLuint shaderId = glCreateShader(type);
		if (shaderId == 0) { throw std::runtime_error("\n**glCreateShader Error"); }
		// Attach the GLSL source code and compile the shader
		const char *shaderchar = source.c_str();
		glShaderSource(shaderId, 1, &shaderchar, NULL);
		glCompileShader(shaderId);
		// Check the compilation status and throw a runtime_error if shader compilation failed
		GLint shaderStatus;
		glGetShaderiv(shaderId, GL_COMPILE_STATUS, &shaderStatus);
		if (shaderStatus == GL_FALSE) {
			printf("shader failed to compile\n");
			if (DEBUG_ON) {
				GLint logMaxSize, logLength;
				glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &logMaxSize);
				printf("printing error message of %d bytes\n", logMaxSize);
				char* logMsg = new char[logMaxSize];
				glGetShaderInfoLog(shaderId, logMaxSize, &logLength, logMsg);
				printf("%d bytes retrieved\n", logLength);
				printf("error message: %s\n", logMsg);
				delete[] logMsg;
			}
			throw std::runtime_error("\n**glCompileShader Error");
		}
		return shaderId;
	}
	void Shader::init(std::string vertex_source, std::string frag_source) {
		// Create the resource
		program_id = glCreateProgram();
		if (program_id == 0) { throw std::runtime_error("\n**glCreateProgram Error"); }
		glUseProgram(program_id);
		// Compile the shaders and return their id values
		vertex_id = compile(vertex_source, GL_VERTEX_SHADER);
		fragment_id = compile(frag_source, GL_FRAGMENT_SHADER);
		// Attach and link the shader program
		glAttachShader(program_id, vertex_id);
		glAttachShader(program_id, fragment_id);
		glLinkProgram(program_id);
		// Once the shader program has the shaders attached and linked, the shaders are no longer required.
		// If the linking failed, then we're going to abort anyway so we still detach the shaders.
		glDetachShader(program_id, vertex_id);
		glDetachShader(program_id, fragment_id);
		// Check the program link status and throw a runtime_error if program linkage failed.
		GLint programLinkSuccess = GL_FALSE;
		glGetProgramiv(program_id, GL_LINK_STATUS, &programLinkSuccess);
		if (programLinkSuccess != GL_TRUE) { throw std::runtime_error("\n**Shader Error: Problem with link"); }
		// Check the validation status and throw a runtime_error if program validation failed.
		// Does NOT work with corearb headers???
		//  glValidateProgram(program_id);
		//  GLint programValidatationStatus;
		//  glGetProgramiv(program_id, GL_VALIDATE_STATUS, &programValidatationStatus);
		//  if( programValidatationStatus != GL_TRUE ){ throw std::runtime_error("\n**Shader Error: Problem with validation"); }
		glUseProgram(0);
	}
	void Shader::init_from_files(std::string vertex_file, std::string frag_file) {
		std::string vert_string, frag_string;
		// Load the vertex shader
		std::ifstream vert_in(vertex_file, std::ios::in | std::ios::binary);
		if (vert_in) { vert_string = (std::string((std::istreambuf_iterator<char>(vert_in)), std::istreambuf_iterator<char>())); }
		else { throw std::runtime_error("\n**Shader Error: failed to load \"" + vertex_file + "\""); }
		// Load the fragement shader
		std::ifstream frag_in(frag_file, std::ios::in | std::ios::binary);
		if (frag_in) { frag_string = (std::string((std::istreambuf_iterator<char>(frag_in)), std::istreambuf_iterator<char>())); }
		else { throw std::runtime_error("\n**Shader Error: failed to load \"" + frag_file + "\""); }
		init(vert_string, frag_string);
	}
	void Shader::enable() {
		if (program_id != 0) { glUseProgram(program_id); }
		else { throw std::runtime_error("\n**Shader Error: Can't enable, not initialized"); }
	}
	GLuint Shader::attribute(const std::string name) {
		// Add the attribute to the map table if it doesn't already exist
		if (attributes.count(name) == 0) {
			attributes[name] = glGetAttribLocation(program_id, name.c_str());
			if (attributes[name] == -1) { throw std::runtime_error("\n**Shader Error: bad attribute (" + name + ")"); }
		}
		return attributes[name];
	}
	GLuint Shader::uniform(const std::string name) {
		// Add the uniform to the map table if it doesn't already exist
		if (uniforms.count(name) == 0) {
			uniforms[name] = glGetUniformLocation(program_id, name.c_str());
			if (uniforms[name] == -1) { throw std::runtime_error("\n**Shader Error: bad uniform (" + name + ")"); }
		}
		return uniforms[name];
	}
} // end namespace mcl
#endif