// Copyright 2016 University of Minnesota
// 
// TRIMESH Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
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

#ifndef TRIMESH_HPP
#define TRIMESH_HPP 1

#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

//
//	Vector Class
//	Not complete, only has functions needed for this sample.
//
template <size_t D, class T> class Vec {
public:
	// Constructors
	Vec(){ data[0]=T(0); data[1]=T(0); data[2]=T(0); }
	Vec(T x_, T y_, T z_){ data[0]=x_; data[1]=y_; data[2]=z_; }

	// Data
	T data[3];

	// Functions
	T operator[](int i) const { return data[i]; }
	T& operator[](int i){ return data[i]; }
	Vec<D,T> &operator += (const Vec<D,T> &x) {
		for(size_t i=0; i<D; ++i){ data[i] += x[i]; }
		return *this;
	}
	double len2() const { return this->dot(*this); } // squared length
	double len() const { return sqrt(len2()); } // length
	T dot(const Vec<D,T> &v) const {
		T r(0);
		for(size_t i=0; i<D; ++i){ r += v[i] * data[i]; }
		return r;
	}
	Vec<3,T> cross(const Vec<3,T> &v) const {
		assert(D == 3); // only defined for 3 dims
		return Vec<3,T>(data[1]*v[2] - data[2]*v[1], data[2]*v[0] - data[0]*v[2], data[0]*v[1] - data[1]*v[0]);
	}
	void normalize() {
		double l = len(); if( l<=0.0 ){ return; }
		for(size_t i=0; i<D; ++i){ data[i] = data[i] / l; }
	}

	// Helpers
	T x() { return data[0] };
	T y() { return data[1] };
	T z() { return data[2] };
};

template <size_t D, class T> static inline const Vec<D,T> operator-(const Vec<D,T> &v1, const Vec<D,T> &v2){
	Vec<D,T> r;
	for(size_t i=0; i<D; ++i){ r[i] = v1[i]-v2[i]; }
	return r;
}

template <size_t D, class T> static inline const Vec<D,T> operator*(const Vec<D,T> &v, const T &x){
	Vec<D,T> r;
	for (size_t i=0; i<D; ++i){ r[i] = v[i]*x; }
	return r;
}

typedef Vec<3,float> Vec3f;
typedef Vec<3, int> Vec3i;
typedef Vec<2, float> Vec2f;


//
//	Triangle Mesh Class
//
class TriMesh {
public:
	std::vector<Vec3f> vertices;
	std::vector<Vec3f> normals;
	std::vector<Vec3f> colors;
	std::vector<Vec3i> faces;

	// Compute normals if not loaded from obj
	// or if recompute is set to true.
	void need_normals( bool recompute=false );

	// Sets a default vertex colors if
	// they haven't been set.
	void need_colors( Vec3f default_color = Vec3f(0.4,0.4,0.4) );

	// Loads an OBJ file
	bool load_obj( std::string file );

	// Prints details about the mesh
	void print_details();
};



//
//	Implementation
//

void TriMesh::print_details(){
	std::cout << "Vertices: " << vertices.size() << std::endl;
	std::cout << "Normals: " << normals.size() << std::endl;
	std::cout << "Colors: " << colors.size() << std::endl;
	std::cout << "Faces: " << faces.size() << std::endl;
}


void TriMesh::need_normals( bool recompute ){
	if( vertices.size() == normals.size() && !recompute ){ return; }
	if( normals.size() != vertices.size() ){ normals.resize( vertices.size() ); }
	std::cout << "Computing TriMesh normals" << std::endl;
	const int nv = normals.size();
	for( int i = 0; i < nv; ++i ){ normals[i][0] = 0.f; normals[i][1] = 0.f; normals[i][2] = 0.f; }
	int nf = faces.size();
	for( int f = 0; f < nf; ++f ){
		Vec3i face = faces[f];
		const Vec3f &p0 = vertices[ face[0] ];
		const Vec3f &p1 = vertices[ face[1] ];
		const Vec3f &p2 = vertices[ face[2] ];
		Vec3f a = p0-p1,  b = p1-p2, c = p2-p0;
		float l2a = a.len2(), l2b = b.len2(), l2c = c.len2();
		if (!l2a || !l2b || !l2c){ continue; } // check for zeros or nans
		Vec3f facenormal = a.cross( b );
		normals[faces[f][0]] += facenormal * (1.0f / (l2a * l2c));
		normals[faces[f][1]] += facenormal * (1.0f / (l2b * l2a));
		normals[faces[f][2]] += facenormal * (1.0f / (l2c * l2b));
	}
	for (int i = 0; i < nv; i++){ normals[i].normalize(); }
} // end need normals


void TriMesh::need_colors( Vec3f default_color ){
	if( vertices.size() == colors.size() ){ return; }
	else{ colors.resize( vertices.size(), default_color ); }
} // end need colors

// Function to split a string into multiple strings, seperated by delimeter
static void split_str( char delim, const std::string &str, std::vector<std::string> *result ){
	std::stringstream ss(str); std::string s;
	while( std::getline(ss, s, delim) ){ result->push_back(s); }
}

bool TriMesh::load_obj( std::string file ){

	std::cout << "\nLoading " << file << std::endl;

	//	README:
	//
	//	The problem with standard obj files and opengl is that
	//	there isn't a good way to make triangles with different indices
	//	for vertices/normals. At least, not any way that I'm aware of.
	//	So for now, we'll do the inefficient (but robust) way:
	//	redundant vertices/normals.
	//

	std::vector<Vec3f> temp_normals;
	std::vector<Vec3f> temp_verts;
	std::vector<Vec3f> temp_colors;

	//
	//	First loop, make buffers
	//
	std::ifstream infile( file.c_str() );
	if( infile.is_open() ){

		std::string line;
		while( std::getline( infile, line ) ){

			std::stringstream ss(line);
			std::string tok; ss >> tok;

			// Vertex
			if( tok == "v" ){

				// First three location
				float x, y, z; ss >> x >> y >> z;
				temp_verts.push_back( Vec3f(x,y,z) );

				// Next three colors
				float cx, cy, cz;
				if( ss >> cx >> cy >> cz ){
					temp_colors.push_back( Vec3f(cx,cy,cz) );
				} else {
					temp_colors.push_back( Vec3f(0.3f,0.3f,0.3f) );
				}
			}

			// Normal
			if( tok == "vn" ){
				float x, y, z; ss >> x >> y >> z;
				temp_normals.push_back( Vec3f(x,y,z) );
			}

		} // end loop lines

	} // end load obj
	else { std::cerr << "\n**TriMesh::load_obj Error: Could not open file " << file << std::endl; return false; }

	//
	//	Second loop, make faces
	//
	std::ifstream infile2( file.c_str() );
	if( infile2.is_open() ){

		std::string line;
		while( std::getline( infile2, line ) ){

			std::stringstream ss(line);
			std::string tok; ss >> tok;

			// Face
			if( tok == "f" ){

				Vec3i face;
				// Get the three vertices
				for( size_t i=0; i<3; ++i ){

					std::string f_str; ss >> f_str;
					std::vector<std::string> f_vals;
					split_str( '/', f_str, &f_vals );
					assert(f_vals.size()>0);

					face[i] = vertices.size();
					int v_idx = std::stoi(f_vals[0])-1;
					vertices.push_back( temp_verts[v_idx] );
					colors.push_back( temp_colors[v_idx] );

					// Check for normal
					if( f_vals.size()>2 ){
						int n_idx = std::stoi(f_vals[2])-1;
						normals.push_back( temp_normals[n_idx] );
					}
				}

				faces.push_back(face);

				// If it's a quad, make another triangle
				std::string last_vert="";
				if( ss >> last_vert ){
					Vec3i face2;
					face2[0] = face[0];
					face2[1] = face[2];

					std::vector<std::string> f_vals;
					split_str( '/', last_vert, &f_vals );
					assert(f_vals.size()>0);

					int v_idx = std::stoi(f_vals[0])-1;
					vertices.push_back( temp_verts[v_idx] );
					colors.push_back( temp_colors[v_idx] );
					face2[2] = vertices.size();

					// Check for normal
					if( f_vals.size()>2 ){
						int n_idx = std::stoi(f_vals[2])-1;
						normals.push_back( temp_normals[n_idx] );
					}

					faces.push_back(face2);
				}

			} // end parse face

		} // end loop lines

	} // end load obj

	// Make sure we have normals
	if( !normals.size() ){
		std::cout << "**Warning: normals not loaded so we'll compute them instead." << std::endl;
		need_normals();
	}

	return true;

} // end load obj


#endif

