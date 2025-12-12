import numpy as np
import math

def generate_naca6412_coordinates(n_points=400, chord_length=1.0, x_offset=-0.5, angle_of_attack=5.0):
    """
    Generate NACA 6412 airfoil coordinates with higher resolution and better point distribution
    """
    # NACA 6412 parameters
    m = 0.06  # 2% maximum camber
    p = 0.4   # 40% position of maximum camber
    t = 0.12  # 24% thickness
    
    # Enhanced cosine spacing with more points near leading/trailing edges
    beta = np.linspace(0, np.pi, n_points)
    x = chord_length * 0.5 * (1 - np.cos(beta))
    
    # Use standard cosine spacing (remove buggy clustering)
    # x is already correctly defined above
    
    # Thickness distribution with higher precision
    y_t = 5 * t * chord_length * (
        0.2969 * np.sqrt(x/chord_length) - 
        0.1260 * (x/chord_length) - 
        0.3516 * (x/chord_length)**2 + 
        0.2843 * (x/chord_length)**3 - 
        0.1015 * (x/chord_length)**4
    )
    
    # Mean camber line
    y_c = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    # Forward of maximum camber position
    idx_front = x <= p * chord_length
    y_c[idx_front] = m * x[idx_front] / (p**2) * (2 * p - x[idx_front]/chord_length)
    dyc_dx[idx_front] = 2 * m / (p**2) * (p - x[idx_front]/chord_length)
    
    # Aft of maximum camber position
    idx_aft = x > p * chord_length
    y_c[idx_aft] = m * (chord_length - x[idx_aft]) / ((1-p)**2) * (1 + x[idx_aft]/chord_length - 2 * p)
    dyc_dx[idx_aft] = 2 * m / ((1-p)**2) * (p - x[idx_aft]/chord_length)
    
    # Angle of camber line
    theta = np.arctan(dyc_dx)
    
    # Upper and lower surface coordinates
    x_upper = x - y_t * np.sin(theta)
    y_upper = y_c + y_t * np.cos(theta)
    x_lower = x + y_t * np.sin(theta)
    y_lower = y_c - y_t * np.cos(theta)
    
    # Apply angle of attack rotation
    alpha = math.radians(angle_of_attack)
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    
    # Rotate upper surface
    x_upper_rot = x_upper * cos_alpha - y_upper * sin_alpha + x_offset
    y_upper_rot = x_upper * sin_alpha + y_upper * cos_alpha
    
    # Rotate lower surface  
    x_lower_rot = x_lower * cos_alpha - y_lower * sin_alpha + x_offset
    y_lower_rot = x_lower * sin_alpha + y_lower * cos_alpha
    
    return x_upper_rot, y_upper_rot, x_lower_rot, y_lower_rot

def calculate_normal(v1, v2, v3):
    """Calculate unit normal vector for a triangle"""
    vec1 = np.array(v2) - np.array(v1)
    vec2 = np.array(v3) - np.array(v1)
    
    normal = np.cross(vec1, vec2)
    norm = np.linalg.norm(normal)
    if norm > 1e-12:
        normal = normal / norm
    else:
        normal = np.array([0.0, 0.0, 1.0])
    
    return normal

def write_stl_triangle(f, v1, v2, v3):
    """Write a triangle to STL file with calculated normal"""
    normal = calculate_normal(v1, v2, v3)
    
    f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
    f.write("    outer loop\n")
    f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
    f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
    f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
    f.write("    endloop\n")
    f.write("  endfacet\n")

def triangulate_polygon(vertices):
    """
    Better triangulation using ear clipping method for end caps
    """
    triangles = []
    n = len(vertices)
    
    if n < 3:
        return triangles
    
    # For convex airfoil end caps, simple fan triangulation from centroid works well
    # Calculate centroid
    cx = sum(v[0] for v in vertices) / n
    cy = sum(v[1] for v in vertices) / n
    cz = vertices[0][2]  # Same z-level for all vertices
    centroid = [cx, cy, cz]
    
    # Create triangles from centroid to each edge
    for i in range(n):
        i_next = (i + 1) % n
        triangles.append([centroid, vertices[i], vertices[i_next]])
    
    return triangles

def write_stl_file(filename, chord_length=1.0, span=0.1, n_points=200, x_offset=-0.5, angle_of_attack=5.0):
    """
    Generate high-resolution NACA6412 STL file
    """
    x_upper, y_upper, x_lower, y_lower = generate_naca6412_coordinates(n_points, chord_length, x_offset, angle_of_attack)
    
    # Create a closed airfoil profile
    x_profile = np.concatenate([x_upper[::-1], x_lower[1:]])
    y_profile = np.concatenate([y_upper[::-1], y_lower[1:]])
    
    n_profile = len(x_profile)
    
    # Create 3D vertices
    vertices_z0 = [[x_profile[i], y_profile[i], 0.0] for i in range(n_profile)]
    vertices_z1 = [[x_profile[i], y_profile[i], span] for i in range(n_profile)]
    
    triangle_count = 0
    
    with open(filename, 'w') as f:
        f.write("solid naca6412_hires\n")
        
        # Side surface triangles
        for i in range(n_profile):
            i_next = (i + 1) % n_profile
            
            v1 = vertices_z0[i]
            v2 = vertices_z1[i]
            v3 = vertices_z0[i_next]
            v4 = vertices_z1[i_next]
            
            # Two triangles per quadrilateral with correct winding
            write_stl_triangle(f, v1, v3, v2)
            triangle_count += 1
            write_stl_triangle(f, v2, v3, v4)
            triangle_count += 1
        
        # End cap triangulation with better method
        triangles_z0 = triangulate_polygon(vertices_z0)
        for tri in triangles_z0:
            # Outward normal (-z direction)
            write_stl_triangle(f, tri[0], tri[2], tri[1])
            triangle_count += 1
        
        triangles_z1 = triangulate_polygon(vertices_z1)
        for tri in triangles_z1:
            # Outward normal (+z direction)
            write_stl_triangle(f, tri[0], tri[1], tri[2])
            triangle_count += 1
        
        f.write("endsolid naca6412_hires\n")
    
    print(f"High-resolution STL file '{filename}' generated with {triangle_count} triangles")
    return triangle_count



# Generate high-resolution STL
if __name__ == "__main__":
    print("=" * 60)
    print("High-Resolution NACA 6412 STL Generator")
    print("=" * 60)
    
    # Parameters for smoother surface
    chord_length = 1.0
    span = 0.1
    n_points = 200  # Increased from 80 to 200
    x_offset = -0.5
    angle_of_attack = 5.0
    filename = "naca6412.stl"
    
    print(f"Generating with {n_points} points for smoother surface...")
    
    # Generate high-resolution STL
    triangle_count = write_stl_file(filename, chord_length, span, n_points, x_offset, angle_of_attack)
    
    print(f"Generated {triangle_count} triangles")
    print(f"File saved as: {filename}")
    

