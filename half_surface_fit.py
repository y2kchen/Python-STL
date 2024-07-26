#goal: To divide the mesh into two halves and fit each half with a surface.

import numpy as np
from stl import mesh
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import stl
from stl.stl import AUTOMATIC
import pymesh

def analyze_stl(file_path):
    your_mesh = mesh.Mesh.from_file(file_path)
    # find the mean of the x coordinates
    x_coords = your_mesh.vectors[:, :, 0].flatten()
    x_mean = np.mean(x_coords)
    # subtract x_mean from x coordinates of the mesh
    your_mesh.x -= x_mean
    #find the mean of the y coordinates
    y_coords = your_mesh.vectors[:, :, 1].flatten()
    y_mean = np.mean(y_coords)
    # subtract x_mean from x coordinates of the mesh
    your_mesh.y -= y_mean
    # find the min and max of the z coordinates
    z_coords = your_mesh.vectors[:, :, 2].flatten()
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    z_range = z_max - z_min
    segment_size = z_range / 10
    segments = [(z_min + i * segment_size, z_min + (i + 1) * segment_size) for i in range(10)]
    return your_mesh, z_min, z_max, segment_size, segments

def remove_duplicate_vertices(your_mesh):
    #remove duplicate faces
    your_mesh.remove_duplicate_faces()
    
    vertices = your_mesh.vectors.reshape(-1, 3)
    unique_vertices, indices = np.unique(vertices, axis=0, return_inverse=True)
    unique_vectors = unique_vertices[indices].reshape(your_mesh.vectors.shape)
    return unique_vectors

def normalize_face(face):
    # Sort the vertices of the face to ensure consistent ordering
    face = np.array(face)
    face.sort(axis=0)
    return face

def remove_duplicate_faces(your_mesh):
    # Flatten the faces to a 2D array and sort the vertices in each face
    #faces = np.sort(your_mesh.vectors.reshape(-1, 3, 3), axis=1)
    # Use np.unique to find unique faces
    #unique_faces, indices = np.unique(faces, axis=0, return_index=True)
    # Reconstruct the unique faces
    #unique_vectors = faces[indices]
    # old: unique_vectors = your_mesh.vectors[indices // 3]
    #return unique_vectors
    # Normalize and sort faces to account for duplicate faces regardless of vertex order
    normalized_faces = np.array([normalize_face(face) for face in your_mesh.vectors])
    unique_faces, indices = np.unique(normalized_faces, axis=0, return_index=True)
    unique_vectors = your_mesh.vectors[indices]
    return unique_vectors

def divide_into_quadrants(mesh_segment):
    quadrants = [[] for _ in range(8)]
    for triangle in mesh_segment:
        centroid = np.mean(triangle, axis=0)
        x, y, z = centroid
        if x >= 0 and y >= 0:
            if x >= y:
                quadrants[0].append(triangle)
            else:
                quadrants[1].append(triangle)
        elif x < 0 and y >= 0:
            if abs(x) >= y:
                quadrants[2].append(triangle)
            else:
                quadrants[3].append(triangle)
        elif x < 0 and y < 0:
            if abs(x) >= abs(y):
                quadrants[4].append(triangle)
            else:
                quadrants[5].append(triangle)
        elif x >= 0 and y < 0:
            if x >= abs(y):
                quadrants[6].append(triangle)
            else:
                quadrants[7].append(triangle)
    return quadrants

def fit_surface_to_quadrant(quadrant):
    def poly_func(X, a, b, c, d, e, f):
        x, y = X
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
    
    x_coords = []
    y_coords = []
    z_coords = []

    for triangle in quadrant:
        for vertex in triangle:
            x_coords.append(vertex[0])
            y_coords.append(vertex[1])
            z_coords.append(vertex[2])

    if len(x_coords) < 6:
        print(f"  Not enough points to fit a surface: {len(x_coords)} points")
        return None
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = np.array(z_coords)
    
    popt, _ = curve_fit(poly_func, (x_coords, y_coords), z_coords)
    return popt, x_coords, y_coords, z_coords

def plot_quadrants(quadrants, fitted_surfaces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']

    # Plot original points in each quadrant
    for i, quadrant in enumerate(quadrants):
        x_coords = []
        y_coords = []
        z_coords = []
        for triangle in quadrant:
            for vertex in triangle:
                x_coords.append(vertex[0])
                y_coords.append(vertex[1])
                z_coords.append(vertex[2])
        ax.scatter(x_coords, y_coords, z_coords, color=colors[i], label=f'Quadrant {i+1}')

    # Plot fitted surfaces
    for i, (popt, x_coords, y_coords, z_coords) in enumerate(fitted_surfaces):
        if popt is not None:
            x_range = np.linspace(min(x_coords), max(x_coords), 30)
            y_range = np.linspace(min(y_coords), max(y_coords), 30)
            x_mesh, y_mesh = np.meshgrid(x_range, y_range)
            z_mesh = (popt[0]*x_mesh**2 + popt[1]*y_mesh**2 + popt[2]*x_mesh*y_mesh +
                      popt[3]*x_mesh + popt[4]*y_mesh + popt[5])
            ax.plot_surface(x_mesh, y_mesh, z_mesh, color=colors[i], alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #plt.legend()
    #plt.show()

#plot the two halves of the mesh and the surfaces fitted to each half
def plot_halves(half1, half2, fitted_surfaces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']

    # Plot original points in each half
    #for i, half in enumerate([half1, half2]):
    for i, half in enumerate([half1]):
        x_coords = []
        y_coords = []
        z_coords = []
        for triangle in half:
            for vertex in triangle:
                x_coords.append(vertex[0])
                y_coords.append(vertex[1])
                z_coords.append(vertex[2])
        ax.scatter(x_coords, y_coords, z_coords, color=colors[i], label=f'Half {i+1}')

    # Plot fitted surfaces
    # for i, (popt, x_coords, y_coords, z_coords) in enumerate(fitted_surfaces):
    #     if popt is not None:
    #         x_range = np.linspace(min(x_coords), max(x_coords), 30)
    #         y_range = np.linspace(min(y_coords), max(y_coords), 30)
    #         x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    #         z_mesh = (popt[0]*x_mesh**2 + popt[1]*y_mesh**2 + popt[2]*x_mesh*y_mesh +
    #                   popt[3]*x_mesh + popt[4]*y_mesh + popt[5])
    #         ax.plot_surface(x_mesh, y_mesh, z_mesh, color=colors[i], alpha=0.5)

    #modify the limits of x-axis of the plot
    ax.set_xlim([-15, 15])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

def divide_mesh_by_half(your_mesh):
    #divide the mesh into two halves, one half is x>=0 and the other half is x<0
    #x-axis is along the semi-minoraxis in top view.
    half1 = []
    half2 = []
    for triangle in your_mesh.vectors:
        centroid = np.mean(triangle, axis=0)
        x, _, _ = centroid
        if x >= 0:
            half1.append(triangle)
        else:
            half2.append(triangle)
    #fit each half with a surface
    fitted_surfaces = []
    popt1, x_coords1, y_coords1, z_coords1 = fit_surface_to_quadrant(half1)
    if popt1 is not None:
        with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output_half_mesh.txt', 'a') as f:
            f.write(f'constraint positive_x nonpositive   \n')
            f.write(f'formula: z = {popt1[0]}* x^2 + {popt1[1]}* y^2 + {popt1[2]}* x * y + {popt1[3]}* x + {popt1[4]}* y + {popt1[5]}  \n')
            f.write(f'\n')
        print(f"    Surface fit coefficients: {popt1}")
        fitted_surfaces.append((popt1, x_coords1, y_coords1, z_coords1))
    else:
        fitted_surfaces.append((None, [], [], []))
    popt2, x_coords2, y_coords2, z_coords2 = fit_surface_to_quadrant(half2)
    if popt2 is not None:
        with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output_half_mesh.txt', 'a') as f:
            f.write(f'constraint negative_x nonpositive   \n')
            f.write(f'formula: z = {popt2[0]}* x^2 + {popt2[1]}* y^2 + {popt2[2]}* x * y + {popt2[3]}* x + {popt2[4]}* y + {popt2[5]}  \n')
            f.write(f'\n')
        print(f"    Surface fit coefficients: {popt2}")
        fitted_surfaces.append((popt2, x_coords2, y_coords2, z_coords2))
    else:
        fitted_surfaces.append((None, [], [], []))
    plot_halves(half1, half2, fitted_surfaces)

def divide_mesh_by_segments_and_quadrants(your_mesh, segments):
    for i, (z_start, z_end) in enumerate(segments):
        segment_triangles = []
        for triangle in your_mesh.vectors:
            if np.any((triangle[:, 2] >= z_start) & (triangle[:, 2] < z_end)):
                segment_triangles.append(triangle)
        quadrants = divide_into_quadrants(segment_triangles)
        #identify the absolute minimal x and y coordinates of the segment
        if i<9:
            min_abs_x = float('inf')
            min_abs_y = float('inf')
            for triangle in segment_triangles:
                for point in triangle:
                    x, y, _ = point
                    #find the minimal y in a narrow range of x
                    if -0.2 <= x <= 0.2:
                        min_abs_y = min(min_abs_y, abs(y))
                    #find the minimal x in a narrow range of y
                    if -0.2 <= y <= 0.2:
                        min_abs_x = min(min_abs_x, abs(x))
        #set up the rough constrains for the segment
        with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output1.txt', 'a') as g:
            if i < 9:
                g.write(f'constraint {100+i+1} nonnegative // seg_{i+1}  \n')
                g.write(f'formula: x^2/{min_abs_x}^2+y^2/{min_abs_y}^2 = 1  \n')
                g.write(f'\n')
            if i == 9:
                g.write(f'constraint {100+i+1} nonpositive // seg_{i+1}  \n')
                g.write(f'formula: x^2/{min_abs_x}^2+y^2/{min_abs_y}^2 + (z-({z_min + i * segment_size}))^2/{segment_size}^2 = 1  \n')
                g.write(f'\n')
        #set up the fine constrains for the segment
        print(f"Z Segment {i+1}: {z_start} to {z_end}")
        fitted_surfaces = []
        for j, quadrant in enumerate(quadrants):
            print(f"  Quadrant {j+1}: {len(quadrant)} triangles")
            if quadrant:
                popt, x_coords, y_coords, z_coords = fit_surface_to_quadrant(quadrant)
                if popt is not None:
                    with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output.txt', 'a') as f:
                        f.write(f'constraint {8*i+j+1} nonpositive // seg_{i+1}_quad_{j+1}  \n')
                        f.write(f'formula: z = {popt[0]}* x^2 + {popt[1]}* y^2 + {popt[2]}* x * y + {popt[3]}* x + {popt[4]}* y + {popt[5]}  \n')
                        f.write(f'\n')
                    print(f"    Surface fit coefficients: {popt}")
                    fitted_surfaces.append((popt, x_coords, y_coords, z_coords))
                else:
                    fitted_surfaces.append((None, [], [], []))
            else:
                fitted_surfaces.append((None, [], [], []))
        plot_quadrants(quadrants, fitted_surfaces)

if __name__ == "__main__":
    #file_path = 'E:\IRPI LLC\Engineering - Syringe Debubbler\CFSC-D alt .75 mm opening-OK rough mesh.stl'  # Replace with your STL file path
    file_path = 'E:\IRPI LLC\Engineering - Syringe Debubbler\CFSC-D alt .75 mm opening-OK.stl'  # Replace with your STL file path
    your_mesh, z_min, z_max, segment_size, segments = analyze_stl(file_path)
    #your_mesh.vectors = remove_duplicate_vertices(your_mesh)
    #old unique_vectors = remove_duplicate_faces(your_mesh)
    #old your_mesh.vectors = unique_vectors
    unique_vectors = remove_duplicate_faces(your_mesh)
    your_mesh = mesh.Mesh(np.zeros(unique_vectors.shape[0], dtype=mesh.Mesh.dtype))
    your_mesh.vectors = unique_vectors
    #save the mesh with the duplicate vertices removed in ASCII format
    #with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output1.txt', 'w') as g:
    #    pass
    #with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output.txt', 'w') as f:
    with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output_half_mesh.txt', 'w') as f:
        pass
    #your_mesh.save('E:\\IRPI LLC\\Engineering - Syringe Debubbler\\CFSC-D alt .75 mm opening-OK rough mesh compact.stl', mode=stl.Mode.ASCII)
    your_mesh.save('E:\\IRPI LLC\\Engineering - Syringe Debubbler\\CFSC-D alt .75 mm opening-OK compact.stl', mode=stl.Mode.ASCII)
    #divide the mesh into segments and quadrants
    #divide_mesh_by_segments_and_quadrants(your_mesh, segments)
    divide_mesh_by_half(your_mesh)


