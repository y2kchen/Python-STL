#goal: To fit a part of the mesh with a surface.
# the part is based on the drawing in Solidworks.

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
    x_min = np.min(x_coords)
    # subtract x_min from x coordinates of the mesh
    #your_mesh.x -= x_min
    #find the mean of the y coordinates
    y_coords = your_mesh.vectors[:, :, 1].flatten()
    y_mean = np.mean(y_coords)
    y_min = np.min(y_coords)
    # subtract y_min from y coordinates of the mesh
    #your_mesh.y -= y_min
    # find the min and max of the z coordinates
    z_coords = your_mesh.vectors[:, :, 2].flatten()
    z_min = np.min(z_coords)
    print(f"Z minimum: {z_min}")
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

def fit_surface_to_mesh(your_mesh):
    def poly_func(X, a, b, c, d, e, f):
        x, y = X
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
    
    x_coords = []
    y_coords = []
    z_coords = []

    for vertex in your_mesh:
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

def plot_surface(your_mesh, fitted_surfaces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']

    # Plot original points
    #for i, in enumerate([half1, half2]):
    for i, mesh in enumerate([your_mesh]):
        x_coords = []
        y_coords = []
        z_coords = []
        for vertex in your_mesh:
            x_coords.append(vertex[0])
            y_coords.append(vertex[1])
            z_coords.append(vertex[2])
        ax.scatter(x_coords, y_coords, z_coords, color=colors[i], label=f'mesh')

    #Plot fitted surfaces
    for i, (popt, x_coords, y_coords, z_coords) in enumerate(fitted_surfaces):
        if popt is not None:
            x_range = np.linspace(min(x_coords), max(x_coords), 30)
            y_range = np.linspace(min(y_coords), max(y_coords), 30)
            x_mesh, y_mesh = np.meshgrid(x_range, y_range)
            z_mesh = (popt[0]*x_mesh**2 + popt[1]*y_mesh**2 + popt[2]*x_mesh*y_mesh +
                      popt[3]*x_mesh + popt[4]*y_mesh + popt[5])
            ax.plot_surface(x_mesh, y_mesh, z_mesh, color=colors[i], alpha=0.5)

    #modify the limits of x-axis of the plot
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #file_path = 'E:\IRPI LLC\Engineering - Syringe Debubbler\CFSC-D alt .75 mm opening-OK rough mesh.stl'  # Replace with your STL file path
    #file_path = 'E:\IRPI LLC\Engineering - Syringe Debubbler\CFSC-D alt .75 mm opening-OK -CY-part3.stl'  # Replace with your STL file path
    file_path = 'E:\IRPI LLC\Engineering - Syringe Debubbler\CFSC-D alt .75 mm opening-OK part4 fine.stl'  # Replace with your STL file path

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
    with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output_part_1.txt', 'w') as f:
        pass
    #your_mesh.save('E:\\IRPI LLC\\Engineering - Syringe Debubbler\\CFSC-D alt .75 mm opening-OK rough mesh compact.stl', mode=stl.Mode.ASCII)
    #your_mesh.save('E:\\IRPI LLC\\Engineering - Syringe Debubbler\\CFSC-D alt .75 mm opening-OK -CY-part3 compact.stl', mode=stl.Mode.ASCII)
    your_mesh.save('E:\IRPI LLC\\Engineering - Syringe Debubbler\CFSC-D alt .75 mm opening-OK part4 fine compact.stl', mode=stl.Mode.ASCII)

    #divide the mesh into segments and quadrants
    #divide_mesh_by_segments_and_quadrants(your_mesh, segments)
    fitted_surfaces = []
    #fit the surface
    popt1, x_coords1, y_coords1, z_coords1 = fit_surface_to_mesh(your_mesh)
    if popt1 is not None:
        with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output_part_1.txt', 'a') as f:
            f.write(f'constraint positive_x nonpositive   \n')
            f.write(f'formula: z = {popt1[0]}* x^2 + {popt1[1]}* y^2 + {popt1[2]}* x * y + {popt1[3]}* x + {popt1[4]}* y + {popt1[5]}  \n')
            f.write(f'\n')
        print(f"    Surface fit coefficients: {popt1}")
        fitted_surfaces.append((popt1, x_coords1, y_coords1, z_coords1))
    else:
        fitted_surfaces.append((None, [], [], []))
    plot_surface(your_mesh, fitted_surfaces)


