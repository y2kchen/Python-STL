#goal: I have an STL file and I would like to divide the surface into patches and for each patch I would like to use surface fitting to find an expression.
#how should I proceed?
# for the STL file, I can use the numpy-stl library to read the file and extract the vertices and faces.
# for the surface fitting, I can use the scipy library to fit a surface to the points.
# for the patches, I can use the scikit-learn library to cluster the points into patches.
# I can use the matplotlib library to visualize the patches and the surface fitting.
# I can use the sympy library to represent the expression of the surface fitting.
# I can use the numpy library to evaluate the expression of the surface fitting.
# I can use the pandas library to store the patches and the expression of the surface fitting.
# I can use the pickle library to save the patches and the expression of the surface fitting.
# I can use the joblib library to load the patches and the expression of the surface fitting.
# I can use the os library to manage the files.
# I can use the time library to measure the time of the computation.
# I can use the logging library to log the information.
# I can use the argparse library to parse the arguments.
# I can use the sys library to manage the exit.
# I can use the unittest library to test the functions.
# I can use the pytest library to test the functions.
# I can use the sphinx library to generate the documentation.
# I can use the numpydoc extension to generate the documentation.
# I can use the readthedocs to host the documentation.
# I can use the github to host the code.
# I can use the travis-ci to test the code.
# I can use the coveralls to measure the coverage.
# I can use the codecov to measure the coverage.
# I can use the landscape to measure the quality.
# I can use the codeclimate to measure the quality.
# I can use the codacy to measure the quality
       
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
    segment_n = 20
    segment_size = z_range / segment_n
    segments = [(z_min + i * segment_size, z_min + (i + 1) * segment_size) for i in range(segment_n)]
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

def divide_mesh_by_segments_and_quadrants(your_mesh, segments):
    for i, (z_start, z_end) in enumerate(segments):
        segment_triangles = []
        for triangle in your_mesh.vectors:
            if np.any((triangle[:, 2] >= z_start) & (triangle[:, 2] < z_end)):
                segment_triangles.append(triangle)
        quadrants = divide_into_quadrants(segment_triangles)
        #identify the absolute minimal x and y coordinates of the segment
        if i<19:
            min_abs_x = float('inf')
            min_abs_y = float('inf')
            for triangle in segment_triangles:
                for point in triangle:
                    x, y, _ = point
                    #find the minimal y in a narrow range of x, make sure the range is large enough to produce non-zero min_abs_y
                    if -0.2 <= x <= 0.2:
                        min_abs_y = min(min_abs_y, abs(y))
                    #find the minimal x in a narrow range of y
                    if -0.2 <= y <= 0.2:
                        min_abs_x = min(min_abs_x, abs(x))
        #set up the rough constrains for the segment
        with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output1.txt', 'a') as g:
            if i < 19:
                g.write(f'constraint {200+i+1} nonpositive // seg_{i+1}  \n')
                g.write(f'formula: x^2/{min_abs_x}^2+y^2/{min_abs_y}^2 = 1  \n')
                g.write(f'\n')
            if i == 19:
                g.write(f'constraint {200+i+1} nonpositive // seg_{i+1}  \n')
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
                        if i < 18:
                            f.write(f'constraint {8*i+j+1} nonnegative // seg_{i+1}_quad_{j+1}  \n')
                        if i >= 18:
                            f.write(f'constraint {8*i+j+1} nonpositive // seg_{i+1}_quad_{j+1}  \n')                            
                        #f.write(f'constraint {8*i+j+1} nonpositive // seg_{i+1}_quad_{j+1}  \n')
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
    #file_path = 'E:\IRPI LLC\Engineering - Syringe Debubbler\CFSC-D alt .75 mm opening-OK rough mesh origin at dome base.stl'  # Replace with your STL file path
    file_path = 'E:\IRPI LLC\Engineering - Syringe Debubbler\CFSC-D alt .75 mm opening-OK.stl'  # Replace with your STL file path
    your_mesh, z_min, z_max, segment_size, segments = analyze_stl(file_path)
    #your_mesh.vectors = remove_duplicate_vertices(your_mesh)
    #old unique_vectors = remove_duplicate_faces(your_mesh)
    #old your_mesh.vectors = unique_vectors
    unique_vectors = remove_duplicate_faces(your_mesh)
    your_mesh = mesh.Mesh(np.zeros(unique_vectors.shape[0], dtype=mesh.Mesh.dtype))
    your_mesh.vectors = unique_vectors
    #save the mesh with the duplicate vertices removed in ASCII format
    with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output1.txt', 'w') as g:
        pass
    with open('E:\IRPI LLC\Engineering - Syringe Debubbler\output.txt', 'w') as f:
        pass
    #your_mesh.save('E:\\IRPI LLC\\Engineering - Syringe Debubbler\\CFSC-D alt .75 mm opening-OK rough mesh origin at dome base compact.stl', mode=stl.Mode.ASCII)
    your_mesh.save('E:\\IRPI LLC\\Engineering - Syringe Debubbler\\CFSC-D alt .75 mm opening-OK compact.stl', mode=stl.Mode.ASCII)
    #divide the mesh into segments and quadrants
    divide_mesh_by_segments_and_quadrants(your_mesh, segments)


