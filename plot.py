import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python plot.py <grid_size>")
        print("Example: python plot.py 100")
        sys.exit(1)

    try:
        N = int(sys.argv[1])
    except ValueError:
        print("Error: Grid size must be an integer")
        sys.exit(1)

    # Configuration
    input_file = 'sol.dat'
    nx, ny = N, N
    D = 0.5
    L = 0.3
    boxC = 0.5

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        print("Make sure to run the Fortran program first to generate the solution data")
        sys.exit(1)

    try:
        # Load and process data
        data = np.loadtxt(input_file)
        
        # Check if data size matches grid size
        expected_size = nx * ny
        if data.size != expected_size:
            print(f"Error: Data size mismatch. Expected {expected_size} points, got {data.size}")
            print(f"Make sure your grid size ({N}) matches the data in {input_file}")
            sys.exit(1)

        # Reshape and transpose to convert to "standard" c order
        data = data.reshape((ny, nx)).T

        # Create coordinate grids
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Calculate electric field
        Ey, Ex = np.gradient(-data, y, x)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Poisson Solution with Electric Field Streamlines')

        # Plot 2d solution with contour lines
        im = ax.imshow(data, extent=[x[0], x[-1], y[0], y[-1]], 
                      origin='lower', interpolation='nearest')
        fig.colorbar(im, ax=ax, label='Potential')

        # Plot streamlines of the gradient field (electric field)
        Em = np.sqrt(Ey**2. + Ex**2.)
        lw = 8. * Em / Em.max()  # linewidth depending on magnitude
        ax.streamplot(x, y, Ex, Ey, color='white', linewidth=lw, 
                     arrowsize=0.7, density=1.2)

        # Plot contour lines
        cntr = ax.contour(data, [-0.7, -0.25, -0.05, 0.05, 0.25, 0.7], 
                         colors='red', extent=[0, 1, 0, 1])
        ax.clabel(cntr, cntr.levels, fontsize=10, colors='red')

        # Plot bars inside domain
        ax.vlines(x=boxC-D/2., ymin=boxC-L/2., ymax=boxC+L/2., 
                 linewidth=2, color='b')
        ax.vlines(x=boxC+D/2., ymin=boxC-L/2., ymax=boxC+L/2., 
                 linewidth=2, color='b')

        plt.show()

    except np.linalg.LinAlgError as e:
        print(f"Error processing data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

