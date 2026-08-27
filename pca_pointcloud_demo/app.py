import os
import json
import numpy as np
from flask import Flask, jsonify, request, render_template
from sklearn.decomposition import PCA

app = Flask(__name__, template_folder='templates', static_folder='static')

import glob

def generate_point_cloud(shape='ellipsoid', n_points=1500, noise=0.2):
    np.random.seed(42)  # For reproducibility, though we could make it random
    
    if shape == 'ellipsoid':
        # Generate an elongated shape so PCA axes are distinct
        rx, ry, rz = 10, 4, 1.5
        
        # Random points in a sphere
        phi = np.random.uniform(0, 2*np.pi, n_points)
        costheta = np.random.uniform(-1, 1, n_points)
        u = np.random.uniform(0, 1, n_points)
        
        theta = np.arccos(costheta)
        r = u**(1/3.0)
        
        x = r * np.sin(theta) * np.cos(phi) * rx
        y = r * np.sin(theta) * np.sin(phi) * ry
        z = r * np.cos(theta) * rz
        
        points = np.vstack((x, y, z)).T
        
    elif shape == 'plane':
        # A 2D plane embedded in 3D
        x = np.random.uniform(-10, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)
        z = np.zeros(n_points)
        points = np.vstack((x, y, z)).T
        
    elif shape == 'real_kitti':
        bin_dir = r"C:\Users\jaip7\Downloads\madhan\sih26\data\raw\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data"
        bin_files = glob.glob(os.path.join(bin_dir, "*.bin"))
        if bin_files:
            # Load the first bin file
            bin_path = bin_files[0]
            scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            points = scan[:, :3]  # Discard reflectance, keep x, y, z
            
            # Filter out points that are too far away to keep the PCA focused on the local environment
            distances = np.linalg.norm(points, axis=1)
            points = points[distances < 30.0]
            
            # Downsample heavily for browser rendering
            if len(points) > n_points:
                indices = np.random.choice(len(points), n_points, replace=False)
                points = points[indices]
        else:
            points = np.random.randn(n_points, 3) * 5
    
    else:
        # random cluster
        points = np.random.randn(n_points, 3) * 5
        
    # Add Gaussian noise
    points += np.random.normal(0, float(noise), points.shape)
    
    # Rotate the cloud so axes are not aligned to world XYZ
    # Use random angles if we want, but fixed for predictability
    theta_x, theta_y, theta_z = np.radians(30), np.radians(-45), np.radians(15)
    
    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    points = points @ R.T
    
    # Translate it off center slightly
    points += np.array([2.0, -1.0, 3.0])
    
    return points

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/pca', methods=['GET'])
def get_pca():
    shape = request.args.get('shape', 'ellipsoid')
    n_points = int(request.args.get('n_points', 1500))
    noise = float(request.args.get('noise', 0.5))
    
    # Get raw points
    points = generate_point_cloud(shape, n_points, noise)
    
    # Run PCA
    pca = PCA(n_components=3)
    pca.fit(points)
    
    # Compute center
    center = np.mean(points, axis=0)
    
    # Eigenvectors (principal axes)
    components = pca.components_
    
    # Eigenvalues (explained variance, indicates "length" or "importance" of axis)
    explained_variance = pca.explained_variance_
    
    return jsonify({
        'points': {
            'x': points[:, 0].tolist(),
            'y': points[:, 1].tolist(),
            'z': points[:, 2].tolist()
        },
        'pca': {
            'center': center.tolist(),
            'components': components.tolist(),
            'explained_variance': explained_variance.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
