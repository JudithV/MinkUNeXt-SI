import numpy as np

class SphericalCoords:
    def to_spherical(points, dataset_name):
        spherical_points = []
        for point in points:
            if (np.abs(point[:3]) < 1e-4).all():
                continue

            r = np.linalg.norm(point[:3])

            # Theta is calculated as an angle measured from the y-axis towards the x-axis
            # Shifted to range (0, 360)
            theta = np.rad2deg(np.arctan2(point[1], point[0]))
            if theta < 0:
                theta += 360

            if dataset_name == "usyd":
                # VLP-16 has 2 deg VRes and (+15, -15 VFoV).
                # Phi calculated from the vertical axis, so (75, 105)
                # Shifted to (0, 30)
                phi = np.rad2deg(np.arccos(point[2] / r)) - 75
            elif dataset_name in ['intensityOxford', 'Oxford']:
                # Oxford scans are built from a 2D scanner.
                # Phi calculated from the vertical axis, so (0, 180)
                phi = np.rad2deg(np.arccos(point[2] / r))

            elif dataset_name == 'KITTI':
                # HDL-64 has 0.4 deg VRes and (+2, -24.8 VFoV).
                # Phi calculated from the vertical axis, so (88, 114.8)
                # Shifted to (0, 26.8)
                phi = np.rad2deg(np.arccos(point[2] / r)) - 88
            elif dataset_name == 'nclt':
                # HDL-32 has 0.4 deg VRes and (+10.67°, -30.67° VFoV).
                # Phi calculated from the vertical axis, so (79.33, 120.67)
                # Shifted to (0, 41.33)
                phi = np.rad2deg(np.arccos(point[2] / r)) - 79.33
            
            elif dataset_name == 'arvc':
                # OS1-128 has 0.4 deg VRes and (+22.5, -22.5 VFoV).
                # Phi calculated from the vertical axis, so (67.5, 112.5)
                # Shifted to (0, 45)
                phi = np.rad2deg(np.arccos(point[2] / r)) - 67.5
            
            if point.shape[-1] == 4:
                spherical_points.append([r, theta, phi, point[3]])
            else:
                spherical_points.append([r, theta, phi])
        return np.array(spherical_points)
