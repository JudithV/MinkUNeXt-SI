# Judith Vilella-Cantos. Miguel Hernández University of Elche

import copy
import time
import numpy as np
import os
import open3d as o3d
import open3d.core as o3c
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
import torch
from config import PARAMS 
from datasets.spherical_coords import SphericalCoords

from datasets.base_datasets import PointCloudLoader

class PNVPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    def read_pc(self, device, file_pathname: str) -> np.ndarray:
        if PARAMS.format_point_cloud == 'csv':
            # Load point cloud from binary file. Does not apply any transform
            if PARAMS.protocol == 'usyd' and PARAMS.spherical_coords:
                file_pathname = file_pathname.replace("USyd/","USyd_downsample/").replace(".bin", ".csv")
            elif PARAMS.protocol == 'usyd' and not PARAMS.spherical_coords:
                file_pathname = file_pathname.replace("USyd/","USyd_downsample/").replace(".bin", "_NO_SP.csv")
            df = pd.read_csv(file_pathname)
            df.columns = df.columns.str.lower().str.strip()
            df = df.query('x != 0 and y != 0 and z != 0')
            points = df[["x", "y", "z"]].to_numpy() 
            intensity = df["intensity"].to_numpy()
            if PARAMS.protocol == 'arvc': # Remove noise using minimun and maximun distance
                [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
                r2 = x ** 2 + y ** 2
                idx = np.where(r2 < PARAMS.max_distance ** 2) and np.where(r2 > PARAMS.min_distance ** 2)
                points = points[idx]
                intensity = intensity[idx]
            if PARAMS.protocol != 'usyd' and PARAMS.spherical_coords:
                spherical_points = SphericalCoords.to_spherical(points, PARAMS.protocol)
            if PARAMS.equalize_intensity:
                intensity = exposure.equalize_hist(intensity)
            return points, intensity
        
        else:
            # Load point cloud from binary file. Does not apply any transform
            file_path = os.path.join(file_pathname)
            if PARAMS.protocol == 'nclt':
                dtype = np.dtype([('x', '<H'), ('y', '<H'), ('z', '<H'), ('intensity', 'B'), ('label', 'B')])
                data = np.fromfile(file_path, dtype=dtype)
                points = np.stack([data['x'] * 0.005 - 100, data['y'] * 0.005 - 100, (data['z'] * 0.005 - 100)], axis=-1)

                intensity =  data['intensity']
                # Remove ground plane by threshold value
                ground_threshold = 0.5  # meters
                [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
                r2 = x ** 2 + y ** 2
                idx = (r2 < 80**2) & (z < -ground_threshold) 
                points = points[idx]
                intensity = intensity[idx]
                pc = np.column_stack((points, intensity))
            elif PARAMS.protocol == 'intensityOxford':
                pc = np.fromfile(file_path, dtype=np.float64).reshape([-1, 4])
            else:
                pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4]) #USYD, NCLT: float32, OXFORD: float64
            np.random.shuffle(pc)
            print(f"Shape of the pointcloud: {pc.shape}")
            if pc.shape[0] == 0:
                pc = np.zeros((1, 4), dtype=np.float32)
            else:
                if PARAMS.spherical_coords:
                    pc = SphericalCoords.to_spherical(pc, PARAMS.protocol)
                if PARAMS.equalize_intensity:
                    pc[:, 3] = exposure.equalize_hist(pc[:, 3])
            return pc[:, :3], pc[:, 3]
