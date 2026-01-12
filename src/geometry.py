import math
import numpy as np

class GeometryUtils:
    @staticmethod
    def euclidean_distance(point1, point2):
        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @staticmethod
    def get_aspect_ratio(eye_points):
        A = GeometryUtils.euclidean_distance(eye_points[1], eye_points[5])
        B = GeometryUtils.euclidean_distance(eye_points[2], eye_points[4])
        C = GeometryUtils.euclidean_distance(eye_points[0], eye_points[3])
        if C == 0: return 0
        return (A + B) / (2.0 * C)
