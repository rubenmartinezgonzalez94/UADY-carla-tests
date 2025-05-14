#! /usr/bin/env python3
# coding: UTF-8

import random
from typing import List, Tuple, Optional, Any
from sklearn.cluster import AgglomerativeClustering
from sklearn import linear_model
import os
import sys
import cv2
import numpy as np
import PyRansac as pr
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from clipLine import *

Paleta = np.load("Paleta.npy")


class ImageInfo:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.distances, self.angles, self.time = self.get_tags(image_path)

    def get_tags(self, image_path: str) -> Tuple[List[float], List[float], float]:
        """
        Extracts tags (distances, angles, and time) from the image filename.
        """
        distances = []
        time = 0.0
        angles = []
        filename = os.path.basename(image_path)
        filename = filename.split('_')
        for word in filename:
            if word.startswith('d'):
                distances.append(float(word[3:]))
            if word.startswith('a'):
                angles.append(float(word[3:]))
            if word.startswith('t'):
                time = float(word[2:])
        return distances, angles, time


class HiperParams:
    def __init__(self):
        self.threshold_image = 210
        self.canny_params = {
            "threshold_1": 100,
            "threshold_2": 200,
        }
        self.hough_params = {
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 50,
            "min_line_length": 50,
            "max_line_gap": 40,
        }
        self.relevant_intersections_horizon_threshold = 5
        self.cluster_n_intersections = 5
        self.cluster_distance_threshold = 150
        self.center_distance_threshold = 30
        self.merge_lines_threshold = 1.


class VanishingPoint:
    def __init__(self, lines: List[np.ndarray], weights: Optional[List[float]] = None):
        self.lines = lines
        self.weights = weights if weights is not None else [1.0] * len(lines)
        self.x, self.y = self.compute_vanishing_point()

    def compute_vanishing_point(self) -> Tuple[float, float]:
        M = np.zeros((3, 3))
        for i, line in enumerate(self.lines):
            weight = self.weights[i]
            M += weight * np.outer(line, line)
        eigenvalues, eigenvectors = np.linalg.eig(M)
        min_eigenvalue_index = np.argmin(eigenvalues)
        vanishing_point_homogeneous = eigenvectors[:, min_eigenvalue_index]
        vanishing_point_homogeneous /= vanishing_point_homogeneous[2]
        return vanishing_point_homogeneous[0], vanishing_point_homogeneous[1]

    def __repr__(self):
        return f"VanishingPoint(x={self.x}, y={self.y})"


class ImageProcessor:
    def __init__(self, hiper_params: HiperParams):
        self.hiper_params = hiper_params
        self.images: List[ImageInfo] = []
        self.current_image_index: int = 0
        self.paused: bool = True
        self.show_contours: bool = False
        self.show_lines: bool = False
        self.show_intersections: bool = False
        self.show_relevant_intersections: bool = False
        self.show_relevant_lines: bool = False
        self.show_clusters: bool = False
        self.show_vanishing_points: bool = False
        self.show_gray_images: bool = False
        self.show_info: bool = False
        self.show_binary_image: bool = False
        self.show_vps = False
        self.errores = []
        self.show_test: bool = False
        self.show_intersections_vps = False
        self.show_lines_vps = False
        self.show_merged_lines_vps = False
        self.show_homography_grond_lines = False

    def load_images(self, directory: str):
        """
        Loads tagged images from a directory.
        """
        self.images = load_tagged_images(directory)
        self.images.sort(key=lambda x: x.time, reverse=False)  # Sort by time

    def process_image(self, image: np.ndarray) -> dict:
        """
        Processes an image to extract bottom_half, binary_gray_image, edges, lines, lineEqs, and intersections.
        """
        # Step 1: Ignore the top half of the image (but keep original dimensions)
        height, width = image.shape[:2]
        bottom_half = image.copy()
        bottom_half[: height // 2, :] = 0  # Set top half to black

        # Step 2: Convert to grayscale and apply binary threshold
        gray_image = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        _, binary_gray_image = cv2.threshold(gray_image, self.hiper_params.threshold_image, 255, cv2.THRESH_BINARY)

        # Step 3: Detect edges using Canny
        edges = cv2.Canny(binary_gray_image, self.hiper_params.canny_params["threshold_1"],
                          self.hiper_params.canny_params["threshold_2"])

        # Step 4: Detect lines using Hough Transform
        # rho: Resolución del parámetro de distancia en píxeles.
        # theta: Resolución del parámetro de ángulo en radianes np.pi / 180 = 1 grado
        # threshold: Número mínimo de intersecciones en el espacio de parámetros para considerar una línea.
        # minLineLength: Longitud mínima de la línea que se detectará.
        # maxLineGap: Distancia máxima entre segmentos de línea que aún se considerarán parte de la misma línea.
        lines = cv2.HoughLinesP(
            edges,
            rho=self.hiper_params.hough_params["rho"],
            theta=self.hiper_params.hough_params["theta"],
            threshold=self.hiper_params.hough_params["threshold"],
            minLineLength=self.hiper_params.hough_params["min_line_length"],
            maxLineGap=self.hiper_params.hough_params["max_line_gap"]
        )

        # Step 5: Compute line equations and collect lines endpoints
        line_eqs, end_pts = self.compute_line_equations(lines)
        
        # print(end_pts)
        # Step 5.5: fit lines to the endpoints
        #end_pts_line_eqs, end_pts_lines, end_pts_info = self.fit_lines_to_endpoints(end_pts,thresh=0.5, max_error=0.5)

        # Add end point lines to the list of lines
        #lines = np.concatenate((lines, end_pts_lines)).astype('int64')
        
        # Add end point lines equations to the list of lines equations
        #line_eqs += end_pts_line_eqs

        # Step 6: Compute intersections between lines
        intersections = self.compute_intersections(lines, line_eqs)

        # Step 7: Filter relevant intersections (near the horizon)
        relevant_intersections = self.filter_relevant_intersections(intersections, height,
                                                                    self.hiper_params.relevant_intersections_horizon_threshold)

        # Step 8: Filter relevant lines
        relevant_lines = self.filter_relevant_lines(lines, relevant_intersections)

        # Step 9: Cluster relevant intersections
        relevant_points = [point for point, _ in relevant_intersections]
        cluster_labels, cluster_centers = self.cluster_intersections(
            relevant_points,
            self.hiper_params.cluster_n_intersections if self.hiper_params.cluster_distance_threshold == 0 else None,
            self.hiper_params.cluster_distance_threshold if self.hiper_params.cluster_distance_threshold != 0 else None
        )

        # Step 10: Compute vanishing point for the strongest cluster
        vanishing_points = []
        if len(cluster_labels) > 0:
            # Find the strongest cluster
            cluster_sizes = np.bincount(cluster_labels)
            strongest_clusters_ids = np.argsort(cluster_sizes)[-1:]  # Obtaining the largest clusters

            # Obtaining the vanishing points of the two largest clusters
            vanishing_points = []
            for cluster_id in strongest_clusters_ids:
                cluster_lines = [line_eqs[indices[0]] for point, indices in relevant_intersections if
                                 cluster_labels[relevant_points.index(point)] == cluster_id]
                if cluster_lines:
                    vp = VanishingPoint(cluster_lines)
                    vanishing_points.append(vp)

        # Step 11: Calculate second vanishing point
        second_vanishing_point = self.get_vp2_by_vp1(vanishing_points[0])

        # Step 12: Filter most relevant intersections (near the two vanishing points)
        vp1 = {"x": vanishing_points[0].x, "y": vanishing_points[0].y}
        intersections_near_vps = self.filter_intersections_near_vps(relevant_intersections,
                                                                    [vp1, second_vanishing_point],
                                                                    self.hiper_params.center_distance_threshold
                                                                    )
        # Step 13: Filter relevant lines near the vanishing points
        lines_near_vps = self.filter_relevant_lines(lines, intersections_near_vps)
        

        merged_lines_near_vps = self.merge_lines(lines_near_vps, self.hiper_params.merge_lines_threshold)
        print("len(lines)          = ", len(lines))
        print("len(lines_near_vps) = ", len(lines_near_vps))
        print("len(merged_lines_near_vps) = ", len(merged_lines_near_vps))
        print("\n")
        # Step 14: parking grid RECONSTRUCTION by camera  image lines
        ground_lines = self.build_ground_lines(lines_near_vps)

        return {
            "bottom_half": bottom_half,
            "binary_gray_image": binary_gray_image,
            "edges": edges,
            "lines": lines,
            "line_eqs": line_eqs,
            "intersections": intersections,
            #"end_point_lines": end_pts_lines,
            "relevant_intersections": relevant_intersections,
            "relevant_lines": relevant_lines,
            "cluster_labels": cluster_labels,
            "cluster_centers": cluster_centers,
            "vanishing_points": vanishing_points,
            "second_vanishing_point": second_vanishing_point,
            "intersections_near_vps": intersections_near_vps,
            "lines_near_vps": lines_near_vps,
            "merged_lines_near_vps": merged_lines_near_vps,
            "ground_lines" : ground_lines
        }

    def detect_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects lines using the Hough Transform.
        """
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=40
        )
        return lines

    def compute_line_equations(self, lines: np.ndarray) -> List[np.ndarray]:
        """
        Computes the equations of the lines in homogeneous coordinates.
        """
        line_eqs = []
        (n, _, _) = lines.shape
        end_pts = np.zeros((2 * n, 2))
        idx = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                end_pts[idx, :] += [x1, y1]
                idx += 1
                end_pts[idx, :] += [x2, y2]
                idx += 1
                M = np.array([[x1, y1, 1], [x2, y2, 1]])
                line_eq = self.null_space(M)[:, 0]
                line_eqs.append(line_eq)
        return line_eqs, end_pts

    def fit_lines_to_endpoints(self, end_pts, thresh=1., max_error=1.):
        X = end_pts[:, 0].reshape(-1, 1)
        Y = end_pts[:, 1].reshape(-1, 1)
        
        Lines=[]
        Coefs =[]
        linReg = pr.Ransac(e=0.2)
        M=np.hstack([X,Y])
        
        N , _ = M.shape
        Idx = np.arange(N)

        [coefs, Error] = linReg.fitRansac(M, thrFact=thresh)
        if Error == None:
            return None

        iX =  X[Idx[linReg.inliersIdx],0]
        iY =  Y[Idx[linReg.inliersIdx],0]
        
        Coefs.append([coefs, Error, iX, iY])
        
        idxMask=np.ones(len(X)).astype('bool')
        idxMask[linReg.inliersIdx]=False
        
        while(Error < max_error):
            sX = end_pts[idxMask, 0].reshape(-1, 1)
            sy = end_pts[idxMask, 1].reshape(-1, 1)
            Idx = np.arange(N)[idxMask]

            M=np.hstack([sX,sy])
            n , _ = M.shape
            if n < 5:
                break
            
            [coefs, Error] = linReg.fitRansac(M, thrFact=1.)
            if Error == None:
                break
            iX =  X[Idx[linReg.inliersIdx],0]
            iY =  Y[Idx[linReg.inliersIdx],0]
            Coefs.append([coefs, Error, iX, iY])
            idxMask[Idx[linReg.inliersIdx]]=False

        end_pts_line_eqs = []
        end_pts_info = []
        end_pts_lines = np.zeros((len(Coefs),1,4))
        idx = 0
        for C in Coefs:
            iX = C[2]
            iY = C[3]
            
            if min(iX) != max(iX):
                indices = sorted(range(len(iX)),key=lambda index: X[index])
                iX = iX[indices]
                iY = iY[indices]
            elif min(iY) != max(iY):
                indices = sorted(range(len(iY)),key=lambda index: Y[index])
                iX = iX[indices]
                iY = iY[indices]
            else:
                print("fit_lines_to_endpoints:I should throw an exception!")
            end_pts_line_eqs.append(C[0][:3])
            end_pts_lines[idx,0,:] = [iX[0], iY[0], iX[-1], iY[-1]]
            end_pts_info.append([C[1], iX, iY])
            idx += 1
        
        return end_pts_line_eqs, end_pts_lines, end_pts_info

    def compute_intersections(
            self,
            lines: np.ndarray,
            line_eqs: List[np.ndarray]
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Computes the intersections between lines and returns a list of tuples containing
        the intersection coordinates and the indices of the lines that generated the intersection.
        """
        intersections = []
        if lines is not None:
            for i in range(len(lines) - 1):
                for j in range(i + 1, len(lines)):
                    homo_p = np.cross(line_eqs[i], line_eqs[j])
                    if not self.are_equal(homo_p[2], 0.0, 8):
                        homo_p /= homo_p[2]
                        intersections.append(((int(homo_p[0]), int(homo_p[1])), (i, j)))
        return intersections

    def filter_relevant_intersections(
            self,
            intersections: List[Tuple[Tuple[int, int], Tuple[int, int]]],
            image_height: int,
            horizon_threshold: int = 5,  # Distance in pixels from the horizon line
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Filters intersections to keep only those near the horizon.
        """

        horizon_line = image_height // 2  # Middle of the image (horizon approximation)
        relevant_intersections = []

        for point, indices in intersections:
            x, y = point
            if abs(y - horizon_line) < horizon_threshold:
                relevant_intersections.append((point, indices))

        return relevant_intersections

    def filter_relevant_lines(
            self,
            lines: np.ndarray,
            relevant_intersections: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> List[np.ndarray]:
        """
        Filters lines that intersect at relevant intersections.
        """
        relevant_lines = []
        mask = np.ones(len(lines)).astype(bool)
        if lines is not None:
            for point, (i, j) in relevant_intersections:
                if mask[i] == True:
                    relevant_lines.append(lines[i])
                    mask[i] = False
                if mask[j] == True:
                    relevant_lines.append(lines[j])
                    mask[j] = False
        return relevant_lines

    def cluster_intersections(
            self,
            intersections: List[Tuple[int, int]],
            n_clusters: int | None = 2,
            distance_threshold: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies AgglomerativeClustering to group intersections into clusters.
        Returns the cluster labels and the cluster centers.
        """
        if not intersections:
            return np.array([]), np.array([])

        # Convert intersections to a numpy array
        points = np.array(intersections)

        # Apply AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            compute_full_tree=True,
            metric='euclidean',
            linkage='ward'
            # ,distance_threshold=2
        )
        labels = clustering.fit_predict(points)

        # Compute cluster centers
        if n_clusters is None:
            n_clusters = len(np.unique(labels))
        cluster_centers = []
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append(center)

        return labels, np.array(cluster_centers)

    @staticmethod
    def null_space(A: np.ndarray, rcond: Optional[float] = None) -> np.ndarray:
        """
        Computes the null space of a matrix.
        """
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]
        if rcond is None:
            rcond = np.finfo(s.dtype).eps * max(M, N)
        tol = np.amax(s) * rcond
        num = np.sum(s > tol, dtype=int)
        Q = vh[num:, :].T.conj()
        return Q

    @staticmethod
    def are_equal(a: float, b: float, ord: int = 7) -> bool:
        """
        Checks if two numbers are equal within a given tolerance.
        """
        if a == b:
            return True
        if a != 0.0 and b != 0.0:
            val = np.abs((a - b) / max(np.abs([a, b])))
        else:
            val = max(np.abs([a, b]))
        return -np.log10(val) > ord

    def show_legend(self):
        """
        Displays the legend with available options.
        """
        for line in self.get_legend():
            print(line)

    def get_legend(self):
        return [
            "\n--- Leyenda de Teclas ---",
            "Arrow keys: adelantar/retroceder secuencia",
            "P: play secuencia (Step 0)",
            "D: detalles (Step 0)",
            "G: grayscale (Step 2)",
            "C: contornos (Step 3 Canny)",
            "L: líneas (Step 4 Hough)",
            "I: intersecciones (Step 6 all)",
            "R: intersecciones (Step 7 near the horizon)",
            "E: líneas relevantes(Step 8 near the horizon).",
            "A: cúmulos (Step 9 AgglomerativeClustering).",
            "F: 1er punto de fuga(Step 10).",
            "V: 1er y 2do punto de fuga(Step 11).",
            "Q: intersecciones (Step 12 near vanishing points)",
            "W: líneas relevantes(Step 13 near vanishing points).",
            "M: líneas relevantes Mezcladas(Step 14 near vanishing points).",
            "H: Reconstrucción de líneas en el plano del suelo.",
            "T: Test.",
            "ESC: Salir."
        ]

    def update_display(self, image: np.ndarray, processed_data: dict) -> np.ndarray:

        if self.show_binary_image:
            display_image = processed_data["binary_gray_image"]
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for consistency
        else:
            display_image = image.copy()

        height, width = display_image.shape[:2]
        center_x, center_y = width // 2, height // 2

        if self.show_info:

            y_offset = 20
            # Show vanishing point coordinates
            for vp in processed_data["vanishing_points"]:
                text = f"VP: ({int(vp.x)}, {int(vp.y)})"
                cv2.putText(display_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20

                # Check if the VP is near the center of the image
                distance_to_center = np.sqrt((vp.x - center_x) ** 2 + (vp.y - center_y) ** 2)
                if distance_to_center < self.hiper_params.center_distance_threshold:
                    text = f"VP ({int(vp.x)}, {int(vp.y)}) is near to the image center ({center_x}, {center_y})"
                    cv2.putText(display_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y_offset += 20

            # show second     vanishing point
            vp = processed_data["second_vanishing_point"]
            text = f"VP2: ({int(vp['x'])}, {int(vp['y'])})"
            cv2.putText(display_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20

            # Show cluster information ordered by size
            cluster_labels = processed_data["cluster_labels"]
            cluster_centers = processed_data["cluster_centers"]
            cluster_sizes = [(i, np.sum(cluster_labels == i)) for i in range(len(cluster_centers))]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            for i, (cluster_id, size) in enumerate(cluster_sizes):
                center = cluster_centers[cluster_id]
                text = f"Cluster {i}: {size} points, Center: ({int(center[0])}, {int(center[1])})"
                cv2.putText(display_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # show current image path
            image_info = self.images[self.current_image_index]
            cv2.putText(display_image, image_info.image_path, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            y_offset += 20

            legend = self.get_legend()
            x_offset = int(1920 / 2)
            for i, line in enumerate(legend):
                cv2.putText(display_image, line, (x_offset, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

            # Show errors
            for error in self.errores:
                cv2.putText(display_image, f"Error: {error}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
                y_offset += 20

        # Show contours (edges)
        if self.show_contours:
            contours, _ = cv2.findContours(processed_data["edges"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_image, contours, -1, (0, 255, 0), 2)

        # Show lines
        if self.show_lines and processed_data["lines"] is not None:
            for line in processed_data["lines"]:
                x1, y1, x2, y2 = line[0]
                cv2.line(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Show all intersections
        if self.show_intersections:
            for point, _ in processed_data["intersections"]:
                cv2.circle(display_image, point, 5, (255, 0, 0), -1)

        # Show relevant intersections
        if self.show_relevant_intersections:
            for point, _ in processed_data["relevant_intersections"]:
                cv2.circle(display_image, point, 5, (0, 255, 255), -1)  # Yellow color for relevant intersections

        # Show relevant lines
        if self.show_relevant_lines:
            # for line in processed_data["relevant_lines"]:
            #     x1, y1, x2, y2 = line[0]
            #     cv2.line(display_image, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta color for relevant lines
            for point, (i, j) in processed_data["relevant_intersections"]:
                x1, y1, x2, y2 = processed_data["lines"][i][0]
                x3, y3, x4, y4 = processed_data["lines"][j][0]
                # Prolongar la línea i
                cv2.line(display_image, (x1, y1), point, (255, 0, 255), 1)  # Magenta color for relevant lines
                # Prolongar la línea j
                cv2.line(display_image, (x3, y3), point, (255, 0, 255), 1)  # Magenta color for relevant lines

        # Show clusters
        if self.show_clusters:
            cluster_labels = processed_data["cluster_labels"]
            cluster_centers = processed_data["cluster_centers"]
            relevant_intersections = [point for point, _ in processed_data["relevant_intersections"]]

            if len(cluster_labels) > 0 and len(relevant_intersections) > 0:
                # Draw each cluster with a different color
                for i, point in enumerate(relevant_intersections):
                    cluster_id = cluster_labels[i]
                    color = Paleta[cluster_id, :].tolist()
                    cv2.circle(display_image, point, 5, color, -1)

                    # Draw cluster centers
                    for center in cluster_centers:
                        cv2.drawMarker(display_image, (int(center[0]), int(center[1])), (255, 255, 255),
                                       cv2.MARKER_CROSS, 10, 5)

        # Show vanishing points
        if self.show_vanishing_points:
            for vp in processed_data["vanishing_points"]:
                cv2.drawMarker(display_image, (int(vp.x), int(vp.y)), (0, 255, 255), cv2.MARKER_CROSS, 30,
                               5)  # Yellow color for vanishing points

        if self.show_vps:
            vp1 = processed_data["vanishing_points"][0]
            cv2.drawMarker(display_image, (int(vp1.x), int(vp1.y)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30,
                           5)  # Red cross for vanishing points
            cv2.line(display_image, (center_x, center_y), (int(vp1.x), int(vp1.y)), (0, 0, 255),
                     2)  # Red line from center to vanishing point

            vp2 = processed_data["second_vanishing_point"]
            cv2.drawMarker(display_image, (int(vp2['x']), int(vp2['y'])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30,
                           5)  # Red cross for second vanishing point
            cv2.line(display_image, (center_x, center_y), (int(vp2['x']), int(vp2['y'])), (0, 0, 255),
                     2)  # Red line from center to second vanishing point

        # Show intersections near vanishing points
        if self.show_intersections_vps:
            for point, _ in processed_data["intersections_near_vps"]:
                cv2.circle(display_image, point, 5, (0, 255, 255), -1)

        # Show lines near     vanishingpoints
        if self.show_lines_vps:
            for point, (i, j) in processed_data["intersections_near_vps"]:
                x1, y1, x2, y2 = processed_data["lines"][i][0]
                x3, y3, x4, y4 = processed_data["lines"][j][0]
                # Prolongar la línea i
                cv2.line(display_image, (x1, y1), point, (255, 0, 255), 1)  # Magenta color for relevant lines
                # Prolongar la línea j
                cv2.line(display_image, (x3, y3), point, (255, 0, 255), 1)  # Magenta color for relevant lines
        if self.show_merged_lines_vps:
            merged_lines_near_vps = processed_data["merged_lines_near_vps"]
            for line in merged_lines_near_vps:
                x1, y1, x2, y2 = line[0]
                pt1 = x1, y1
                pt2 = x2, y2
                cv2.line(display_image, pt1, pt2, (0, 255, 0), 1)

        if self.show_homography_grond_lines:
            ground_lines = processed_data["ground_lines"]
            # Crear una imagen en blanco para dibujar
            canvas_width, canvas_height = 800, 800
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # Fondo blanco

            # Escalar las coordenadas al tamaño del canvas
            scale = 50  # Escala para convertir metros a píxeles
            offset_x, offset_z = canvas_width // 2, canvas_height // 2  # Centrar en el canvas

            # Dibujar las líneas del suelo
            for p1, p2 in ground_lines:
                x1, z1 = int(p1[0] * scale + offset_x), int(offset_z - p1[2] * scale)
                x2, z2 = int(p2[0] * scale + offset_x), int(offset_z - p2[2] * scale)
                cv2.line(canvas, (x1, z1), (x2, z2), (255, 0, 0), 2)  # Azul para las líneas

            # Dibujar el rectángulo azul (carrito)
            car_x, car_z = -1, -4  # Coordenadas iniciales del carrito
            car_width, car_length = 2, 4  # Ancho y largo del carrito
            rect_x1 = int(car_x * scale + offset_x)
            rect_z1 = int(offset_z - car_z * scale)
            rect_x2 = int((car_x + car_width) * scale + offset_x)
            rect_z2 = int(offset_z - (car_z + car_length) * scale)
            cv2.rectangle(canvas, (rect_x1, rect_z1), (rect_x2, rect_z2), (0, 0, 255), -1)  # Rojo relleno

            # Mostrar la imagen en una ventana de OpenCV
            cv2.imshow("Homography Ground Lines", canvas)
            cv2.waitKey(1)  # Refrescar la ventana

        if self.show_test:
            image_info = self.images[self.current_image_index]
            # print(image_info.image_path)
            lines_near_vps = processed_data["lines_near_vps"]
            np.save('lines_near_vps.npy', lines_near_vps)

            merged_lines_near_vps = processed_data["merged_lines_near_vps"]
            np.save('merged_lines_near_vps.npy', merged_lines_near_vps)

            end_point_lines = processed_data["end_point_lines"]
            #print(end_point_lines)
            with open("end_point_lines.pkl", "wb") as fp:   #Pickling
                pickle.dump(end_point_lines, fp)
            fp.close()
            

            print("lines_near_vps saved")
            self.show_test = False

        return display_image

    def create_trackbars(self):
        cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
        # cv2.createButton("Pause", self.toggle_pause, None, cv2.QT_PUSH_BUTTON, 0)
        cv2.createTrackbar("Horizon Threshold", "Trackbars", self.hiper_params.relevant_intersections_horizon_threshold,
                           100, self.update_horizon_threshold)
        cv2.createTrackbar("Cluster Intersections", "Trackbars", self.hiper_params.cluster_n_intersections, 50,
                           self.update_cluster_intersections)
        cv2.createTrackbar("Cluster Distance Threshold", "Trackbars", self.hiper_params.cluster_distance_threshold,
                           1000,
                           self.update_distance_threshold)  # Add Trackbar for distance_thresholdackbar for distance_threshold
        cv2.createTrackbar("Canny Threshold 1", "Trackbars", self.hiper_params.canny_params["threshold_1"], 500,
                           self.update_canny_threshold_1)
        cv2.createTrackbar("Canny Threshold 2", "Trackbars", self.hiper_params.canny_params["threshold_2"], 500,
                           self.update_canny_threshold_2)
        cv2.createTrackbar("Hough Threshold", "Trackbars", self.hiper_params.hough_params["threshold"], 200,
                           self.update_hough_threshold)
        cv2.createTrackbar("Hough Min Line Length", "Trackbars", self.hiper_params.hough_params["min_line_length"], 200,
                           self.update_min_line_length)
        cv2.createTrackbar("Hough Max Line Gap", "Trackbars", self.hiper_params.hough_params["max_line_gap"], 200,
                           self.update_max_line_gap)
        cv2.createTrackbar("Threshold Image", "Trackbars", self.hiper_params.threshold_image, 255,
                           self.update_threshold_image)
        cv2.createTrackbar("Distance FOV to center", "Trackbars", self.hiper_params.center_distance_threshold, 500,
                           self.update_center_distance_threshold)
        cv2.createTrackbar("Merged lines Threshold", "Trackbars", 3, 100,
                           self.update_merge_lines_threshold)

    def toggle_pause(self, *args):
        self.paused = not self.paused

    def update_threshold_image(self, value):
        self.hiper_params.threshold_image = value
        self.process_and_display_current_image()

    def update_distance_threshold(self, value):
        self.hiper_params.cluster_distance_threshold = value
        self.process_and_display_current_image()

    def update_canny_threshold_1(self, value):
        self.hiper_params.canny_params["threshold_1"] = value
        self.process_and_display_current_image()

    def update_canny_threshold_2(self, value):
        self.hiper_params.canny_params["threshold_2"] = value
        self.process_and_display_current_image()

    def update_hough_threshold(self, value):
        self.hiper_params.hough_params["threshold"] = value
        self.process_and_display_current_image()

    def update_min_line_length(self, value):
        self.hiper_params.hough_params["min_line_length"] = value
        self.process_and_display_current_image()

    def update_max_line_gap(self, value):
        self.hiper_params.hough_params["max_line_gap"] = value
        self.process_and_display_current_image()

    def update_horizon_threshold(self, value):
        self.hiper_params.relevant_intersections_horizon_threshold = value
        self.process_and_display_current_image()

    def update_cluster_intersections(self, value):
        self.hiper_params.cluster_n_intersections = value
        self.process_and_display_current_image()

    def update_center_distance_threshold(self, value):
        self.hiper_params.center_distance_threshold = value
        self.process_and_display_current_image()

    def update_merge_lines_threshold(self, value):
        self.hiper_params.merge_lines_threshold = value
        self.process_and_display_current_image()

    def process_and_display_current_image(self):
        if self.images:
            image_info = self.images[self.current_image_index]
            image = cv2.imread(image_info.image_path, cv2.IMREAD_COLOR)
            processed_data = self.process_image(image)
            self.update_display(image, processed_data)

    def get_vp2_by_vp1(self, vp1):
        vp1x = float(vp1.x)
        vp1y = vp1.y
        image_width = 1920
        fov = 90
        f = focal_length = image_width / (2 * np.tan(fov * np.pi / 360))
        Cx = image_width / 2

        vp2x = Cx - f ** 2 / (vp1x - Cx)
        vp2y = vp1y
        vp2 = {"x": vp2x, "y": vp2y}
        return vp2

    def filter_intersections_near_vps(
            self,
            intersections: List[Tuple[Tuple[int, int], Tuple[int, int]]],
            vps,
            threshold=50
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        intersections_near_vps = []
        # return intersections near the vanishing points
        for vp in vps:
            for point, index in intersections:
                x, y = point
                if abs(x - vp['x']) < threshold and abs(y - vp['y']) < threshold:
                    intersections_near_vps.append((point, index))
        return intersections_near_vps

    def merge_lines(self, lines, merge_threshold=0.01):
        n = len(lines)
        checked_lines = np.ones(n)
        similarities = np.zeros((n, n + 1), dtype=int)
        Identicas_DBG=0
        #print("len(lines) = ",len(lines))
        for i in range(n):
            similarities[i, 0] = 0 #Number of similar elements.
            idx = 1
            line_i = points_to_homogeneous_line(lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3])
            
            LI_DBG=np.array(lines[i][0,:]).reshape(2,2).transpose()
            Sim = np.zeros(n)
            Sidx=0
            for j in range(i + 1, n):
                if checked_lines[j] == 1:
                    line_j = points_to_homogeneous_line(lines[j][0][0], lines[j][0][1], lines[j][0][2], lines[j][0][3])
                    LJ_DBG=np.array(lines[j][0,:]).reshape(2,2).transpose()
                    if np.max(np.abs(LJ_DBG-LI_DBG)) != 0:
                        simil, dist, _ = line_similarity(line_i, line_j, merge_threshold, normType=0 )
                        if simil:
                            #print("Dist(%d,%d)=%f" % (i,j,dist))
                            Sim[Sidx]=dist
                            Sidx+=1
                            checked_lines[j] = 0
                            similarities[i, idx] = j
                            similarities[i, 0] += 1  # i, 0
                            idx += 1
                            #print('similar:', (i,j), dist)
                        #else:
                            #print('no similar:', dist)
                    else:
                        #print("*"*80)
                        #print("Iguales:\n",LJ_DBG,"\n",LI_DBG)
                        #print("*"*80+"\n")
                        Identicas_DBG+=1
            #print(i, Sidx, np.sort(Sim))
        #print("Similarities = \n",similarities)
        #print("Se Encontraron %d lineas identicas." % (Identicas_DBG))                


        # call to fusiona_lines
        merged_lines = []
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if similarities[i, 0] > 0:
                group_indices = similarities[i, 1: similarities[i, 0] + 1]
                group = [lines[k][0] for k in group_indices]
                homog_line = fusiona_lines(group)
                #merged_line = lineHomo_to_linePoint(homog_line)
                merged_line = lineHomo_to_linePoint2(homog_line,group)
                merged_lines.append([np.array(merged_line, dtype=int)])
                visited[group_indices] = True
            else:
                merged_lines.append(lines[i][0])
                
        return merged_lines

    def build_ground_lines(self, lines_near_vps):
        image_width, image_height = 1920, 1080
        fov = 90  # grados
        camera_height = 1.3  # metros
        ground_lines = []

        # Calcular focal en píxeles
        f = image_width / (2 * np.tan(np.radians(fov / 2)))
        cx, cy = image_width / 2, image_height / 2

        # Matriz intrínseca
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        K_inv = np.linalg.inv(K)

        for line in lines_near_vps:
            x1, y1, x2, y2 = line[0]

            # Proyección al espacio
            p1_img = np.array([x1, y1, 1])
            p2_img = np.array([x2, y2, 1])

            # Rayos en cámara
            r1_cam = K_inv @ p1_img
            r2_cam = K_inv @ p2_img

            # Escalar rayos para intersectar el plano Z=0 (suelo)
            scale1 = camera_height / r1_cam[1]
            scale2 = camera_height / r2_cam[1]

            p1_ground = r1_cam * scale1
            p2_ground = r2_cam * scale2

            ground_lines.append((p1_ground, p2_ground))
        return ground_lines


def fusiona_lines(lines):
    n = len(lines)
    if n == 1:
        x1, y1, x2, y2 = lines[0]
        l = points_to_homogeneous_line(x1, y1, x2, y2)
        return l / l[2]
    w = np.zeros(n)
    acum = 0
    for i in range(n):
        x1 = lines[i][0]
        y1 = lines[i][1]
        x2 = lines[i][2]
        y2 = lines[i][3]
        dx = x1 - x2
        dy = y1 - y2
        w[i] = np.sqrt(dx * dx + dy * dy)
        acum += w[i]
    w /= acum

    acumL = np.zeros((3))
    for i in range(n):
        x1 = lines[i][0]
        y1 = lines[i][1]
        x2 = lines[i][2]
        y2 = lines[i][3]
        l = points_to_homogeneous_line(x1, y1, x2, y2).astype('float')
        l /= l[2]
        acumL += w[i] * l

    result_line = acumL / acumL[2]
    return result_line


def points_to_homogeneous_line(x1, y1, x2, y2):
    return np.cross([x1, y1, 1], [x2, y2, 1])



def sortPts(P):
    theta = np.zeros(4)
    mX = np.mean(P[0,: ])
    mY = np.mean(P[1, :])
    for k in range(4):
        Dx = P[0, k] - mX
        Dy = P[1, k] - mY
        theta[k] = np.arctan2(Dy, Dx)
    
    indices = sorted(range(len(theta)), key = lambda index: theta[index])
    sP = P[:, indices]
    return sP, indices

def sortPtsIdx(p,i,j):
    P = np.array([[p[i, 0, 0],p[i, 0, 2],p[j, 0, 0],p[j, 0, 2]],
                  [p[i, 0, 1],p[i, 0, 3],p[j, 0, 1],p[j, 0, 3]],
                  [  1,                1,         1,        1]]).astype('float64')
    return sortPts(P)

def line_similarity(line_a, line_b, threshold=1, normType = 0, region=[1920, 1080]):
    distance = np.inf
    if normType == 1:
        # Normalize the lines as homogeneous variable
        linea_n = line_a / line_a[2]
        lineb_n = line_b / line_b[2]
        tmp = linea_n[:2] - lineb_n[:2]
        distance = np.dot(tmp, tmp)
    elif normType == 2:
        # Normalize the lines according to their size
        linea_n = line_b / np.linalg.norm(line_b)
        linea_n = line_a / np.linalg.norm(line_a)
        tmp = lineb_n - linea_n
        distance = np.dot(tmp, tmp)
    else:
        
        pl1, pl2, success = clipLine(line_a, (0,region[1]//2), (region[0],region[1]//2))
        
        if success == True:
            pl3, pl4, success = clipLine(line_b, (0,region[1]//2), (region[0],region[1]//2))
            
            if success == True:
                
                P = np.hstack([pl1,pl2,pl3,pl4]).reshape(4,3).transpose()
                
                sP, idx = sortPts(P)
                
                d=[]
                tmp = sP[:2,0]-sP[:2,1]
                d.append(np.dot(tmp, tmp)) #Squared Distance between P[0,:] and P[1,:]
                tmp = sP[:2,2]-sP[:2,3]
                d.append(np.dot(tmp, tmp)) #Squared Distance between P[0,:] and P[1,:]
                pts = P.copy()
                if d[0] > d[1]:
                    distance = d[0]
                    pts = np.hstack([pts, np.array(sP[:,0],ndmin=2).transpose()])
                    pts = np.hstack([pts, np.array(sP[:,1],ndmin=2).transpose()])
                else:
                    distance = d[1]
                    pts = np.hstack([pts, np.array(sP[:,2],ndmin=2).transpose()])
                    pts = np.hstack([pts, np.array(sP[:,3],ndmin=2).transpose()])

    # Compute similarity
    return distance <= (threshold * threshold), distance, pts


def lineHomo_to_linePoint2(homo_line, group):
    #TODO: Hay que optimizar esto.
    n = len(group)
    p = np.zeros((2,2*n))
    idx = 0
    for i in range(n):
        p[:,idx] = group[i][:2]
        idx += 1
        p[:,idx] = group[i][2:4]
    
    n *= 2
    dMax = np.dot(p[:,0], p[:,1])
    pMax = (0,1)
    for i in range(n-1):
        for j in range(i+1,n):
            d = np.dot(p[:,i], p[:,j])
            if d < dMax:
                dMax=d
                pMax = (i,j)
    a = homo_line[0]
    b = homo_line[1]
    c = homo_line[2]
    den = a*a+b*b
    
    x0 = p[0,i]
    y0 = p[1,i]
    X0=(b*(b*x0-a*y0)-a*c)/den
    Y0=(a*(-b*x0+a*y0)-b*c)/den

    x0 = p[0,j]
    y0 = p[1,j]
    X1=(b*(b*x0-a*y0)-a*c)/den
    Y1=(a*(-b*x0+a*y0)-b*c)/den

    return (X0, Y0, X1, Y1)

def lineHomo_to_linePoint(homo_line, x_range=(0, 1000)):
    a, b, c = homo_line

    if abs(b) > 1e-6:
        x1, x2 = x_range
        y1 = -(a * x1 + c) / b
        y2 = -(a * x2 + c) / b
    else:
        # Línea vertical
        x1 = x2 = -c / a if abs(a) > 1e-6 else 0
        y1, y2 = 0, 1000

    return (x1, y1, x2, y2)


def load_tagged_images(directory: str) -> List[ImageInfo]:
    """
    Loads tagged images from a directory.
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            route = os.path.join(directory, filename)
            images.append(ImageInfo(route))
    return images


def main(sequence='../manual_sequence/sec4/'):

    print("*"*80)
    print(("*"+" "*78+"*"+"\n")*5,end='')
    print("*"*80)
    # Create an instance of ImageProcessor
    hiper_params = HiperParams()
    processor = ImageProcessor(hiper_params)

    # Load images from the directory
    processor.load_images(sequence)

    # Display the legend
    processor.show_legend()

    # Create trackbars
    processor.create_trackbars()

    # Create an OpenCV window to capture keys
    cv2.namedWindow("Image Sequence", cv2.WINDOW_NORMAL)

    # Main loop to interact with the options
    while True:
        # processor.errores = []
        # Get the current image
        image_info = processor.images[processor.current_image_index]
        image = cv2.imread(image_info.image_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Error: Unable to load image {image_info.image_path}.")
            continue

        try:
            # Process the image
            processed_data = processor.process_image(image)

            # Update the display image with the current options
            display_image = processor.update_display(image, processed_data)

            # Show the image
            cv2.imshow("Image Sequence", display_image)

        except Exception as e:
            print(f"Error processing image {image_info.image_path}: {e}")
            # processor.errores.append(str(e))
            processor.paused = True

        # Wait for a key press
        key = cv2.waitKey(30) & 0xFF

        # Handle key presses
        if key == ord('p'):  # P: Pause/Resume the sequence
            processor.paused = not processor.paused
        elif key == ord('d'):  # D: Toggle detailed information
            processor.show_info = not processor.show_info
        elif key == ord('c'):  # C: Toggle contours
            processor.show_contours = not processor.show_contours
        elif key == ord('l'):  # L: Toggle lines
            processor.show_lines = not processor.show_lines
        elif key == ord('i'):  # I: Toggle intersections
            processor.show_intersections = not processor.show_intersections
        elif key == ord('r'):  # R: Toggle relevant intersections
            processor.show_relevant_intersections = not processor.show_relevant_intersections
        elif key == ord('e'):  # E: Toggle relevant lines
            processor.show_relevant_lines = not processor.show_relevant_lines
        elif key == ord('a'):  # A: Toggle clusters
            processor.show_clusters = not processor.show_clusters
        elif key == ord('f'):  # F: Toggle vanishing points
            processor.show_vanishing_points = not processor.show_vanishing_points
        elif key == ord('g'):  # G: Toggle between original and binary image
            processor.show_binary_image = not processor.show_binary_image
        elif key == ord('v'):  # V: Toggle vanishing points with red lines
            processor.show_vps = not processor.show_vps
        elif key == ord('t'):  # t: Toggle print test
            processor.show_test = not processor.show_test
        elif key == ord('q'):  # Q: Toggle intersections near vanishing points
            processor.show_intersections_vps = not processor.show_intersections_vps
        elif key == ord('w'):  # W: Toggle relevant lines near vanishing points
            processor.show_lines_vps = not processor.show_lines_vps
        elif key == ord('m'):
            processor.show_merged_lines_vps = not processor.show_merged_lines_vps
        elif key == ord('h'):  # H: Show homography
            processor.show_homography_grond_lines = not processor.show_homography_grond_lines
        elif key == 81 or key == 52:  # Left arrow key
            processor.current_image_index = (processor.current_image_index - 1) % len(processor.images)
        elif key == 83 or key == 54:  # Right arrow key
            processor.current_image_index = (processor.current_image_index + 1) % len(processor.images)
        elif key == 27:  # ESC: Exit
            break

        # Move to the next image only if not paused
        if not processor.paused:
            processor.current_image_index = (processor.current_image_index + 1) % len(processor.images)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()

# def get_calibration_matrix(image_width, image_height, fov=90):
#     focal_length = image_width / (2 * np.tan(fov * np.pi / 360))
#     calibration_matrix = np.array([[focal_length, 0, image_width / 2],
#                                    [0, focal_length, image_height / 2],
#                                    [0, 0, 1]])
#     return calibration_matrix
#
#
# +-+-----+-----+-----+
# |   | 0     | 1     | 2     |
# +-+-----+-----+-----+
# | 0 | 960.0 | 0.0   | 960.0 |
# | 1 | 0.0   | 960.0 | 540.0 |
# | 2 | 0.0   | 0.0   | 1.0   |
# +-+-----+-----+-----+
