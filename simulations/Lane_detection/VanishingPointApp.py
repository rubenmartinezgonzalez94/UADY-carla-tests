#! /usr/bin/env python3
# coding: UTF-8

import random
from typing import List, Tuple, Optional, Any
from sklearn.cluster import AgglomerativeClustering
import os
import sys
import cv2
import numpy as np

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
        self.distance_threshold = 0


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
        self.show_vanishing_points: bool = True
        self.show_info: bool = False
        self.errores = []

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
        lines = cv2.HoughLinesP(
            edges,
            rho=self.hiper_params.hough_params["rho"],
            theta=self.hiper_params.hough_params["theta"],
            threshold=self.hiper_params.hough_params["threshold"],
            minLineLength=self.hiper_params.hough_params["min_line_length"],
            maxLineGap=self.hiper_params.hough_params["max_line_gap"]
        )

        # Step 5: Compute line equations
        line_eqs = self.compute_line_equations(lines)

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
            self.hiper_params.cluster_n_intersections if self.hiper_params.distance_threshold is 0 else None,
            self.hiper_params.distance_threshold if self.hiper_params.distance_threshold is not 0 else None
        )

        # Step 10: Compute vanishing point for the strongest cluster
        vanishing_points = []
        if len(cluster_labels) > 0:
            # Find the strongest cluster
            cluster_sizes = np.bincount(cluster_labels)
            strongest_clusters_ids = np.argsort(cluster_sizes)[-2:]  # Obtaining the two largest clusters

            # Obtaining the vanishing points of the two largest clusters
            vanishing_points = []
            for cluster_id in strongest_clusters_ids:
                cluster_lines = [line_eqs[indices[0]] for point, indices in relevant_intersections if
                                 cluster_labels[relevant_points.index(point)] == cluster_id]
                if cluster_lines:
                    vp = VanishingPoint(cluster_lines)
                    vanishing_points.append(vp)

        return {
            "bottom_half": bottom_half,
            "binary_gray_image": binary_gray_image,
            "edges": edges,
            "lines": lines,
            "line_eqs": line_eqs,
            "intersections": intersections,
            "relevant_intersections": relevant_intersections,
            "relevant_lines": relevant_lines,
            "cluster_labels": cluster_labels,
            "cluster_centers": cluster_centers,
            "vanishing_points": vanishing_points,
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
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                M = np.array([[x1, y1, 1], [x2, y2, 1]])
                line_eq = self.null_space(M)[:, 0]
                line_eqs.append(line_eq)
        return line_eqs

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
        if lines is not None:
            for point, (i, j) in relevant_intersections:
                relevant_lines.append(lines[i])
                relevant_lines.append(lines[j])
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
        print("\n--- Leyenda de Teclas ---")
        print("P: Iniciar/Pausar la secuencia de imágenes.")
        print("C: Mostrar/Ocultar contornos.")
        print("L: Mostrar/Ocultar líneas.")
        print("I: Mostrar/Ocultar intersecciones.")
        print("R: Mostrar/Ocultar intersecciones relevantes.")
        print("E: Mostrar/Ocultar líneas relevantes.")
        print("A: Mostrar/Ocultar cúmulos de intersecciones.")
        print("F: Mostrar/Ocultar puntos de fuga.")
        print("ESC: Salir.")

    def update_display(self, image: np.ndarray, processed_data: dict) -> np.ndarray:
        display_image = image.copy()
        height, width = display_image.shape[:2]
        center_x, center_y = width // 2, height // 2

        if self.show_info:
            y_offset = 20
            threshold_distance = 5
            # Show vanishing point coordinates
            for vp in processed_data["vanishing_points"]:
                text = f"VP: ({int(vp.x)}, {int(vp.y)})"
                cv2.putText(display_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20

                # Check if the VP is near the center of the image
                distance_to_center = np.sqrt((vp.x - center_x) ** 2 + (vp.y - center_y) ** 2)
                if distance_to_center < threshold_distance:
                    text = f"VP ({int(vp.x)}, {int(vp.y)}) is near to the image center"
                    cv2.putText(display_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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

        return display_image

    def create_trackbars(self):
        cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
        cv2.createButton("Pause", self.toggle_pause, None, cv2.QT_PUSH_BUTTON, 0)
        cv2.createTrackbar("Horizon Threshold", "Trackbars", self.hiper_params.relevant_intersections_horizon_threshold,
                           100, self.update_horizon_threshold)
        cv2.createTrackbar("Cluster Intersections", "Trackbars", self.hiper_params.cluster_n_intersections, 20,
                           self.update_cluster_intersections)
        cv2.createTrackbar("Cluster Distance Threshold", "Trackbars", self.hiper_params.distance_threshold, 500,
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

    def toggle_pause(self, *args):
        self.paused = not self.paused

    def update_threshold_image(self, value):
        self.hiper_params.threshold_image = value
        self.process_and_display_current_image()

    def update_distance_threshold(self, value):
        self.hiper_params.distance_threshold = value
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

    def process_and_display_current_image(self):
        if self.images:
            image_info = self.images[self.current_image_index]
            image = cv2.imread(image_info.image_path, cv2.IMREAD_COLOR)
            processed_data = self.process_image(image)
            self.update_display(image, processed_data)


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
        elif key == 52:  # Left arrow key
            print ("Key", key, chr(key))
            processor.current_image_index = (processor.current_image_index - 1) % len(processor.images)
        elif key == 54:  # Right arrow key
            print ("Key", key, chr(key))
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
