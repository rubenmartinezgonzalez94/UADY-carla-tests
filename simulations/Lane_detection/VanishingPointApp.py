from typing import List, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering
import os
import cv2
import numpy as np


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

class vanishingPoint:
   def __init__(self, x, y, linesIdx, lines):
      self.x, self.y = x, y
      self.linesIdex = linesIdx
      self.lines = lines
      self.w=[]

      #Calculamos los pesos de cada linea en función de su longitud
      for l in lines[:]:
         self.w.append(l[3])
      sw = sum(self.w)
      for i in range(len(self.w)):
         self.w[i] /= sw

      # Calculamos la matriz de productos externos ponderados.
      M=np.zeros((3,3))
      i=0
      for l in self.lines:
         M += self.w[i] * np.outer(l[:3],l[:3])
         i += 1

      # Definimos el Punto de Fuga como el eigenvector asociado al eigenvalor
      # menor de la matriz acumuladora de productos exteriores.
      [l,V]=np.linalg.eig(M)
      mn = min(l)
      idxMin = l.tolist().index(mn)
      V[:,idxMin] /= V[2,idxMin]
      self.x = V[0, idxMin]
      self.y = V[1, idxMin]

   def __repr__(self):
      s = "VP=(%06.3f, %06.3f)\n" % (int(self.x), int(self.y))
      i = 0
      for l in self.lines[:]:
         s += "w=%f, line: %fx+%fy+%f=0\n" % (self.w[i], l[0], l[1], l[2])
         i += 1
      return s
   def __str__(self):
      s = "VP=(%f, %06.3f)\n" % (int(self.x), int(self.y))
      i = 0
      for l in self.lines[:]:
         s += "w=%f, line: %fx+%fy+%f=0\n" % (self.w[i], l[0], l[1], l[2])
         i += 1
      return s

class ImageProcessor:
    def __init__(self):
        self.images: List[ImageInfo] = []
        self.current_image_index: int = 0
        self.paused: bool = False
        self.show_contours: bool = False
        self.show_lines: bool = False
        self.show_intersections: bool = False
        self.show_relevant_intersections: bool = False
        self.show_clusters: bool = False
        self.show_vanishing_points: bool = False

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
        _, binary_gray_image = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY)

        # Step 3: Detect edges using Canny
        edges = cv2.Canny(binary_gray_image, 100, 200)

        # Step 4: Detect lines using Hough Transform
        lines = self.detect_lines(edges)

        # Step 5: Compute line equations
        line_eqs = self.compute_line_equations(lines)

        # Step 6: Compute intersections between lines
        intersections = self.compute_intersections(lines, line_eqs)

        # Step 7: Filter relevant intersections (near the horizon)
        relevant_intersections = self.filter_relevant_intersections(intersections, height)

        # Step 8: Cluster relevant intersections
        cluster_labels, cluster_centers = self.cluster_intersections(relevant_intersections, n_clusters=2)

        # Step 9: Compute vanishing points for each cluster
        vanishing_points = []

        return {
            "bottom_half": bottom_half,
            "binary_gray_image": binary_gray_image,
            "edges": edges,
            "lines": lines,
            "line_eqs": line_eqs,
            "intersections": intersections,
            "relevant_intersections": relevant_intersections,
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

    def compute_intersections(self, lines: np.ndarray, line_eqs: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Computes the intersections between lines.
        """
        intersections = []
        if lines is not None:
            for i in range(len(lines) - 1):
                for j in range(i + 1, len(lines)):
                    homo_p = np.cross(line_eqs[i], line_eqs[j])
                    if not self.are_equal(homo_p[2], 0.0, 8):
                        homo_p /= homo_p[2]
                        intersections.append((int(homo_p[0]), int(homo_p[1])))
        return intersections

    def filter_relevant_intersections(self, intersections: List[Tuple[int, int]], image_height: int) -> List[Tuple[int, int]]:
        """
        Filters intersections to keep only those near the horizon.
        """
        horizon_threshold = 5  # Distance in pixels from the horizon line
        horizon_line = image_height // 2  # Middle of the image (horizon approximation)
        relevant_intersections = []

        for point in intersections:
            x, y = point
            if abs(y - horizon_line) < horizon_threshold:
                relevant_intersections.append(point)

        return relevant_intersections

    def cluster_intersections(self, intersections: List[Tuple[int, int]], n_clusters: int = 2) -> Tuple[np.ndarray, np.ndarray]:
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
            n_clusters=2,
            compute_full_tree=True,
            metric='euclidean',
            linkage='ward'
            #,distance_threshold=2
        )
        labels = clustering.fit_predict(points)

        # Compute cluster centers
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
        print("A: Mostrar/Ocultar cúmulos de intersecciones.")
        print("F: Mostrar/Ocultar puntos de fuga.")
        print("ESC: Salir.")

    def update_display(self, image: np.ndarray, processed_data: dict) -> np.ndarray:
        display_image = image.copy()

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
            for point in processed_data["intersections"]:
                cv2.circle(display_image, point, 5, (255, 0, 0), -1)

        # Show relevant intersections
        if self.show_relevant_intersections:
            for point in processed_data["relevant_intersections"]:
                cv2.circle(display_image, point, 5, (0, 255, 255), -1)  # Yellow color for relevant intersections

        # Show clusters
        if hasattr(self, 'show_clusters') and self.show_clusters:
            cluster_labels = processed_data["cluster_labels"]
            cluster_centers = processed_data["cluster_centers"]
            relevant_intersections = processed_data["relevant_intersections"]

            if len(cluster_labels) > 0 and len(relevant_intersections) > 0:
                # Define colors for clusters
                colors = [(0, 255, 0), (0, 0, 255)]  # Green and Red for two clusters

                # Draw each cluster with a different color
                for i, point in enumerate(relevant_intersections):
                    cluster_id = cluster_labels[i]
                    color = colors[cluster_id % len(colors)]
                    cv2.circle(display_image, point, 5, color, -1)

                # Draw cluster centers
                for center in cluster_centers:
                    cv2.drawMarker(display_image, (int(center[0]), int(center[1])), (255, 255, 255), cv2.MARKER_CROSS,
                                   30, 5)
        # Show vanishing points
        if hasattr(self, 'show_vanishing_points') and self.show_vanishing_points:
            print('vanishing points not implemented yet')

        return display_image


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


def main():
    # Create an instance of ImageProcessor
    processor = ImageProcessor()

    # Load images from the directory
    processor.load_images('../manual_sequence/sec4/')

    # Display the legend
    processor.show_legend()

    # Create an OpenCV window to capture keys
    cv2.namedWindow("Image Sequence", cv2.WINDOW_NORMAL)

    # Main loop to interact with the options
    while True:
        # Get the current image
        image_info = processor.images[processor.current_image_index]
        image = cv2.imread(image_info.image_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Error: Unable to load image {image_info.image_path}.")
            continue

        # Process the image
        processed_data = processor.process_image(image)

        # Update the display image with the current options
        display_image = processor.update_display(image, processed_data)

        # Show the image
        cv2.imshow("Image Sequence", display_image)

        # Wait for a key press
        key = cv2.waitKey(30) & 0xFF

        # Handle key presses
        if key == ord('p'):  # P: Pause/Resume the sequence
            processor.paused = not processor.paused
        elif key == ord('c'):  # C: Toggle contours
            processor.show_contours = not processor.show_contours
        elif key == ord('l'):  # L: Toggle lines
            processor.show_lines = not processor.show_lines
        elif key == ord('i'):  # I: Toggle intersections
            processor.show_intersections = not processor.show_intersections
        elif key == ord('r'):  # R: Toggle relevant intersections
            processor.show_relevant_intersections = not processor.show_relevant_intersections
        elif key == ord('a'):  # A: Toggle clusters
            processor.show_clusters = not processor.show_clusters
        elif key == ord('f'):  # F: Toggle vanishing points
            processor.show_vanishing_points = not processor.show_vanishing_points
        elif key == 27:  # ESC: Exit
            break

        # Move to the next image only if not paused
        if not processor.paused:
            processor.current_image_index = (processor.current_image_index + 1) % len(processor.images)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()