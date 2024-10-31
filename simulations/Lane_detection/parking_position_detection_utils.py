import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


class ImageInfo:
    def __init__(self, image_path):
        self.image_path = image_path
        self.distances, self.angles, self.time = self.get_tags(image_path)

    def get_tags(self, image_path):
        distances = []
        angles = []
        time = 0
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


class ImageProcessing:
    def __init__(self, image):
        self.image = image

    def get_vanishing_points(self, print_intersections=False):
        bottom_half = self.cut_below_horizon(self.image)
        binary_gray_image = self.treshold_image(bottom_half)
        edges = self.canny_coutours(binary_gray_image)
        lines = self.get_hough_lines(edges)
        lineEqs = self.get_null_space_from_lines(lines)
        intersections, intLines = self.get_intersection(lines, lineEqs)
        centroids, labels = self.find_intersection_centroids(intersections)

        if print_intersections:
            self.print_intersections(intersections, intLines, lines, lineEqs)
        # self.draw_centroids(centroids, intersections)
        # self.draw_centroids_in_image(centroids, intersections, self.image)
        return centroids, intersections, labels

    def cut_below_horizon(self, image):
        height, width = image.shape
        half_width = height // 2
        bottom_half = image.copy()
        bottom_half[0:half_width, :] = 0
        return bottom_half

    def treshold_image(self, image, threshold=210):
        _, binary_gray_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary_gray_image

    def canny_coutours(self, image):
        edges = cv2.Canny(image, 100, 200)
        return edges

    def get_hough_lines(self, image):
        lines = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=40)
        return lines

    def get_null_space_from_lines(self, lines):
        lineEqs = np.zeros((len(lines), 3))
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                M = np.array([[x1, y1, 1], [x2, y2, 1]])
                lineEqs[i, :] = self.null_space(M)[:, 0]
        return lineEqs

    @staticmethod
    def null_space(A, rcond=None):
        u, s, vh = la.svd(A, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]
        if rcond is None:
            rcond = np.finfo(s.dtype).eps * max(M, N)
        tol = np.amax(s) * rcond
        num = np.sum(s > tol, dtype=int)
        Q = vh[num:, :].T.conj()
        return Q

    @staticmethod
    def areEqual(a, b, ord=7):
        if a == b:
            return True
        if a != 0. and b != 0.:
            val = np.abs((a - b) / max(np.abs([a, b])))
        else:
            val = max(np.abs([a, b]))
        return -np.log10(val) > ord

    def get_intersection(self, lines, lineEqs):
        nLines = len(lines)
        nComb = ((nLines * (nLines - 1)) // 2)
        pn = np.zeros((nComb, 3))
        intLines = np.zeros((nComb, 2), dtype='int32')
        idx = 0
        for i in range(nLines - 1):
            for j in range(i + 1, nLines):
                homoP = np.cross(lineEqs[i, :], lineEqs[j, :])
                if not self.areEqual(homoP[2], 0., 8):
                    if self.get_angle_between(lineEqs[i, :], lineEqs[j, :]) < 45:
                        print("angle=", self.get_angle_between(lineEqs[i, :], lineEqs[j, :]))
                        homoP /= homoP[2]
                        pn[idx, :] = homoP
                        intLines[idx, :] = [i, j]
                        idx += 1
        return pn[:idx, :], intLines[:idx, :]

    def get_angle_between(self, line1, line2):
        # dot_product = np.dot(line1, line2)
        # dot_product = line1[0] * line2[0] + line1[1] * line2[1] + line1[2] * line2[2]
        # theta_rad = np.arccos(dot_product)  # Aseguramos que cos_theta esté en el rango [-1, 1]
        # theta_deg = np.degrees(theta_rad)

        # Extraer A y B de cada recta
        A1, B1, _ = line1
        A2, B2, _ = line2

        # Calcular el coseno del ángulo entre las rectas
        cos_theta = abs(A1 * A2 + B1 * B2) / (np.sqrt(A1 ** 2 + B1 ** 2) * np.sqrt(A2 ** 2 + B2 ** 2))
        theta_rad = np.arccos(cos_theta)  # ángulo en radianes
        theta_deg = np.degrees(theta_rad)  # ángulo en grados

        print("line1=", line1)
        print("line2=", line2)
        print("theta_rad=", theta_rad)
        print("theta_deg=", theta_deg)

        return theta_deg

    def find_intersection_centroids(self, intersections):
        kmeans = KMeans(n_clusters=2, n_init='auto')
        kmeans.fit(intersections[:, :2])
        return kmeans.cluster_centers_, kmeans.labels_

    def draw_centroids(self, centroids, intersections):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(intersections[:, 0], intersections[:, 1], c='b')
        ax.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x', s=100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

    def draw_centroids_in_image(self, centroids, intersections, image):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(intersections[:, 0], intersections[:, 1], c='b')
        ax.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x', s=100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.imshow(image, cmap='gray')
        plt.show()

    @staticmethod
    def draw_centroids_in_image_CV2(centroids, intersections, image):
        # Dibuja las intersecciones en color azul
        for point in intersections:
            cv2.circle(image, (int(point[0]), int(point[1])), radius=5, color=(255, 0, 0), thickness=-1)

        # Dibuja los centroides en color rojo con una 'x'
        for point in centroids:
            cv2.drawMarker(image, (int(point[0]), int(point[1])), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=10, thickness=2)

        return image

    def print_intersections(self, intersections, intLines, lines, lineEqs):
        cv2.namedWindow("Intersections", cv2.WINDOW_NORMAL)
        for i in range(len(intersections)):
            locImage = self.image.copy()
            idx1, idx2 = intLines[i]
            print("idx1 = ", idx1)
            print("idx2 = ", idx2)
            point = intersections[i]
            x11, y11, x12, y12 = lines[idx1][0]
            x21, y21, x22, y22 = lines[idx2][0]

            angle = self.get_angle_between(lineEqs[idx1, :], lineEqs[idx2, :])
            cv2.putText(locImage, "Angle: %.2f" % angle, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.line(locImage, (x11, y11), (x12, y12), (0, 255, 0), 2)
            cv2.line(locImage, (x21, y21), (x22, y22), (0, 255, 0), 2)
            cv2.circle(locImage, (int(point[0]), int(point[1])), radius=5, color=(255, 0, 0), thickness=-1)
            cv2.imshow("Intersections", locImage)
            cv2.waitKey(0)
        cv2.destroyWindow("Intersections")


def load_tagged_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            route = os.path.join(directory, filename)
            images.append(ImageInfo(route))
    return images


def draw_centroids_in_image(centroids, intersections, labels, image):
    # Dibuja las intersecciones en color azul
    n = len(intersections)
    print("n=", n)
    print("labels.shape=", labels.shape)
    for idx in range(n):
        # print("labels[%d]=" % idx, labels[idx])
        point = intersections[idx]
        # print("point", point)
        if labels[idx] == 0:
            cv2.circle(image, (int(point[0]), int(point[1])), radius=5, color=(255, 0, 0), thickness=-1)
        else:
            cv2.circle(image, (int(point[0]), int(point[1])), radius=5, color=(0, 255, 0), thickness=-1)

    # Dibuja los centroides en color rojo con una 'x'
    for point in centroids:
        cv2.drawMarker(image, (int(point[0]), int(point[1])), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=10, thickness=2)

    return image


images_info = load_tagged_images('../parking_sequence')
images_info.sort(key=lambda x: x.time, reverse=False)

cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
for i in range(0):  # Repeat 1 time
    for idx in range(len(images_info)):
        image = cv2.imread(images_info[idx].image_path, cv2.IMREAD_COLOR)
        cv2.imshow("Images", image)
        cv2.waitKey(100)
cv2.destroyAllWindows()

cv2.namedWindow("Centroids", cv2.WINDOW_NORMAL)
for i in range(1):  # Repeat 1 time
    for idx in range(len(images_info)):
        image = cv2.imread(images_info[idx].image_path, cv2.IMREAD_COLOR)

        imageGray = cv2.imread(images_info[idx].image_path, cv2.IMREAD_GRAYSCALE)

        image_processing = ImageProcessing(imageGray)  # Instantiate the class

        centroids, intersections, labels = image_processing.get_vanishing_points(print_intersections=True)

        image_with_centroids = draw_centroids_in_image(centroids, intersections, labels, image.copy())

        cv2.imshow("Centroids", image_with_centroids)
        cv2.waitKey(1)

cv2.destroyAllWindows()
