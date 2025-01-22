#! /usr/bin/env python3
# encoding: utf-8

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

Paleta = np.load("Paleta.npy")


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

    def get_vanishing_points(self, n=0, n_clusters=1, print_intersections=False):
        bottom_half = self.cut_below_horizon(self.image)
        binary_gray_image = self.treshold_image(bottom_half)
        edges = self.canny_coutours(binary_gray_image)
        lines = self.get_hough_lines(edges)
        lineEqs = self.get_null_space_from_lines(lines)
        intersections, intLines, IntersectionsInf, intLineInf = self.get_intersection(lines, lineEqs)
        print("IntersectionsInf:",IntersectionsInf)

        # fileName="Intersecciones_%05d" % n
        # np.savez(fileName, lines=lines, lineEqs=lineEqs, intersections=intersections, intLines=intLines)

        centroids, labels = self.find_intersection_centroids(intersections, n_clusters)

        if print_intersections:
            self.print_intersections(intersections, intLines, lines, lineEqs)
        # self.draw_centroids(centroids, intersections)
        # self.draw_centroids_in_image(centroids, intersections, self.image)
        return centroids, intersections, labels, IntersectionsInf

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
        # rho: Resolución del parámetro de distancia en píxeles.
        # theta: Resolución del parámetro de ángulo en radianes np.pi / 180 = 1 grado
        # threshold: Número mínimo de intersecciones en el espacio de parámetros para considerar una línea.
        # minLineLength: Longitud mínima de la línea que se detectará.
        # maxLineGap: Distancia máxima entre segmentos de línea que aún se considerarán parte de la misma línea.
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
        pn =  np.zeros((nComb, 3)) #Aqui almacenamos las lineas que no estan en el infinito
        pInf = np.zeros((nComb, 3)) #Aqui almacenamos las lineas que si estan en el infinito
        intLines = np.zeros((nComb, 2), dtype='int32')
        intLinesInf = np.zeros((nComb, 2), dtype='int32')
        idx  = 0
        idxInf = 0
        for i in range(nLines - 1):
            for j in range(i + 1, nLines):
                homoP = np.cross(lineEqs[i, :], lineEqs[j, :])
                if not self.areEqual(homoP[2], 0., 8):
                    homoP /= homoP[2]
                    pn[idx, :] = homoP
                    intLines[idx, :] = [i, j]
                    idx += 1
                else:
                    pInf[idxInf, :] = homoP
                    intLinesInf[idx, :] = [i, j]
                    idxInf += 1

        return pn[:idx, :], intLines[:idx, :], pInf[:idxInf, :], intLinesInf[:idxInf, :]

    def find_intersection_centroids(self, intersections, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
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

            angle = get_angle_between(lineEqs[idx1, :], lineEqs[idx2, :])
            cv2.putText(locImage, "Angle: %.2f" % angle, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.line(locImage, (x11, y11), (x12, y12), (0, 255, 0), 2)
            cv2.line(locImage, (x21, y21), (x22, y22), (0, 255, 0), 2)
            print("Lines = ", lines[idx1][0], lines[idx2][0])
            print("LineEqs = ", lineEqs[idx1, :], lineEqs[idx2, :])
            cv2.circle(locImage, (int(point[0]), int(point[1])), radius=5, color=(255, 0, 0), thickness=-1)
            cv2.imshow("Intersections", locImage)
            key = cv2.waitKey(0)
            if key == 27:
                break
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
        cv2.circle(image, (int(point[0]), int(point[1])), radius=2, color=Paleta[labels[idx], :].tolist(), thickness=-1)

    # Dibuja los centroides en color rojo con una 'x'
    for point in centroids:
        cv2.drawMarker(image, (int(point[0]), int(point[1])), color=(255, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=20, thickness=4)

    return image

def get_angle_between(line1, line2):
    # dot_product = np.dot(line1, line2)
    # dot_product = line1[0] * line2[0] + line1[1] * line2[1] + line1[2] * line2[2]
    # theta_rad = np.arccos(dot_product)  # Aseguramos que cos_theta esté en el rango [-1, 1]
    # theta_deg = np.degrees(theta_rad)

    # Extraer A y B de cada recta
    A1, B1, C1 = line1
    A2, B2, C2 = line2

    # Calcular el coseno del ángulo entre las rectas
    cos_theta = abs(A1 * A2 + B1 * B2 + C1 * C2) / (
            np.sqrt(A1 ** 2 + B1 ** 2 + C1 ** 2) * np.sqrt(A2 ** 2 + B2 ** 2 + C2 ** 2))
    theta_rad = np.arccos(cos_theta)  # ángulo en radianes
    theta_deg = np.degrees(theta_rad)  # ángulo en grados

    print("line1=", line1)
    print("line2=", line2)
    print("theta_rad=", theta_rad)
    print("theta_deg=", theta_deg)

    return theta_deg

def count_Clusters(n, labels):
    cont = np.zeros(n)
    for idx in range(n):
        cont[idx] = np.sum(labels == idx)
    return cont


if __name__ == "__main__":
    images_info = load_tagged_images('../manual_sequence/sec3/')
    # images_info = load_tagged_images('../parking_sequence/')
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

            centroids, intersections, labels, IntersectionsInf = image_processing.get_vanishing_points(
                idx,
                n_clusters=50,
                print_intersections=False
            )



            nInter, _ = intersections.shape

            dt = [('x', 'float'), ('y', float), ('w', float), ('label', 'S5')]
            interInfo=np.zeros((nInter,), dtype=dt)
            for i in range(nInter):
                interInfo[i][0] = intersections[i,0]
                interInfo[i][1] = intersections[i,1]
                interInfo[i][2] = intersections[i,2]
                interInfo[i][3] = "%05d" % labels[i]

            interInfo = np.sort(interInfo, order='label')
            print (interInfo)


            print(centroids.shape, intersections.shape, labels.shape)
            clustersSize = count_Clusters(len(centroids), labels)

            idxMax = np.argmax(clustersSize)



            cv2.drawMarker(image, (int(centroids[idxMax, 0]), int(centroids[idxMax][1])), color=(0, 0, 0),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=60, thickness=8)

            print(clustersSize[idxMax])

            fig1 = plt.figure(1)
            ax = fig1.add_subplot(111)
            ax.plot(IntersectionsInf[:,0],IntersectionsInf[:,1], '.')



            image_with_centroids = draw_centroids_in_image(centroids, intersections, labels, image.copy())

            cv2.imshow("Centroids", image_with_centroids)

            key = cv2.waitKeyEx  (0)
            print ("key = ",(key, int(key)))
            if chr(key) == 'a':
                plt.show()
            if key == 27:
                break

    cv2.destroyAllWindows()
