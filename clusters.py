import numpy as np
import cv2
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(260, 350)
        MainWindow.setMinimumSize(QtCore.QSize(260, 350))
        MainWindow.setMaximumSize(QtCore.QSize(260, 350))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 181, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(200, 10, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 40, 141, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(160, 40, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(140, 270, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.buttonClick)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 100, 181, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 70, 261, 21))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(10, 120, 241, 141))
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 270, 31, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(50, 270, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(140, 310, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.buttonClick_2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Кол-во классов и кластеров ="))
        self.label_2.setText(_translate("MainWindow", "Кол-во изображений ="))
        self.pushButton.setText(_translate("MainWindow", "Распределить"))
        self.label_3.setText(_translate("MainWindow", "Распределение по кластерам:"))
        self.label_4.setText(_translate("MainWindow", "КК ="))
        self.pushButton_2.setText(_translate("MainWindow", "Визуализировать"))

    def buttonClick(self):
        self.plainTextEdit.clear()
        self.result = Program(int(self.lineEdit.text()), int(self.lineEdit_2.text()))
        self.result.get_result()

    def buttonClick_2(self):
        self.result.visualization()

class Program():
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.true_labels = []
        self.image_paths = []
        self.num_clusters = self.n

    def get_database(self):
        for i in range(self.n):
            for j in range(self.m):
                self.true_labels.append(i) # Исходные метки классов для каждого изображения
                self.image_paths.append(f'faces94/s{i + 1}/{j + 1}.jpg')

    def preprocess_image(self, image):
        # Преобразование изображения в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Применение фильтра Гаусса для сглаживания изображения
        smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

        return smoothed


    def extract_face_features(self, image):
        # Предварительная обработка изображения
        preprocessed = self.preprocess_image(image)

        # Извлечение признаков с помощью метода LBPH
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(preprocessed, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_features = []
        for (x, y, w, h) in faces:
            roi = preprocessed[y:y + h, x:x + w]
            resized_roi = cv2.resize(roi, (100, 100))
            face_features.append(resized_roi.flatten())

        return np.array(face_features)


    def improve_clustering_quality(self):
        # Извлечение признаков из всех изображений
        all_features = []
        for image_path in self.image_paths:
            image = cv2.imread(image_path)
            features = self.extract_face_features(image)
            all_features.extend(features)

        all_features = np.array(all_features)

        # Применение алгоритма K-means для кластеризации
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(all_features)

        # Разделение изображений на кластеры
        clusters = [[] for _ in range(self.num_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append((self.image_paths[i], all_features[i]))

        # Рассчет расстояний между центрами классов и внутри классов
        class_centers = kmeans.cluster_centers_
        min_between = np.inf
        max_within = -np.inf

        for i in range(self.num_clusters):
            for j in range(i + 1, self.num_clusters):
                distance = np.linalg.norm(class_centers[i] - class_centers[j])
                if distance < min_between:
                    min_between = distance

            cluster = np.array([item[1] for item in clusters[i]])
            distances = np.linalg.norm(cluster - class_centers[i], axis=1)
            max_distance = np.max(distances)
            if max_distance > max_within:
                max_within = max_distance

        '''
        # Вычисление ARI
        predicted_labels = []
        for cluster in clusters:
            cluster_labels = [self.true_labels[self.image_paths.index(item[0])] for item in cluster]
            predicted_labels.extend(cluster_labels)
        '''

        quality = min_between / (2 * max_within)

        return clusters, quality

    def get_result(self):
        self.get_database()

        self.clusters, quality = self.improve_clustering_quality()

        # Вывод информации о принадлежности изображений к кластерам
        for i, cluster in enumerate(self.clusters):
            ui.plainTextEdit.appendPlainText(f'Cluster {i + 1}:')
            for item in cluster:
                image_path, _ = item
                ui.plainTextEdit.appendPlainText(f'Image: {image_path}')
            ui.plainTextEdit.appendPlainText('\n')

        # Вывод значения качества кластеризации
        ui.lineEdit_3.setText(str(round(quality, 1)))

    def visualization(self):
        # Визуализация кластеров в трехмерном пространстве
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, cluster in enumerate(self.clusters):
            cluster = np.array([item[1] for item in cluster])
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], label=f'Cluster {i + 1}')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.legend()

        plt.show()

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec_())

