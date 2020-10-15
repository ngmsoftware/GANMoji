import sys, time

import numpy as np

from PyQt5.QtCore import pyqtSlot, Qt, QRect, QPoint, QSize, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QFrame, QVBoxLayout
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont, QColor
 
import tensorflow as tf
import numpy as np

randn = np.random.randn
rand = np.random.rand




#
# Canvas custom widget
#
#  The drawing are made in paintEvent
#
class Canvas(QWidget):
    def __init__(self, parent):
        super().__init__()

        # this is the main "screen buffer"
        self.mainPixmap = None

        # remember where you are placed (to get stuff from the window or frame that the canvas is on)
        self.parent = parent

        # call to selt itself up (set self size, etc...)
        self.initUI()



    #
    # This is where you set yourself 
    #
    #   if the the parent widget change its geometry, this has to e called again
    def initUI(self):
        w = self.parent.geometry().width()
        h = self.parent.geometry().height()


        # set myself size
        self.setGeometry(0, 0, w, h)

        # adjust the screen buffer with the new size
        self.mainPixmap = QPixmap('emojis/1-1.png')


        # image = self.pixmap().toImage()
        # w, h = image.width(), image.height()
        
        # # Get our target color from origin.
        # target_color = image.pixel(x,y)




    #
    # Where the magic happens!!!
    #
    def paintEvent(self, event):


        w = self.window().width()
        h = self.window().height()

        # main painter the first QPainter that is instanciated here will reflect the screen
        # draws can be made direct here but they will be erased every time.
        # to avoid that we allocate a second painter with the main buffer (mainPixmap) that always 
        # contain the current draws
        mainPainter = QPainter(self)

        rMin = -3.0
        rMax = 3.0

        f1 = rMin + (rMax-rMin)*self.parent.feature1Slider.value()/100.0
        f2 = rMin + (rMax-rMin)*self.parent.feature2Slider.value()/100.0
        f3 = rMin + (rMax-rMin)*self.parent.feature3Slider.value()/100.0
        f4 = rMin + (rMax-rMin)*self.parent.feature4Slider.value()/100.0
        f5 = rMin + (rMax-rMin)*self.parent.feature5Slider.value()/100.0
        f6 = rMin + (rMax-rMin)*self.parent.feature6Slider.value()/100.0
        f7 = rMin + (rMax-rMin)*self.parent.feature7Slider.value()/100.0
        f8 = rMin + (rMax-rMin)*self.parent.feature8Slider.value()/100.0


        I = self.parent.gamojiModel.predict( np.array([f1, f2, f3, f4, f5, f6, f7, f8]).reshape(1,8)  )[0]
        
        image = self.mainPixmap.toImage()
        wI, hI = image.width(), image.height()

        for j in range(hI):
            scanLine = image.scanLine(j).asarray(4*wI)
            for i in range(wI):
                if I[j][i][0] > 1.0:
                    I[j][i][0] = 1.0
                if I[j][i][1] > 1.0:
                    I[j][i][1] = 1.0
                if I[j][i][2] > 1.0:
                    I[j][i][2] = 1.0
                if I[j][i][0] < 0.0:
                    I[j][i][0] = 0.0
                if I[j][i][1] < 0.0:
                    I[j][i][1] = 0.0
                if I[j][i][2] < 0.0:
                    I[j][i][2] = 0.0
                    
                    
                scanLine[4*i+0] = int(255*I[j][i][2])
                scanLine[4*i+1] = int(255*I[j][i][1])
                scanLine[4*i+2] = int(255*I[j][i][0])



        self.mainPixmap.fromImage(image)


        # draw the main buffer in the first allocated painter (so it will be shown in the actual screen)
        mainPainter.drawImage(QRect(0, 0, w, w), image)




    def mouseMoveEvent(self, e):
        
        print(e.x(),  e.y())





#
# main window
#
class Window(QDialog):
    def __init__(self, *args):
        super(Window, self).__init__(*args)
        
        # load UI file
        #
        # In the ui, the names of the widgets becomes instance opbjects. For insntace
        # self.pushButton is defined in the .ui file
        #
        loadUi('gui.ui', self)

        # connect event in the .ui defined element to a method that can be called
        self.feature1Slider.valueChanged.connect(self.sliderChanged)
        self.feature2Slider.valueChanged.connect(self.sliderChanged)
        self.feature3Slider.valueChanged.connect(self.sliderChanged)
        self.feature4Slider.valueChanged.connect(self.sliderChanged)
        self.feature5Slider.valueChanged.connect(self.sliderChanged)
        self.feature6Slider.valueChanged.connect(self.sliderChanged)
        self.feature7Slider.valueChanged.connect(self.sliderChanged)
        self.feature8Slider.valueChanged.connect(self.sliderChanged)

        self.zeroButton.clicked.connect(self.zeroButtonClicked)
        self.randomButton.clicked.connect(self.randomButtonClicked)


        # create our custon canvas and add to the app window. Geometry will be defined by
        # the widget itself (see Canvas.InitUI). Pass this window widget to be set as parent
        # in the canvas __init__
        self.canvas = Canvas(self)
        self.layout.addWidget(self.canvas)



        self.setFixedSize(self.width(), self.height())



        # tensorflow trained model:
        self.gamojiModel = tf.keras.models.load_model('saved/modelGenerator')



    #
    # KeyPress function (this ine is automatically connected with the app window)
    #
    def keyPressEvent(self, e):
        self.canvas.keyPressEvent(e)
        if e.key() == Qt.Key_Escape:
            self.close()


    #
    # pushButton_clicked function (NOT automaticalle connected)
    #
    def sliderChanged(self):
        self.update()
        

    def zeroButtonClicked(self):
        self.feature1Slider.setValue(50)
        self.feature2Slider.setValue(50)
        self.feature3Slider.setValue(50)
        self.feature4Slider.setValue(50)
        self.feature5Slider.setValue(50)
        self.feature6Slider.setValue(50)
        self.feature7Slider.setValue(50)
        self.feature8Slider.setValue(50)


    def randomButtonClicked(self):
        self.feature1Slider.setValue(int(50+randn()*15))
        self.feature2Slider.setValue(int(50+randn()*15))
        self.feature3Slider.setValue(int(50+randn()*15))
        self.feature4Slider.setValue(int(50+randn()*15))
        self.feature5Slider.setValue(int(50+randn()*15))
        self.feature6Slider.setValue(int(50+randn()*15))
        self.feature7Slider.setValue(int(50+randn()*15))
        self.feature8Slider.setValue(int(50+randn()*15))






app = QApplication(sys.argv)
widget = Window()
widget.show()
app.exec_()
