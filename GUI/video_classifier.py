import numpy as np
import torch

def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script

import pytorchvideo.models
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QFileDialog, QMessageBox, QLineEdit, QPushButton, QLabel, QListWidget, QTextBrowser, QProgressBar, QAction, QInputDialog, qApp
from pytorchvideo.data.encoded_video import EncodedVideo
from video_dataset import VideoClassificationData
from video_utils import *

VERSION = '0.0.1'


class UIDialog(object):

    def setupUi(self, InputDialog):
        self.LoadingMsgBox = QMessageBox()
        self.LoadingMsgBox.setWindowTitle("Video Classification Test Program (Version: {})".format(VERSION))

        InputDialog.setObjectName("InputDialog")
        InputDialog.setFixedSize(810, 654)

        self.InputPathLine = QLineEdit(InputDialog)
        self.InputPathLine.setGeometry(QtCore.QRect(10, 30, 345, 24))
        self.InputPathLine.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.InputPathLine.setReadOnly(True)
        self.InputPathLine.setObjectName("InputPathLine")
        self.InputSelectButton = QPushButton(InputDialog)
        self.InputSelectButton.setGeometry(QtCore.QRect(10 + 345 + 10, 30, 128, 24))
        self.InputSelectButton.setObjectName("InputSelectButton")

        self.OutputimageLabel = QLabel(InputDialog)
        self.OutputimageLabel.setGeometry(QtCore.QRect(10, 30 + 24 + 10, 480, 480))
        self.OutputimageLabel.setAcceptDrops(False)
        self.OutputimageLabel.setObjectName("OutputimageLabel")
        self.OutputimageLabel.setStyleSheet("background-color: black")

        self.pbar = QProgressBar(InputDialog)
        self.pbar.setGeometry(10, 30 + 24 + 10 + 480 + 10, 480, 18)
        self.pbar.setValue(0)

        self.ExtractButton = QPushButton(InputDialog)
        self.ExtractButton.setGeometry(QtCore.QRect(10, 30 + 24 + 10 + 480 + 10 + 18 + 10, 480, 62))
        self.ExtractButton.setObjectName("ExtractButton")

        self.LogLabel = QTextBrowser(InputDialog)
        self.LogLabel.setGeometry(QtCore.QRect(10 + 480 + 10, 30, 300, 514))
        self.LogLabel.setAcceptDrops(False)
        self.LogLabel.setObjectName("LogLabel")
        self.OutputListView = QListWidget(InputDialog)
        self.OutputListView.setGeometry(QtCore.QRect(10 + 480 + 10, 30 + 514 + 10, 300, 90))
        self.OutputListView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.OutputListView.setMouseTracking(False)
        self.OutputListView.setObjectName("OutputListView")

        self.retranslateUi(InputDialog)
        QtCore.QMetaObject.connectSlotsByName(InputDialog)

    def retranslateUi(self, InputDialog):
        _translate = QtCore.QCoreApplication.translate
        InputDialog.setWindowTitle(_translate("InputDialog", "Dialog"))
        self.InputSelectButton.setText(_translate("InputDialog", "?????? ??????"))
        self.ExtractButton.setText(_translate("InputDialog", "?????? ??????"))


class ExtractorDialog(QMainWindow, QDialog, UIDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("Video Classification Test Program (Version: {})".format(VERSION))

        self.LoadingMsgBox.show()
        self.write_log('Initializing...')

        self.model_path = './data/model.pth'
        self.label_path = './data/labels.csv'
        self.video_path = ''
        self.video_data = None
        self.video_frames = []
        self.extract_num = 5
        self.max_clips = 5
        self.clip_duration = 10
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'

        # Set GUI
        self.InputSelectButton.clicked.connect(self.onclick_input_button)
        self.ExtractButton.clicked.connect(self.onclick_extract_button)
        self.set_menu_bar()
        self.write_log('Interface initialized.')

        # Set Dataloader
        self.dataloader = VideoClassificationData()
        self.write_log('Data loader initialized.')

        # Load Model and Label Names
        self.model = torch.load(self.model_path)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.label_names = load_labels(self.label_path)
        self.write_log('Model initialized.')

        self.LoadingMsgBox.close()
        self.write_log('Initializing done.')

    def set_menu_bar(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('&??????')
        settingmenu = menubar.addMenu('&??????')
        helpmenu = menubar.addMenu('&?????????')

        exitAction = QAction('??????', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('??????????????? ???????????????.')
        exitAction.triggered.connect(qApp.quit)
        filemenu.addAction(exitAction)

        setMaxClipAction = QAction('Max Clip Setting', self)
        setMaxClipAction.setStatusTip('Max clip??? ???????????????.')
        setMaxClipAction.triggered.connect(self.SetMaxClipDialog)
        settingmenu.addAction(setMaxClipAction)

        setClipDurationAction = QAction('Clip Duration', self)
        setClipDurationAction.setStatusTip('Clip duration??? ???????????????.')
        setClipDurationAction.triggered.connect(self.SetClipDurationDialog)
        settingmenu.addAction(setClipDurationAction)

        infoAction = QAction('??????', self)
        infoAction.setShortcut('Ctrl+H')
        infoAction.setStatusTip('???????????? ???????????????.')
        infoAction.triggered.connect(self.InfoMsgBox)
        helpmenu.addAction(infoAction)

    def SetMaxClipDialog(self):
        MaxClipDialog = QInputDialog(None)
        MaxClipDialog.setInputMode(QInputDialog.TextInput)
        MaxClipDialog.setWindowTitle('Max Clip ??????')
        MaxClipDialog.setLabelText('Max Clip (1 ~ 30, Default: 5)')
        MaxClipDialog.setFixedSize(256, 128)
        MaxClipDialog.setIntRange(1, 30)
        MaxClipDialog.setIntValue(self.max_clips)
        ok = MaxClipDialog.exec_()
        input_num = MaxClipDialog.intValue()
        if ok:
            self.max_clips = int(input_num)
            self.write_log('Set Max Clip as {}'.format(input_num))
        else:
            self.write_log('Canceled Max Clip Setting')

    def SetClipDurationDialog(self):
        ClipDurationDialog = QInputDialog(None)
        ClipDurationDialog.setInputMode(QInputDialog.TextInput)
        ClipDurationDialog.setWindowTitle('Clip Duration ??????')
        ClipDurationDialog.setLabelText('Clip Duration (1 ~ 30, Default: 10)')
        ClipDurationDialog.setFixedSize(256, 128)
        ClipDurationDialog.setIntRange(1, 30)
        ClipDurationDialog.setIntValue(self.clip_duration)
        ok = ClipDurationDialog.exec_()
        input_num = ClipDurationDialog.intValue()
        if ok:
            self.clip_duration = int(input_num)
            self.write_log('Set Clip Duration as {}'.format(input_num))
        else:
            self.write_log('Canceled Clip Duration Setting')

    def InfoMsgBox(self):
        msgbox = QMessageBox(self)
        msgbox.setWindowTitle('Information')
        info = 'Video Classification Test Program (Version: {})\n\n' \
               '1. ??????\n' \
               ' ??? ??????????????? ????????? ?????????????????? ?????? ?????? ??????\n' \
               ' ???????????? ????????? ?????? ??? ??????????????? ?????? ?????????????????????.\n\n' \
               '2. ?????? ??????\n' \
               ' ScopVC_Tester.exe ??????\n\n' \
               '3. ?????? ??????\n' \
               ' 3.1 \'?????? ??????\' ?????? ?????? - ????????? ?????? ??????\n' \
               ' 3.2 \'?????? ??????\' ?????? ??????'.format(VERSION)
        msgbox.setText(info)
        msgbox.addButton(QPushButton('??????'), QMessageBox.RejectRole)
        msgbox.show()

    def warningMsgBox(self, msg):
        self.write_log('??????: {}'.format(msg))
        msgbox = QMessageBox(self)
        msgbox.setWindowTitle('??????')
        msgbox.setText(msg)
        msgbox.addButton(QPushButton('??????'), QMessageBox.RejectRole)
        msgbox.setFixedSize(256, 128)
        msgbox.show()

    def write_log(self, line):
        print(line)
        self.LogLabel.insertPlainText(str(line) + '\n')

    def setPixmap(self, p):
        self.OutputimageLabel.setPixmap(p)

    def setProgressBar(self, v):
        self.pbar.setValue(v)

    def setThumbnail(self, sec):
        thumbnail = self.video_data.get_clip(start_sec=sec, end_sec=sec + 1)['video']
        thumbnail = np.array(thumbnail.permute(1, 2, 3, 0)[0]).astype('uint8')
        thumbnail_pixmap = cv_to_pixmap(thumbnail, 480)
        self.setPixmap(thumbnail_pixmap)

    def onclick_input_button(self):
        try:
            self.video_path = QFileDialog.getOpenFileName(self, 'OpenFile')[0]
            self.video_data = EncodedVideo.from_path(self.video_path)
            self.InputPathLine.setText(self.video_path)
            self.OutputListView.clear()
            self.pbar.setValue(0)
            self.setThumbnail(0)
            self.write_log('Succeed to open file {}'.format(self.video_path))
        except Exception as e:
            self.warningMsgBox('Failed to open file {}'.format(self.video_path))

    def onclick_extract_button(self):
        self.write_log('Start label extraction')
        self.pbar.setValue(0)

        duration = max(self.video_data.duration, self.clip_duration)
        num_clips = min(int(duration / self.clip_duration), self.max_clips)
        start_clip = 0

        preds = torch.zeros((1, 400), device=self.device)
        for i in range(start_clip, num_clips):
            self.setThumbnail(i * self.clip_duration)
            self.update()

            clear_memory()

            input = self.dataloader(self.video_data, start_sec=i * self.clip_duration, end_sec=(i + 1) * self.clip_duration)
            input = [i.to(self.device)[None, ...] for i in input["video"]]
            preds += self.model(input) / num_clips

            self.pbar.setValue(int((i + 1) / (num_clips - start_clip) * 100))

        pred_classes = get_topk_classes(preds, topk=self.extract_num)
        pred_class_names = [self.label_names[int(i)] for i in pred_classes[0]]
        for pred_label in pred_class_names:
            self.OutputListView.addItem(pred_label)

        self.setThumbnail(0)
        self.pbar.setValue(100)
        self.write_log('Succeed: Check the output below.')


if __name__ == '__main__':
    app = QApplication([])
    diag = ExtractorDialog()
    diag.show()
    app.exec()
