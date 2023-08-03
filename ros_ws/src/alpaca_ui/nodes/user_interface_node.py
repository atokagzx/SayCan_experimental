#! /usr/bin/env python3

from time import sleep
import rospy
import cv2
import traceback
import numpy as np
from threading import Thread
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap
from flask import Flask, request
import requests
from ui.saycan_ui import Ui_MainWindow
from prompt_tools.msg import Prompt, PromptMonitoring

class UserInterface:
    class MainWindow(QMainWindow, Ui_MainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setupUi(self)

    def __init__(self):
        self._app = QApplication([])
        self._window = self.MainWindow()
        self._user_task = ""
        self._window.microToggle.stateChanged.connect(self._on_micro_toggle)
        # QDialogButtonBox
        # self.promptConfirm.setStandardButtons(QtWidgets.QDialogButtonBox.Abort|QtWidgets.QDialogButtonBox.Ok)
        self._window.promptConfirm.accepted.connect(self._on_prompt_confirm)
        self._window.promptConfirm.rejected.connect(self._on_prompt_reject)
        self._window.cancelButton.clicked.connect(self._on_cancel_button)
        self._request_thread = Thread(target=self._request_thread_loop, daemon=True)
        self._request_thread.start()
        # self._window.setWindowTitle('SayCan demo')
        self._window.show()
        self._is_running = True

    def run(self):
        rospy.loginfo('running application')
        self._window.imageViewer.setPixmap(QPixmap.fromImage(QImage(640, 480, QImage.Format_RGB888)))
        self._app.exec()

    def set_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # rotate image
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self._window.imageViewer.setPixmap(QPixmap.fromImage(image))
    
    def set_actions(self, actions):
        if not self._is_running:
            self._window.scoredActions.setText(f"available actions:\n{actions}")
    
    def set_user_input(self, eng, rus):
        self._user_task = eng
        if self._window.microToggle.isChecked() and not self._is_running:
            rospy.loginfo(f'setting user input: "{rus}"')
            self._window.userInput.setText(rus)
            self._window.promptConfirm.setEnabled(True)
        else:
            rospy.loginfo(f'micro off, not setting user input: "{rus}"')
    
    def _on_micro_toggle(self, state):
        if state == Qt.Unchecked:
            self._window.promptConfirm.setEnabled(False)
            self._window.userInput.setText('')
    
    def _on_prompt_confirm(self):
        rospy.loginfo('prompt confirmed')
        scenario_id = int(self._window.scenarioIDSpin.value())
        self._window.promptConfirm.setEnabled(False)
        request = requests.post('http://localhost:5225/execute', json={'task': self._user_task, 'scenario_id': scenario_id})
        self._window.scenarioIDSpin.setValue(scenario_id + 1)

    def _on_prompt_reject(self):
        rospy.loginfo('prompt rejected')
        self._window.promptConfirm.setEnabled(False)
        self._window.userInput.setText('')
    
    def set_prompt_monitoring(self, prompt_monitoring):
        history_text = f"task: {prompt_monitoring['user_input']}\ndone actions:\n"
        history_text += "\n".join(prompt_monitoring['done_actions'])
        self._window.actionsHistory.setText(history_text)
        action_text = "\n".join(prompt_monitoring['actions'])
        self._window.scoredActions.setText(action_text)
    
    def _request_thread_loop(self):
        while True:
                # rospy.loginfo('requesting prompt monitoring')
            try:
                response = requests.get('http://localhost:5225/status')
            except requests.exceptions.ConnectionError as e:
                rospy.logerr(f"connection error: {e}")
                sleep(2)
            except Exception as e:
                rospy.logerr(f"exception: {e}")
                rospy.logerr(traceback.format_exc())
                sleep(2)
                continue
            # rospy.loginfo(f'got response: {response}')
            else:
                self._window.status.setText(response.json()['status'])
                code = response.json()['code']
                
                if code == 0:
                    # not running
                    self._is_running = False
                    self._window.microToggle.setEnabled(True)
                    self._window.cancelButton.setEnabled(False)
                elif code == 1:
                    # running
                    self._window.cancelButton.setEnabled(True)
                    self._window.microToggle.setEnabled(False)
                    self._is_running = True
                if code == 2:
                    # stopping
                    self._window.cancelButton.setEnabled(False)
                    self._window.microToggle.setEnabled(False)
                    self._is_running = True
                self._window.selectedAction.setText(response.json()['selected_action'])
            sleep(0.2)

    def _on_cancel_button(self):
        rospy.loginfo('canceling')
        try:
            response = requests.post('http://localhost:5225/force_stop')
            rospy.loginfo(f'got response: {response}')
        except requests.exceptions.ConnectionError as e:
            rospy.logerr(f"connection error: {e}")

    def close(self):
        self._app.quit()

class ROSConnector:
    def __init__(self):
        # subscribe to the image topic
        self._image = None
        self._bridge = CvBridge()
        self._thread = None
        self._subscribe()
        self._image_callback = None
        self._prompt_callback = None
        self._prompt_monitoring_callback = None
    
    def _subscribe(self):
        rospy.Subscriber("/alpaca/detector/camera/detected", Image, self._image_cb, queue_size=2)
        rospy.Subscriber("/alpaca/prompt/actions", Prompt, self._prompt_cb, queue_size=2)
        rospy.Subscriber("/alpaca/prompt/monitoring", PromptMonitoring, self._prompt_monitoring_cb, queue_size=2)

    def _image_cb(self, msg):
        color = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self._image = color
        if self._image_callback is not None:
            self._image_callback(color)
    
    def _prompt_cb(self, msg):
        if self._prompt_callback is not None:
            self._prompt_callback("\n".join(msg.actions))

    def _prompt_monitoring_cb(self, msg):
        if self._prompt_monitoring_callback is not None:
            if sum (msg.probabilities) > 0:
                # actions_text = [f"{action} ({prob})" for action, prob in zip(msg.actions, msg.probabilities)]
                # actions_dict = {action: prob for action, prob in zip(msg.actions, msg.probabilities)}
                #sort by probability
                # format prob to 2 decimal places
                actions_dict = {action: prob for action, prob in zip(msg.actions, msg.probabilities)}
                actions_dict = {action: f"{prob:.3f}" for action, prob in sorted(actions_dict.items(), key=lambda item: item[1], reverse=True)}
                # actions_text = [f"{actions_dict[action]} - {action}" for action in sorted(actions_dict, key=actions_dict.get, reverse=True)]
                actions_text = [f"{score} - {action}" for action, score in actions_dict.items()]
            else:
                actions_text = [f"{action}" for action in msg.actions]
            self._prompt_monitoring_callback({
                'user_input': msg.user_prompt,
                'actions': actions_text,
                'done_actions': msg.done_actions,
            })
    def set_callbacks(self, image = None, prompt = None, prompt_monitoring = None):
        self._image_callback = image
        self._prompt_callback = prompt
        self._prompt_monitoring_callback = prompt_monitoring

    def run(self):
        rospy.loginfo('running ros connector')
        self._thread = Thread(target=rospy.spin, daemon=True)

class RESTApplication:
    def __init__(self, port):
        self._port = port
        self._app = Flask(__name__)
        self._configure_endpoints()
        self._thread = None
        self._executor = None
    
    def _configure_endpoints(self):
        self._app.add_url_rule("/execute", methods=["POST"], view_func=self._execute)

    def _execute(self):
        task, rus = request.json["task"], request.json["rus"]
        rospy.loginfo(f'received task: "{task}"')
        if self._executor:
            self._executor(task, rus)
        else:
            rospy.logwarn("executor not set")
        return "OK", 200
    
    def set_callbacks(self, executor = None):
        self._executor = executor

    def run(self):
        self._flask_thread = Thread(target=self._app.run, daemon=True, name="flask_thread", args=(), kwargs={"host": "0.0.0.0", "port": self._port})  
        self._flask_thread.start()
    
if __name__ == "__main__":
    rospy.init_node('user_interface_node')
    ros_connector = ROSConnector()
    app = UserInterface()
    rest_app = RESTApplication(5000)
    rospy.on_shutdown(app.close)
    ros_connector.run()
    rest_app.run()
    ros_connector.set_callbacks(**{
        'image': app.set_image,
        'prompt': app.set_actions,
        'prompt_monitoring': app.set_prompt_monitoring})
    rest_app.set_callbacks(**{
        'executor': app.set_user_input})
    app.run()
