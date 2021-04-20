from image_alterations_detector.app.controller import Controller
from image_alterations_detector.app.gui.gui import Gui

if __name__ == '__main__':
    controller = Controller()
    gui = Gui(controller)
    controller.set_ui(gui)
    controller.start()
