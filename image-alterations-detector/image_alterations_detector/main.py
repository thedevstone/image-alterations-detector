from image_alterations_detector.app.controller.controller import Controller
from image_alterations_detector.app.view.view import View

if __name__ == '__main__':
    controller = Controller()
    gui = View(controller)
    controller.set_ui(gui)
    controller.start()
