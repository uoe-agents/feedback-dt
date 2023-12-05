from src.utils.argparsing import get_args
from src.utils.manual_control import DemoManualControl

if __name__ == "__main__":
    args = get_args()
    manual_control = DemoManualControl(args["demo_config"], args["demo_seed"], record=True, speed=0.8, save_log=True)
    manual_control.start()
