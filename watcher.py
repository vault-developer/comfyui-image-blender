import sys
import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import subprocess

class MyHandler(PatternMatchingEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process = None
        self.restart_process()

    def restart_process(self):
        if self.process:
            self.process.terminate()
        self.process = subprocess.Popen([sys.executable, 'ComfyUI\\main.py'])

    def on_modified(self, event):
        self.restart_process()

if __name__ == "__main__":
    patterns = ["*.py"]
    ignore_patterns = None
    ignore_directories = False
    case_sensitive = True
    my_event_handler = MyHandler(patterns=patterns, ignore_patterns=ignore_patterns, ignore_directories=ignore_directories, case_sensitive=case_sensitive)

    path = ".\\custom_nodes\\image-enchanser"
    go_recursively = True
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)

    my_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        my_observer.stop()
    my_observer.join()