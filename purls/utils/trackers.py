from .logs import debug


class TensorboardTracker:
    def __init__(self, writer):
        self.writer = writer

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.writer is not None:
            self.writer.close()

    def push(self, update, debug_message, **kwargs):
        if self.writer is not None:
            for key, value in kwargs.items():
                self.writer.add_scalar(key, value, update)
        debug(debug_message)
