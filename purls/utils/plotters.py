import matplotlib.pyplot as plt
import numpy as np


class MultiprocessPlotter:
    def __init__(self, processes, env):
        self.processes = processes
        self.width = int(np.ceil(np.sqrt(self.processes)))
        self.heigth = self.width
        self.env = env
        self.closed = True

    def create_figure(self):
        self.fig, ax = plt.subplots(self.width, self.heigth)
        self.fig.canvas.draw()
        self.closed = False

        def handle_close(event):
            self.closed = True

        self.fig.canvas.mpl_connect("close_event", handle_close)
        self.handles = []
        for rgb_array, axis in zip(self.env.render(), ax.flatten()):
            self.handles.append(axis.imshow(rgb_array))
            axis.axis("off")

        self.closed = False

    def render(self, i):
        try:
            for rgb_array, handle in zip(self.env.render(), self.handles):
                handle.set_data(rgb_array)

            self.fig.canvas.set_window_title(f"frame {i}")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.000_000_000_001)
        except Exception:
            self.closed = True
