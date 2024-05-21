import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as co
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from filter_math import *

def plot_3D_signal(signals, name, fignum=1):
    """
    Plots multiple 3D signals together

    Parameters:
        signals: list of tuples with signal (x, y, z, "name")
        name: figure title
    """
    fig = plt.figure(fignum, figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for (xs, ys, zs, label) in signals:
        ax.plot(xs, ys, zs, lw=0.5, label=label)
    ax.set_title(name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    leg = ax.legend()
    # plt.show()
    return leg

def plot_3D_signal_slider(signals, name, fignum=1):
    """
    Plots multiple 3D signals together with a slider to control time

    Parameters:
        signals: list of tuples with signal (x, y, z, "name")
        name: figure title
    """
    fig = plt.figure(fignum, figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    slider_ax = fig.add_axes([0.1, 0.02, 0.8, 0.03])  # Position of the slider

    lines = []  # Store the plotted lines

    for (xs, ys, zs, label) in signals:
        line, = ax.plot(xs, ys, zs, lw=0.5, label=label)
        lines.append(line)
    ax.set_title(name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    leg = ax.legend()

    time_slider = Slider(slider_ax, 'Time', 0, len(signals[0][0])-1, valinit=0)  # Initialize slider

    def update(val):
        time_index = int(time_slider.val)
        for line, (xs, ys, zs, label) in zip(lines, signals):
            line.set_data(xs[:time_index+1], ys[:time_index+1])
            line.set_3d_properties(zs[:time_index+1])
        ax.set_title(name)
        fig.canvas.draw_idle()  # Redraw the plot

    time_slider.on_changed(update)
    
    return fig, time_slider, leg

def plot_signals_ensemble(*args, title="Signal Plot", xlabel="Time", ylabel="Amplitude", fignum=3):
    """
    Plots multiple signals on the same figure

    Parameters:
        *args: variable list of tuple (signal, "name")
        title: Title of the plot
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
    """

    plt.figure(fignum)
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (signal, name) in enumerate(args):
        rgb_color = co.hsv_to_rgb([i / len(args), 1, 1])
        ax.plot(signal, label=name, color=rgb_color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    leg = ax.legend()

    plt.grid(True)
    # plt.show()
    return fig, leg


def plot_signals(*args, title="Signal Plot", xlabel="Time", fignum=2):
    """
    Plots multiple signals on subplots

    Parameters:
        *args: variable list of tuple ([signalX1, signalX2, ...], ["name1", "name2", ...], "title") ...
        title: Title of the plot
        xlabel: Label for the x-axis
    """

    # plt.figure(fignum)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    legs = []

    for i, (signal_list, names, title) in enumerate(args):
        for j, signal in enumerate(signal_list):
            rgb_color = co.hsv_to_rgb([j / len(signal_list), 1, 1])
            axes[i].plot(signal, label=names[j], color=rgb_color)
        axes[i].set_title(title + " Component")
        axes[i].set_ylabel(title)
        legs.append(axes[i].legend())

    axes[-1].set_xlabel(xlabel)

    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    return legs

def plot_rmse(ground_truth, *args, title="Signal RMSE Plot", xlabel="Time", ylabel="Amplitude RMSE", fignum=5):
    """
    Plots the RMSE for multiple signals on the same figure

    Parameters:
        ground_truth: the ground truth signal
        *args: variable list of tuple (signal, "name")
        title: Title of the plot
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
    """

    plt.figure(fignum)
    fig, ax = plt.subplots(figsize=(12, 8))
    ground_truth = np.array(ground_truth)

    for i, (signal, name) in enumerate(args):
        signal = np.array(signal)
        assert ground_truth.shape == signal.shape
        rmse = calc_rmse(ground_truth, signal)
        rgb_color = co.hsv_to_rgb([i / len(args), 1, 1])
        ax.plot(rmse, label=(name + " RMSE"), color=rgb_color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    leg = ax.legend()

    plt.grid(True)
    # plt.show()
    return fig, leg


def plot_asymptotic_matrix_spectral_norm(*args, title="Signal RMSE Plot", xlabel="Time", ylabel="Amplitude RMSE", fignum=5):
    """
    Plots spectral norm of matrix over timestep on same figure

    Parameters:
        *args: variable list of tuple (matrix_signal, "name") - note matrix_signal should be (NxNxM)
        title: Title of the plot
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
    """

    plt.figure(fignum)
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (matrices, name) in enumerate(args):
        M = matrices.shape[2]
        signal = np.empty(M)
        for i in range(M):
            signal[i] = np.linalg.norm(matrices[:, :, i], ord=2)
        rgb_color = co.hsv_to_rgb([i / len(args), 1, 1])
        ax.plot(signal, label=name, color=rgb_color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    leg = ax.legend()

    plt.grid(True)
    # plt.show()
    return fig, leg


def plot_asymptotic_matrix_max_eigen(*args, title="Signal RMSE Plot", xlabel="Time", ylabel="Amplitude RMSE", fignum=5):
    """
    Plots spectral norm of matrix over timestep on same figure

    Parameters:
        *args: variable list of tuple (matrix_signal, "name") - note matrix_signal should be (NxNxM)
        title: Title of the plot
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
    """

    plt.figure(fignum)
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (matrices, name) in enumerate(args):
        M = matrices.shape[2]
        signal = np.empty(M)
        for i in range(M):
            signal[i] = calc_max_eigenval(matrices[:, :, i])
        rgb_color = co.hsv_to_rgb([i / len(args), 1, 1])
        ax.plot(signal, label=name, color=rgb_color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    leg = ax.legend()

    plt.grid(True)
    # plt.show()
    return fig, leg


#-------------------------------------Code From External Sources--------------------------------------

"""
The following class was code found on stack overflow attributed to @Basj
link:
https://stackoverflow.com/a/64752658
"""
class InteractiveLegend(object):
    def __init__(self, legend=None):
        if legend == None:
            legend = plt.gca().get_legend()
        self.legend = legend
        self.fig = legend.axes.figure
        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()
        self.update()
    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))
        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist
        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))
        return lookup_artist, lookup_handle
    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()
    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return
        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()
    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()