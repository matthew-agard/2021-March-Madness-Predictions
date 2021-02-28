import matplotlib.pyplot as plt


def line_plot(x, y, color, label, linewidth=3):
    plt.plot(x=x, y=y, color=color, linewidth=linewidth, label=label)

def vertical_bar_plot():
    #pass

def horizontal_bar_plot():
    #pass

def format_plot(title, xlabel, ylabel, yticks):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')