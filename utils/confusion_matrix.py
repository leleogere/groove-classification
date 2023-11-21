import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, class_names, minimum=None, maximum=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cm: Confusion matrix
    :param class_names: List of classes
    :param minimum: Minimum value for the color map
    :param maximum: Maximum value for the color map
    :param cmap: Color map to use
    """

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    fmt = '.2f' if cm.dtype == np.float64 else 'd'
    for true, pred in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            pred, true, format(cm[true, pred], fmt),
            horizontalalignment="center",
            color="gainsboro" if cm[true, pred] > thresh else "dimgray",
            fontsize=8,
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.clim(minimum, maximum)
    plt.tight_layout()
    return fig


def plot_to_image(fig, close_fig: bool = True):
    """
    Convert a Matplotlib figure to a PNG image and return it.
    :param fig: Matplotlib figure
    :param close_fig: Whether to close the figure
    :return: image as an array
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='raw')
    if close_fig:
        plt.close(fig)
    buf.seek(0)
    array = np.reshape(
        np.frombuffer(buf.getvalue(), dtype=np.uint8),
        (int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
    )
    return array


def confusion_matrix_pipeline(y_true, y_pred, class_names, normalization=None, cmap=plt.cm.Blues):
    """
    Compute confusion matrix and plot it.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param class_names: List of classes
    :param normalization: Normalization to apply (None, 'true', 'pred', 'all')
    :param cmap: Color map to use
    """
    minimum = 0.
    maximum = 1. if normalization is not None else None
    cm = confusion_matrix(y_true, y_pred, normalize=normalization)
    fig = plot_confusion_matrix(cm, class_names, minimum=minimum, maximum=maximum, cmap=cmap)
    return plot_to_image(fig)


if __name__ == '__main__':
    y_true = np.array([1, 1, 2, 1, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    y_pred = np.array([0, 1, 0, 2, 1, 1, 0, 2, 2, 3, 4, 5, 6, 7, 7])
    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    classes = [f"Class {i}" for i in range(num_classes)]
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig = plot_confusion_matrix(cm, classes)
    plt.show()
