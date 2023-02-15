"""
Created by Wang Han on 2020/7/17 22:50.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2020 Wang Han. SCU. All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(cm, fname):
    assert cm.shape[0] == cm.shape[1]
    num_classes = cm.shape[0]

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title('Confusion Matrix')

    xlocations = np.array(range(num_classes))
    labels = [str(i) for i in range(num_classes)]
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)

    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for j in range(num_classes):
        for i in range(num_classes):
            v = int(cm[j][i])
            plt.text(i, j, "{}".format(v), color='red',
                     fontsize=14, va='center', ha='center')

    plt.savefig(fname, dpi=300)
    plt.close(fig)
