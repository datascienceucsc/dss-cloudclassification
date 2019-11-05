# diagnostics.py
#
# Anders Poirel
#
# 05-19-2019

import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('../../models/loss.png')
    return

def plot_acc(history):
    plt.plot(history.history['dice_coeff'])
    plt.plot(history.history['val_dice_coeff'])
    plt.title('Model dice coefficient')
    plt.ylabel('Dice coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.savefig('../../models/dice_coeff.png')
    return