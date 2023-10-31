import os
import sys

import matplotlib.pyplot as plt

def parse_log(log_path):
    losses = []
    high_freq_ddpm_losses = []
    low_freq_pixel_losses = []
    pixel_losses = []

    file = open(log_path, 'r')
    lines = file.readlines()
    for line in lines:
        loss, high_freq_ddpm_loss, low_freq_pixel_loss, pixel_loss = [float(i.split('=')[1]) for i in line.split(', ')]

        losses.append(loss)
        high_freq_ddpm_losses.append(high_freq_ddpm_loss)
        low_freq_pixel_losses.append(low_freq_pixel_loss)
        pixel_losses.append(pixel_loss)
    
    return losses, high_freq_ddpm_losses, low_freq_pixel_losses, pixel_losses
    
def plot(log_path):
    if not os.path.exists(log_path):
        return

    dir = os.path.dirname(log_path)
    basename = os.path.basename(log_path)[:-4]

    losses = parse_log(log_path)
    labels = ['loss', 'high_freq_ddpm_loss', 'low_freq_pixel_loss', 'pixel_loss']

    plt.figure(figsize=(15, 10), tight_layout=True)
    for i in range(len(losses)):
        loss = losses[i]
        label = labels[i]

        plt.plot(range(len(loss)), loss, label=label)

        if label == 'loss':
            min_loss = min(loss)
            min_index = len(loss) - loss[::-1].index(min_loss) - 1
            plt.plot(min_index, min_loss, '*', label='best value')

    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{dir}/{basename}.png')

if __name__ == '__main__':
    idx = sys.argv[1]
    log_path = [f'Training/{idx}/log/train_log.txt', f'Training/{idx}/log/val_log.txt']
    [plot(log) for log in log_path] 
