import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curves(log_file_path):
    try:
        data = pd.read_csv(log_file_path)
    except FileNotFoundError:
        print(f'Error: Log file not found at {log_file_path}')
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(data['epoch'], data['train_loss'], label='Training Loss', color='blue')
    ax1.plot(data['epoch'], data['val_loss'], label='Validation Loss', color='orange')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(data['epoch'], data['val_iou'], label='Validation IoU', color='green')
    ax2.plot(data['epoch'], data['val_f1'], label='Validation F1 Score', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric Value')
    ax2.legend()
    ax2.set_title('Validation Metrics (IoU and F1 Score)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot learning curves from training log file.'
    )
    parser.add_argument('log_file', type=str, help='Path to the training log CSV file.')
    args = parser.parse_args()
    plot_learning_curves(args.log_file)
