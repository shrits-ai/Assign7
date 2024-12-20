import matplotlib.pyplot as plt

def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accuracies)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accuracies)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()

