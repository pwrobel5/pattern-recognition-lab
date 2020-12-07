import glob

import matplotlib.pyplot as plt


def preprocess_file(input_file):
    results = [entry.split() for entry in input_file.read().split("\n")]
    results = list(filter(lambda entry: len(entry) > 0, results))
    raw_x, raw_train_loss, raw_train_accuracies, raw_val_losses, raw_val_accuracies = zip(*results)

    x = [float(value) for value in raw_x]
    train_loss = [float(value) for value in raw_train_loss]
    train_accuracies = [float(value) for value in raw_train_accuracies]
    val_loss = [float(value) for value in raw_val_losses]
    val_accuracies = [float(value) for value in raw_val_accuracies]

    return x, train_loss, train_accuracies, val_loss, val_accuracies


def read_data(pattern):
    file_names = glob.glob(pattern)
    results = {}

    for file_name in file_names:
        file = open(file_name, "r")
        results[file_name] = preprocess_file(file)
        file.close()

    return results


def make_graph_from_column(output_name, column_index, results, graph_title, y_title):
    x = results["results/accuracy_1.txt"][0]
    normal_values = results["results/accuracy_1.txt"][column_index]
    without_dropout_values = results["results/accuracy_2.txt"][column_index]
    separable_convolutions_values = results["results/accuracy_3.txt"][column_index]
    convolutions_5x5_values = results["results/accuracy_4.txt"][column_index]

    plt.figure()
    plt.plot(x, normal_values, "-o", label="bez modyfikacji")
    plt.plot(x, without_dropout_values, "-o", label="bez dropout")
    plt.plot(x, separable_convolutions_values, "-o", label="separable convolutions")
    plt.plot(x, convolutions_5x5_values, "-o", label="konwolucje 5x5")
    plt.title(graph_title)
    plt.xlabel("Epoka uczenia")
    plt.ylabel(y_title)
    plt.legend()
    plt.savefig(output_name)


def make_graphs():
    results = read_data("results/accuracy_[0-9].txt")
    make_graph_from_column("results/train_loss.png", 1, results, "Loss (zbiór uczący)", "Loss")
    make_graph_from_column("results/train_accuracy.png", 2, results, "Accuracy (zbiór uczący)", "Accuracy")
    make_graph_from_column("results/val_loss.png", 3, results, "Loss (zbiór walidacyjny)", "Loss")
    make_graph_from_column("results/val_accuracy.png", 4, results, "Accuracy (zbiór walidacyjny)", "Accuracy")


if __name__ == '__main__':
    make_graphs()
