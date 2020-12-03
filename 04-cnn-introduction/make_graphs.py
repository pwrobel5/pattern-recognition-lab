import glob
import statistics

import matplotlib.pyplot as plt


def preprocess_file(input_file):
    results = [entry.split() for entry in input_file.read().split("\n")]
    results = list(filter(lambda entry: len(entry) > 0, results))
    raw_x, raw_y = zip(*results)
    x = [float(value) for value in raw_x]
    y = [float(value) for value in raw_y]

    return x, y


def read_data(pattern):
    file_names = glob.glob(pattern)
    results = {}

    for file_name in file_names:
        file = open(file_name, "r")
        results[file_name] = preprocess_file(file)
        file.close()

    return results


def make_statistics(data):
    x_values = []
    y_values = []

    for x, y in data.values():
        x_values.append(x)
        y_values.append(y)

    x_merged = [x[0] for x in zip(*x_values)]  # for all calculation x values are the same
    y_merged = list(zip(*y_values))

    y_averages = [statistics.mean(y) for y in y_merged]
    y_stddevs = [statistics.stdev(y) for y in y_merged]

    return x_merged, y_averages, y_stddevs


def make_graph_10_epochs():
    big_aug_results = read_data("results/*10_*[a-zA-Z].txt")
    normal_results = read_data("results/*10_[0-9].txt")
    x_normal, y_normal, y_err_normal = make_statistics(normal_results)

    x_big, y_big = big_aug_results["results/accuracy_10_0_big.txt"]
    x_aug, y_aug = big_aug_results["results/accuracy_10_0_big_aug.txt"]

    fig, ax = plt.subplots()
    ax.errorbar(x_normal, y_normal, label="zwykła sieć", linestyle="-", yerr=y_err_normal, fmt="o")
    ax.plot(x_big, y_big, "-o", label="3 neurony więcej")
    ax.plot(x_aug, y_aug, "-o", label="3 neurony więcej + augmentacja")
    ax.set_ylabel("Dokładność dla zbioru walidacyjnego")
    ax.set_xlabel("Epoka uczenia")
    ax.legend()

    fig.savefig("graphs/graph_10_epochs.png")


def make_graph_50_epochs():
    results = read_data("results/*50*.txt")

    x_normal, y_normal = results["results/accuracy_50_0.txt"]
    x_4class, y_4class = results["results/accuracy_50_0_4_class.txt"]

    fig, ax = plt.subplots()
    ax.plot(x_normal, y_normal, "-o", label="sieć z 3 klasami")
    ax.plot(x_4class, y_4class, "-o", label="sieć z 4 klasami")
    ax.set_ylabel("Dokładność dla zbioru walidacyjnego")
    ax.set_xlabel("Epoka uczenia")
    ax.legend()

    fig.savefig("graphs/graph_50_epochs.png")


if __name__ == '__main__':
    make_graph_10_epochs()
    make_graph_50_epochs()
