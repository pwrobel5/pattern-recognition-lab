import matplotlib.pyplot as plt


def preprocess_file(input_file):
    results = [entry.split() for entry in input_file.read().split("\n")]
    results = list(filter(lambda entry: len(entry) > 0, results))
    raw_x, raw_y = zip(*results)
    x = [float(value) for value in raw_x]
    y = [float(value) for value in raw_y]

    return x, y


def load_data_from_file(file_name, data_label):
    input_file = open(file_name, "r")
    x, y = preprocess_file(input_file)
    input_file.close()

    ax.scatter(x, y, label=data_label)
    ax.plot(x, y)


if __name__ == '__main__':
    fig, ax = plt.subplots()

    load_data_from_file("reconstructed_random/lenin.txt", "losowy wybór")
    load_data_from_file("reconstructed_small_random/lenin.txt", "mała rozdzielczość")
    load_data_from_file("reconstructed_missing_random/lenin.txt", "brakujący fragment")
    load_data_from_file("reconstructed_every_nth/lenin.txt", "co n-ty piksel")

    ax.set_ylabel("Norma z błędu")
    ax.set_xlabel("Ułamek próbek użytych do rekonstrukcji")
    ax.legend()

    fig.savefig("lenin_errors.png")
