import sys

def normalize(line):
    pixelArray = [int(n) for n in line.split(",")]
    normalizedArray = []
    normalizedArray.append(pixelArray[0])

    for num in pixelArray[1:]:
        normalizedArray.append(num/255)

    return normalizedArray

def preProcess():
    # Read file line by line
    train_filepath = "./trainingdata/mnist_train.csv"
    train_normalized_filepath = "./trainingdata/mnist_train_normalized.csv"

    read_fd = open(train_filepath, "r")
    write_fd = open(train_normalized_filepath, "w")

    for line in read_fd:
        normalizedArray = normalize(line.strip())
        normalizedString = ','.join(str(e) for e in normalizedArray)
        write_fd.write(normalizedString + "\n")

    read_fd.close()
    write_fd.close()