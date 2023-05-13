import os
import random

import split_image
from os import listdir
from os.path import isfile, join

OUTPUT_DIR = "out"
mypath = "puzzles/raw-data"
subdirs = ["/train", "/validation", "/test"]


def generate_sequence():
    sequence = [*range(16)]
    random.shuffle(sequence)

    return sequence, "-".join([str(seq) for seq in sequence])


for i in subdirs:
    onlyfiles = [f for f in listdir(mypath + i) if isfile(join(mypath + i, f))]
    os.makedirs(mypath + i + "/out")
    counter = 0
    sequence, string_sequence = generate_sequence()
    for image in onlyfiles:
        split_image.split_image(mypath + i + "/" + image, 4, 4, False, False, output_dir=mypath + i + "/out")

        if counter % 100 == 0:
            sequence, string_sequence = generate_sequence()
            os.makedirs(mypath + i + "/" + string_sequence)
            os.makedirs("puzzles/" + i + "/" + string_sequence)

        current_images = [f for f in listdir(mypath + i + "/out") if isfile(join(mypath + i + "/out", f))]
        for f in current_images:
            name, index = f.split(".")[0].split("_")
            os.rename(mypath + i + "/out/" + f,
                      mypath + i + "/" + string_sequence + "/" + name + "_" + str(sequence[int(index)]) + ".jpg")
        current_images = [(mypath + i + "/" + string_sequence + "/" + f) for f in
                          listdir(mypath + i + "/" + string_sequence) if
                          isfile(join(mypath + i + "/" + string_sequence, f))]
        current_images.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        split_image.reverse_split(current_images, 4, 4,
                                  "puzzles/" + i + "/" + string_sequence + "/" + current_images[0].split("/")[
                                      -1].split("_")[0] + ".jpg",
                                  True)

        counter += 1
