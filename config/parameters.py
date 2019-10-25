import json
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2 as cv


class Parameters:
    def __init__(self, config_file="./config/parameters.json"):
        with open(config_file) as config:
            config_dict = json.load(config)
            self.image_path = config_dict["image_path"]
            self.grayscale = config_dict["grayscale"] == "True"
            self.small_images_dir = config_dict["small_images_dir"]
            self.small_images_type = config_dict["small_images_type"]
            self.num_pieces_horizontal = config_dict["num_pieces_horizontal"]
            self.show_small_images = config_dict["show_small_images"] == "True"
            self.layout = config_dict["layout"]
            self.different = config_dict["different"] == "True"
            self.criterion = config_dict["criterion"]
            self.hexagon = config_dict["hexagon"]
            self.results_dir = config_dict["results_dir"]
            self.add_checkpoint_perc = config_dict["add_checkpoint_perc"]

            self.image = cv.imread(
                self.image_path,
                cv.IMREAD_GRAYSCALE if self.grayscale else cv.IMREAD_COLOR,
            )
            self.small_images = [
                cv.imread(
                    img, cv.IMREAD_GRAYSCALE if self.grayscale else cv.IMREAD_COLOR
                )
                for img in glob.glob(
                    self.small_images_dir + "*." + self.small_images_type
                )
            ]
            self.__compute_dimensions()
            if self.show_small_images:
                self.__print_small_images()

    def __compute_dimensions(self):
        self.num_pieces_vertical = int(
            np.ceil(
                (self.image.shape[0] / self.image.shape[1])
                * (self.small_images[0].shape[1] / self.small_images[0].shape[0])
                * self.num_pieces_horizontal
            )
        )
        self.height_small = self.small_images[0].shape[0]
        self.width_small = self.small_images[0].shape[1]
        self.height = self.num_pieces_vertical * self.height_small
        self.width = self.num_pieces_horizontal * self.width_small
        self.image = cv.resize(self.image, (self.width, self.height))

    def __print_small_images(self):
        for i in range(0, 10):
            for j in range(0, 10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # swap channels BGR (opencv) to RGB (matplotlib)
                im = self.small_images[i * 10 + j].copy()
                im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()
