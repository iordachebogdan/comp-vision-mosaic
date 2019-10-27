import json
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import cv2 as cv
from util.cifar_business import CifarBusiness


class Parameters:
    def __init__(self, config_file="./config/parameters.json"):
        with open(config_file) as config:
            config_dict = json.load(config)
            self.image_path = config_dict["image_path"]
            self.grayscale = config_dict["grayscale"] == "True"
            self.small_images_dir = config_dict["small_images_dir"]
            self.small_images_type = config_dict["small_images_type"]
            self.cifar = config_dict["cifar"] == "True"
            self.num_pieces_horizontal = config_dict["num_pieces_horizontal"]
            self.show_small_images = config_dict["show_small_images"] == "True"
            self.layout = config_dict["layout"]
            if self.layout != "aleator":
                self.different = config_dict["different"] == "True"
            self.criterion = config_dict["criterion"]
            if self.layout == "caroiaj":
                self.hexagon = config_dict["hexagon"] == "True"
            self.results_dir = config_dict["results_dir"]
            self.add_checkpoint_perc = config_dict["add_checkpoint_perc"]

            self.image = cv.imread(
                self.image_path,
                cv.IMREAD_GRAYSCALE if self.grayscale else cv.IMREAD_COLOR,
            )
            if not self.cifar:
                self.small_images = [
                    cv.imread(
                        img, cv.IMREAD_GRAYSCALE if self.grayscale else cv.IMREAD_COLOR
                    )
                    for img in glob.glob(
                        os.path.join(
                            self.small_images_dir, "*." + self.small_images_type
                        )
                    )
                ]
            else:
                self.cifar_type = config_dict["cifar_type"]
                cifar_business = CifarBusiness(self.small_images_dir, self.grayscale)
                self.small_images = cifar_business.read_images(self.cifar_type)

            self.__compute_dimensions()
            if self.show_small_images:
                self.__print_small_images()

    def __compute_dimensions(self):
        self.height_small = self.small_images[0].shape[0]
        self.width_small = self.small_images[0].shape[1]
        if self.layout == "caroiaj" and self.hexagon:
            # in cazul in care acoperim cu hexagoane vom considera num_pieces_horizontal
            # ca fiind numarul de coloane de hexagoane pe care le vom avea
            self.width = (
                self.width_small
                + (self.num_pieces_horizontal - 1) * self.width_small * 3 // 4
            )
            self.height = int(
                np.ceil((self.image.shape[0] / self.image.shape[1]) * self.width)
            )
            self.height -= self.height % self.height_small
            self.num_pieces_vertical = self.height // self.height_small
        else:
            self.num_pieces_vertical = int(
                np.ceil(
                    (self.image.shape[0] / self.image.shape[1])
                    * (self.small_images[0].shape[1] / self.small_images[0].shape[0])
                    * self.num_pieces_horizontal
                )
            )
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
