import cv2 as cv
import sys
import numpy as np
from config.parameters import Parameters
import random


class MosaicBuilder:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.means = self.__compute_means()

    def build_grid(self):
        print("Start building grid ...")
        # afisam imagini intermediare
        add_checkpoint = self.parameters.add_checkpoint_perc
        next_checkpoint = self.parameters.add_checkpoint_perc
        result = np.full(self.parameters.image.shape, 255, np.uint8)
        used_images = np.full((self.parameters.num_pieces_vertical,
                               self.parameters.num_pieces_horizontal),
                              -1)
        for i in range(0, self.parameters.num_pieces_vertical):
            for j in range(0, self.parameters.num_pieces_horizontal):
                mean_fragment = np.mean(
                    self.parameters.image[
                        i * self.parameters.height_small:
                            (i+1) * self.parameters.height_small,
                        j * self.parameters.width_small:
                            (j+1) * self.parameters.width_small,
                        ...
                    ],
                    axis=(0, 1)
                )

                if self.parameters.criterion == "euclid":
                    selected_image = self.__choose_euclid(
                        i, j, mean_fragment, used_images
                    )
                else:
                    selected_image = self.__choose_random()

                result[
                    i * self.parameters.height_small:
                        (i+1) * self.parameters.height_small,
                    j * self.parameters.width_small:
                        (j+1) * self.parameters.width_small,
                    ...
                ] = selected_image

                percentage = (i * self.parameters.num_pieces_horizontal + j) /\
                    (self.parameters.num_pieces_horizontal *
                     self.parameters.num_pieces_vertical)
                sys.stdout.write("\r")
                sys.stdout.write("Progress: %d%%" % (int(percentage * 100)))
                sys.stdout.flush()
                if percentage >= next_checkpoint:
                    next_checkpoint = \
                        min(next_checkpoint + add_checkpoint, 1)
                    cv.imwrite(
                        self.parameters.results_dir + str(int(100*percentage))
                        + "." + self.parameters.small_images_type,
                        result
                    )

        sys.stdout.write("\n")
        print("Writing solution ...")
        cv.imwrite(self.parameters.results_dir + "result."
                   + self.parameters.small_images_type, result)
        print("Done!")

    def __compute_means(self):
        return [np.mean(image, axis=(0, 1))
                for image in self.parameters.small_images]

    def __choose_euclid(self,
                        i: int, j: int,
                        mean_fragment: np.ndarray,
                        used_images: np.ndarray) -> np.ndarray:
        best = -1
        best_mean = np.zeros(3)

        for idx in range(0, len(self.means)):
            curr_mean = self.means[idx]
            if best == -1 or \
                    np.linalg.norm(mean_fragment-curr_mean) < \
                    np.linalg.norm(mean_fragment-best_mean):
                if self.parameters.different and \
                        i > 0 and used_images[i-1, j] == idx:
                    continue
                if self.parameters.different and \
                        j > 0 and used_images[i, j-1] == idx:
                    continue
                best = idx
                best_mean = curr_mean

        used_images[i, j] = best
        return self.parameters.small_images[best]

    def __choose_random(self) -> np.ndarray:
        return self.parameters.small_images[
            random.randrange(len(self.parameters.small_images))
        ]
