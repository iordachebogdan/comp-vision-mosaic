import cv2 as cv
import sys
import numpy as np
from config.parameters import Parameters


class MosaicBuilder:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

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

                best = -1
                best_mean = np.zeros(3)
                means = self.__compute_means()
                for idx in range(0, len(means)):
                    curr_mean = means[idx]
                    if best == -1 or \
                            np.linalg.norm(mean_fragment-curr_mean) < \
                            np.linalg.norm(mean_fragment-best_mean):
                        if i > 0 and used_images[i-1, j] == idx:
                            continue
                        if j > 0 and used_images[i, j-1] == idx:
                            continue
                        best = idx
                        best_mean = curr_mean

                used_images[i, j] = best
                result[
                    i * self.parameters.height_small:
                        (i+1) * self.parameters.height_small,
                    j * self.parameters.width_small:
                        (j+1) * self.parameters.width_small,
                    ...
                ] = self.parameters.small_images[best]

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

        cv.imwrite(self.parameters.results_dir + "result."
                   + self.parameters.small_images_type, result)

    def __compute_means(self):
        return [np.mean(image, axis=(0, 1))
                for image in self.parameters.small_images]
