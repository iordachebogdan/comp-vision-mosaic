from config.parameters import Parameters
import cv2 as cv
import numpy as np
from sklearn.neighbors import KDTree
import random
from util.progress import Progress


class MosaicBuilderRandom:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.means = self.__compute_means()
        self.kdtree = KDTree(self.means)

    def build_random(self):
        # add border to image
        image = self.parameters.image.copy()
        image = cv.copyMakeBorder(
            image,
            top=0,
            bottom=self.parameters.height_small - 1,
            left=0,
            right=self.parameters.width_small - 1,
            borderType=cv.BORDER_REFLECT,
        )

        print("Start building random layout ...")
        print("Generating random sequence ...")
        # generate a random list of positions to insert small images
        positions = [
            (i, j)
            for i in range(self.parameters.image.shape[0])
            for j in range(self.parameters.image.shape[1])
        ]
        random.shuffle(positions)

        # this is a mask for positions that can no longer be used
        blocked = np.zeros(
            (self.parameters.image.shape[0], self.parameters.image.shape[1])
        )
        total_blocked = 0

        print("Start patching ...")
        result = np.zeros(image.shape)
        progress = Progress(self.parameters.add_checkpoint_perc, self.parameters)
        for idx, position in enumerate(positions):
            if blocked[position[0], position[1]] == 1:
                continue
            patch = image[
                position[0] : position[0] + self.parameters.height_small,
                position[1] : position[1] + self.parameters.width_small,
                ...,
            ]
            curr_mean = np.mean(patch, axis=(0, 1))
            if self.parameters.criterion == "euclid":
                idx = self.__choose_euclid(curr_mean)
            else:
                idx = self.__choose_random()

            result[
                position[0] : position[0] + self.parameters.height_small,
                position[1] : position[1] + self.parameters.width_small,
                ...,
            ] = self.parameters.small_images[idx]

            count_blocked = self.__block(
                blocked,
                top=position[0],
                bottom=min(
                    blocked.shape[0], position[0] + self.parameters.height_small
                ),
                left=position[1],
                right=min(blocked.shape[1], position[1] + self.parameters.width_small),
            )

            total_blocked += count_blocked
            progress.update(
                1.0 * total_blocked / len(positions),
                result[
                    : -self.parameters.height_small + 1,
                    : -self.parameters.width_small + 1,
                    ...,
                ],
            )

    def __compute_means(self) -> np.ndarray:
        return np.array(
            [
                np.ravel([np.mean(image, axis=(0, 1))])
                for image in self.parameters.small_images
            ]
        )

    def __choose_euclid(self, curr_mean: np.ndarray) -> int:
        _, ind = self.kdtree.query([np.ravel([curr_mean])], k=1)
        return ind[0][0]

    def __choose_random(self) -> int:
        return random.randrange(len(self.parameters.small_images))

    def __block(
        self, blocked: np.ndarray, top: int, bottom: int, left: int, right: int
    ) -> int:
        blocked_patch = blocked[top:bottom, left:right]
        count_blocked = np.count_nonzero(blocked_patch)
        blocked_patch[:, :] = 1
        count_blocked = np.count_nonzero(blocked_patch) - count_blocked
        blocked[top:bottom, left:right] = blocked_patch
        return count_blocked
