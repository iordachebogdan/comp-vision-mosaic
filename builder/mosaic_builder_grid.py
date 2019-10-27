import numpy as np
from config.parameters import Parameters
import random
from util.progress import Progress
from sklearn.neighbors import KDTree


class MosaicBuilderGrid:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.means = self.__compute_means()
        self.kdtree = KDTree(self.means)

    def build_grid(self):
        print("Start building grid ...")
        # afisam imagini intermediare
        progress = Progress(self.parameters.add_checkpoint_perc, self.parameters)
        result = np.zeros(self.parameters.image.shape, dtype=np.uint8)
        used_images = np.full(
            (
                self.parameters.num_pieces_vertical,
                self.parameters.num_pieces_horizontal,
            ),
            -1,
        )
        for i in range(0, self.parameters.num_pieces_vertical):
            for j in range(0, self.parameters.num_pieces_horizontal):
                mean_fragment = np.mean(
                    self.parameters.image[
                        i
                        * self.parameters.height_small : (i + 1)
                        * self.parameters.height_small,
                        j
                        * self.parameters.width_small : (j + 1)
                        * self.parameters.width_small,
                        ...,
                    ],
                    axis=(0, 1),
                )

                if self.parameters.criterion == "euclid":
                    selected_image = self.__choose_euclid(
                        i, j, mean_fragment, used_images
                    )
                else:
                    selected_image = self.__choose_random()

                result[
                    i
                    * self.parameters.height_small : (i + 1)
                    * self.parameters.height_small,
                    j
                    * self.parameters.width_small : (j + 1)
                    * self.parameters.width_small,
                    ...,
                ] = selected_image

                percentage = (i * self.parameters.num_pieces_horizontal + j) / (
                    self.parameters.num_pieces_horizontal
                    * self.parameters.num_pieces_vertical
                )
                progress.update(percentage, result)
        progress.update(1, result)

    def __compute_means(self) -> np.ndarray:
        return np.array(
            [
                np.ravel([np.mean(image, axis=(0, 1))])
                for image in self.parameters.small_images
            ]
        )

    def __choose_euclid(
        self, i: int, j: int, mean_fragment: np.ndarray, used_images: np.ndarray
    ) -> np.ndarray:
        _, ind = self.kdtree.query([np.ravel([mean_fragment])], k=5)

        for idx in ind[0]:
            if self.parameters.different and i > 0 and used_images[i - 1, j] == idx:
                continue
            if self.parameters.different and j > 0 and used_images[i, j - 1] == idx:
                continue
            used_images[i, j] = idx
            return self.parameters.small_images[idx]
        return None

    def __choose_random(self) -> np.ndarray:
        return self.parameters.small_images[
            random.randrange(len(self.parameters.small_images))
        ]
