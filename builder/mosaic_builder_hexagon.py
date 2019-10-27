from config.parameters import Parameters
import numpy as np
from sklearn.neighbors import KDTree
import cv2 as cv
import random
from util.progress import Progress


class MosaicBuilderHexagon:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.mask = self.__buid_mask()
        self.means = self.__compute_means()
        self.kdtree = KDTree(self.means)
        self.di = [-2, -1, -1, 1, 1, 2]
        self.dj = [0, -1, 1, -1, 1, 0]

    def build_hexagon_grid(self):
        image = self.parameters.image
        image = cv.copyMakeBorder(
            image,
            top=self.parameters.height_small // 2,
            bottom=self.parameters.height_small // 2,
            left=self.parameters.width_small * 3 // 4,
            right=self.parameters.width_small * 3 // 4,
            borderType=cv.BORDER_REFLECT,
        )
        # matricea used va retine ce imagine am folosit pentru fiecare hexagon
        # si este dispusa asemenea unei table de sah pentru a putea accesa usor vecinii
        # unui hexagon
        used = np.zeros(
            (
                2 * self.parameters.num_pieces_vertical + 1,
                self.parameters.num_pieces_horizontal + 2,
            )
        )
        result = np.zeros(image.shape)
        print("Start building hexagon grid ...")
        progress = Progress(self.parameters.add_checkpoint_perc, self.parameters)
        total_hexagons = (
            (self.parameters.num_pieces_horizontal + 2)
            * self.parameters.num_pieces_vertical
            + self.parameters.num_pieces_horizontal // 2
            + 1
            + self.parameters.num_pieces_horizontal % 2
        )
        current_hexagons = 0
        for column in range(self.parameters.num_pieces_horizontal + 2):
            steps = self.parameters.num_pieces_vertical + (0 if column % 2 == 1 else 1)
            line = column % 2
            i = 0 if column % 2 == 0 else self.parameters.height_small // 2
            j = column * self.parameters.width_small * 3 // 4

            for step in range(steps):
                patch = image[
                    i : i + self.parameters.height_small,
                    j : j + self.parameters.width_small,
                    ...,
                ]
                mean_fragment = np.sum(
                    patch * self.mask, axis=(0, 1)
                ) / np.count_nonzero(self.mask, axis=(0, 1))

                if self.parameters.criterion == "euclid":
                    idx = self.__choose_euclid(mean_fragment, used, line, column)
                else:
                    idx = self.__choose_random()

                res_patch = result[
                    i : i + self.parameters.height_small,
                    j : j + self.parameters.width_small,
                    ...,
                ]
                np.putmask(res_patch, self.mask, self.parameters.small_images[idx])
                result[
                    i : i + self.parameters.height_small,
                    j : j + self.parameters.width_small,
                    ...,
                ] = res_patch

                current_hexagons += 1
                progress.update(
                    current_hexagons / total_hexagons,
                    result[
                        self.parameters.height_small
                        // 2 : -self.parameters.height_small
                        // 2,
                        self.parameters.width_small
                        * 3
                        // 4 : -self.parameters.width_small
                        * 3
                        // 4,
                        ...,
                    ],
                )
                line += 2
                i += self.parameters.height_small

    def __buid_mask(self) -> np.ndarray:
        h = self.parameters.height_small
        w = self.parameters.width_small
        mask = np.ones(self.parameters.small_images[0].shape)
        top_left = np.array([0, w // 4])
        top_right = np.array([0, w * 3 // 4])
        left = np.array([h // 2, 0])
        right = np.array([h // 2, w - 1])
        bottom_left = np.array([h - 1, w // 4])
        bottom_right = np.array([h - 1, w * 3 // 4])
        for i in range(h):
            for j in range(w):
                curr = np.array([i, j])
                if np.cross(curr - left, top_left - left) < 0:
                    mask[i, j] = 0
                if np.cross(curr - left, bottom_left - left) > 0:
                    mask[i, j] = 0
                if np.cross(curr - right, bottom_right - right) < 0:
                    mask[i, j] = 0
                if np.cross(curr - right, top_right - right) > 0:
                    mask[i, j] = 0

        # adaugam sau stergem pixeli din masca
        # pentru a ne asigura ca avem imbinare perfecta
        # top-left ~ bottom-right
        origin = (
            -self.parameters.height_small // 2,
            -self.parameters.width_small * 3 // 4,
        )
        for i in range(0, h // 2):
            for j in range(0, w // 4):
                if np.all(mask[i, j]) and np.all(mask[i - origin[0], j - origin[1]]):
                    mask[i, j] = 0
                elif not np.any(mask[i, j]) and not np.any(
                    mask[i - origin[0], j - origin[1]]
                ):
                    mask[i, j] = 1
        # bottom-left ~ top-right
        origin = (
            self.parameters.height_small // 2,
            -self.parameters.width_small * 3 // 4,
        )
        for i in range(h // 2 + 1, h):
            for j in range(0, w // 4):
                if np.all(mask[i, j]) and np.all(mask[i - origin[0], j - origin[1]]):
                    mask[i, j] = 0
                elif not np.any(mask[i, j]) and not np.any(
                    mask[i - origin[0], j - origin[1]]
                ):
                    mask[i, j] = 1

        return mask

    def __compute_means(self) -> np.ndarray:
        return np.array(
            [
                np.ravel(
                    [
                        np.sum(image * self.mask, axis=(0, 1))
                        / np.count_nonzero(self.mask, axis=(0, 1))
                    ]
                )
                for image in self.parameters.small_images
            ]
        )

    def __choose_random(self) -> int:
        return random.randrange(len(self.parameters.small_images))

    def __choose_euclid(
        self, mean_fragment: np.ndarray, used: np.ndarray, line: int, column: int
    ) -> int:
        _, ind = self.kdtree.query([np.ravel([mean_fragment])], k=7)
        if not self.parameters.different:
            return ind[0][0]
        for idx in ind[0]:
            ok = True
            for i in range(6):
                lvec = line + self.di[i]
                cvec = column + self.dj[i]
                if (
                    0 <= lvec
                    and lvec < used.shape[0]
                    and 0 <= cvec
                    and cvec < used.shape[1]
                    and used[lvec, cvec] == idx
                ):
                    ok = False
                    break
            if ok:
                used[line, column] = idx
                return idx
        return -1
