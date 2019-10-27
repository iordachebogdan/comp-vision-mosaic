from config.parameters import Parameters
from builder import MosaicBuilderGrid, MosaicBuilderRandom, MosaicBuilderHexagon


if __name__ == "__main__":
    print("Configuring script and reading images ...")
    parameters = Parameters()

    if parameters.layout == "caroiaj":
        if parameters.hexagon:
            mosaic_builder = MosaicBuilderHexagon(parameters)
            mosaic_builder.build_hexagon_grid()
        else:
            mosaic_builder = MosaicBuilderGrid(parameters)
            mosaic_builder.build_grid()
    elif parameters.layout == "aleator":
        mosaic_builder = MosaicBuilderRandom(parameters)
        mosaic_builder.build_random()
