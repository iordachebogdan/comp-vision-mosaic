from config.parameters import Parameters
from builder import MosaicBuilderGrid


if __name__ == "__main__":
    print("Configuring script and reading images ...")
    parameters = Parameters()
    mosaic_builder = MosaicBuilderGrid(parameters)

    if parameters.layout == "caroiaj":
        mosaic_builder.build_grid()
