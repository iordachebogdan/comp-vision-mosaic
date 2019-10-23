from config.parameters import Parameters
from builder.mosaic_builder import MosaicBuilder


if __name__ == "__main__":
    print("Configuring script and reading images ...")
    parameters = Parameters()
    mosaic_builder = MosaicBuilder(parameters)

    if parameters.layout == "caroiaj":
        if parameters.layout == "aleator":
            pass
        else:
            mosaic_builder.build_grid()
