import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

if __name__ == "__main__":
    REQUIREMENTS = (
        "ggplot2",
        "earth",
        "plotmo",
        "plotrix",
        "TeachingDemo",
    )

    # Choosing a CRAN Mirror
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    # Installing required packages
    utils.install_packages(StrVector(REQUIREMENTS))
