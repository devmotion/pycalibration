from julia import Main
import os


def install():
    """
    Install Julia packages required by pycalibration.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    Main.include(os.path.join(script_dir, "install.jl"))
