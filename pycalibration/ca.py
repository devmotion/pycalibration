from julia import Main
import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
Main.include(os.path.join(script_dir, "PyCalibration.jl"))

from julia import PyCalibration
sys.modules[__name__] = PyCalibration
