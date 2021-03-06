from julia import Main, CalibrationErrorsDistributions
import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
Main.include(os.path.join(script_dir, "setup.jl"))

sys.modules[__name__] = CalibrationErrorsDistributions
