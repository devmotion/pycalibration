import subprocess


def install():
    """
    Install and load Julia packages required by pycalibration.
    """
    script = """import Pkg;
    Pkg.add("CalibrationErrors");
    Pkg.add("CalibrationErrorsDistributions");
    Pkg.add("CalibrationTests");
    Pkg.add("PyCall");
    using CalibrationErrors;
    using CalibrationErrorsDistributions;
    using CalibrationTests;
    using PyCall;"""
    subprocess.check_call(['julia', '-e', script])
