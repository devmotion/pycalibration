# A lightweight Pythonic version of CalibrationAnalysis.jl
module PyCalibration
    
using CalibrationErrors
using CalibrationTests

# Use Julia wrapper for ColVecs and RowVecs in Python
# Otherwise the automatic conversions in PyCall will convert it to a matrix!
using PyCall: PyCall
function PyCall.PyObject(::Type{<:ColVecs})
    return PyCall.pyfunctionret(ColVecs, Any, PyCall.PyAny)
end
function PyCall.PyObject(::Type{<:RowVecs})
    return PyCall.pyfunctionret(RowVecs, Any, PyCall.PyAny)
end

end # module