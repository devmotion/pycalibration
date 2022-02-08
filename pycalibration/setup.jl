using CalibrationAnalysis: ColVecs, RowVecs
using PyCall: PyCall

# use Julia wrapper for ColVecs and RowVecs in Python
# otherwise the automatic conversions in PyCall will convert it to a matrix!
function PyCall.PyObject(::Type{<:ColVecs})
    return PyCall.pyfunctionret(ColVecs, Any, PyCall.PyAny)
end
function PyCall.PyObject(::Type{<:RowVecs})
    return PyCall.pyfunctionret(RowVecs, Any, PyCall.PyAny)
end
