using KernelFunctions: KernelFunctions
using PyCall: PyCall

# use Julia wrapper for ColVecs and RowVecs in Python
# otherwise the automatic conversions in PyCall will convert it to a matrix!
function PyCall.PyObject(::Type{<:KernelFunctions.ColVecs})
    return PyCall.pyfunctionret(KernelFunctions.ColVecs, Any, PyCall.PyAny)
end
function PyCall.PyObject(::Type{<:KernelFunctions.RowVecs})
    return PyCall.pyfunctionret(KernelFunctions.RowVecs, Any, PyCall.PyAny)
end
