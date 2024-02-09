#include "NodeBindings.hpp"
#include "Common.hpp"

#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/Node.hpp"
#include "depthai/pipeline/node/DepthEncoder.hpp"


void bind_depthencoder(pybind11::module& m, void* pCallstack){

    using namespace dai;
    using namespace dai::node;

    // Node and Properties declare upfront
    py::class_<DepthEncoderProperties> depthEncoderProperties(m, "DepthEncoderProperties", DOC(dai, DepthEncoderProperties));
    auto depthEncoder = ADD_NODE(DepthEncoder);

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    // Call the rest of the type defines, then perform the actual bindings
    Callstack* callstack = (Callstack*) pCallstack;
    auto cb = callstack->top();
    callstack->pop();
    cb(m, pCallstack);
    // Actual bindings
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    // DepthEncoder properties
    depthEncoderProperties
        .def_readwrite("outputType", &DepthEncoderProperties::outputType, DOC(dai, DepthEncoderProperties, outputType))
        .def_readwrite("numFramesPool", &DepthEncoderProperties::numFramesPool, DOC(dai, DepthEncoderProperties, numFramesPool))
        .def_readwrite("numShaves", &DepthEncoderProperties::numShaves, DOC(dai, DepthEncoderProperties, numShaves))
        .def_readwrite("lutR", &DepthEncoderProperties::lutR, DOC(dai, DepthEncoderProperties, lutR))
        .def_readwrite("lutG", &DepthEncoderProperties::lutG, DOC(dai, DepthEncoderProperties, lutG))
        .def_readwrite("lutB", &DepthEncoderProperties::lutB, DOC(dai, DepthEncoderProperties, lutB))
        ;

    // DepthEncoder Node
    depthEncoder
        .def_readonly("input", &DepthEncoder::input,
                      DOC(dai, node, DepthEncoder, input))
        .def_readonly("output", &DepthEncoder::output,
                      DOC(dai, node, DepthEncoder, output))
        .def_readonly("passthroughInput", &DepthEncoder::passthroughInput,
                      DOC(dai, node, DepthEncoder, passthroughInput))
        .def("setLut", &DepthEncoder::setLut,
             DOC(dai, node, DepthEncoder, setLut))
        // .def("setHueLut", &DepthEncoder::setHueLut, py::arg("minIn"),
        // py::arg("maxIn"), py::arg("scaleFactor") = 1,
        // py::arg("bufferPercentage") = 0.0f, DOC(dai, node, DepthEncoder,
        // setHueLut))
        .def("setOutputType", &DepthEncoder::setOutputType,
             DOC(dai, node, DepthEncoder, setOutputType))
        .def("setNumFramesPool", &DepthEncoder::setNumFramesPool,
             DOC(dai, node, DepthEncoder, setNumFramesPool))
        .def("setNumShaves", &DepthEncoder::setNumShaves,
             DOC(dai, node, DepthEncoder, setNumShaves),
             DOC(dai, node, DepthEncoder, setNumShaves))
        .def("setHueLutGeneric", &DepthEncoder::setHueLutGeneric,
             py::arg("minDepthIn"), py::arg("maxDepthIn"),
             py::arg("bufferAmount"), py::arg("getMinDisparity"),
             py::arg("getMaxDisparity"), py::arg("getHueValueFromDisparity"),
             DOC(dai, node, DepthEncoder, setHueLutGeneric))
        // Binding for setHueLutDisparity
        .def("setHueLutDisparity", &DepthEncoder::setHueLutDisparity,
             py::arg("minInDepth"), py::arg("maxInDepth"), py::arg("scale"),
             py::arg("bufferAmount"),
             DOC(dai, node, DepthEncoder, setHueLutDisparity))
        // Binding for setHueLutDepth
        .def("setHueLutDepth", &DepthEncoder::setHueLutDepth,
             py::arg("minInDepth"), py::arg("maxInDepth"), py::arg("scale"),
             py::arg("bufferAmount"),
             DOC(dai, node, DepthEncoder, setHueLutDepth))
        .def(
            "setHueLutDepthNormalized",
            [](DepthEncoder &self, float minInDepth, float maxInDepth,
               float scale, float bufferAmount) {
              auto result = self.setHueLutDepthNormalized(
                  minInDepth, maxInDepth, scale, bufferAmount);
              return py::make_tuple(std::get<0>(result), std::get<1>(result));
            },
            py::arg("minInDepth"), py::arg("maxInDepth"), py::arg("scale"),
            py::arg("bufferAmount"),
            DOC(dai, node, DepthEncoder, setHueLutDepthNormalized));
    daiNodeModule.attr("DepthEncoder").attr("Properties") = depthEncoderProperties;

}
