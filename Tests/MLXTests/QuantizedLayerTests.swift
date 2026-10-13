// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import XCTest

/// Fixed weights from a formula, so that each run uses the same values.
private func patternWeights(_ rows: Int, _ cols: Int, scale: Float = 0.1) -> MLXArray {
    var values = [Float]()
    for i in 0 ..< rows {
        for j in 0 ..< cols {
            values.append(scale * sin(Float(i * cols + j) * 0.7 + 0.3))
        }
    }
    return MLXArray(values, [rows, cols])
}

/// Fixed inputs from a formula.
private func patternInputs(_ rows: Int, _ cols: Int) -> MLXArray {
    var values = [Float]()
    for i in 0 ..< rows {
        for j in 0 ..< cols {
            values.append(cos(Float(i * cols + j) * 0.37))
        }
    }
    return MLXArray(values, [rows, cols])
}

/// A model with two linear layers and one embedding.
private class QuantizeTestModel: Module {
    @ModuleInfo var a: Linear = Linear(weight: patternWeights(16, 64), bias: MLXArray.zeros([16]))
    @ModuleInfo var b: Linear = Linear(weight: patternWeights(8, 64), bias: nil)
    @ModuleInfo var e: Embedding = Embedding(weight: patternWeights(10, 64))
}

/// Tests for QuantizedLinear, QuantizedEmbedding and the quantize(model:) functions in
/// MLXNN/Quantized.swift. Each output is compared with a matmul or a lookup on the
/// weights from `dequantized(...)`, and with the float layer within the quantization error.
class QuantizedLayerTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Helpers

    /// Check that `dequantized` is within one quantization step of `original`, for each group.
    private func checkDequantized(
        _ dequantized: MLXArray, _ original: MLXArray, scales: MLXArray, groupSize: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(dequantized.shape, original.shape, file: file, line: line)
        let cols = original.dim(1)
        let d = dequantized.asType(.float32).asArray(Float.self)
        let o = original.asType(.float32).asArray(Float.self)
        let s = scales.asType(.float32).asArray(Float.self)
        let groups = cols / groupSize
        var worst: Float = 0
        for i in 0 ..< original.dim(0) {
            for j in 0 ..< cols {
                let step = Swift.abs(s[i * groups + j / groupSize])
                let error = Swift.abs(d[i * cols + j] - o[i * cols + j])
                worst = Swift.max(worst, error - step)
            }
        }
        XCTAssertLessThanOrEqual(worst, 2e-5, file: file, line: line)
    }

    /// Check `quantized` against `float` with the bound sum_j |x_j| * |scale(o, j)|.
    private func checkWithinQuantizationError(
        _ quantized: MLXArray, _ float: MLXArray, x: MLXArray, scales: MLXArray,
        groupSize: Int, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(quantized.shape, float.shape, file: file, line: line)
        let rows = x.dim(0)
        let inputs = x.dim(1)
        let outputs = float.dim(1)
        let groups = inputs / groupSize
        let xv = x.asArray(Float.self)
        let s = scales.asType(.float32).asArray(Float.self)
        let q = quantized.asArray(Float.self)
        let f = float.asArray(Float.self)
        for r in 0 ..< rows {
            for o in 0 ..< outputs {
                var bound: Float = 1e-4
                for j in 0 ..< inputs {
                    bound +=
                        Swift.abs(xv[r * inputs + j]) * Swift.abs(s[o * groups + j / groupSize])
                }
                let error = Swift.abs(q[r * outputs + o] - f[r * outputs + o])
                XCTAssertLessThanOrEqual(error, bound, "row \(r) out \(o)", file: file, line: line)
            }
        }
    }

    private func names(_ parameters: ModuleParameters) -> Set<String> {
        Set(parameters.flattened().map { $0.0 })
    }

    // MARK: - QuantizedLinear

    func testQuantizedLinearAffine() {
        let w = patternWeights(16, 64)
        let bias = MLXArray((0 ..< 16).map { Float($0) * 0.01 }, [16])
        let layer = QuantizedLinear(weight: w, bias: bias, groupSize: 32, bits: 4)

        XCTAssertEqual(layer.groupSize, 32)
        XCTAssertEqual(layer.bits, 4)
        XCTAssertEqual(layer.mode, .affine)
        XCTAssertEqual(layer.weight.shape, [16, 64 * 4 / 32])
        XCTAssertEqual(layer.weight.dtype, .uint32)
        XCTAssertEqual(layer.scales.shape, [16, 2])
        XCTAssertEqual(layer.biases?.shape, [16, 2])
        XCTAssertEqual(layer.shape.0, 16)
        XCTAssertEqual(layer.shape.1, 64)

        let dq = dequantized(
            layer.weight, scales: layer.scales, biases: layer.biases, groupSize: 32, bits: 4)
        checkDequantized(dq, w, scales: layer.scales, groupSize: 32)

        let x = patternInputs(3, 64)
        let y = layer(x)
        XCTAssertEqual(y.shape, [3, 16])
        XCTAssertEqual(y.dtype, .float32)
        assertEqual(y, matmul(x, dq.T) + bias, rtol: 1e-4, atol: 1e-4)

        let float = Linear(weight: w, bias: bias)(x)
        checkWithinQuantizationError(y, float, x: x, scales: layer.scales, groupSize: 32)
    }

    func testQuantizedLinearFromLinear() {
        let w = patternWeights(16, 64)
        let linear = Linear(weight: w, bias: nil)
        let layer = QuantizedLinear(linear, groupSize: 64, bits: 8)

        XCTAssertEqual(layer.groupSize, 64)
        XCTAssertEqual(layer.bits, 8)
        XCTAssertNil(layer.bias)
        XCTAssertEqual(layer.weight.shape, [16, 64 * 8 / 32])
        XCTAssertEqual(layer.scales.shape, [16, 1])
        XCTAssertEqual(layer.shape.0, 16)
        XCTAssertEqual(layer.shape.1, 64)

        let dq = dequantized(
            layer.weight, scales: layer.scales, biases: layer.biases, groupSize: 64, bits: 8)
        checkDequantized(dq, w, scales: layer.scales, groupSize: 64)

        let x = patternInputs(2, 64)
        let y = layer(x)
        assertEqual(y, matmul(x, dq.T), rtol: 1e-4, atol: 1e-4)
        checkWithinQuantizationError(y, linear(x), x: x, scales: layer.scales, groupSize: 64)

        // toQuantized with a mode gives the same layer
        let viaProtocol = linear.toQuantized(groupSize: 64, bits: 8, mode: .affine)
        guard let fromProtocol = viaProtocol as? QuantizedLinear else {
            XCTFail("toQuantized did not return a QuantizedLinear")
            return
        }
        assertEqual(fromProtocol(x), y)
    }

    func testQuantizedLinearRandomInit() {
        let layer = QuantizedLinear(64, 16, bias: false, groupSize: 32, bits: 4)
        XCTAssertNil(layer.bias)
        XCTAssertEqual(layer.shape.0, 16)
        XCTAssertEqual(layer.shape.1, 64)
        XCTAssertEqual(layer.weight.shape, [16, 8])

        let withBias = QuantizedLinear(64, 16)
        XCTAssertEqual(withBias.groupSize, 64)
        XCTAssertEqual(withBias.bits, 4)
        XCTAssertEqual(withBias.bias?.shape, [16])
        XCTAssertEqual(withBias.bias?.asArray(Float.self), [Float](repeating: 0, count: 16))

        // the output is the matmul with the dequantized weight
        let x = patternInputs(2, 64)
        let dq = dequantized(
            layer.weight, scales: layer.scales, biases: layer.biases, groupSize: 32, bits: 4)
        assertEqual(layer(x), matmul(x, dq.T), rtol: 1e-4, atol: 1e-4)
    }

    func testQuantizedLinearMXFP4() {
        let w = patternWeights(16, 64)
        let bias = MLXArray((0 ..< 16).map { Float($0) * -0.02 }, [16])
        let layer = QuantizedLinear(weight: w, bias: bias, groupSize: 32, bits: 4, mode: .mxfp4)

        XCTAssertEqual(layer.mode, .mxfp4)
        XCTAssertNil(layer.biases)
        XCTAssertEqual(layer.weight.shape, [16, 8])
        XCTAssertEqual(layer.scales.shape, [16, 2])

        let dq = dequantized(
            layer.weight, scales: layer.scales, biases: nil, groupSize: 32, bits: 4,
            mode: .mxfp4, dtype: .float32)
        XCTAssertEqual(dq.shape, [16, 64])

        let x = patternInputs(3, 64)
        assertEqual(layer(x), matmul(x, dq.T) + bias, rtol: 1e-4, atol: 1e-4)
    }

    func testQuantizedLinearArrayInit() {
        // the initializer for subclasses takes the quantized arrays as they are
        let w = patternWeights(16, 64)
        let bias = MLXArray((0 ..< 16).map { Float($0) * 0.03 }, [16])
        let (wq, scales, biases) = MLX.quantized(w, groupSize: 32, bits: 4)
        let direct = QuantizedLinear(
            weight: wq, bias: bias, scales: scales, biases: biases, groupSize: 32, bits: 4)
        let reference = QuantizedLinear(weight: w, bias: bias, groupSize: 32, bits: 4)

        XCTAssertEqual(direct.groupSize, 32)
        XCTAssertEqual(direct.bits, 4)
        XCTAssertEqual(direct.mode, .affine)

        let x = patternInputs(2, 64)
        assertEqual(direct(x), reference(x))

        // this initializer does not freeze, the other one does
        XCTAssertEqual(
            names(direct.trainableParameters()), ["weight", "bias", "scales", "biases"])
        XCTAssertEqual(names(reference.trainableParameters()), [])
        XCTAssertEqual(names(reference.parameters()), ["weight", "bias", "scales", "biases"])
    }

    func testQuantizedLinearStaysFrozen() throws {
        let layer = QuantizedLinear(
            weight: patternWeights(16, 64), bias: nil, groupSize: 32, bits: 4)
        XCTAssertEqual(names(layer.trainableParameters()), [])

        // unfreeze freezes the layer's own parameters again
        try layer.unfreeze(recursive: true, keys: nil, strict: false)
        XCTAssertEqual(names(layer.trainableParameters()), [])
        XCTAssertEqual(names(layer.parameters()), ["weight", "scales", "biases"])
    }

    func testQuantizedLinearUpdateParameters() throws {
        // bfloat16 constants with a float32 input use the cast cache; an update
        // must clear the cache so that the new scales are used.
        let w = patternWeights(16, 64).asType(.bfloat16)
        let bias = MLXArray((0 ..< 16).map { Float($0) * 0.01 }, [16]).asType(.bfloat16)
        let layer = QuantizedLinear(weight: w, bias: bias, groupSize: 32, bits: 4)
        XCTAssertEqual(layer.scales.dtype, .bfloat16)

        let x = patternInputs(3, 64)
        // the layer widens the constants to float32 before the matmul
        let dq = dequantized(
            layer.weight, scales: layer.scales.asType(.float32),
            biases: layer.biases?.asType(.float32), groupSize: 32, bits: 4)
        let y1 = layer(x)
        XCTAssertEqual(y1.dtype, .float32)
        assertEqual(y1, matmul(x, dq.T) + bias.asType(.float32), rtol: 1e-3, atol: 1e-3)
        // call again so that the cached casts are used
        assertEqual(layer(x), y1)

        // double the scales and the offsets: the dequantized weight doubles
        let newScales = layer.scales * 2
        let newBiases = layer.biases! * 2
        try layer.update(
            parameters: ModuleParameters.unflattened([
                ("scales", newScales), ("biases", newBiases),
            ]), verify: .none)
        assertEqual(layer.scales, newScales)

        let y2 = layer(x)
        assertEqual(
            y2, 2 * matmul(x, dq.T) + bias.asType(.float32), rtol: 1e-3, atol: 1e-3)
    }

    @available(*, deprecated)
    func testQuantizedLinearDeprecatedFrom() {
        let linear = Linear(weight: patternWeights(16, 64), bias: MLXArray.ones([16]))
        let layer = QuantizedLinear.from(linear: linear, groupSize: 32, bits: 8)
        XCTAssertEqual(layer.groupSize, 32)
        XCTAssertEqual(layer.bits, 8)

        let x = patternInputs(2, 64)
        assertEqual(layer(x), QuantizedLinear(linear, groupSize: 32, bits: 8)(x))

        // the two argument toQuantized uses the affine mode
        guard let other = (linear as Quantizable).toQuantized(groupSize: 32, bits: 8) as? Quantized
        else {
            XCTFail("toQuantized did not return a Quantized layer")
            return
        }
        XCTAssertEqual(other.mode, .affine)
        XCTAssertEqual(other.groupSize, 32)
        XCTAssertEqual(other.bits, 8)
    }

    @available(*, deprecated)
    func testQuantizedLinearDeprecatedStaticQuantize() {
        let model = QuantizeTestModel()
        // only layers with more than 8 outputs
        QuantizedLinear.quantize(model: model, groupSize: 32, bits: 8) { $0.weight.dim(0) > 8 }

        XCTAssertTrue(model.a is QuantizedLinear)
        XCTAssertFalse(model.b is QuantizedLinear)
        XCTAssertFalse(model.e is QuantizedEmbedding)
        XCTAssertEqual((model.a as? QuantizedLinear)?.groupSize, 32)
        XCTAssertEqual((model.a as? QuantizedLinear)?.bits, 8)
    }

    // MARK: - QuantizedEmbedding

    func testQuantizedEmbedding() {
        let w = patternWeights(10, 64)
        let layer = QuantizedEmbedding(weight: w, groupSize: 32, bits: 4)

        XCTAssertEqual(layer.groupSize, 32)
        XCTAssertEqual(layer.bits, 4)
        XCTAssertEqual(layer.mode, .affine)
        XCTAssertEqual(layer.weight.shape, [10, 8])
        XCTAssertEqual(layer.scales.shape, [10, 2])
        XCTAssertEqual(layer.biases?.shape, [10, 2])
        XCTAssertEqual(layer.shape.0, 10)
        XCTAssertEqual(layer.shape.1, 64)
        XCTAssertEqual(names(layer.trainableParameters()), [])

        let dq = dequantized(
            layer.weight, scales: layer.scales, biases: layer.biases, groupSize: 32, bits: 4)
        checkDequantized(dq, w, scales: layer.scales, groupSize: 32)

        // lookup: each output row is the dequantized row of the index
        let indices: [Int32] = [1, 3, 9, 0]
        let y = layer(MLXArray(indices, [2, 2]))
        XCTAssertEqual(y.shape, [2, 2, 64])
        let table = dq.asArray(Float.self)
        let expected = indices.flatMap { i in table[(Int(i) * 64) ..< (Int(i) * 64 + 64)] }
        assertEqual(y, MLXArray(expected, [2, 2, 64]), rtol: 1e-5, atol: 1e-6)

        // the lookup is within one step of the float embedding
        let floatRows = Embedding(weight: w)(MLXArray(indices, [4]))
        let scaleRows = layer.scales[MLXArray(indices, [4])]
        checkDequantized(y.reshaped([4, 64]), floatRows, scales: scaleRows, groupSize: 32)

        // asLinear is the matmul with the dequantized table
        let x = patternInputs(3, 64)
        let projected = layer.asLinear(x)
        XCTAssertEqual(projected.shape, [3, 10])
        assertEqual(projected, matmul(x, dq.T), rtol: 1e-4, atol: 1e-4)
        checkWithinQuantizationError(
            projected, Embedding(weight: w).asLinear(x), x: x, scales: layer.scales,
            groupSize: 32)
    }

    func testQuantizedEmbeddingInitializers() {
        let random = QuantizedEmbedding(embeddingCount: 8, dimensions: 64, groupSize: 64, bits: 8)
        XCTAssertEqual(random.weight.shape, [8, 16])
        XCTAssertEqual(random.scales.shape, [8, 1])
        XCTAssertEqual(random.shape.0, 8)
        XCTAssertEqual(random.shape.1, 64)

        let defaults = QuantizedEmbedding(embeddingCount: 4, dimensions: 128)
        XCTAssertEqual(defaults.groupSize, 64)
        XCTAssertEqual(defaults.bits, 4)
        XCTAssertEqual(defaults.scales.shape, [4, 2])

        let w = patternWeights(6, 64)
        let fromEmbedding = QuantizedEmbedding(Embedding(weight: w), groupSize: 32, bits: 8)
        XCTAssertEqual(fromEmbedding.bits, 8)
        let dq = dequantized(
            fromEmbedding.weight, scales: fromEmbedding.scales, biases: fromEmbedding.biases,
            groupSize: 32, bits: 8)
        let index = MLXArray([5, 2] as [Int32], [2])
        assertEqual(fromEmbedding(index), dq[index], rtol: 1e-5, atol: 1e-6)
    }

    func testQuantizedEmbeddingMXFP4() {
        let w = patternWeights(6, 64)
        let layer = QuantizedEmbedding(weight: w, groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(layer.mode, .mxfp4)
        XCTAssertNil(layer.biases)

        let index = MLXArray([4, 0, 2] as [Int32], [3])
        let rows = dequantized(
            layer.weight[index], scales: layer.scales[index], biases: nil, groupSize: 32,
            bits: 4, mode: .mxfp4)
        let y = layer(index)
        XCTAssertEqual(y.shape, [3, 64])
        assertEqual(y.asType(.float32), rows.asType(.float32))

        let x = patternInputs(2, 64)
        let table = dequantized(
            layer.weight, scales: layer.scales, biases: nil, groupSize: 32, bits: 4,
            mode: .mxfp4, dtype: .float32)
        assertEqual(layer.asLinear(x), matmul(x, table.T), rtol: 1e-4, atol: 1e-4)
    }

    // MARK: - quantizeSingle and quantize(model:)

    func testQuantizeSingle() {
        let linear = Linear(weight: patternWeights(16, 64), bias: nil)
        let q = quantizeSingle(layer: linear, groupSize: 32, bits: 8)
        XCTAssertTrue(q is QuantizedLinear)
        XCTAssertEqual(q?.groupSize, 32)
        XCTAssertEqual(q?.bits, 8)
        XCTAssertEqual(q?.mode, .affine)

        let mx = quantizeSingle(layer: linear, groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(mx?.mode, .mxfp4)

        let e = quantizeSingle(layer: Embedding(weight: patternWeights(4, 64)))
        XCTAssertTrue(e is QuantizedEmbedding)
        XCTAssertEqual(e?.groupSize, 64)
        XCTAssertEqual(e?.bits, 4)

        // an already quantized layer and a layer that can not be quantized give nil
        XCTAssertNil(quantizeSingle(layer: QuantizedLinear(64, 16)))
        XCTAssertNil(quantizeSingle(layer: ReLU()))
    }

    func testQuantizeModelFilter() {
        let model = QuantizeTestModel()
        let x = patternInputs(2, 64)
        let floatB = model.b(x)

        var seen = [String]()
        quantize(
            model: model, groupSize: 32, bits: 8, mode: .affine,
            filter: { path, _ in
                seen.append(path)
                return path != "b"
            })

        XCTAssertEqual(seen.sorted(), ["a", "b", "e"])
        let a = model.a as? QuantizedLinear
        XCTAssertNotNil(a)
        XCTAssertEqual(a?.groupSize, 32)
        XCTAssertEqual(a?.bits, 8)
        XCTAssertFalse(model.b is QuantizedLinear)
        assertEqual(model.b(x), floatB)
        let e = model.e as? QuantizedEmbedding
        XCTAssertEqual(e?.groupSize, 32)
        XCTAssertEqual(e?.bits, 8)

        // the quantized layer gives the dequantized matmul
        if let a {
            let dq = dequantized(
                a.weight, scales: a.scales, biases: a.biases, groupSize: 32, bits: 8)
            assertEqual(model.a(x), matmul(x, dq.T), rtol: 1e-4, atol: 1e-4)
        }

        // a second pass with the defaults quantizes only b; a stays as it was
        quantize(model: model)
        XCTAssertEqual((model.a as? QuantizedLinear)?.bits, 8)
        XCTAssertEqual((model.b as? QuantizedLinear)?.groupSize, 64)
        XCTAssertEqual((model.b as? QuantizedLinear)?.bits, 4)
        XCTAssertEqual((model.e as? QuantizedEmbedding)?.bits, 8)
    }

    func testQuantizeModelApply() {
        // an apply function that returns nil changes nothing
        let model = QuantizeTestModel()
        quantize(
            model: model, groupSize: 64, bits: 4, mode: .affine, filter: { _, _ in true },
            apply: { _, _, _, _ in nil })
        XCTAssertFalse(model.a is QuantizedLinear)
        XCTAssertFalse(model.b is QuantizedLinear)
        XCTAssertFalse(model.e is QuantizedEmbedding)

        // an apply function that receives the arguments of the call
        var calls = [String]()
        quantize(
            model: model, groupSize: 32, bits: 8, mode: .affine, filter: { _, _ in true },
            apply: { layer, groupSize, bits, mode in
                calls.append("\(groupSize)/\(bits)/\(mode.rawValue)")
                if layer is Linear {
                    return quantizeSingle(
                        layer: layer, groupSize: groupSize, bits: bits, mode: mode)
                }
                return nil
            })
        XCTAssertEqual(calls, ["32/8/affine", "32/8/affine", "32/8/affine"])
        XCTAssertTrue(model.a is QuantizedLinear)
        XCTAssertTrue(model.b is QuantizedLinear)
        XCTAssertFalse(model.e is QuantizedEmbedding)
    }

    func testQuantizeModelTupleFilter() {
        let model = QuantizeTestModel()
        quantize(
            model: model,
            filter: {
                (path: String, _: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                switch path {
                case "a": return (32, 4, .affine)
                case "e": return (64, 8, .affine)
                default: return nil
                }
            })

        XCTAssertEqual((model.a as? QuantizedLinear)?.groupSize, 32)
        XCTAssertEqual((model.a as? QuantizedLinear)?.bits, 4)
        XCTAssertFalse(model.b is QuantizedLinear)
        XCTAssertEqual((model.e as? QuantizedEmbedding)?.groupSize, 64)
        XCTAssertEqual((model.e as? QuantizedEmbedding)?.bits, 8)
    }

    @available(*, deprecated)
    func testQuantizeModelDeprecatedOverloads() {
        // filter that returns (groupSize, bits)
        let model = QuantizeTestModel()
        quantize(
            model: model,
            filter: { (path: String, _: Module) -> (groupSize: Int, bits: Int)? in
                if path == "b" {
                    return (32, 8)
                }
                return nil
            })
        XCTAssertFalse(model.a is QuantizedLinear)
        XCTAssertEqual((model.b as? QuantizedLinear)?.groupSize, 32)
        XCTAssertEqual((model.b as? QuantizedLinear)?.bits, 8)
        XCTAssertEqual((model.b as? QuantizedLinear)?.mode, .affine)

        // apply with three arguments
        let model2 = QuantizeTestModel()
        var calls = 0
        quantize(
            model: model2, groupSize: 32, bits: 4, filter: { path, _ in path == "a" },
            apply: { (layer: Module, groupSize: Int, bits: Int) -> Module? in
                calls += 1
                return quantizeSingle(layer: layer, groupSize: groupSize, bits: bits)
            })
        XCTAssertEqual(calls, 1)
        XCTAssertEqual((model2.a as? QuantizedLinear)?.groupSize, 32)
        XCTAssertEqual((model2.a as? QuantizedLinear)?.bits, 4)
        XCTAssertFalse(model2.b is QuantizedLinear)
    }
}
