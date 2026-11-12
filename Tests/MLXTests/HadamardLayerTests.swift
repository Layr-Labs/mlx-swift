// Adapted from PrismML-Eng/mlx-swift 6d3a84de28225d1f5bc0a56f5c781596997242f9 (MIT).
import Foundation
import MLX
import MLXNN
import XCTest

class HadamardLayerTests: XCTestCase {
    override class func setUp() {}

    func testHalfTransformsUseFloat32Intermediates() throws {
        let width = 1024
        let signs = (0 ..< width).map { Float($0 % 3 == 0 ? -1 : 1) }
        let transform = try SignedBlockHadamard(blockSize: width, signs: signs)
        // A named [Float] keeps the MLXArray initializer overloads cheap to type-check.
        let values: [Float] = (0 ..< width).map { Float(($0 * 13) % 127 - 63) / 7 }
        let input = MLXArray(values).asType(.float16)
        let forward = hadamardTransform(input.asType(.float32) * MLXArray(signs))
            .asType(.float16)
        let inverse = (hadamardTransform(input.asType(.float32)) * MLXArray(signs))
            .asType(.float16)
        XCTAssertEqual(transform(input).asArray(Float16.self), forward.asArray(Float16.self))
        XCTAssertEqual(
            transform.inverse(input).asArray(Float16.self), inverse.asArray(Float16.self))
    }

    func testEmbeddingLookupAndTiedProjection() throws {
        let width = 512
        let transform = try SignedBlockHadamard(
            blockSize: 128, signs: (0 ..< width).map { $0 % 3 == 0 ? -1 : 1 })
        let weights = MLXArray(
            (0 ..< 4 * width).map { Float(($0 * 13) % 31 - 15) / 16 }, [4, width])
        for bits in [2] {
            for dtype in [DType.float32, .float16, .bfloat16] {
                let (packed, scales, biases) = quantized(
                    weights.asType(dtype), groupSize: 128, bits: bits)
                let embedding = try HadamardQuantizedEmbedding(
                    weight: packed, scales: scales, biases: biases,
                    groupSize: 128, bits: bits, transform: transform)
                let unfolded = transform.inverse(
                    dequantized(
                        packed, scales: scales, biases: biases, groupSize: 128, bits: bits))
                let ids = MLXArray([3, 0, 3, 1], [2, 2])
                let lookup = embedding(ids)
                XCTAssertEqual(lookup.shape, [2, 2, width])
                XCTAssertEqual(lookup.dtype, dtype)
                XCTAssertLessThan(abs(lookup - unfolded[ids]).max().item(Float.self), 1e-5)
                let input = weights[0 ..< 2].asType(dtype)
                // Published runtime contract: rotate activations then use packed
                // MM. Dense unfolded GEMM is a different reduction path,
                // not an exact oracle for the published packed operation.
                let reference = quantizedMM(
                    transform(input), packed, scales: scales,
                    biases: biases, groupSize: 128, bits: bits
                ).asType(.float32)
                let actual = embedding.asLinear(input).asType(.float32)
                XCTAssertEqual(actual.asArray(Float.self), reference.asArray(Float.self))
                XCTAssertTrue(embedding.trainableParameters().flattened().isEmpty)
                XCTAssertNil(quantizeSingle(layer: embedding))
            }
        }
    }

    func testGroupedGDNBeforeTransform() throws {
        let width = 512
        let layout = try HadamardGDNLayout(width: width, keyHeads: 2, valueHeads: 4)
        let values = (0 ..< 2 * width).map { Float($0 % 71) / 71 }
        let input = MLXArray(values, [1, 2, width])
        let permutation = [0, 2, 1, 3]
        var expected = [Float]()
        for row in 0 ..< 2 {
            for head in permutation {
                expected.append(
                    contentsOf: values[
                        (row * width + head * 128) ..< (row * width + (head + 1) * 128)])
            }
        }
        let grouped = MLXArray(expected, input.shape)
        XCTAssertEqual(layout(input).asArray(Float.self), expected)
        let transform = try SignedBlockHadamard(
            blockSize: 512,
            signs: (0 ..< width).map { $0 % 5 == 0 ? -1 : 1 })
        let weights = MLXArray(values, [2, width])
        let (packed, scales, biases) = quantized(weights, groupSize: 128, bits: 2)
        let layer = try HadamardQuantizedLinear(
            weight: packed,
            bias: MLXArray([Float(0.25), -0.25]), scales: scales, biases: biases,
            groupSize: 128, bits: 2, transform: transform, gdnLayout: layout)
        let reference =
            quantizedMM(
                transform(grouped), packed,
                scales: scales, biases: biases, groupSize: 128, bits: 2)
            + MLXArray([Float(0.25), -0.25])
        XCTAssertEqual(layer(input).asArray(Float.self), reference.asArray(Float.self))
        XCTAssertThrowsError(try HadamardGDNLayout(width: 512, keyHeads: 3, valueHeads: 4))
    }

    /// With `independentRows`, compare with independent original singleton
    /// calls, never a dense dequantized GEMM or the same broadcast
    /// implementation as the candidate.
    func testBonsaiFloat32RowsMatchSingletonPackedArithmetic() throws {
        let width = 1024
        let outputs = 32
        let transform = try SignedBlockHadamard(
            blockSize: width,
            signs: (0 ..< width).map { $0 % 3 == 0 ? -1 : 1 })
        let original = MLXArray(
            (0 ..< outputs * width).map {
                Float(($0 * 13) % 127 - 63) / 63
            }, [outputs, width]
        ).asType(.float16)
        let (weight, scales, biases) = quantized(original, groupSize: 128, bits: 2)
        for grouped in [false, true] {
            let layout =
                grouped ? try HadamardGDNLayout(width: width, keyHeads: 2, valueHeads: 4) : nil
            for hasBias in [false, true] {
                let bias =
                    hasBias
                    ? (MLXArray(0 ..< outputs).asType(.float32) / 100).asType(.float16) : nil
                let layer = try HadamardQuantizedLinear(
                    weight: weight, bias: bias,
                    scales: scales, biases: biases, groupSize: 128, bits: 2,
                    transform: transform, gdnLayout: layout, independentRows: true)
                XCTAssertTrue(layer.independentRows)
                let names = layer.parameters().flattened().map(\.0).sorted()
                for shape in [
                    [width], [1, width], [2, 3, width], [1, 511, width], [1, 512, width],
                ] {
                    let elements = shape.reduce(1, *)
                    let input = MLXArray(
                        (0 ..< elements).map {
                            Float(($0 * 29) % 251 - 125) / 127
                        }, shape)
                    let flat = input.reshaped([-1, width])
                    let expected = concatenated(
                        (0 ..< flat.dim(0)).map { row in
                            layer(flat[row ..< (row + 1)])
                        }, axis: 0
                    ).reshaped(Array(shape.dropLast()) + [outputs])
                    let actual = layer(input)
                    XCTAssertEqual(actual.dtype, .float32)
                    XCTAssertEqual(actual.shape, expected.shape)
                    XCTAssertEqual(
                        actual.asData(access: .copy).data, expected.asData(access: .copy).data)
                }
                XCTAssertEqual(layer.parameters().flattened().map(\.0).sorted(), names)
                XCTAssertTrue(layer.trainableParameters().flattened().isEmpty)
            }
        }
    }

    /// By default a prompt chunk of `minimumBatchedRows` or more FP32 rows
    /// takes one packed matmul over all rows, with the exactly widened FP32
    /// constants: single-Mac prefill speed. `independentRows` changes only the
    /// block1024 profile.
    func testBonsaiFloat32RowsDefaultToOneBatchedPackedMatmul() throws {
        let width = 1024
        let outputs = 32
        let original = MLXArray(
            (0 ..< outputs * width).map {
                Float(($0 * 13) % 127 - 63) / 63
            }, [outputs, width]
        ).asType(.float16)
        let (weight, scales, biases) = quantized(original, groupSize: 128, bits: 2)
        let bias = (MLXArray(0 ..< outputs).asType(.float32) / 100).asType(.float16)
        let grouped = try HadamardGDNLayout(width: width, keyHeads: 2, valueHeads: 4)
        for (blockSize, independentRows) in [(1024, false), (512, true)] {
            let transform = try SignedBlockHadamard(
                blockSize: blockSize,
                signs: (0 ..< width).map { $0 % 3 == 0 ? -1 : 1 })
            for layout in [nil, grouped] {
                let layer = try HadamardQuantizedLinear(
                    weight: weight, bias: bias,
                    scales: scales, biases: biases, groupSize: 128, bits: 2,
                    transform: transform, gdnLayout: layout,
                    independentRows: independentRows)
                XCTAssertEqual(layer.independentRows, independentRows)
                for shape in [[1, 512, width], [4, 128, width]] {
                    let elements = shape.reduce(1, *)
                    let input = MLXArray(
                        (0 ..< elements).map {
                            Float(($0 * 29) % 251 - 125) / 127
                        }, shape)
                    let expected =
                        quantizedMM(
                            transform(layout.map { $0(input) } ?? input), weight,
                            scales: scales.asType(.float32), biases: biases?.asType(.float32),
                            transpose: true, groupSize: 128, bits: 2)
                        + bias.asType(.float32)
                    let actual = layer(input)
                    XCTAssertEqual(actual.dtype, .float32)
                    XCTAssertEqual(actual.shape, expected.shape)
                    XCTAssertEqual(
                        actual.asData(access: .copy).data, expected.asData(access: .copy).data)
                }
            }
        }
    }

    /// A single Mac's call with fewer rows than `minimumBatchedRows` takes the
    /// per-row path (the bytes of separate singleton calls); from the threshold
    /// on it takes one batched packed matmul. The same for a decode step
    /// (`[B, 1, H]`, one position per request) as for a prompt (`[1, S, H]`).
    func testSingleMacRowsBatchFromTheThreshold() throws {
        let fixture = try BonsaiRowsFixture()
        let threshold = 3
        XCTAssertEqual(HadamardQuantizedLinear.minimumBatchedRows, threshold)
        for layout in [nil, fixture.grouped] {
            let layer = try fixture.layer(layout: layout)
            XCTAssertFalse(layer.independentRows)
            for rows in [2, 3, 4, 8, 11, 12, 13, 16, 24, 512] {
                for shape in [[rows, 1, fixture.width], [1, rows, fixture.width]] {
                    let input = fixture.input(shape)
                    let expected =
                        rows < threshold
                        ? fixture.singletons(layer, input) : fixture.batched(input, layout: layout)
                    XCTAssertEqual(
                        fixture.bytes(layer(input)), fixture.bytes(expected), "shape \(shape)")
                }
            }
        }
    }

    func testConfigurationValidation() throws {
        var metadata: [String: Any] = [
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 512,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["output.weight"],
            "prism.hadamard.inverse_weight_names": ["token_embd.weight"],
            "prism.hadamard.sign_widths": [512],
            "prism.hadamard.sign_values": Array(repeating: 1, count: 512),
            "prism.hadamard.gdn_v_grouped": true,
        ]
        func decode(_ object: [String: Any]) throws -> PrismHadamardConfiguration {
            try JSONDecoder().decode(
                PrismHadamardConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
        }
        let config = try decode(metadata)
        XCTAssertTrue(config.gdnVGrouped)
        XCTAssertEqual(try config.transform(forWidth: 512).width, 512)
        XCTAssertThrowsError(try config.transform(forWidth: 1024))
        for (key, invalid) in [
            ("prism.hadamard.version", 2),
            ("prism.hadamard.block_size", 513),
            ("prism.hadamard.sign_values", [0]),
            ("prism.hadamard.sign_widths", [Int.max]),
            ("prism.hadamard.sign_mode", "identity"),
            ("prism.hadamard.inverse_weight_names", ["output.weight"]),
        ] as [(String, Any)] {
            var candidate = metadata
            candidate[key] = invalid
            XCTAssertThrowsError(try decode(candidate), key)
        }
        metadata.removeValue(forKey: "prism.hadamard.gdn_v_grouped")
        XCTAssertFalse(try decode(metadata).gdnVGrouped)
    }

    func testRejectInvalidTransformAndPackedShape() throws {
        XCTAssertThrowsError(try SignedBlockHadamard(blockSize: 0, signs: []))
        XCTAssertThrowsError(try SignedBlockHadamard(blockSize: 16384, signs: [1]))
        XCTAssertThrowsError(try SignedBlockHadamard(blockSize: 2, signs: [1, .nan]))
        let transform = try SignedBlockHadamard(
            blockSize: 128, signs: Array(repeating: 1, count: 128))
        XCTAssertThrowsError(
            try HadamardQuantizedLinear(
                weight: MLXArray.zeros([2, 8]), scales: MLXArray.ones([2, 1]), biases: nil,
                groupSize: 128, bits: 2, transform: transform))
    }

    /// The bundled quantized kernels support {2, 3, 4, 5, 6, 8}, so the layer
    /// initializers must accept the non-power-of-two widths too. The packed width
    /// `width / 32 * bits` is the exact MLX affine layout for every one of them.
    func testAcceptsEveryKernelSupportedBitWidth() throws {
        let width = 512
        let transform = try SignedBlockHadamard(
            blockSize: 128, signs: (0 ..< width).map { $0 % 3 == 0 ? -1 : 1 })
        let weights = MLXArray(
            (0 ..< 4 * width).map { Float(($0 * 13) % 31 - 15) / 16 }, [4, width])
        for bits in [2, 3, 4, 5, 6, 8] {
            let (packed, scales, biases) = quantized(weights, groupSize: 128, bits: bits)
            XCTAssertEqual(packed.dim(1), width / 32 * bits, "packed width for \(bits) bits")
            let linear = try HadamardQuantizedLinear(
                weight: packed, scales: scales, biases: biases,
                groupSize: 128, bits: bits, transform: transform)
            let input = weights[0 ..< 2]
            let reference = quantizedMM(
                transform(input), packed,
                scales: scales, biases: biases, groupSize: 128, bits: bits)
            XCTAssertEqual(
                linear(input).asArray(Float.self), reference.asArray(Float.self), "bits \(bits)")
        }
    }

    /// Retain the unfolded-basis algebra check on CPU, separate from the exact
    /// GPU packed-operation oracle above. Do not loosen a GPU error threshold
    /// to disguise the different dense GEMM/reduction path.
    func testUnfoldedBasisAlgebraOnCPU() throws {
        try Device.withDefaultDevice(.cpu) {
            let width = 512
            let transform = try SignedBlockHadamard(
                blockSize: 128,
                signs: (0 ..< width).map { $0 % 3 == 0 ? -1 : 1 })
            let weight = MLXArray(
                (0 ..< 4 * width).map { Float(($0 * 13) % 31 - 15) / 16 }, [4, width])
            let (packed, scales, biases) = quantized(weight, groupSize: 128, bits: 2)
            let layer = try HadamardQuantizedLinear(
                weight: packed, scales: scales,
                biases: biases, groupSize: 128, bits: 2, transform: transform)
            let unfolded = transform.inverse(
                dequantized(
                    packed, scales: scales,
                    biases: biases, groupSize: 128, bits: 2))
            let reference = weight[0 ..< 2].matmul(unfolded.T)
            let delta = layer(weight[0 ..< 2]) - reference
            let relative = (delta * delta).sum().sqrt() / (reference * reference).sum().sqrt()
            XCTAssertLessThan(relative.item(Float.self), 1e-4)
        }
    }
}

/// Bonsai 27B's block1024/affine2 profile with FP16 scales and a bias, 4,352
/// outputs wide (above 4,096, so MLX's vector limit is the one its wide
/// projections meet), and its two oracles: separate singleton calls, and one
/// `quantizedMM` over every row with the exactly widened FP32 constants.
private struct BonsaiRowsFixture {
    let width = 1024
    let outputs = 4352
    let transform: SignedBlockHadamard
    let grouped: HadamardGDNLayout
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let bias: MLXArray

    init() throws {
        // Named [Float] values keep the MLXArray initializer overloads cheap to type-check.
        let signs: [Float] = (0 ..< 1024).map { $0 % 3 == 0 ? -1 : 1 }
        transform = try SignedBlockHadamard(blockSize: 1024, signs: signs)
        grouped = try HadamardGDNLayout(width: 1024, keyHeads: 2, valueHeads: 4)
        let values: [Float] = (0 ..< 4352 * 1024).map { Float(($0 * 13) % 127 - 63) / 63 }
        let original = MLXArray(values, [4352, 1024]).asType(.float16)
        (weight, scales, biases) = quantized(original, groupSize: 128, bits: 2)
        bias = (MLXArray(0 ..< 4352).asType(.float32) / 1000).asType(.float16)
    }

    /// The single-Mac loader's call: no `independentRows` argument.
    func layer(layout: HadamardGDNLayout?) throws -> HadamardQuantizedLinear {
        try HadamardQuantizedLinear(
            weight: weight, bias: bias, scales: scales, biases: biases,
            groupSize: 128, bits: 2, transform: transform, gdnLayout: layout)
    }

    func input(_ shape: [Int]) -> MLXArray {
        let values: [Float] = (0 ..< shape.reduce(1, *)).map { Float(($0 * 29) % 251 - 125) / 127 }
        return MLXArray(values, shape)
    }

    func singletons(_ layer: HadamardQuantizedLinear, _ input: MLXArray) -> MLXArray {
        let flat = input.reshaped([-1, width])
        return concatenated((0 ..< flat.dim(0)).map { layer(flat[$0 ..< ($0 + 1)]) }, axis: 0)
            .reshaped(Array(input.shape.dropLast()) + [outputs])
    }

    func batched(_ input: MLXArray, layout: HadamardGDNLayout?) -> MLXArray {
        quantizedMM(
            transform(layout.map { $0(input) } ?? input), weight,
            scales: scales.asType(.float32), biases: biases?.asType(.float32),
            transpose: true, groupSize: 128, bits: 2) + bias.asType(.float32)
    }

    func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }
}
