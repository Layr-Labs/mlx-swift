// Copyright © 2026 Eigen Labs.

import MLX
import XCTest

final class SortedGatherQuantizedTailTests: XCTestCase {
    func testSortedExpertProjectionKeepsPartialRowsAndReductionTiles() {
        for mode: QuantizationMode in [.affine, .mxfp4, .mxfp8] {
            for dtype: DType in [.float32, .float16, .bfloat16] {
                for k in [32, 64, 96, 128] {
                    for n in [32, 64, 96, 128] {
                        check(
                            k: k, n: n, groupSize: 32, mode: mode, dtype: dtype,
                            bits: mode == .mxfp8 ? 8 : 4, counts: [17, 3, 31, 14])
                    }
                }
            }
        }
    }

    func testNVFP4PartialActivationFragmentsAndBothRowTiles() {
        for dtype: DType in [.float16, .bfloat16] {
            for k in [16, 48, 80] {
                for n in [32, 96] {
                    for counts in [[17, 3, 31, 14], [65, 64, 65, 66]] {
                        check(
                            k: k, n: n, groupSize: 16, mode: .nvfp4,
                            dtype: dtype, bits: 4, counts: counts)
                    }
                }
            }
        }
    }

    func testAffineOtherPackingWidthsAndScaleGroups() {
        for dtype: DType in [.float16, .bfloat16] {
            for bits in [2, 3, 4, 5, 6, 8] {
                for groupSize in [64, 128] {
                    for n in [32, 96] {
                        for counts in [[17, 3, 31, 14], [65, 64, 65, 66]] {
                            check(
                                k: 128, n: n, groupSize: groupSize, mode: .affine,
                                dtype: dtype, bits: bits, counts: counts)
                        }
                    }
                }
            }
        }
    }

    func testReductionTailCannotReadFollowingNaNRow() {
        for mode: QuantizationMode in [.affine, .mxfp4, .mxfp8, .nvfp4] {
            for dtype: DType in [.float16, .bfloat16] {
                check(
                    k: mode == .nvfp4 ? 48 : 96, n: 96,
                    groupSize: mode == .nvfp4 ? 16 : 32, mode: mode, dtype: dtype,
                    bits: mode == .mxfp8 ? 8 : 4, counts: [17, 3, 31, 14], poison: true)
            }
        }
    }

    private func check(
        k: Int, n: Int, groupSize: Int, mode: QuantizationMode,
        dtype: DType, bits: Int, counts: [Int], poison: Bool = false,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        // Powers of two and integral inputs have exactly representable sums.
        // NaN rows detect reads beyond a finite row's reduction extent.
        let experts: [Float] = [0.25, 0.5, 1, 2]
        let indices = counts.enumerated().flatMap { expert, count in
            Array(repeating: UInt32(expert), count: count)
        }
        let weights = MLXArray(
            experts.flatMap {
                Array(repeating: $0, count: n * k)
            }, [experts.count, n, k]
        ).asType(dtype)
        let packed: (wq: MLXArray, scales: MLXArray, biases: MLXArray?)
        if mode == .nvfp4 {
            // E2M1 code 2 is 1; these E4M3 scales are exactly 1/4, 1/2, 1, 2.
            // Direct packing isolates tile bounds from lossy FP8 quantization.
            packed = (
                MLXArray(
                    Array(repeating: UInt32(0x2222_2222), count: 4 * n * k / 8), [4, n, k / 8]),
                MLXArray(
                    [UInt8(40), 48, 56, 64].flatMap {
                        Array(repeating: $0, count: n * k / 16)
                    }, [4, n, k / 16]), nil
            )
        } else {
            packed = quantized(weights, groupSize: groupSize, bits: bits, mode: mode)
        }
        let scalars = indices.indices.map { row in
            poison && row % 2 == 1 ? Float.nan : Float(row % 3 + 1)
        }
        let x = MLXArray(
            scalars.flatMap { Array(repeating: $0, count: k) },
            [indices.count, 1, k]
        ).asType(dtype)
        let actual = gatherQuantizedMM(
            x, packed.wq, scales: packed.scales,
            biases: packed.biases, rhsIndices: MLXArray(indices), transpose: true,
            groupSize: groupSize, bits: bits, mode: mode, sortedIndices: true)
        let expected = indices.enumerated().flatMap { row, expert in
            Array(repeating: Float(k) * scalars[row] * experts[Int(expert)], count: n)
        }
        eval(actual)
        XCTAssertEqual(actual.shape, [indices.count, 1, n], file: file, line: line)
        let result = actual.asType(.float32).asArray(Float.self)
        let mismatch = zip(result, expected).enumerated().first {
            $0.element.0 != $0.element.1 && !($0.element.0.isNaN && $0.element.1.isNaN)
        }
        XCTAssertNil(
            mismatch,
            "mode=\(mode) dtype=\(dtype) K=\(k) N=\(n) group=\(groupSize) bits=\(bits) rows=\(indices.count) poison=\(poison)",
            file: file, line: line)
    }
}
