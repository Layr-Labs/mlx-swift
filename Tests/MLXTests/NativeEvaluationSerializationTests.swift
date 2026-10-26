// Copyright © 2026 Eigen Labs.

import Cmlx
import Foundation
import XCTest

@_spi(Execution) @testable import MLX

final class NativeEvaluationSerializationTests: XCTestCase {
    /// Each queue exclusively owns one set of graphs, constructed before either
    /// queue starts. Only evaluation overlaps, as in a model and native intake.
    private final class Work: @unchecked Sendable {
        let arrays: [MLXArray]
        init(offset: Int) {
            arrays = (0 ..< 128).map { index in
                let value = Float(index + offset)
                return MLXArray([value, value + 1]) * 2 + 3
            }
        }
    }

    func testCheckedNativeFencesShareSwiftEvaluationSerialization() {
        let swift = Work(offset: 0)
        let native = Work(offset: 1000)
        let finished = DispatchGroup()
        finished.enter()
        DispatchQueue.global().async {
            defer { finished.leave() }
            for (index, array) in swift.arrays.enumerated() {
                eval(array)
                XCTAssertEqual(
                    array.asArray(Float.self), [Float(index * 2 + 3), Float(index * 2 + 5)])
            }
        }
        finished.enter()
        DispatchQueue.global().async {
            defer { finished.leave() }
            for (index, array) in native.arrays.enumerated() {
                withSerializedEvaluation {
                    XCTAssertEqual(mlx_array_eval(array.ctx), 0)
                    XCTAssertEqual(mlx_synchronize(StreamOrDevice.cpu.ctx), 0)
                    XCTAssertEqual(mlx_synchronize(StreamOrDevice.gpu.ctx), 0)
                    // The lock is recursive: a checked integration can inspect
                    // the completed result through Swift's ordinary read API.
                    XCTAssertEqual(
                        array.asArray(Float.self),
                        [Float((index + 1000) * 2 + 3), Float((index + 1000) * 2 + 5)])
                }
            }
        }
        finished.wait()
    }
}
