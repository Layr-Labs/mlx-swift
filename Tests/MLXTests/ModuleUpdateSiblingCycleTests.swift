import Foundation
import MLX
import MLXNN
import XCTest

/// Buffer accounting for `Module.update` on a quantized layer whose weight, scales and
/// biases are all replaced. `Memory.numResources` is the live Metal buffer count.
class ModuleUpdateSiblingCycleTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// `Memory.numResources` counts Metal buffers. Without Metal the count is always 0 and
    /// these tests would assert nothing, so that is a failure, not a pass or a skip.
    private func requireMetalResourceCount(
        file: StaticString = #filePath, line: UInt = #line
    ) -> Bool {
        XCTAssertGreaterThan(
            Memory.resourceLimit, 0, "these tests need the Metal backend", file: file, line: line)
        return Memory.resourceLimit > 0
    }

    private func makeLayer() -> QuantizedLinear {
        QuantizedLinear(Linear(64, 64, bias: false))
    }

    /// Runs `cycles` build/update/release cycles and asserts the count returns to the
    /// baseline after each. `mutate` receives the live layer.
    private func runCycles(
        _ cycles: Int, file: StaticString = #filePath, line: UInt = #line,
        mutate: (QuantizedLinear) throws -> Void
    ) rethrows {
        // Allocate one-time global state (random key) before taking the baseline.
        _ = Linear(8, 8, bias: false)
        Memory.clearCache()
        let baseline = Memory.numResources

        for cycle in 1 ... cycles {
            var layer: QuantizedLinear? = makeLayer()
            // AT2: positive control. A backend that never allocates cannot pass this test.
            XCTAssertGreaterThan(
                Memory.numResources, baseline,
                "cycle \(cycle): live layer should own buffers", file: file, line: line)
            try mutate(layer!)
            layer = nil
            Memory.clearCache()
            XCTAssertEqual(
                Memory.numResources, baseline,
                "cycle \(cycle): buffer count did not return to baseline", file: file, line: line)
        }
    }

    /// AT1 (with AT2 as the positive control inside `runCycles`).
    func testOverwritingAllQuantizedOutputsReleasesTheirBuffers() throws {
        guard requireMetalResourceCount() else { return }

        try runCycles(3) { layer in
            try layer.update(
                parameters: ModuleParameters.unflattened([
                    ("weight", MLXArray.zeros([1])),
                    ("scales", MLXArray.zeros([1])),
                    ("biases", MLXArray.zeros([1])),
                ]),
                verify: [.noUnusedKeys])
        }
    }

    /// Control: no overwrite at all.
    func testReleasingAnUntouchedQuantizedLayerReleasesItsBuffers() throws {
        guard requireMetalResourceCount() else { return }

        runCycles(3) { _ in }
    }

    /// Control: the leak is specific to overwriting every output, not two of three.
    func testOverwritingTwoOfThreeOutputsReleasesTheirBuffers() throws {
        guard requireMetalResourceCount() else { return }

        runCycles(3) { layer in
            layer.weight._updateInternal(MLXArray.zeros([1]))
            layer.scales._updateInternal(MLXArray.zeros([1]))
        }
    }

    /// AT3: update still installs the new values.
    func testUpdateStillInstallsTheNewValues() throws {
        let layer = makeLayer()
        let biases = try XCTUnwrap(layer.biases)

        let newWeight = MLXArray.ones(layer.weight.shape, dtype: layer.weight.dtype)
        let newScales = MLXArray.ones(layer.scales.shape, dtype: layer.scales.dtype) * 3
        let newBiases = MLXArray.ones(biases.shape, dtype: biases.dtype) * 5
        eval(newWeight, newScales, newBiases)

        try layer.update(
            parameters: ModuleParameters.unflattened([
                ("weight", newWeight),
                ("scales", newScales),
                ("biases", newBiases),
            ]),
            verify: [.all])

        XCTAssertTrue(layer.weight.arrayEqual(newWeight).item(Bool.self))
        XCTAssertTrue(layer.scales.arrayEqual(newScales).item(Bool.self))
        XCTAssertTrue(try XCTUnwrap(layer.biases).arrayEqual(newBiases).item(Bool.self))
    }

    /// An update whose source cannot be copied reports the error and keeps the old value.
    func testFailedUpdateKeepsTheOldValue() {
        let parameter = MLXArray([1, 2, 3] as [Float])
        XCTAssertThrowsError(try withError { parameter._updateInternal(.mlxNone) })
        XCTAssertEqual(parameter.asArray(Float.self), [1, 2, 3])
    }
}
