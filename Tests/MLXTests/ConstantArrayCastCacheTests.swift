import Foundation
@_spi(QuantizedConstantCache) import MLX
import XCTest

final class ConstantArrayCastCacheTests: XCTestCase {
    func testReusesBackingAndInvalidatesSameSwiftObjectUpdates() {
        let cache = ConstantArrayCastCache()
        let source = MLXArray([1.25, -2.5] as [Float]).asType(.bfloat16)
        let first = cache.cachedCast(source, to: .float32)!
        XCTAssertTrue(first === cache.cachedCast(source, to: .float32)!)
        XCTAssertEqual(first.asArray(Float.self), [1.25, -2.5])

        source._updateInternal(MLXArray([3.0, 4.0] as [Float]).asType(.bfloat16))
        let updated = cache.cachedCast(source, to: .float32)!
        XCTAssertFalse(first === updated)
        XCTAssertEqual(updated.asArray(Float.self), [3, 4])
        XCTAssertEqual(first.asArray(Float.self), [1.25, -2.5])

        source[0] = MLXArray(Float(7)).asType(.bfloat16)
        XCTAssertEqual(cache.cachedCast(source, to: .float32)!.asArray(Float.self), [7, 4])
        cache.clear()
        XCTAssertFalse(updated === cache.cachedCast(source, to: .float32)!)
    }

    func testOtherDtypesAreUnchanged() {
        let cache = ConstantArrayCastCache()
        for dtype in [DType.uint8, .float16, .float32] {
            let source = MLXArray([1, 2]).asType(dtype)
            XCTAssertNil(cache.cachedCast(source, to: .float32))
        }
        let source = MLXArray([1, 2]).asType(.bfloat16)
        XCTAssertNil(cache.cachedCast(source, to: .bfloat16))
        XCTAssertNil(ConstantArrayCastCache(enabled: false).cachedCast(source, to: .float32))
    }

    func testTracingDoesNotRetainConstantsOrGradients() {
        let cache = ConstantArrayCastCache()
        let f: (MLXArray) -> MLXArray = { x in
            (cache.cachedCast(x, to: .float32) ?? x.asType(.float32)).square().sum()
        }
        let compiled = compile(f)
        for value in [Float(2), Float(3)] {
            let x = MLXArray([value, value]).asType(.bfloat16)
            XCTAssertEqual(compiled(x).item(Float.self), value * value * 2)
            XCTAssertEqual(grad(f)(x).asArray(Float.self), [value * 2, value * 2])
        }
    }
}
