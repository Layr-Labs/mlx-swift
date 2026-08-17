// Copyright © 2024 Apple Inc.

//
//  Copyright © 2023 Apple. All rights reserved.
//

import Foundation
import XCTest

@testable import Cmlx

class CmlxTests: XCTestCase {

    private func graphDOT(_ output: mlx_array) -> String {
        guard let file = tmpfile() else {
            XCTFail("unable to create graph output file")
            return ""
        }
        defer { fclose(file) }

        let namer = mlx_node_namer_new()
        defer { XCTAssertEqual(mlx_node_namer_free(namer), 0) }
        let outputs = mlx_vector_array_new_value(output)
        defer { XCTAssertEqual(mlx_vector_array_free(outputs), 0) }

        XCTAssertEqual(mlx_export_to_dot(file, namer, outputs), 0)
        fflush(file)
        rewind(file)

        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let count = buffer.withUnsafeMutableBytes { bytes in
                fread(bytes.baseAddress, 1, bytes.count, file)
            }
            guard count > 0 else { break }
            data.append(contentsOf: buffer.prefix(count))
        }
        return String(decoding: data, as: UTF8.self)
    }

    func testMinimal() throws {
        // smoke test making sure we can build, link & call C api
        //
        // note: there are convenience wrappers in MLX + the entire
        // wrapping of the API in swift

        var data: [Float] = [1, 2, 3, 4, 5, 6]
        var shape: [Int32] = [2, 3]

        let arr = mlx_array_new_data(&data, &shape, 2, MLX_FLOAT32)
        defer { mlx_array_free(arr) }

        var str = mlx_string_new()
        mlx_array_tostring(&str, arr)
        defer { mlx_string_free(str) }
        let description = String(cString: mlx_string_data(str))

        print(description)
    }

    func testForceFusedSDPASelectsPrimitive() {
        var queryData = [UInt16](repeating: 0, count: 9 * 256)
        var keyValueData = [UInt16](repeating: 0, count: 17 * 256)
        var queryShape: [Int32] = [1, 1, 9, 256]
        var keyValueShape: [Int32] = [1, 1, 17, 256]

        let queries = mlx_array_new_data(&queryData, &queryShape, 4, MLX_FLOAT16)
        defer { mlx_array_free(queries) }
        let keys = mlx_array_new_data(&keyValueData, &keyValueShape, 4, MLX_FLOAT16)
        defer { mlx_array_free(keys) }
        let values = mlx_array_new_data(&keyValueData, &keyValueShape, 4, MLX_FLOAT16)
        defer { mlx_array_free(values) }
        let none = mlx_array_new()
        defer { mlx_array_free(none) }
        let stream = mlx_default_gpu_stream_new()
        defer { mlx_stream_free(stream) }

        var defaultResult = mlx_array_new()
        defer { mlx_array_free(defaultResult) }
        XCTAssertEqual(
            mlx_fast_scaled_dot_product_attention(
                &defaultResult, queries, keys, values, 1.0, "", none, none, stream),
            0)

        var forcedResult = mlx_array_new()
        defer { mlx_array_free(forcedResult) }
        XCTAssertEqual(
            mlx_fast_scaled_dot_product_attention_v2(
                &forcedResult, queries, keys, values, 1.0, "", none, none, true, stream),
            0)

        XCTAssertFalse(graphDOT(defaultResult).contains("ScaledDotProductAttention"))
        XCTAssertTrue(graphDOT(forcedResult).contains("ScaledDotProductAttention"))
    }

}
