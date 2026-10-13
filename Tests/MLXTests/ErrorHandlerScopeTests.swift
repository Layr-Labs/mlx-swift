// Copyright © 2026 Eigen Labs.

import Foundation
import XCTest

@testable import MLX

/// The message prefix of the broadcast error made by `makeBroadcastError()`.
private let broadcastMessage = "[broadcast_shapes] Shapes (2,5) and (3,5) cannot be broadcast"

/// Make an op with shapes that cannot broadcast. MLX reports the error to the active handler.
private func makeBroadcastError() -> MLXArray {
    MLXArray(0 ..< 10, [2, 5]) + MLXArray(0 ..< 15, [3, 5])
}

/// Thread safe store for messages that a handler receives.
private final class MessageRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _messages = [String]()
    private var _released = false

    var messages: [String] { lock.withLock { _messages } }
    var released: Bool { lock.withLock { _released } }

    func record(_ message: String) {
        lock.withLock { _messages.append(message) }
    }

    func markReleased() {
        lock.withLock { _released = true }
    }
}

/// Tests for `ErrorHandler.swift`: scoped handlers, `withError`, `ErrorBox`, `MLXError` and the
/// global handler.
class ErrorHandlerScopeTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - MLXError and ErrorBox

    func testMLXErrorDescription() {
        let error = MLXError.caught("bad shape")
        XCTAssertEqual(error.errorDescription, "MLX Error: bad shape")
        XCTAssertEqual(error.localizedDescription, "MLX Error: bad shape")
        XCTAssertEqual(error, MLXError.caught("bad shape"))
        XCTAssertNotEqual(error, MLXError.caught("other"))
    }

    func testErrorBoxKeepsFirstError() throws {
        let box = ErrorBox()
        XCTAssertNil(box.firstError)

        // no error: check() does not throw
        try box.check()

        box.firstError = MLXError.caught("first")
        box.firstError = MLXError.caught("second")
        XCTAssertEqual(box.firstError as? MLXError, MLXError.caught("first"))

        XCTAssertThrowsError(try box.check()) { error in
            XCTAssertEqual(error as? MLXError, MLXError.caught("first"))
        }
    }

    // MARK: - withError (sync)

    func testWithErrorReturnsValueWhenNoError() throws {
        let value = try withError { (error: ErrorBox) throws -> Int in
            let a = MLXArray([1, 2, 3] as [Int32]) + 1
            try error.check()
            return a.sum().item(Int.self)
        }
        XCTAssertEqual(value, 9)

        let plain = try withError {
            (MLXArray([2, 3] as [Int32]) * 2).asArray(Int32.self)
        }
        XCTAssertEqual(plain, [4, 6])
    }

    func testWithErrorThrowsMLXError() {
        XCTAssertThrowsError(
            try withError { (error: ErrorBox) throws -> Void in
                _ = makeBroadcastError()
                try error.check()
                XCTFail("check() should throw")
            }
        ) { error in
            guard case .caught(let message)? = error as? MLXError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(message.hasPrefix(broadcastMessage), message)
        }
    }

    // MARK: - async forms

    func testWithErrorHandlerAsync() async {
        let recorder = MessageRecorder()

        let handler: @Sendable (String) -> Void = { recorder.record($0) }
        let result = await withErrorHandler(handler) { () async -> Int in
            await Task.yield()
            _ = makeBroadcastError()
            return 7
        }

        XCTAssertEqual(result, 7)
        XCTAssertFalse(recorder.messages.isEmpty)
        XCTAssertTrue(recorder.messages[0].hasPrefix(broadcastMessage), "\(recorder.messages)")
    }

    func testWithErrorAsyncWithErrorBox() async {
        do {
            try await withError { (error: ErrorBox) async throws -> Void in
                await Task.yield()
                _ = makeBroadcastError()
                try error.check()
                XCTFail("check() should throw")
            }
            XCTFail("withError should throw")
        } catch {
            guard case .caught(let message)? = error as? MLXError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(message.hasPrefix(broadcastMessage), message)
        }

        // no error: the value is returned
        do {
            let value = try await withError { (error: ErrorBox) async throws -> Int in
                await Task.yield()
                try error.check()
                return 3
            }
            XCTAssertEqual(value, 3)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    // MARK: - global handler

    @available(*, deprecated)
    func testGlobalHandlerReceivesErrors() {
        let recorder = MessageRecorder()
        let data = Unmanaged.passRetained(recorder).toOpaque()

        setErrorHandler(
            { message, data in
                let recorder = Unmanaged<MessageRecorder>.fromOpaque(data!).takeUnretainedValue()
                recorder.record(message.map { String(cString: $0) } ?? "")
            },
            data: data,
            dtor: { data in
                let recorder = Unmanaged<MessageRecorder>.fromOpaque(data!)
                recorder.takeUnretainedValue().markReleased()
                recorder.release()
            })

        // always put back the default handler; this also calls the dtor for `data`
        var restored = false
        defer {
            if !restored {
                setErrorHandler(nil)
            }
        }

        // no scoped handler is active, so the global handler gets the error
        _ = makeBroadcastError()
        XCTAssertFalse(recorder.messages.isEmpty)
        XCTAssertTrue(recorder.messages[0].hasPrefix(broadcastMessage), "\(recorder.messages)")

        // a scoped handler takes precedence over the global handler
        let count = recorder.messages.count
        let scoped = MessageRecorder()
        let scopedHandler: @Sendable (String) -> Void = { scoped.record($0) }
        withErrorHandler(scopedHandler) {
            _ = makeBroadcastError()
        }
        XCTAssertFalse(scoped.messages.isEmpty)
        XCTAssertEqual(recorder.messages.count, count)

        // replacing the handler calls the dtor of the old data
        XCTAssertFalse(recorder.released)
        setErrorHandler(nil)
        restored = true
        XCTAssertTrue(recorder.released)
    }
}
