// Copyright © 2024 Apple Inc.

import Cmlx

/// Marker protocol for types that can be used in the `template` of a kernel call.
///
/// Currently:
/// - `Int`
/// - `Bool`
/// - `DType`
///
/// See also: ``MLXFast/MLXFastKernel``
public protocol KernelTemplateArg {}

extension Bool: KernelTemplateArg {}
extension Int: KernelTemplateArg {}
extension DType: KernelTemplateArg {}

#if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)

    extension MLXFast {

        /// Container for a kernel created by
        /// ``callAsFunction(_:template:grid:threadGroup:outputShapes:outputDTypes:initValue:verbose:stream:)``
        ///
        /// The ``callAsFunction(_:template:grid:threadGroup:outputShapes:outputDTypes:initValue:verbose:stream:)`` can be used to evaluate the kernel with inputs:
        ///
        /// ```swift
        /// let a = normal([2, 2])
        /// let kernel = MLXFast.metalKernel(
        ///     name: "basic",
        ///     inputNames: ["a"],
        ///     outputNames: ["out1"],
        ///     source: """
        ///         uint elem = thread_position_in_grid.x;
        ///         out1[elem] = a[elem];
        ///     """,
        ///     grid: (4, 1, 1),
        ///     threadGroup: (2, 1, 1),
        ///     outputShapes: [[2, 2]],
        ///     outputDTypes: [.float32])
        ///
        /// let out = kernel([a])
        /// ```
        final public class MLXFastKernel: @unchecked Sendable {
            let kernel: mlx_fast_metal_kernel
            public let outputNames: [String]

            init(
                name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
                source: String, header: String = "",
                ensureRowContiguous: Bool = true,
                atomicOutputs: Bool = false,
                mutableInputs: [String] = [],
                fencedInputs: [String] = []
            ) {
                self.outputNames = Array(outputNames)

                let input_names = mlx_vector_string_new()
                defer { mlx_vector_string_free(input_names) }
                for name in inputNames {
                    mlx_vector_string_append_value(input_names, name)
                }

                let output_names = mlx_vector_string_new()
                defer { mlx_vector_string_free(output_names) }
                for name in self.outputNames {
                    mlx_vector_string_append_value(output_names, name)
                }

                if !fencedInputs.isEmpty {
                    let mutable_names = mlx_vector_string_new()
                    defer { mlx_vector_string_free(mutable_names) }
                    for name in mutableInputs {
                        mlx_vector_string_append_value(mutable_names, name)
                    }
                    let fenced_names = mlx_vector_string_new()
                    defer { mlx_vector_string_free(fenced_names) }
                    for name in fencedInputs {
                        mlx_vector_string_append_value(fenced_names, name)
                    }
                    self.kernel = mlx_fast_metal_kernel_new_fenced(
                        name.cString(using: .utf8), input_names, output_names,
                        source.cString(using: .utf8), header.cString(using: .utf8),
                        ensureRowContiguous, atomicOutputs, mutable_names, fenced_names)
                } else if mutableInputs.isEmpty {
                    self.kernel = mlx_fast_metal_kernel_new(
                        name.cString(using: .utf8), input_names, output_names,
                        source.cString(using: .utf8), header.cString(using: .utf8),
                        ensureRowContiguous, atomicOutputs)
                } else {
                    let mutable_names = mlx_vector_string_new()
                    defer { mlx_vector_string_free(mutable_names) }
                    for name in mutableInputs {
                        mlx_vector_string_append_value(mutable_names, name)
                    }
                    self.kernel = mlx_fast_metal_kernel_new_mutable(
                        name.cString(using: .utf8), input_names, output_names,
                        source.cString(using: .utf8), header.cString(using: .utf8),
                        ensureRowContiguous, atomicOutputs, mutable_names)
                }
            }

            deinit {
                mlx_fast_metal_kernel_free(kernel)
            }

            /// Call the prepared metal kernel.
            ///
            /// See ``MLXFastKernel`` for example.  Use
            /// ``metalKernel(name:inputNames:outputNames:source:header:ensureRowContiguous:atomicOutputs:)``
            /// to create an instance.
            ///
            /// - Parameters:
            ///   - inputs: inputs passed to the metal kernel
            ///   - template: template arguments
            ///   - grid: 3-tuple specifying the grid to launch the kernel with
            ///   - threadGroup: 3-tuple specifying the threadgroup size to use
            ///   - outputShapes: list of shapes for each output in ``outputNames``
            ///   - outputDTypes: list of data types for each output in ``outputNames``
            ///   - initValue: optional value to use to initialize all of the output arrays
            ///   - verbose: if true will print the full generated source code of the kernel when run
            ///   - stream: stream to run on
            /// - Returns: array of `MLXArray`
            public func callAsFunction(
                _ inputs: [any ScalarOrArray],
                template: [(String, any KernelTemplateArg)]? = nil,
                grid: (Int, Int, Int),
                threadGroup: (Int, Int, Int),
                outputShapes: some Sequence<[Int]>,
                outputDTypes: some Sequence<DType>,
                initValue: Float? = nil,
                verbose: Bool = false,
                stream: StreamOrDevice = .default
            ) -> [MLXArray] {
                callAsFunction(
                    inputs, template: template, grid: grid, threadGroup: threadGroup,
                    outputShapes: Array(outputShapes), outputDTypes: Array(outputDTypes),
                    initValue: initValue, verbose: verbose, stream: stream)
            }

            /// ``callAsFunction(_:template:grid:threadGroup:outputShapes:outputDTypes:initValue:verbose:stream:)``
            /// with the output shapes and dtypes as arrays: the same launch without
            /// the generic sequence iteration, which costs more host time than the
            /// rest of the call's Swift side.
            public func callAsFunction(
                _ inputs: [any ScalarOrArray],
                template: [(String, any KernelTemplateArg)]? = nil,
                grid: (Int, Int, Int),
                threadGroup: (Int, Int, Int),
                outputShapes: [[Int]],
                outputDTypes: [DType],
                initValue: Float? = nil,
                verbose: Bool = false,
                stream: StreamOrDevice = .default
            ) -> [MLXArray] {
                let config = Self.makeConfig(
                    template: template, grid: grid, threadGroup: threadGroup,
                    outputShapes: outputShapes, outputDTypes: outputDTypes,
                    initValue: initValue, verbose: verbose)
                defer { mlx_fast_metal_kernel_config_free(config) }

                let inputs = new_mlx_vector_array(inputs.map { $0.asMLXArray(dtype: nil) })
                defer { mlx_vector_array_free(inputs) }

                var result = mlx_vector_array_new()
                mlx_fast_metal_kernel_apply(&result, kernel, inputs, config, stream.ctx)
                defer { mlx_vector_array_free(result) }

                return mlx_vector_array_values(result)
            }

            /// A launch of this kernel with the template arguments, grid,
            /// threadgroup and outputs bound once: calling it with inputs is
            /// ``callAsFunction(_:template:grid:threadGroup:outputShapes:outputDTypes:initValue:verbose:stream:)``
            /// with the same arguments, without rebuilding them per call.
            public final class Launch: @unchecked Sendable {
                let kernel: MLXFastKernel
                let handle: mlx_fast_metal_kernel_launch

                init(kernel: MLXFastKernel, config: mlx_fast_metal_kernel_config) {
                    self.kernel = kernel
                    self.handle = mlx_fast_metal_kernel_launch_new(kernel.kernel, config)
                }

                deinit {
                    mlx_fast_metal_kernel_launch_free(handle)
                }

                public func callAsFunction(_ inputs: [MLXArray], stream: StreamOrDevice = .default)
                    -> [MLXArray]
                {
                    let inputs = new_mlx_vector_array(inputs)
                    defer { mlx_vector_array_free(inputs) }

                    var result = mlx_vector_array_new()
                    mlx_fast_metal_kernel_launch_apply(&result, handle, inputs, stream.ctx)
                    defer { mlx_vector_array_free(result) }

                    return mlx_vector_array_values(result)
                }
            }

            /// Binds a launch's template arguments, grid, threadgroup and
            /// outputs once (see ``Launch``).
            public func prepare(
                template: [(String, any KernelTemplateArg)]? = nil,
                grid: (Int, Int, Int),
                threadGroup: (Int, Int, Int),
                outputShapes: [[Int]],
                outputDTypes: [DType],
                initValue: Float? = nil,
                verbose: Bool = false
            ) -> Launch {
                let config = Self.makeConfig(
                    template: template, grid: grid, threadGroup: threadGroup,
                    outputShapes: outputShapes, outputDTypes: outputDTypes,
                    initValue: initValue, verbose: verbose)
                defer { mlx_fast_metal_kernel_config_free(config) }
                return Launch(kernel: self, config: config)
            }

            /// A new config; the caller frees it.
            private static func makeConfig(
                template: [(String, any KernelTemplateArg)]?,
                grid: (Int, Int, Int),
                threadGroup: (Int, Int, Int),
                outputShapes: [[Int]],
                outputDTypes: [DType],
                initValue: Float?,
                verbose: Bool
            ) -> mlx_fast_metal_kernel_config {
                let config = mlx_fast_metal_kernel_config_new()

                if let template {
                    for (name, arg) in template {
                        name.withCString { name in
                            Self.addTemplateArg(config, name, arg)
                        }
                    }
                }

                mlx_fast_metal_kernel_config_set_grid(
                    config, grid.0.int32, grid.1.int32, grid.2.int32)
                mlx_fast_metal_kernel_config_set_thread_group(
                    config, threadGroup.0.int32, threadGroup.1.int32, threadGroup.2.int32)

                for index in 0 ..< min(outputShapes.count, outputDTypes.count) {
                    let shape = outputShapes[index]
                    withUnsafeTemporaryAllocation(of: Int32.self, capacity: max(1, shape.count))
                    { dims in
                        for (i, d) in shape.enumerated() { dims[i] = Int32(d) }
                        _ = mlx_fast_metal_kernel_config_add_output_arg(
                            config, dims.baseAddress, shape.count, outputDTypes[index].cmlxDtype)
                    }
                }

                if let initValue {
                    mlx_fast_metal_kernel_config_set_init_value(config, initValue)
                }

                mlx_fast_metal_kernel_config_set_verbose(config, verbose)
                return config
            }

            /// Adds one template argument. Opening the existential and comparing
            /// the type is much cheaper than a dynamic cast per supported type.
            private static func addTemplateArg<T: KernelTemplateArg>(
                _ config: mlx_fast_metal_kernel_config, _ name: UnsafePointer<CChar>, _ arg: T
            ) {
                if T.self == Int.self {
                    mlx_fast_metal_kernel_config_add_template_arg_int(
                        config, name, Int32(unsafeBitCast(arg, to: Int.self)))
                } else if T.self == Bool.self {
                    mlx_fast_metal_kernel_config_add_template_arg_bool(
                        config, name, unsafeBitCast(arg, to: Bool.self))
                } else if T.self == DType.self {
                    mlx_fast_metal_kernel_config_add_template_arg_dtype(
                        config, name, unsafeBitCast(arg, to: DType.self).cmlxDtype)
                } else {
                    fatalError(
                        "Unable to handle KernelTemplateArg \(String(cString: name)) with type: \(T.self)."
                    )
                }
            }
        }

        /// A jit-compiled custom Metal kernel defined from a source string.
        ///
        /// - Parameters:
        ///   - name: name for the kernel
        ///   - inputNames: parameter names of the inputs in the function signature
        ///   - outputNames: parameter names of the outputs in the function signature
        ///   - source: source code -- this is the body of a function in Metal,
        ///   the function signature will be automatically generated.
        ///   - header: header source code to include before the main function.  Useful
        ///   for helper functions or includes that should live outside of the main function body.
        ///   - ensureRowContiguous: whether to ensure the inputs are row contiguous
        ///   before the kernel runs (at a performance cost)
        ///   - atomicOutputs: whether to use atomic outputs in the function signature,
        ///   e.g. `device atomic<float>`
        ///   - mutableInputs: owned input buffers written by this kernel. The
        ///   caller must order mutations in the graph; this declares Metal
        ///   write access and forbids implicit contiguous copies of those inputs.
        /// - Returns: an ``MLXFastKernel`` -- see that for information on how to call it
        public static func metalKernel(
            name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
            source: String, header: String = "", ensureRowContiguous: Bool = true,
            atomicOutputs: Bool = false, mutableInputs: [String] = []
        ) -> MLXFastKernel {
            MLXFastKernel(
                name: name, inputNames: inputNames, outputNames: outputNames,
                source: source, header: header,
                ensureRowContiguous: ensureRowContiguous, atomicOutputs: atomicOutputs,
                mutableInputs: mutableInputs
            )
        }

        /// ``metalKernel(name:inputNames:outputNames:source:header:ensureRowContiguous:atomicOutputs:mutableInputs:)``
        /// with fenced inputs.
        ///
        /// A fenced input can come from a CPU stream in the same evaluation, e.g. a
        /// CPU-stream all-sum. With `MLX_METAL_FENCED_INPUTS=1` (and
        /// `MLX_METAL_FAST_SYNCH=1`) the kernel waits for that CPU work itself and MLX
        /// encodes no separate fence-wait dispatch. The source must call
        /// `MLX_FENCED_INPUT_WAIT(name);` for each fenced input before it reads it, at a
        /// point that every thread of the threadgroup reaches (it contains a threadgroup
        /// barrier). Without the variable the call does nothing and MLX waits as before.
        /// Synchronization only: the outputs do not change.
        public static func metalKernel(
            name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
            source: String, header: String = "", ensureRowContiguous: Bool = true,
            atomicOutputs: Bool = false, mutableInputs: [String] = [], fencedInputs: [String]
        ) -> MLXFastKernel {
            MLXFastKernel(
                name: name, inputNames: inputNames, outputNames: outputNames,
                source: source, header: header,
                ensureRowContiguous: ensureRowContiguous, atomicOutputs: atomicOutputs,
                mutableInputs: mutableInputs, fencedInputs: fencedInputs
            )
        }

    }  // MLXFast

#else

    // Stub allowing compilation for platforms without Metal.

    extension MLXFast {

        final public class MLXFastKernel: @unchecked Sendable {
            public let outputNames: [String]

            init(
                name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
                source: String, header: String = "",
                ensureRowContiguous: Bool = true,
                atomicOutputs: Bool = false, mutableInputs: [String] = [],
                fencedInputs: [String] = []
            ) {
                self.outputNames = []
                fatalError("MLXFastKernel is not available without Metal")
            }

            public func callAsFunction(
                _ inputs: [any ScalarOrArray],
                template: [(String, any KernelTemplateArg)]? = nil,
                grid: (Int, Int, Int),
                threadGroup: (Int, Int, Int),
                outputShapes: some Sequence<[Int]>,
                outputDTypes: some Sequence<DType>,
                initValue: Float? = nil,
                verbose: Bool = false,
                stream: StreamOrDevice = .default
            ) -> [MLXArray] {
                fatalError("MLXFastKernel is not available without Metal")
            }

            public final class Launch: @unchecked Sendable {
                public func callAsFunction(_ inputs: [MLXArray], stream: StreamOrDevice = .default)
                    -> [MLXArray]
                {
                    fatalError("MLXFastKernel is not available without Metal")
                }
            }

            public func prepare(
                template: [(String, any KernelTemplateArg)]? = nil,
                grid: (Int, Int, Int),
                threadGroup: (Int, Int, Int),
                outputShapes: [[Int]],
                outputDTypes: [DType],
                initValue: Float? = nil,
                verbose: Bool = false
            ) -> Launch {
                fatalError("MLXFastKernel is not available without Metal")
            }
        }

        public static func metalKernel(
            name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
            source: String, header: String = "", ensureRowContiguous: Bool = true,
            atomicOutputs: Bool = false, mutableInputs: [String] = []
        ) -> MLXFastKernel {
            fatalError("MLXFastKernel is not available without Metal")
        }

        public static func metalKernel(
            name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
            source: String, header: String = "", ensureRowContiguous: Bool = true,
            atomicOutputs: Bool = false, mutableInputs: [String] = [], fencedInputs: [String]
        ) -> MLXFastKernel {
            fatalError("MLXFastKernel is not available without Metal")
        }

    }  // MLXFast

#endif
