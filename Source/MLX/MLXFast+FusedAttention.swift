import Cmlx

extension MLXFast {
    /// Explicitly select a fused attention kernel when its bounded allocation
    /// behavior is required. With `forceFused: true`, unsupported devices or
    /// shapes report an MLX error instead of composing a score tensor.
    ///
    /// The existing overloads preserve their default routing. Use `withError`
    /// when a requested fused kernel may be unavailable. This controls kernel
    /// selection, not the multiplication precision chosen by the device.
    public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: ScaledDotProductAttentionMaskMode, sinks: MLXArray? = nil,
        forceFused: Bool, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_scaled_dot_product_attention_with_force_fused(
            &result, queries.ctx, keys.ctx, values.ctx, scale,
            mask.mode, mask.mask?.ctx ?? MLXArray.mlxNone.ctx,
            (sinks ?? .mlxNone).ctx, forceFused, stream.ctx)
        return MLXArray(result)
    }

    /// Array-mask counterpart of the explicit fused-routing overload.
    public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXArray?, sinks: MLXArray? = nil, forceFused: Bool,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: mask.map(ScaledDotProductAttentionMaskMode.array) ?? .none,
            sinks: sinks, forceFused: forceFused, stream: stream)
    }
}
