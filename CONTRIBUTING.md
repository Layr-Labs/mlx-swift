# Contributing to MLX Swift

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

1. Fork and submit pull requests to the repo.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Every PR should have passing tests and at least one review.
5. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   If needed you may need to `brew install swift-format`.

   You can also run the formatters manually as follows:

     ```shell
     swift-format -i Source/MLX/*.swift
     ```

   or run `pre-commit run --all-files` to check all files in the repo.

## Running the fork tests

The Fork Tests job sets MLX_ENABLE_TF32=0 because its FP32 reference comparisons
require full-precision products on TF32-capable GPUs. This setting is for tests;
it does not change the runtime default or relax numerical assertions.

The process allocation-cap test installs a one-time cap before any allocation.
CI runs it alone in a fresh process with MLX_TEST_PROCESS_CAP=1, then excludes
it from the ordinary pass. The ordinary pass rejects unexpected skips. It
allows the opt-in timing test and the sortedness-retract test whose AOT,
non-NAX hardware condition does not apply on NAX devices.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX Swift, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
