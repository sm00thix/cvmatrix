# Contribution Guidelines

We welcome any kind of contribution to our software, from a simple comment or question to a full-fledged [pull request](https://help.github.com/articles/about-pull-requests/).

A contribution can be one of the following cases:

1. You have a question;
2. You think you may have found a bug (including unexpected behavior);
3. You want to change the code base (e.g., to fix a bug, add a new feature, or update documentation).

The sections below outline the steps in each case.

## You have a question

1. Use the search functionality [here](https://github.com/Sm00thix/CVMatrix/issues) to see if someone already filed the same issue;
2. If your issue search did not yield any relevant results, make a new issue;
3. Apply the "Question" label; apply other labels when relevant.

## You think you may have found a bug

1. Use the search functionality [here](https://github.com/Sm00thix/CVMatrix/issues) to see if someone already filed the same issue;
2. If your issue search does not yield any relevant results, make a new issue, providing enough information to the rest of the community to understand the cause and context of the problem. Depending on the issue, you may want to include:
    - the [SHA hashcode](https://help.github.com/articles/autolinked-references-and-urls/#commit-shas) of the commit that is causing your problem;
    - some identifying information (name and version number) for dependencies you're using;
    - information about the operating system;
3. Apply relevant labels to the newly created issue.

## You want to make some kind of change to the code base

1. (**Important**) Announce your plan to the rest of the community *before you start working*. This announcement should be in the form of a (new) issue;
2. (**Important**) Wait until some kind of consensus is reached about your idea being a good idea;
3. If needed, fork the repository to your own Github profile and create your own feature branch off of the latest main commit. While working on your feature branch, make sure to stay up to date with the main branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions [here](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [here](https://help.github.com/articles/syncing-a-fork/));
4. Follow the [Design guidelines](#design-guidelines) when changing `CVMatrix`;
5. Make sure the existing tests still work by following the instructions in [Run the test suite](#run-the-test-suite);
6. Add your own tests (if necessary);
7. If you added your own tests, make sure they pass by following the instructions in [Run the test suite](#run-the-test-suite);
8. If your contribution is a performance enhancement, make sure to include benchmarks. See [Benchmarking](#benchmarking) for more information;
9. Update or expand the documentation;
10. Make sure the documentation builds without errors by following the instructions in [Build the documentation](#build-the-documentation);
11. Check that you can build the package locally and that it passes twine check. See [Build from source](#build-from-source) for more information;
12. [Push](http://rogerdudler.github.io/git-guide/) your feature branch to (your fork of) the CVMatrix repository on GitHub;
13. Create the pull request, e.g. following the instructions [here](https://help.github.com/articles/creating-a-pull-request/).

If you feel like you've made a valuable contribution, but you don't know how to write or run tests for it or generate the documentation, don't let this discourage you from making the pull request; we can help you! Just submit the pull request, but remember that you might be asked to append additional commits to your pull request.

## Design guidelines

`CVMatrix` is used in several execution settings: traced by `jax.jit`/`jax.vmap` with `backend="jax"`, called concurrently from multiple threads (including on free-threaded Python, which has no GIL), and pickled to worker processes (e.g., with `joblib`, `multiprocessing`, or `concurrent.futures.ProcessPoolExecutor`). To keep it correct and fast in all of them:

- **The per-fold methods only read the fitted state.** `training_XTX`, `training_XTY`, `training_XTX_XTY`, and `training_statistics` must not modify the instance, e.g., by caching results on it. Compute anything dataset-wide in `fit`. A cache written during a per-fold call would leak tracers under `jax.jit`, race between threads, and, in worker processes, be filled in a throwaway copy, i.e., recomputed for every task.
- **Route array operations through `self.xp`**, not `np`, so that the NumPy and JAX backends share one implementation and the per-fold methods stay traceable. Data-dependent checks that cannot run under tracing are skipped for tracer inputs (see `_TRACER_TYPES`). The NumPy backend's results must not change.
- **Keep instances picklable with the standard library's `pickle`.** Do not store modules, locks, open files, or other unpicklable objects on the instance. If one is unavoidable, leave it out in `__reduce__` and restore it in `_reconstruct`, as is done for the array namespace `xp`. `test_pickle` and `test_process_pool` check this.
- **Test both backends.** Every test runs with both `backend="numpy"` and `backend="jax"` through the `_set_backend` fixture in `tests/test_cvmatrix.py`; JAX-only tests skip themselves under the NumPy parametrization.

## Build from source

CVMatrix uses [uv](https://docs.astral.sh/uv/) to manage its dependencies and packaging. To build the package from source, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/Sm00thix/CVMatrix.git
    ```

2. Change to the repository directory:

    ```shell
    cd CVMatrix
    ```

3. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), e.g.:

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

4. Build the package. uv builds in an isolated environment, so no separate dependency-install step is needed:

    ```shell
    uv build
    ```

5. Check the package with twine:

    ```shell
    uvx twine check dist/*
    ```

## Run the test suite

To run the test suite, follow these steps:

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), e.g.:

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Install the project together with its development dependencies. To also exercise the optional JAX backend (the `backend="jax"` tests), include the `jax` extra:

    ```shell
    uv sync --extra jax --group dev
    ```

    To test the numpy-only installation instead (the `backend="jax"` tests are skipped automatically), omit the extra:

    ```shell
    uv sync --group dev
    ```

3. Now, the tests can be run with the following command:

    ```shell
    uv run pytest tests --doctest-modules
    ```

    The pytest options in `pyproject.toml` enable runtime type checking with [typeguard](https://typeguard.readthedocs.io/) (`--typeguard-packages=cvmatrix`) and coverage reports. Note that `--typeguard-packages` takes the package name `cvmatrix`, not a path: `cvmatrix/` silently instruments nothing.

### Test on free-threaded Python

CVMatrix also runs on free-threaded builds of Python (e.g., `3.15t`), which have no GIL. To run the test suite on one without touching your default environment, let uv create a temporary, isolated environment:

```shell
uv run --isolated --python 3.15t --extra jax --group dev pytest tests --doctest-modules
```

A free-threaded interpreter silently re-enables the GIL if it imports an extension module that does not support free threading. Check that the GIL stays disabled with all dependencies imported; this must print `False`:

```shell
uv run --isolated --python 3.15t --extra jax python -c "import sys, cvmatrix, jax; print(sys._is_gil_enabled())"
```

## Build the documentation

1. Install sphinx, the sphinx-rtd-theme, and the MyST-Parser:

    ```shell
    pip3 install sphinx sphinx-rtd-theme myst-parser
    ```

2. Change to the docs directory:

    ```shell
    cd docs
    ```

3. Build the documentation:

    ```shell
    make html
    ```

## Benchmarking

To run benchmarks, follow the instructions [here](https://github.com/Sm00thix/CVMatrix/blob/main/benchmarks/README.md).

`benchmarks/benchmark.py` can be configured with environment variables, e.g., `BENCH_CONFIGS=plot` (only the three preprocessing combinations in the plots), `BENCH_PS=3,100,10000` (the numbers of folds), `BENCH_N`/`BENCH_K`/`BENCH_M` (the data size), and `BENCH_CSV` (the output file):

```shell
OPENBLAS_NUM_THREADS=1 BENCH_CONFIGS=plot BENCH_PS=3,100,10000 BENCH_CSV=my_results.csv uv run python benchmarks/benchmark.py
```

When you compare the performance of your change against `main`, run both versions on the same machine with a single BLAS thread, as above, and alternate between them over several rounds, also with the order reversed. The per-fold timings depend on the memory allocation history of the process, so identical code can differ by several percent between runs, and a fixed run order can look like a consistent difference.