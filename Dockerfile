# Builds, tests and packages the mlnative extension. Nothing is installed on
# the host; everything happens inside the PHP image.
#
#   docker build --target test -t mlnative:test .   # build + run phpt tests
#   docker build -t mlnative .                        # runtime image
#   docker build --target artifact --output type=local,dest=dist .   # just mlnative.so
#   docker run --rm mlnative php examples/quickstart.php

ARG PHP_VERSION=8.3

FROM php:${PHP_VERSION}-cli-bookworm AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends $PHPIZE_DEPS \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/mlnative
COPY config.m4 php_mlnative.h ./
COPY src ./src
RUN phpize \
    && ./configure --enable-mlnative \
    && make -j"$(nproc)"

FROM build AS test
COPY tests ./tests
COPY docs ./docs
COPY README.md ./
ENV NO_INTERACTION=1 REPORT_EXIT_STATUS=1
RUN make test TESTS="--show-diff tests"

# Only the compiled module, for `docker build --target artifact --output type=local,dest=dist .`
FROM scratch AS artifact
COPY --from=build /usr/src/mlnative/modules/mlnative.so /

FROM php:${PHP_VERSION}-cli-bookworm AS runtime
COPY --from=build /usr/src/mlnative/modules/mlnative.so /tmp/mlnative.so
RUN mv /tmp/mlnative.so "$(php-config --extension-dir)/" \
    && docker-php-ext-enable mlnative
WORKDIR /app
COPY examples ./examples
COPY benchmarks ./benchmarks
CMD ["php", "--ri", "mlnative"]
