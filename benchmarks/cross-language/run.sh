#!/bin/sh
# Cross-language benchmark: mlnative against the same algorithms written in plain PHP,
# Python, JavaScript, Ruby and Lua, plus NumPy, scikit-learn and PyTorch for reference.
# Everything runs in Docker, one container at a time, each limited to one CPU.
# The method and results are in docs/BENCHMARK_REPORT.md.
#
#   sh benchmarks/cross-language/run.sh            # every runtime
#   sh benchmarks/cross-language/run.sh php node   # a subset: native php node ruby lua python
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DATA="$HERE/data"
OUT="$HERE/results.txt"
SELECTED=${*:-native php node ruby lua python}

echo "building images" >&2
docker build -q -t mlnative "$ROOT" > /dev/null
case " $SELECTED " in *" python "*) docker build -q -t mlbench-python -f "$HERE/docker/python.Dockerfile" "$HERE/docker" > /dev/null ;; esac
case " $SELECTED " in *" lua "*) docker build -q -t mlbench-lua -f "$HERE/docker/lua.Dockerfile" "$HERE/docker" > /dev/null ;; esac

if [ ! -f "$DATA/mlp_perm.txt" ]; then
    mkdir -p "$DATA"
    docker run --rm -v "$HERE:/bench:ro" -v "$DATA:/data" mlnative php /bench/gen.php /data >&2
fi

D="docker run --rm --cpus=1 -v $HERE:/bench:ro -v $DATA:/data:ro"
FAST="$D -e REPEATS=5"
SLOW="$D -e REPEATS=3"
ONE_THREAD="-e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1"
PHP="php -d memory_limit=-1"
JIT="-d zend_extension=opcache -d opcache.enable_cli=1 -d opcache.jit=tracing -d opcache.jit_buffer_size=128M"

: > "$OUT"
run() { echo ">> $1" >&2; shift; "$@" 2>&1 | grep '|' | tee -a "$OUT" >&2; }

for group in $SELECTED; do
    case $group in
        native) run "mlnative" $FAST mlnative $PHP /bench/bench.php /data native "PHP+mlnative" ;;
        php)
            run "PHP"     $SLOW mlnative $PHP /bench/bench.php /data pure "PHP 8.3"
            run "PHP JIT" $SLOW mlnative $PHP $JIT /bench/bench.php /data pure "PHP 8.3 JIT" ;;
        node) run "Node.js" $FAST node:22-slim node /bench/bench.js /data "Node.js 22" ;;
        ruby)
            run "Ruby"      $SLOW ruby:3.3-slim ruby /bench/bench.rb /data "Ruby 3.3"
            run "Ruby YJIT" $SLOW ruby:3.3-slim ruby --yjit /bench/bench.rb /data "Ruby 3.3 YJIT" ;;
        lua) run "Lua" $SLOW mlbench-lua lua5.4 /bench/bench.lua /data "Lua 5.4" ;;
        python)
            run "Python"       $SLOW mlbench-python python /bench/bench.py /data pure "Python 3.12"
            run "NumPy"        $FAST $ONE_THREAD mlbench-python python /bench/bench.py /data numpy "Python+NumPy"
            run "scikit-learn" $FAST $ONE_THREAD mlbench-python python /bench/bench.py /data sklearn "Python+sklearn"
            run "PyTorch"      $FAST $ONE_THREAD mlbench-python python /bench/bench.py /data torch "Python+PyTorch" ;;
        *) echo "unknown group: $group" >&2; exit 1 ;;
    esac
done

docker run --rm -v "$HERE:/bench:ro" mlnative php /bench/summarize.php /bench/results.txt
