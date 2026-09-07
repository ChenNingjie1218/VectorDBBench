VENV_BIN := $(CURDIR)/.venv/bin
PYTHON := $(VENV_BIN)/python
VECTORDBBENCH := $(VENV_BIN)/vectordbbench

LABEL_FILTER_PERCENTAGE ?= 0.01
LABEL_FILTER_DATASET_1M := Medium Cohere (768dim, 1M)
LABEL_FILTER_DATASET_10M := Large Cohere (768dim, 10M)

unittest:
	PYTHONPATH=$(CURDIR) $(PYTHON) -m pytest tests/test_dataset.py::TestDataSet::test_download_small -svv

format:
	PYTHONPATH=$(CURDIR) $(PYTHON) -m black vectordb_bench
	PYTHONPATH=$(CURDIR) $(PYTHON) -m ruff check vectordb_bench --fix

lint:
	PYTHONPATH=$(CURDIR) $(PYTHON) -m black vectordb_bench --check
	PYTHONPATH=$(CURDIR) $(PYTHON) -m ruff check vectordb_bench

load-search-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M $(ARGS)

load-search-1m-non-inline-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-non-inline --case-type Performance768D1M --spfresh-build-mode non-inline $(ARGS)

load-search-1m-split-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-split --case-type Performance768D1M --spfresh-build-mode split --spfresh-split-ratio 0.8 $(ARGS)

build-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-build-1m --case-type Performance768D1M --skip-load --build --spfresh-build-mode non-inline $(ARGS)

build-only-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-build-only-1m --case-type Performance768D1M --skip-load --build --skip-search-concurrent --spfresh-build-mode non-inline $(ARGS)

search-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M --skip-load $(ARGS)

load-search-10m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M $(ARGS)

load-search-10m-non-inline-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m-non-inline --case-type Performance768D10M --spfresh-build-mode non-inline $(ARGS)

load-search-10m-split-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m-split --case-type Performance768D10M --spfresh-build-mode split --spfresh-split-ratio 0.8 $(ARGS)

search-10m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M --skip-load

load-search-label-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-label-1m --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_1M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) $(ARGS)

load-search-label-1m-non-inline-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-label-1m-non-inline --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_1M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) --spfresh-build-mode non-inline $(ARGS)

load-search-label-1m-split-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-label-1m-split --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_1M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) --spfresh-build-mode split --spfresh-split-ratio 0.8 $(ARGS)

search-label-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-label-1m --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_1M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) --skip-load $(ARGS)

build-label-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-label-build-1m --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_1M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) --skip-load --build --spfresh-build-mode non-inline $(ARGS)

build-only-label-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-label-build-only-1m --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_1M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) --skip-load --build --skip-search-concurrent --spfresh-build-mode non-inline $(ARGS)

load-search-label-10m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-label-10m --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_10M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) $(ARGS)

load-search-label-10m-non-inline-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-label-10m-non-inline --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_10M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) --spfresh-build-mode non-inline $(ARGS)

load-search-label-10m-split-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-label-10m-split --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_10M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) --spfresh-build-mode split --spfresh-split-ratio 0.8 $(ARGS)

search-label-10m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-label-10m --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_10M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) --skip-load $(ARGS)

build-label-10m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-label-build-10m --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_10M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) --skip-load --build --spfresh-build-mode non-inline $(ARGS)

build-only-label-10m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-label-build-only-10m --case-type LabelFilterPerformanceCase --dataset-with-size-type "$(LABEL_FILTER_DATASET_10M)" --label-percentage $(LABEL_FILTER_PERCENTAGE) --skip-load --build --skip-search-concurrent --spfresh-build-mode non-inline $(ARGS)

build-10m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-build-10m --case-type Performance768D10M --skip-load --build --spfresh-build-mode non-inline $(ARGS)

build-only-10m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-build-only-10m --case-type Performance768D10M --skip-load --build --skip-search-concurrent --spfresh-build-mode non-inline $(ARGS)

build-100m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test100m --task-label tidb-spfresh-build-100m --case-type Performance768D100M --skip-load --build --spfresh-build-mode non-inline $(ARGS)

build-only-100m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test100m --task-label tidb-spfresh-build-only-100m --case-type Performance768D100M --skip-load --build --skip-search-concurrent --spfresh-build-mode non-inline $(ARGS)

delete-plain-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-delete-plain --case-type Performance768D1M --delete --skip-search-serial --skip-search-concurrent --skip-build-spfresh-index

delete-spfresh-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-delete-indexed --case-type Performance768D1M --delete --skip-search-serial --skip-search-concurrent $(ARGS)

delete-compare-1m-local:
	$(MAKE) delete-plain-1m-local
	$(MAKE) delete-spfresh-1m-local
	$(PYTHON) scripts/compare_tidb_delete_results.py

load-search-1m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M $(ARGS)

search-1m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M --skip-load

load-search-10m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M $(ARGS)

search-10m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M --skip-load

load-search-100m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test100m --task-label tidb-spfresh-100m --case-type Performance768D10M $(ARGS)

search-100m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test100m --task-label tidb-spfresh-100m --case-type Performance768D10M --skip-load
