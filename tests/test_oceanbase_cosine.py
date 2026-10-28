import pytest

from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.clients.oceanbase.config import OceanBaseHNSWConfig
from vectordb_bench.backend.clients.oceanbase.oceanbase import OceanBase


@pytest.mark.parametrize("index", [IndexType.HNSW, IndexType.HNSW_SQ, IndexType.HNSW_BQ])
def test_cosine_uses_native_metric_without_client_normalization(index):
    config = OceanBaseHNSWConfig(index=index, metric_type=MetricType.COSINE, m=16, efConstruction=200)
    client = OceanBase.__new__(OceanBase)
    client.db_case_config = config

    assert config.index_param()["metric_type"] == "cosine"
    assert config.search_param()["metric_type"] == "cosine_distance"
    assert client.need_normalize_cosine() is False


def test_bq_l2_metric_remains_l2():
    config = OceanBaseHNSWConfig(index=IndexType.HNSW_BQ, metric_type=MetricType.L2, m=16, efConstruction=200)

    assert config.index_param()["metric_type"] == "l2"
    assert config.search_param()["metric_type"] == "l2_distance"
