from unittest.mock import Mock, patch

import pytest

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients.milvus.milvus import Milvus
from vectordb_bench.backend.clients.oceanbase.oceanbase import OceanBase
from vectordb_bench.backend.clients.tidb.tidb import TiDB
from vectordb_bench.backend.filter import CohereLabelAndFilter, LabelFilter, non_filter


def test_cohere_label_and_uses_the_original_cohort_and_ground_truth():
    single, double = [CaseType.CohereLabelAndPerformanceCase.case_cls({"label_conditions": n}) for n in (1, 2)]
    ids = list(range(1000))
    labels = ["label_1p" if i % 100 == 0 else "label_foo" for i in ids]
    extra = double.filters.extra_labels_for_batch(ids, labels)
    one_matches = {i for i in ids if labels[i] == single.filters.label_value}
    left, right = [{i for i in ids if extra[field][i] == value} for field, value in double.filters.conditions]
    assert one_matches == left & right
    assert left - right
    assert right - left
    assert single.filters.extra_labels_for_batch(ids, labels) == extra
    assert single.filters.groundtruth_file == double.filters.groundtruth_file == LabelFilter(0.01).groundtruth_file


@pytest.mark.parametrize(
    ("count", "sql", "params", "milvus_expr"),
    [
        (1, "WHERE labels = %s", ("label_1p",), "label == 'label_1p'"),
        (2, "WHERE label_b = %s AND label_c = %s", ("yes", "yes"), "label_b == 'yes' and label_c == 'yes'"),
    ],
)
def test_adapters_apply_the_same_label_predicates(count: int, sql: str, params: tuple[str, ...], milvus_expr: str):
    filters = CohereLabelAndFilter(count)
    ob = OceanBase.__new__(OceanBase)
    ob._label_fields = ("labels", *filters.extra_fields)
    ob.prepare_filter(filters)
    assert (ob.expr, ob.where_params) == (sql, params)

    tidb = TiDB.__new__(TiDB)
    tidb.with_scalar_labels = True
    tidb._extra_label_fields = filters.extra_fields
    tidb.prepare_filter(filters)
    assert (tidb.where_clause, tidb.where_params) == (sql, params)

    milvus = Milvus.__new__(Milvus)
    milvus.with_scalar_labels = True
    milvus._is_fts = False
    milvus._extra_label_fields = filters.extra_fields
    milvus._scalar_payload_label_field = "label"
    milvus.prepare_filter(filters)
    assert milvus.expr == milvus_expr


def test_oceanbase_creates_inserts_and_filters_the_native_label_column():
    client = OceanBase.__new__(OceanBase)
    client.dim = 2
    client.db_case_config = Mock(partitions=0)
    client.table_name = "vectors"
    client.load_batch_size = 1
    client._cursor = Mock()
    client._label_fields = ("labels",)

    client._create_table()
    assert "labels VARCHAR(64)" in client._cursor.execute.call_args.args[0]
    assert client.insert_embeddings([[0.0, 1.0]], [123], labels_data=["label_1p"]) == (1, None)
    sql, params = client._cursor.execute.call_args.args
    assert "(id, embedding, labels)" in sql
    assert params == ("label_1p",)

    client.prepare_filter(LabelFilter(0.01))
    assert (client.expr, client.where_params) == ("WHERE labels = %s", ("label_1p",))
    client.prepare_filter(non_filter)
    assert (client.expr, client.where_params) == ("", ())


def test_oceanbase_uses_the_pure_mysql_connector():
    client = OceanBase.__new__(OceanBase)
    client.db_config = {"host": "localhost", "port": 2881, "user": "root", "password": "", "database": "bench"}
    with patch("vectordb_bench.backend.clients.oceanbase.oceanbase.mysql.connect") as connect:
        client._connect()

    assert connect.call_args.kwargs["use_pure"] is True
