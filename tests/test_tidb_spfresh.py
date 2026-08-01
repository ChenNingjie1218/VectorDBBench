from contextlib import contextmanager
from types import SimpleNamespace, TracebackType
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.tidb.config import TiDBIndexConfig
from vectordb_bench.backend.clients.tidb.tidb import (
    MAX_ALLOWED_PACKET_BYTES,
    SPFreshIndexStatus,
    TiDB,
)
from vectordb_bench.backend.runner import SerialInsertRunner
from vectordb_bench.backend.task_runner import CaseRunner


class FakeCursor:
    def __init__(self, fetchone_results: list[tuple[int]] | None = None):
        self.execute_calls: list[tuple[str, Any]] = []
        self.fetchone_results = list(fetchone_results or [])

    def execute(self, sql: str, params: Any = None):
        self.execute_calls.append((sql, params))

    def fetchone(self):
        if not self.fetchone_results:
            return None
        return self.fetchone_results.pop(0)


class FakeConnection:
    def __init__(self):
        self.committed = False

    def commit(self):
        self.committed = True


class FakeInsertDB:
    def __init__(self):
        self.inserted_metadata: list[int] = []

    @contextmanager
    def init(self):
        yield

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
    ) -> tuple[int, None]:
        self.inserted_metadata.extend(metadata)
        return len(metadata), None


class FakeDataset:
    def __init__(self):
        self.data = SimpleNamespace(train_id_field="id", train_vector_field="emb", scalar_labels_file_separated=True)
        self.scalar_labels = None

    def __iter__(self):
        return iter(
            [
                pd.DataFrame({"id": list(range(5)), "emb": [[float(i)] for i in range(5)]}),
                pd.DataFrame({"id": list(range(5, 10)), "emb": [[float(i)] for i in range(5, 10)]}),
            ]
        )


def make_tidb(dim: int = 3) -> TiDB:
    return TiDB(
        dim=dim,
        db_config={
            "host": "127.0.0.1",
            "port": 4000,
            "user": "root",
            "password": "",
            "database": "test",
            "ssl_verify_cert": False,
            "ssl_verify_identity": False,
        },
        db_case_config=TiDBIndexConfig(metric_type=MetricType.L2),
    )


class TestTiDBSPFresh:
    def test_serial_insert_runner_loads_requested_row_range(self):
        db = FakeInsertDB()
        runner = SerialInsertRunner(
            db=db,
            dataset=FakeDataset(),
            normalize=False,
            start_offset=3,
            limit=4,
        )

        count, load_state = runner.task()

        assert count == 4
        assert load_state is None
        assert db.inserted_metadata == [3, 4, 5, 6]

    def test_insert_embeddings_keeps_existing_worker_sharding_when_batches_are_small(self):
        tidb = make_tidb()
        embeddings = [[] for _ in range(200)]
        metadata = list(range(200))
        insert_calls: list[tuple[int, int]] = []

        def capture_insert_call(
            _embeddings: list[list[float]],
            _metadata: list[int],
            offset: int,
            size: int,
        ) -> None:
            insert_calls.append((offset, size))

        with patch.object(tidb, "_insert_embeddings_serial", side_effect=capture_insert_call):
            insert_count, error = tidb.insert_embeddings(embeddings=embeddings, metadata=metadata)

        assert (insert_count, error) == (len(metadata), None)
        assert sorted(insert_calls) == [(offset, 20) for offset in range(0, len(metadata), 20)]

    def test_insert_embeddings_keeps_existing_worker_sharding_for_larger_loads(self):
        tidb = make_tidb()
        embeddings = [[] for _ in range(6000)]
        metadata = list(range(6000))
        insert_calls: list[tuple[int, int]] = []

        def capture_insert_call(
            _embeddings: list[list[float]],
            _metadata: list[int],
            offset: int,
            size: int,
        ) -> None:
            insert_calls.append((offset, size))

        with patch.object(tidb, "_insert_embeddings_serial", side_effect=capture_insert_call):
            insert_count, error = tidb.insert_embeddings(embeddings=embeddings, metadata=metadata)

        assert (insert_count, error) == (len(metadata), None)
        assert sorted(insert_calls) == [(offset, 600) for offset in range(0, len(metadata), 600)]

    def test_insert_embeddings_caps_transactions_by_max_allowed_packet_bytes(self):
        tidb = make_tidb(dim=30_000)
        embeddings = [[] for _ in range(1000)]
        metadata = list(range(1000))
        insert_calls: list[tuple[int, int]] = []
        expected_batch_size = MAX_ALLOWED_PACKET_BYTES // 24 // tidb.dim

        def capture_insert_call(
            _embeddings: list[list[float]],
            _metadata: list[int],
            offset: int,
            size: int,
        ) -> None:
            insert_calls.append((offset, size))

        with patch.object(tidb, "_insert_embeddings_serial", side_effect=capture_insert_call):
            insert_count, error = tidb.insert_embeddings(embeddings=embeddings, metadata=metadata)

        assert expected_batch_size < len(metadata) // 10
        assert (insert_count, error) == (len(metadata), None)
        assert sorted(insert_calls) == [
            (offset, min(expected_batch_size, len(metadata) - offset))
            for offset in range(0, len(metadata), expected_batch_size)
        ]

    def test_insert_embeddings_tracks_max_commit_ts_across_worker_sessions(self):
        tidb = make_tidb()
        embeddings = [[] for _ in range(6000)]
        metadata = list(range(6000))

        def capture_insert_call(
            _embeddings: list[list[float]],
            _metadata: list[int],
            offset: int,
            _size: int,
        ) -> int:
            return 1000 + offset

        with patch.object(tidb, "_insert_embeddings_serial", side_effect=capture_insert_call):
            insert_count, error = tidb.insert_embeddings(embeddings=embeddings, metadata=metadata)

        assert (insert_count, error) == (len(metadata), None)
        assert tidb._max_insert_commit_ts == 6400

    def test_insert_embeddings_does_not_probe_incremental_backlog_during_load(self):
        tidb = make_tidb()
        tidb.cursor = FakeCursor()
        tidb.conn = FakeConnection()
        batch_size = tidb._max_insert_rows_per_transaction()
        total_rows = batch_size * 20

        with (
            patch.object(tidb, "_insert_embeddings_serial", return_value=None),
            patch.object(
                tidb,
                "_fetch_spfresh_index_status",
                side_effect=AssertionError("should not query SPFresh status during load"),
            ),
        ):
            insert_count, error = tidb.insert_embeddings(
                embeddings=[[] for _ in range(total_rows)],
                metadata=list(range(total_rows)),
            )

        assert (insert_count, error) == (total_rows, None)

    def test_insert_embeddings_serial_reads_commit_ts_from_writer_session(self):
        tidb = make_tidb()
        cursor = FakeCursor(fetchone_results=[("123456",)])
        conn = FakeConnection()

        class ConnectionContext:
            def __enter__(self_inner) -> tuple[FakeConnection, FakeCursor]:
                return conn, cursor

            def __exit__(
                self_inner,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                return False

        with patch.object(tidb, "_get_connection", return_value=ConnectionContext()):
            commit_ts = tidb._insert_embeddings_serial(embeddings=[[1.0, 2.0, 3.0]], metadata=[7], offset=0, size=1)

        assert commit_ts == 123456

    def test_create_table_inlines_spfresh_vector_index(self):
        tidb = make_tidb()
        cursor = FakeCursor()
        conn = FakeConnection()

        class ConnectionContext:
            def __enter__(self_inner) -> tuple[FakeConnection, FakeCursor]:
                return conn, cursor

            def __exit__(
                self_inner,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                return False

        with patch.object(tidb, "_get_connection", return_value=ConnectionContext()):
            tidb._create_table()

        assert conn.committed is True
        sql, _ = cursor.execute_calls[0]
        assert "CREATE TABLE vector_bench_test" in sql
        assert "VECTOR INDEX idx_embedding_spfresh_l2 ((vec_l2_distance(embedding))) USING SPFRESH" in sql

    def test_create_table_inlines_spfresh_vector_index_param(self):
        tidb = TiDB(
            dim=3,
            db_config={
                "host": "127.0.0.1",
                "port": 4000,
                "user": "root",
                "password": "",
                "database": "test",
                "ssl_verify_cert": False,
                "ssl_verify_identity": False,
            },
            db_case_config=TiDBIndexConfig(
                metric_type=MetricType.L2,
                spfresh_vector_index_param="max_partition_size=256,write_beam_size=8",
            ),
        )
        cursor = FakeCursor()
        conn = FakeConnection()

        class ConnectionContext:
            def __enter__(self_inner) -> tuple[FakeConnection, FakeCursor]:
                return conn, cursor

            def __exit__(
                self_inner,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                return False

        with patch.object(tidb, "_get_connection", return_value=ConnectionContext()):
            tidb._create_table()

        sql, _ = cursor.execute_calls[0]
        assert (
            "VECTOR INDEX idx_embedding_spfresh_l2 ((vec_l2_distance(embedding))) USING SPFRESH "
            "VECTOR_INDEX_PARAM 'max_partition_size=256,write_beam_size=8'"
        ) in sql

    def test_create_table_skips_inline_index_for_non_inline_spfresh_mode(self):
        tidb = TiDB(
            dim=3,
            db_config={
                "host": "127.0.0.1",
                "port": 4000,
                "user": "root",
                "password": "",
                "database": "test",
                "ssl_verify_cert": False,
                "ssl_verify_identity": False,
            },
            db_case_config=TiDBIndexConfig(
                metric_type=MetricType.L2,
                spfresh_build_mode="non-inline",
            ),
        )
        cursor = FakeCursor()
        conn = FakeConnection()

        class ConnectionContext:
            def __enter__(self_inner) -> tuple[FakeConnection, FakeCursor]:
                return conn, cursor

            def __exit__(
                self_inner,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                return False

        with patch.object(tidb, "_get_connection", return_value=ConnectionContext()):
            tidb._create_table()

        sql, _ = cursor.execute_calls[0]
        assert "CREATE TABLE vector_bench_test" in sql
        assert "VECTOR INDEX" not in sql

    def test_build_spfresh_index_records_alter_table_duration(self):
        tidb = TiDB(
            dim=3,
            db_config={
                "host": "127.0.0.1",
                "port": 4000,
                "user": "root",
                "password": "",
                "database": "test",
                "ssl_verify_cert": False,
                "ssl_verify_identity": False,
            },
            db_case_config=TiDBIndexConfig(
                metric_type=MetricType.L2,
                spfresh_build_mode="non-inline",
                spfresh_vector_index_param="min_partition_size=32,max_partition_size=256,write_beam_size=8",
            ),
        )
        tidb.cursor = FakeCursor()
        tidb.conn = FakeConnection()

        with patch("vectordb_bench.backend.clients.tidb.tidb.time.perf_counter", side_effect=[10.0, 13.5]):
            result = tidb.build_spfresh_index()

        sql, _ = tidb.cursor.execute_calls[0]
        assert "ALTER TABLE vector_bench_test ADD VECTOR INDEX idx_embedding_spfresh_l2" in sql
        assert "VECTOR_INDEX_PARAM 'min_partition_size=32,max_partition_size=256,write_beam_size=8'" in sql
        assert result.optimize_duration == 0.0
        assert result.spfresh_build_duration == 3.5
        assert result.spfresh_incremental_catchup_duration == 0.0
        assert tidb.conn.committed is True

    def test_spfresh_vector_index_param_empty_string_is_ignored(self):
        cfg = TiDBIndexConfig(metric_type=MetricType.L2, spfresh_vector_index_param=" ")
        tidb = TiDB(
            dim=3,
            db_config={
                "host": "127.0.0.1",
                "port": 4000,
                "user": "root",
                "password": "",
                "database": "test",
                "ssl_verify_cert": False,
                "ssl_verify_identity": False,
            },
            db_case_config=cfg,
        )

        assert cfg.spfresh_vector_index_param is None
        assert "VECTOR_INDEX_PARAM" not in tidb._spfresh_index_definition()

    def test_optimize_waits_for_recorded_insert_barrier(self):
        tidb = make_tidb()
        tidb._max_insert_commit_ts = 123456
        tidb.cursor = FakeCursor()
        tidb.conn = FakeConnection()

        with patch.object(tidb, "_wait_for_spfresh_ready") as wait_mock:
            tidb.optimize()

        wait_mock.assert_called_once_with(barrier_ts=123456)

    def test_incremental_catchup_records_only_checkpoint_wait_duration(self):
        tidb = make_tidb()
        tidb._max_insert_commit_ts = 123456
        tidb.cursor = FakeCursor()
        tidb.conn = FakeConnection()

        with (
            patch.object(tidb, "_wait_for_spfresh_ready") as wait_mock,
            patch("vectordb_bench.backend.clients.tidb.tidb.time.perf_counter", side_effect=[10.0, 11.5]),
        ):
            result = tidb.wait_spfresh_incremental_catchup()

        wait_mock.assert_called_once_with(barrier_ts=123456)
        assert result.spfresh_incremental_catchup_duration == 1.5
        assert result.spfresh_build_duration == 0.0
        assert result.optimize_duration == 0.0

    def test_load_train_data_restores_tidb_load_state_from_insert_runner(self):
        class FakeDB:
            def __init__(self):
                self.imported_states: list[dict[str, int] | None] = []

            def need_normalize_cosine(self) -> bool:
                return False

            def import_load_state(self, state: dict[str, int] | None) -> None:
                self.imported_states.append(state)

        case_runner = CaseRunner.construct(
            run_id="run",
            config=None,
            ca=SimpleNamespace(
                dataset=SimpleNamespace(data=SimpleNamespace(metric_type=MetricType.L2)),
                filters=None,
                load_timeout=None,
            ),
            status=None,
            dataset_source=None,
            db=FakeDB(),
        )

        with patch("vectordb_bench.backend.task_runner.SerialInsertRunner") as runner_cls:
            runner_cls.return_value.run.return_value = (1000, {"max_insert_commit_ts": 123456})
            case_runner._load_train_data()

        assert case_runner.db.imported_states == [{"max_insert_commit_ts": 123456}]

    def test_fetch_spfresh_index_status_reads_system_table(self):
        tidb = make_tidb()
        tidb.cursor = FakeCursor(fetchone_results=[(123, "ready", 1, 3, 999, "2026-04-24 14:00:00")])

        status = tidb._fetch_spfresh_index_status()

        assert status == SPFreshIndexStatus(
            applied_base_ts=123,
            index_state="READY",
            is_ready=True,
            lag_seconds=3,
            owner_lease_expire_ts=999,
            observed_at="2026-04-24 14:00:00",
        )
        sql, params = tidb.cursor.execute_calls[0]
        assert "FROM information_schema.TIDB_SPFRESH_INDEX_STATUS" in sql
        assert "index_state" in sql.lower()
        assert "is_ready" in sql.lower()
        assert params == ("vector_bench_test", "idx_embedding_spfresh_l2")

    def test_wait_for_spfresh_ready_succeeds_after_polling(self):
        tidb = make_tidb()
        responses = [
            SPFreshIndexStatus(
                applied_base_ts=149,
                index_state="READY",
                is_ready=True,
                lag_seconds=1,
                owner_lease_expire_ts=999,
                observed_at="2026-04-24 14:00:02",
            ),
            SPFreshIndexStatus(
                applied_base_ts=150,
                index_state="READY",
                is_ready=True,
                lag_seconds=0,
                owner_lease_expire_ts=999,
                observed_at="2026-04-24 14:00:03",
            ),
        ]

        with (
            patch.object(tidb, "_fetch_spfresh_index_status", side_effect=responses) as fetch_mock,
            patch("vectordb_bench.backend.clients.tidb.tidb.time.sleep"),
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=5,
                poll_interval_seconds=0.01,
            )

        assert fetch_mock.call_count == 2

    def test_wait_for_spfresh_ready_succeeds_when_checkpoint_is_above_barrier(self):
        tidb = make_tidb()
        with (
            patch.object(
                tidb,
                "_fetch_spfresh_index_status",
                return_value=SPFreshIndexStatus(
                    applied_base_ts=151,
                    index_state="READY",
                    is_ready=True,
                    lag_seconds=0,
                    owner_lease_expire_ts=999,
                    observed_at="2026-04-24 14:00:04",
                ),
            ) as fetch_mock,
            patch("vectordb_bench.backend.clients.tidb.tidb.time.sleep") as sleep_mock,
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

        fetch_mock.assert_called_once_with()
        sleep_mock.assert_not_called()

    def test_wait_for_spfresh_ready_succeeds_at_barrier_when_lag_is_zero(self):
        tidb = make_tidb()
        with (
            patch.object(
                tidb,
                "_fetch_spfresh_index_status",
                return_value=SPFreshIndexStatus(
                    applied_base_ts=150,
                    index_state="READY",
                    is_ready=True,
                    lag_seconds=0,
                    owner_lease_expire_ts=None,
                    observed_at="2026-04-24 14:00:05",
                ),
            ) as fetch_mock,
            patch("vectordb_bench.backend.clients.tidb.tidb.time.sleep") as sleep_mock,
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

        fetch_mock.assert_called_once_with()
        sleep_mock.assert_not_called()

    def test_wait_for_spfresh_ready_fails_immediately_on_unknown_null_checkpoint(self):
        tidb = make_tidb()
        with (
            patch.object(
                tidb,
                "_fetch_spfresh_index_status",
                return_value=SPFreshIndexStatus(
                    applied_base_ts=None,
                    index_state="UNKNOWN",
                    is_ready=False,
                    lag_seconds=None,
                    owner_lease_expire_ts=None,
                    observed_at="2026-04-24 14:00:06",
                ),
            ) as fetch_mock,
            patch("vectordb_bench.backend.clients.tidb.tidb.time.sleep") as sleep_mock,
            pytest.raises(RuntimeError) as exc_info,
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

        fetch_mock.assert_called_once_with()
        sleep_mock.assert_not_called()
        message = str(exc_info.value)
        assert "table=vector_bench_test" in message
        assert "index=idx_embedding_spfresh_l2" in message
        assert "index_state=UNKNOWN" in message
        assert "is_ready=0" in message
        assert "applied_base_ts=None" in message
        assert "lag_seconds=None" in message
        assert "owner_lease_expire_ts=None" in message
        assert "observed_at=2026-04-24 14:00:06" in message
        assert "waited_seconds=" in message

    def test_wait_for_spfresh_ready_fails_immediately_on_needs_rebuild(self):
        tidb = make_tidb()
        status = SPFreshIndexStatus(
            applied_base_ts=149,
            index_state="NEEDS_REBUILD",
            is_ready=False,
            lag_seconds=1,
            owner_lease_expire_ts=999,
            observed_at="2026-04-24 14:00:07",
        )
        with (
            patch.object(tidb, "_fetch_spfresh_index_status", return_value=status) as fetch_mock,
            patch("vectordb_bench.backend.clients.tidb.tidb.time.sleep") as sleep_mock,
            pytest.raises(RuntimeError, match="index_state=NEEDS_REBUILD"),
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

        fetch_mock.assert_called_once_with()
        sleep_mock.assert_not_called()

    def test_wait_for_spfresh_ready_rejects_ready_without_checkpoint(self):
        tidb = make_tidb()
        null_checkpoint = SPFreshIndexStatus(
            applied_base_ts=None,
            index_state="READY",
            is_ready=True,
            lag_seconds=None,
            owner_lease_expire_ts=None,
            observed_at="2026-04-24 14:00:08",
        )
        with (
            patch.object(tidb, "_fetch_spfresh_index_status", return_value=null_checkpoint),
            patch("vectordb_bench.backend.clients.tidb.tidb.time.sleep") as sleep_mock,
            pytest.raises(RuntimeError) as exc_info,
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

        sleep_mock.assert_not_called()
        assert "READY status is missing applied_base_ts" in str(exc_info.value)
        assert "applied_base_ts=None" in str(exc_info.value)
        assert "observed_at=2026-04-24 14:00:08" in str(exc_info.value)

    def test_wait_for_spfresh_ready_fails_when_status_row_is_missing(self):
        tidb = make_tidb()
        with (
            patch.object(tidb, "_fetch_spfresh_index_status", return_value=None) as fetch_mock,
            patch("vectordb_bench.backend.clients.tidb.tidb.time.sleep") as sleep_mock,
            pytest.raises(RuntimeError) as exc_info,
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

        fetch_mock.assert_called_once_with()
        sleep_mock.assert_not_called()
        assert "status row is missing" in str(exc_info.value)
        assert "index_state=None" in str(exc_info.value)

    def test_wait_for_spfresh_ready_times_out_with_last_checkpoint_below_barrier(self):
        tidb = make_tidb()
        below_barrier = SPFreshIndexStatus(
            applied_base_ts=149,
            index_state="READY",
            is_ready=True,
            lag_seconds=0,
            owner_lease_expire_ts=999,
            observed_at="2026-04-24 14:00:09",
        )
        time_points = iter([0.0, 0.5, 1.1])
        with (
            patch.object(tidb, "_fetch_spfresh_index_status", return_value=below_barrier),
            patch("vectordb_bench.backend.clients.tidb.tidb.time.monotonic", side_effect=lambda: next(time_points)),
            patch("vectordb_bench.backend.clients.tidb.tidb.time.sleep"),
            pytest.raises(RuntimeError) as exc_info,
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

        assert "barrier_ts=150" in str(exc_info.value)
        assert "index_state=READY" in str(exc_info.value)
        assert "is_ready=1" in str(exc_info.value)
        assert "applied_base_ts=149" in str(exc_info.value)
        assert "waited_seconds=1.100" in str(exc_info.value)

    def test_wait_for_spfresh_ready_rejects_checkpoint_observed_at_deadline(self):
        tidb = make_tidb()
        reached_barrier = SPFreshIndexStatus(
            applied_base_ts=150,
            index_state="READY",
            is_ready=True,
            lag_seconds=0,
            owner_lease_expire_ts=999,
            observed_at="2026-04-24 14:00:10",
        )
        time_points = iter([0.0, 0.5, 1.0])
        with (
            patch.object(tidb, "_fetch_spfresh_index_status", return_value=reached_barrier),
            patch("vectordb_bench.backend.clients.tidb.tidb.time.monotonic", side_effect=lambda: next(time_points)),
            pytest.raises(RuntimeError) as exc_info,
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

        assert "barrier_ts=150" in str(exc_info.value)
        assert "applied_base_ts=150" in str(exc_info.value)

    def test_wait_for_spfresh_ready_caps_poll_interval_to_remaining_timeout(self):
        tidb = make_tidb()
        time_points = iter([0.0, 0.25, 0.75, 1.0])
        with (
            patch.object(
                tidb,
                "_fetch_spfresh_index_status",
                return_value=SPFreshIndexStatus(
                    applied_base_ts=149,
                    index_state="READY",
                    is_ready=True,
                    lag_seconds=1,
                    owner_lease_expire_ts=999,
                    observed_at="2026-04-24 14:00:11",
                ),
            ) as fetch_mock,
            patch("vectordb_bench.backend.clients.tidb.tidb.time.monotonic", side_effect=lambda: next(time_points)),
            patch("vectordb_bench.backend.clients.tidb.tidb.time.sleep") as sleep_mock,
            pytest.raises(RuntimeError, match="barrier_ts=150"),
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=1,
                poll_interval_seconds=10,
            )

        fetch_mock.assert_called_once_with()
        sleep_mock.assert_called_once_with(0.25)

    def test_wait_for_spfresh_ready_reports_status_query_errors(self):
        tidb = make_tidb()
        query_error = RuntimeError("TiDB status query failed")
        with (
            patch.object(tidb, "_fetch_spfresh_index_status", side_effect=query_error),
            pytest.raises(RuntimeError) as exc_info,
        ):
            tidb._wait_for_spfresh_ready(
                barrier_ts=150,
                timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

        assert exc_info.value.__cause__ is query_error
        message = str(exc_info.value)
        assert "status query failed: TiDB status query failed" in message
        assert "table=vector_bench_test" in message
        assert "index=idx_embedding_spfresh_l2" in message
        assert "waited_seconds=" in message

    def test_fetch_spfresh_index_status_rejects_missing_columns(self):
        tidb = make_tidb()
        tidb.cursor = FakeCursor(fetchone_results=[(123, "READY")])

        with pytest.raises(RuntimeError, match="expected 6 columns, got 2"):
            tidb._fetch_spfresh_index_status()

    def test_fetch_spfresh_index_status_rejects_missing_required_fields(self):
        tidb = make_tidb()
        tidb.cursor = FakeCursor(fetchone_results=[(123, None, None, 0, None, None)])

        with pytest.raises(RuntimeError, match="missing required fields: index_state, is_ready, observed_at"):
            tidb._fetch_spfresh_index_status()
