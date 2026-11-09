import logging
import struct
import time
from collections.abc import Generator
from contextlib import contextmanager
from copy import copy
from typing import Any

import mysql.connector as mysql

from vectordb_bench.backend.filter import CohereLabelAndFilter, Filter, FilterOp

from ..api import IndexType, VectorDB
from .config import OceanBaseConfigDict, OceanBaseHNSWConfig

log = logging.getLogger(__name__)

OCEANBASE_DEFAULT_LOAD_BATCH_SIZE = 256


class OceanBase(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
        FilterOp.LabelAnd,
    ]
    # mysql-connector cursor cannot be shared across threads
    thread_safe: bool = False

    def __init__(
        self,
        dim: int,
        db_config: OceanBaseConfigDict,
        db_case_config: OceanBaseHNSWConfig,
        collection_name: str = "items",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        extra_label_fields: tuple[str, ...] = (),
        **kwargs,
    ):
        self.name = "OceanBase"
        self.dim = dim
        self.db_config = db_config
        self.db_case_config = db_case_config
        self.table_name = collection_name
        self.load_batch_size = OCEANBASE_DEFAULT_LOAD_BATCH_SIZE
        self._index_name = "vidx"
        self._primary_field = "id"
        self._vector_field = "embedding"
        self._label_fields = ("labels", *extra_label_fields) if with_scalar_labels else ()
        self.expr = ""
        self.where_params = ()

        log.info("OceanBase initialized for %s with case config %s", self.table_name, self.db_case_config)

        self._conn = None
        self._cursor = None

        try:
            self._connect()
            if drop_old:
                self._drop_table()
                self._create_table()
        finally:
            self._disconnect()

    def _connect(self):
        try:
            self._conn = mysql.connect(
                host=self.db_config["host"],
                user=self.db_config["user"],
                port=self.db_config["port"],
                password=self.db_config["password"],
                database=self.db_config["database"],
            )
            self._cursor = self._conn.cursor()
        except mysql.Error:
            log.exception("Failed to connect to the database")
            raise

    def _disconnect(self):
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._conn:
            self._conn.close()
            self._conn = None

    def copy_for_thread(self) -> "VectorDB":
        # mysql.connector holds an open socket that can't be deep-copied; shallow-copy
        # and drop the connection so init() reconnects inside the worker thread.
        db_copy = copy(self)
        db_copy._conn = None
        db_copy._cursor = None
        return db_copy

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        try:
            self._connect()
            self._cursor.execute("SET autocommit=1")

            if self.db_case_config.index in {IndexType.HNSW, IndexType.HNSW_SQ, IndexType.HNSW_BQ}:
                self._cursor.execute(
                    f"SET ob_hnsw_ef_search={(self.db_case_config.search_param())['params']['ef_search']}"
                )
            else:
                self._cursor.execute(
                    f"SET ob_ivf_nprobes={(self.db_case_config.search_param())['params']['ivf_nprobes']}"
                )
            yield
        finally:
            self._disconnect()

    def _drop_table(self):
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        log.info(f"Dropping table {self.table_name}")
        self._cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")

    def _create_table(self):
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        partitions = getattr(self.db_case_config, "partitions", 0)
        log.info(f"Creating table {self.table_name} (partitions={partitions})")

        label_sql = "".join(f", {field} VARCHAR(64) COLLATE utf8mb4_bin NOT NULL" for field in self._label_fields)
        create_table_query = (
            f"CREATE TABLE {self.table_name} (id INT PRIMARY KEY, embedding VECTOR({self.dim}){label_sql})"
        )
        if partitions > 1:
            create_table_query += f" PARTITION BY KEY(id) PARTITIONS {partitions}"
        create_table_query += ";"
        self._cursor.execute(create_table_query)

    def optimize(self, data_size: int):
        index_params = self.db_case_config.index_param()
        index_args = ", ".join(f"{k}={v}" for k, v in index_params["params"].items())
        index_query = (
            f"CREATE /*+ PARALLEL({self.db_case_config.create_index_parallel}) */ VECTOR INDEX idx1 "
            f"ON {self.table_name}(embedding) "
            f"WITH (distance={self.db_case_config.parse_metric()}, "
            f"type={index_params['index_type']}, lib={index_params['lib']}, {index_args}"
        )

        extra_info = getattr(self.db_case_config, "extra_info_max_size", None)
        if extra_info is not None:
            index_query += f", extra_info_max_size={extra_info}"

        index_query += ")"

        log.info("Create index query: %s", index_query)

        try:
            log.info("Creating index...")
            start_time = time.time()
            self._cursor.execute(index_query)
            log.info(f"Index created in {time.time() - start_time:.2f} seconds")

            log.info("Performing major freeze...")
            self._cursor.execute("ALTER SYSTEM MAJOR FREEZE;")
            time.sleep(10)
            self._wait_for_major_compaction()

            log.info("Gathering schema statistics...")
            self._cursor.execute(
                "CALL dbms_stats.gather_schema_stats(%s, degree => %s);",
                (self.db_config["database"], self.db_case_config.create_index_parallel),
            )
        except mysql.Error:
            log.exception("Failed to optimize index")
            raise

    def need_normalize_cosine(self) -> bool:
        return False

    def _wait_for_major_compaction(self):
        while True:
            self._cursor.execute(
                "SELECT IF(COUNT(*) = COUNT(STATUS = 'IDLE' OR NULL), 'TRUE', 'FALSE') "
                "AS all_status_idle FROM oceanbase.DBA_OB_ZONE_MAJOR_COMPACTION;"
            )
            all_status_idle = self._cursor.fetchone()[0]
            if all_status_idle == "TRUE":
                break
            time.sleep(10)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        extra_labels_data: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        label_columns = {"labels": labels_data, **(extra_labels_data or {})}
        if any(
            label_columns.get(field) is None or len(label_columns[field]) != len(embeddings)
            for field in self._label_fields
        ):
            raise ValueError("OceanBase label data must match the inserted embeddings")
        fields = ", ".join(("id", "embedding", *self._label_fields))
        insert_count = 0
        try:
            for batch_start in range(0, len(embeddings), self.load_batch_size):
                batch_end = min(batch_start + self.load_batch_size, len(embeddings))
                label_placeholders = ", %s" * len(self._label_fields)
                values = ", ".join(
                    f"({metadata[i]}, '[{','.join(map(str, embeddings[i]))}]'{label_placeholders})"
                    for i in range(batch_start, batch_end)
                )
                sql = f"INSERT INTO {self.table_name} ({fields}) VALUES {values}"
                if self._label_fields:
                    params = tuple(
                        label_columns[field][i] for i in range(batch_start, batch_end) for field in self._label_fields
                    )
                    self._cursor.execute(sql, params)
                else:
                    self._cursor.execute(sql)
                insert_count += batch_end - batch_start
        except mysql.Error:
            log.exception("Failed to insert embeddings")
            raise

        return insert_count, None

    def prepare_filter(self, filters: Filter):
        self.where_params = ()
        if filters.type == FilterOp.NonFilter:
            self.expr = ""
        elif filters.type == FilterOp.NumGE:
            self.expr = f"WHERE id >= {filters.int_value}"
        elif isinstance(filters, CohereLabelAndFilter) or filters.type == FilterOp.StrEqual:
            conditions = (
                filters.conditions
                if isinstance(filters, CohereLabelAndFilter)
                else ((filters.label_field, filters.label_value),)
            )
            if any(field not in self._label_fields for field, _ in conditions):
                raise ValueError("OceanBase label filters require the corresponding label columns")
            self.expr = "WHERE " + " AND ".join(f"{field} = %s" for field, _ in conditions)
            self.where_params = tuple(value for _, value in conditions)
        else:
            msg = f"Not support Filter for Oceanbase - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
    ) -> list[int]:
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        packed = struct.pack(f"<{len(query)}f", *query)
        hex_vec = packed.hex()
        query_str = (
            f"SELECT id FROM {self.table_name} "
            f"{self.expr} ORDER BY "
            f"{self.db_case_config.parse_metric_func_str()}(embedding, X'{hex_vec}') "
            f"APPROXIMATE LIMIT {k}"
        )

        try:
            if self.where_params:
                self._cursor.execute(query_str, self.where_params)
            else:
                self._cursor.execute(query_str)
            return [row[0] for row in self._cursor.fetchall()]
        except mysql.Error:
            log.exception("Failed to execute search query")
            raise
