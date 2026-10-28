import concurrent
import hashlib
import logging
import re
import time
import traceback
from enum import Enum, auto

import numpy as np

from .. import config
from ..base import BaseModel
from ..metric import Metric, OptimizeResult
from ..models import PerformanceTimeoutError, TaskConfig, TaskStage
from . import utils
from .cases import Case, CaseLabel, StreamingPerformanceCase
from .clients import DB, MetricType, api
from .data_source import DatasetSource
from .filter import FilterOp
from .payload import FLOAT32_BYTES_PER_DIM
from .runner import (
    ColdWarmSearchRunner,
    ConcurrentInsertRunner,
    MultiProcessingSearchRunner,
    ReadWriteRunner,
    SerialDeleteRunner,
    SerialInsertRunner,
    SerialSearchRunner,
)
from .utils import kill_proc_tree
from .workload import WorkloadKind

log = logging.getLogger(__name__)
# Milvus and Zilliz Cloud cap topK at this value unless the collection opts into Large TopK mode.
MILVUS_DEFAULT_TOPK_LIMIT = 16_384
LARGE_TOPK_QUERY_MODE_DBS = frozenset({DB.Milvus, DB.ZillizCloud})


class RunningStatus(Enum):
    PENDING = auto()
    FINISHED = auto()


class CaseRunner(BaseModel):
    """DataSet, filter_rate, db_class with db config

    Fields:
        run_id(str): run_id of this case runner,
            indicating which task does this case belong to.
        config(TaskConfig): task configs of this case runner.
        ca(Case): case for this case runner.
        status(RunningStatus): RunningStatus of this case runner.

        db(api.VectorDB): The vector database for this case runner.
    """

    run_id: str
    config: TaskConfig
    ca: Case
    status: RunningStatus
    dataset_source: DatasetSource

    db: api.VectorDB | None = None
    test_emb: list[list[float]] | None = None
    test_texts: list[str] | None = None
    serial_search_runner: SerialSearchRunner | None = None
    search_runner: MultiProcessingSearchRunner | None = None
    final_search_runner: MultiProcessingSearchRunner | None = None
    read_write_runner: ReadWriteRunner | None = None
    cold_warm_search_runner: ColdWarmSearchRunner | None = None

    def __eq__(self, obj: any):
        if isinstance(obj, CaseRunner):
            key = self.load_reuse_key()
            return key is not None and key == obj.load_reuse_key()
        return False

    def __hash__(self) -> int:
        """Hash method to maintain consistency with __eq__ method."""
        return hash(self.load_reuse_key())

    def load_reuse_key(self) -> tuple | None:
        if self.ca.label != CaseLabel.Performance:
            return None
        return (
            self.config.db.value,
            self._db_config_hash_key(),
            self._db_case_config_hash_key(),
            self._collection_name_hash_key(),
            self._dataset_hash_key(),
            self.config.insert_batch_size,
            self._hashable_value(self._collection_properties()),
            self.ca.with_scalar_labels,
            self.ca.is_multitenant,
            self._multitenant_routing_hash_key(),
        )

    @classmethod
    def _hashable_value(cls, value: object) -> object:
        if isinstance(value, dict):
            hashable = tuple(sorted((str(k), cls._hashable_value(v)) for k, v in value.items()))
        elif isinstance(value, (list, tuple)):
            hashable = tuple(cls._hashable_value(v) for v in value)
        elif isinstance(value, (set, frozenset)):
            hashable = tuple(sorted((cls._hashable_value(v) for v in value), key=repr))
        elif isinstance(value, Enum):
            hashable = value.value
        elif hasattr(value, "model_dump"):
            hashable = cls._hashable_value(value.model_dump(mode="json"))
        elif hasattr(value, "get_secret_value"):
            hashable = value.get_secret_value()
        else:
            hashable = value
        return hashable

    def _db_config_hash_key(self) -> object:
        db_config = self.config.db_config
        if hasattr(db_config, "to_dict"):
            return self._hashable_value(db_config.to_dict())
        return self._hashable_value(db_config)

    def _db_case_config_hash_key(self) -> object:
        return self._hashable_value(self.config.db_case_config)

    def _collection_name_hash_key(self) -> str | None:
        return self._doris_collection_name()

    def _dataset_hash_key(self) -> object:
        return self._hashable_value(self.ca.dataset.data)

    def _multitenant_routing_hash_key(self) -> tuple | None:
        if not self.ca.is_multitenant:
            return None
        return (
            getattr(self.ca, "tenant_count", None),
            getattr(self.ca, "tenant_prefix", None),
            getattr(self.ca, "tenant_id_width", None),
            getattr(self.ca, "tenant_distribution", None),
        )

    def _doris_collection_name(self) -> str | None:
        if self.config.db != DB.Doris:
            return None
        case_type_name = self.config.case_config.case_id.name
        base = f"{case_type_name.lower()}"
        base = re.sub(r"[^a-z0-9_]+", "_", base).strip("_")
        if len(base) > 63:
            h = hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()[:6]
            base = f"{base[: (63 - 7)]}_{h}"
        return base

    def display(self) -> dict:
        dataset_include = {
            "name": True,
            "size": True,
            "label": True,
            "metric_type": True,
        }
        if self.ca.label != CaseLabel.FullTextSearchPerformance:
            dataset_include["dim"] = True
        c_dict = self.ca.dict(
            include={
                "label": True,
                "name": True,
                "filters": True,
                "dataset": {
                    "data": dataset_include,
                },
            },
        )
        c_dict["db"] = self.config.db_name
        return c_dict

    @property
    def normalize(self) -> bool:
        assert self.db
        if self.is_fts:
            return False
        return self.db.need_normalize_cosine() and self.ca.dataset.data.metric_type == MetricType.COSINE

    @property
    def workload_kind(self) -> WorkloadKind:
        if getattr(self.ca, "label", None) == CaseLabel.FullTextSearchPerformance:
            return WorkloadKind.FULL_TEXT
        return WorkloadKind.VECTOR

    @property
    def is_fts(self) -> bool:
        return self.workload_kind == WorkloadKind.FULL_TEXT

    def _collection_properties(self, *, log_selection: bool = False) -> dict[str, str]:
        requested_k = self.config.case_config.k or config.K_DEFAULT
        if (
            self.config.db not in LARGE_TOPK_QUERY_MODE_DBS
            or self.ca.label != CaseLabel.Performance
            or requested_k <= MILVUS_DEFAULT_TOPK_LIMIT
        ):
            return {}

        # Large TopK mode must be applied at collection creation, before the vector index is created.
        if log_selection:
            log.info(
                "%s requested K=%d exceeds the default TopK limit %d; using query_mode=large_topk",
                self.config.db.value,
                requested_k,
                MILVUS_DEFAULT_TOPK_LIMIT,
            )
        return {"query_mode": "large_topk"}

    def init_db(self, drop_old: bool = True) -> None:
        db_cls = self.config.db.init_cls
        # Compose a compact, case-unique collection/table name for Doris to avoid cross-case interference
        collection_name = None
        try:
            collection_name = self._doris_collection_name()
        except Exception:
            # If anything goes wrong, fall back silently; Doris will use its default name logic
            collection_name = None

        # Check if collection_name is in the db_config (e.g., for Zilliz, Milvus)
        db_config_dict = self.config.db_config.to_dict()
        if "collection_name" in db_config_dict and not collection_name:
            collection_name = db_config_dict.pop("collection_name")

        extra_db_kwargs = {}
        if collection_name:
            extra_db_kwargs["collection_name"] = collection_name
        if self.ca.is_multitenant:
            extra_db_kwargs["multitenant_tenant_labels"] = self.ca.tenant_labels()
        if self.is_fts:
            extra_db_kwargs["fts_filter_enabled"] = self.ca.filters.type != FilterOp.NonFilter
        if self.config.db is DB.AWSOpenSearch:
            extra_db_kwargs["insert_batch_size"] = self.config.insert_batch_size
        collection_properties = self._collection_properties(log_selection=True)
        if collection_properties:
            extra_db_kwargs["collection_properties"] = collection_properties

        self.db = db_cls(
            dim=getattr(self.ca.dataset.data, "dim", 0),
            db_config=db_config_dict,
            db_case_config=self.config.db_case_config,
            drop_old=drop_old,
            with_scalar_labels=self.ca.with_scalar_labels,
            **extra_db_kwargs,
        )

    def _validate_vector_payload_profile(self) -> None:
        if self.ca.label != CaseLabel.Performance or self.is_fts:
            return
        if not self.config.db.init_cls.supports_payload_profile(self.ca.payload_profile):
            msg = f"{self.config.db_name} does not support payload_profile={self.ca.payload_profile.value}"
            raise NotImplementedError(msg)

    def _pre_run(self, drop_old: bool = True):
        try:
            self._validate_nq()
            self._validate_cloud_cold_latency_config(drop_old)
            requested_k = self.config.case_config.k or config.K_DEFAULT
            ground_truth_k = (
                requested_k
                if self.ca.label == CaseLabel.Performance and getattr(self.ca, "measure_recall", True)
                else config.K_DEFAULT
            )
            creates_multitenant_collection = (
                TaskStage.DROP_OLD in self.config.stages or TaskStage.LOAD in self.config.stages
            )
            if (
                self.ca.is_multitenant
                and self.config.db in {DB.Milvus, DB.ZillizCloud}
                and creates_multitenant_collection
                and not getattr(self.config.db_case_config, "use_partition_key", False)
            ):
                msg = "CloudMultiTenantSearchCase requires use_partition_key=True for Milvus/ZillizCloud"
                raise ValueError(msg)

            if self.is_fts:
                self.ca.dataset.prepare(
                    self.dataset_source,
                    filters=self.ca.filters,
                )
                self.init_db(drop_old)
                return

            self._validate_vector_payload_profile()
            if self.ca.dataset.data.with_gt and self._requires_perf_dataset_prepare():
                self.ca.dataset.resolve_search_files(k=ground_truth_k, filters=self.ca.filters)
            self.init_db(drop_old)
            if self.ca.is_multitenant and self.db is not None:
                if not self.db.supports_multitenant():
                    msg = f"{self.config.db_name} does not support CloudMultiTenantSearchCase"
                    raise NotImplementedError(msg)
                self.db.set_multitenant_context(self.ca.tenant_labels())
                if self.config.db in {DB.Milvus, DB.ZillizCloud} and not creates_multitenant_collection:
                    self.db.validate_multitenant_schema()
            if self.ca.label != CaseLabel.Performance or self._requires_perf_dataset_prepare():
                self.ca.dataset.prepare(
                    self.dataset_source,
                    filters=self.ca.filters,
                    with_train_files=any(stage in self.config.stages for stage in (TaskStage.LOAD, TaskStage.DELETE)),
                    with_scalar_labels=self.ca.with_scalar_labels,
                    k=ground_truth_k,
                )
            else:
                log.info("Dataset preparation skipped")
        except ModuleNotFoundError as e:
            log.warning(f"pre run case error: please install client for db: {self.config.db}, error={e}")
            raise e from None

    def _validate_nq(self) -> None:
        if self.config.case_config.nq == 1:
            return
        if self.ca.label != CaseLabel.Performance or self.ca.is_multitenant:
            raise ValueError("nq > 1 is only supported for single-tenant vector performance cases")
        if not self.config.db.init_cls.supports_batch_search:
            msg = f"{self.config.db.value} does not support nq > 1"
            raise NotImplementedError(msg)

    def _validate_cloud_cold_latency_config(self, drop_old: bool) -> None:
        if getattr(self.ca, "label", None) != CaseLabel.CloudColdLatency:
            return
        if drop_old:
            msg = "CloudColdLatencyCase requires an existing cold collection. Run with --skip-drop-old and --skip-load."
            raise ValueError(msg)
        if TaskStage.LOAD in self.config.stages:
            msg = "CloudColdLatencyCase is search-only. Run with --skip-load."
            raise ValueError(msg)

    def run(self, drop_old: bool = True) -> Metric:
        log.info("Starting run")

        self._pre_run(drop_old)

        if self.ca.label == CaseLabel.Load:
            return self._run_capacity_case()
        if self.ca.label in {CaseLabel.Performance, CaseLabel.FullTextSearchPerformance}:
            return self._run_perf_case(drop_old)
        if self.ca.label == CaseLabel.Streaming:
            return self._run_streaming_case()
        if self.ca.label == CaseLabel.CloudInsert:
            return self._run_cloud_insert_case()
        if self.ca.label == CaseLabel.CloudColdLatency:
            return self._run_cloud_cold_latency_case(drop_old)
        msg = f"unknown case type: {self.ca.label}"
        log.warning(msg)
        raise ValueError(msg)

    def _run_capacity_case(self) -> Metric:
        """run capacity cases

        Returns:
            Metric: the max load count
        """
        assert self.db is not None
        log.info("Start capacity case")
        try:
            runner = SerialInsertRunner(
                self.db,
                self.ca.dataset,
                self.normalize,
                self.ca.filters,
                self.ca.load_timeout,
                batch_size=self.config.insert_batch_size,
            )
            count = runner.run_endlessness()
        except Exception as e:
            log.warning(f"Failed to run capacity case, reason = {e}")
            raise e from None
        else:
            log.info(f"Capacity case loading dataset reaches VectorDB's limit: max capacity = {count}")
            return Metric(max_load_count=count)

    def _estimated_payload_bytes_per_query(self) -> int:
        return self.ca.estimated_payload_bytes_per_query(
            self.config.case_config.k,
            vector_bytes_per_dim=getattr(self.config.db_case_config, "vector_bytes_per_dim", FLOAT32_BYTES_PER_DIM),
        )

    def _run_perf_case(self, drop_old: bool = True) -> Metric:
        """run performance cases

        Returns:
            Metric: load_duration, recall, serial_latency_p99, and, qps
        """

        log.info("Start performance case")
        try:
            m = Metric()
            if self.is_fts and getattr(self.ca.dataset, "filter_stats", None):
                m.additional_parameters["fts_filter"] = dict(self.ca.dataset.filter_stats)
                m.additional_parameters["fts_recall"] = {
                    "skipped": bool(getattr(self.ca.dataset, "recall_skipped", False)),
                    "reason": getattr(self.ca.dataset, "recall_skip_reason", None),
                    "serial_query_count": len(getattr(self.ca.dataset, "recall_queries_data", []) or []),
                    "full_query_count": len(getattr(self.ca.dataset, "queries_data", []) or []),
                }
            if drop_old:
                self._run_perf_load_stage(m)
                if TaskStage.LOAD in self.config.stages:
                    m.additional_parameters.update(
                        {
                            "insert_batch_size": self.config.insert_batch_size,
                            "load_concurrency": self.config.load_concurrency,
                        }
                    )
            if TaskStage.BUILD in self.config.stages and TaskStage.LOAD not in self.config.stages:
                self._run_perf_build_only_stage(m)
            if TaskStage.DELETE in self.config.stages:
                self._run_perf_delete_stage(m)
            if TaskStage.SEARCH_SERIAL in self.config.stages or TaskStage.SEARCH_CONCURRENT in self.config.stages:
                self._init_search_runners()
                if TaskStage.SEARCH_CONCURRENT in self.config.stages:
                    search_results = self._conc_search()
                    (
                        m.qps,
                        m.conc_num_list,
                        m.conc_qps_list,
                        m.conc_latency_p99_list,
                        m.conc_latency_p95_list,
                        m.conc_latency_avg_list,
                        m.conc_latency_p50_list,
                    ) = search_results
                if TaskStage.SEARCH_SERIAL in self.config.stages:
                    cooldown = self.config.case_config.concurrency_search_config.serial_cooldown
                    if TaskStage.SEARCH_CONCURRENT in self.config.stages and cooldown > 0:
                        log.info(
                            f"Cooldown {cooldown}s before serial search to ensure a stable measurement environment"
                        )
                        time.sleep(cooldown)
                    search_results = self._serial_search()
                    if self.is_fts:
                        m.recall, m.ndcg, m.mrr, m.serial_latency_p99, m.serial_latency_p95 = search_results
                    else:
                        (
                            m.recall,
                            m.ndcg,
                            m.serial_latency_p99,
                            m.serial_latency_p95,
                            m.serial_latency_p50,
                            m.recall_at,
                        ) = search_results
            gt_data = getattr(self.ca.dataset, "gt_data", None)
            if gt_data is not None and hasattr(gt_data, "path"):
                m.additional_parameters["ground_truth"] = {
                    "file": gt_data.path.name,
                    "query_count": gt_data.row_count,
                    "width": gt_data.width,
                }
            if hasattr(self.ca, "payload_profile"):
                m.payload_profile = self.ca.payload_profile.value
                m.payload_estimated_bytes_per_query = self._estimated_payload_bytes_per_query()
        except Exception as e:
            log.warning(f"Failed to run performance case, reason = {e}")
            traceback.print_exc()
            raise e from None
        else:
            log.info(f"Performance case got result: {m}")
            return m

    def _requires_perf_dataset_prepare(self) -> bool:
        return any(
            stage in self.config.stages
            for stage in (
                TaskStage.LOAD,
                TaskStage.DELETE,
                TaskStage.SEARCH_SERIAL,
                TaskStage.SEARCH_CONCURRENT,
            )
        )

    def _run_perf_load_stage(self, metric: Metric) -> None:
        if TaskStage.LOAD not in self.config.stages:
            log.info("Data loading skipped")
            return

        if self._is_tidb_spfresh_split_load():
            self._run_spfresh_split_load(metric)
            log.info(
                "Finish SPFRESH split load, insert_duration=%s, "
                "spfresh_build_duration=%s, spfresh_incremental_catchup_duration=%s, "
                "load_duration=%s",
                metric.insert_duration,
                metric.spfresh_build_duration,
                metric.spfresh_incremental_catchup_duration,
                metric.load_duration,
            )
        else:
            self._run_standard_load_stage(metric)

        if self._tidb_spfresh_build_mode() is not None:
            log.info(
                "Finish loading the entire dataset into VectorDB, insert_duration=%s, "
                "spfresh_build_duration=%s, spfresh_incremental_catchup_duration=%s, load_duration=%s",
                metric.insert_duration,
                metric.spfresh_build_duration,
                metric.spfresh_incremental_catchup_duration,
                metric.load_duration,
            )
        else:
            log.info(
                "Finish loading the entire dataset into VectorDB, insert_duration=%s, "
                "optimize_duration=%s, load_duration(insert + optimize)=%s",
                metric.insert_duration,
                metric.optimize_duration,
                metric.load_duration,
            )

    def _run_standard_load_stage(self, metric: Metric) -> None:
        load_start = time.perf_counter()
        count, load_dur = self._load_data()
        optimize_result = self._coerce_optimize_result(self._optimize())
        if self._tidb_spfresh_build_mode() is not None:
            load_duration = time.perf_counter() - load_start
        else:
            load_duration = load_dur + optimize_result.optimize_duration

        metric.inserted_count = count
        metric.insert_duration = round(load_dur, 4)
        self._record_optimize_result(metric, optimize_result)
        metric.load_duration = round(load_duration, 4)

    def _run_perf_build_only_stage(self, metric: Metric) -> None:
        build_result = self._optimize()
        self._record_optimize_result(metric, build_result)
        if self._tidb_spfresh_build_mode() is not None:
            log.info(
                "Finish building VectorDB index, spfresh_build_duration=%s",
                metric.spfresh_build_duration,
            )
        else:
            log.info("Finish building VectorDB index, optimize_duration=%s", metric.optimize_duration)

    def _run_perf_delete_stage(self, metric: Metric) -> None:
        _, delete_dur = self._delete_data()
        metric.delete_duration = round(delete_dur, 4)
        log.info(f"Finish deleting the entire dataset from VectorDB, delete_duration={delete_dur}")
        (
            immediate_results,
            final_results,
            settle_duration,
            poll_count,
            deadline_exceeded,
        ) = self._verify_search_after_delete()
        metric.delete_search_immediate_results = immediate_results
        metric.delete_search_immediate_result_count = len(immediate_results)
        metric.delete_search_final_results = final_results
        metric.delete_search_final_result_count = len(final_results)
        metric.delete_search_settle_duration = round(settle_duration, 4)
        metric.delete_search_poll_count = poll_count
        log.info(
            "Post-delete ANN search finished, immediate_count=%s, immediate_results=%s, "
            "final_count=%s, final_results=%s, settle_duration=%s, polls=%s",
            metric.delete_search_immediate_result_count,
            metric.delete_search_immediate_results,
            metric.delete_search_final_result_count,
            metric.delete_search_final_results,
            metric.delete_search_settle_duration,
            metric.delete_search_poll_count,
        )
        if deadline_exceeded:
            self._raise_post_delete_verification_error(final_results)

    def _tidb_spfresh_build_mode(self) -> str | None:
        if self.config.db != DB.TiDB or not self.config.db_case_config.build_spfresh_index:
            return None
        return self.config.db_case_config.spfresh_build_mode.value

    def _is_tidb_spfresh_split_load(self) -> bool:
        return self._tidb_spfresh_build_mode() == "split"

    def _spfresh_split_counts(self) -> tuple[int, int]:
        ratio = self.config.db_case_config.spfresh_split_ratio
        total = int(self.ca.dataset.data.size)
        if total < 2:
            raise ValueError("SPFRESH split load requires at least two rows")
        base_count = int(total * ratio)
        base_count = min(max(1, base_count), total - 1)
        return base_count, total - base_count

    @staticmethod
    def _coerce_optimize_result(result: OptimizeResult | float | None, wall_duration: float | None = None):
        if isinstance(result, OptimizeResult):
            return result
        if isinstance(result, int | float):
            return OptimizeResult(optimize_duration=float(result))
        return OptimizeResult(optimize_duration=0.0 if wall_duration is None else wall_duration)

    def _record_optimize_result(self, metric: Metric, result: OptimizeResult | float | None) -> None:
        optimize_result = self._coerce_optimize_result(result)
        metric.optimize_duration = round(optimize_result.optimize_duration, 4)
        metric.spfresh_build_duration = round(optimize_result.spfresh_build_duration, 4)
        metric.spfresh_incremental_catchup_duration = round(
            optimize_result.spfresh_incremental_catchup_duration,
            4,
        )

    def _run_spfresh_split_load(self, metric: Metric) -> None:
        base_count, delta_count = self._spfresh_split_counts()

        load_start = time.perf_counter()
        _, base_insert_dur = self._load_data(limit=base_count)
        build_result = self._coerce_optimize_result(self._build_spfresh_index())

        _, delta_insert_dur = self._load_data(start_offset=base_count, limit=delta_count)
        wait_result = self._coerce_optimize_result(self._wait_spfresh_incremental_catchup())
        load_duration = time.perf_counter() - load_start

        spfresh_build_dur = build_result.spfresh_build_duration
        incremental_catchup_dur = wait_result.spfresh_incremental_catchup_duration

        metric.inserted_count = base_count + delta_count
        metric.spfresh_base_insert_duration = round(base_insert_dur, 4)
        metric.spfresh_delta_insert_duration = round(delta_insert_dur, 4)
        metric.insert_duration = round(base_insert_dur + delta_insert_dur, 4)
        metric.spfresh_build_duration = round(spfresh_build_dur, 4)
        metric.spfresh_incremental_catchup_duration = round(incremental_catchup_dur, 4)
        metric.load_duration = round(load_duration, 4)

    @utils.time_it
    def _delete_data(self):
        """Delete train data and get the delete_duration"""
        try:
            delete_timeout = getattr(self.config.db_case_config, "delete_timeout", None)
            if delete_timeout is None:
                delete_timeout = self.ca.optimize_timeout
            runner = SerialDeleteRunner(
                self.db,
                self.ca.dataset,
                delete_timeout,
            )
            count, delete_state = runner.run()
        except Exception as e:
            raise e from None
        else:
            if hasattr(self.db, "import_delete_state"):
                self.db.import_delete_state(delete_state)
            return count
        finally:
            runner = None

    def _get_delete_search_query(self) -> list[float]:
        assert self.db is not None

        if self.normalize:
            query = np.array(self.ca.dataset.test_data[0])
            query = query / np.linalg.norm(query)
            return query.tolist()
        return self.ca.dataset.test_data[0]

    def _search_once_after_delete(self, search_query: list[float]) -> list[int]:
        assert self.db is not None
        try:
            with self.db.init():
                self.db.prepare_filter(self.ca.filters)
                return self.db.search_embedding(search_query, k=1)
        except Exception as e:
            log.exception("Post-delete ANN search failed, aborting delete benchmark")
            raise RuntimeError("Post-delete ANN search failed") from e

    @staticmethod
    def _raise_post_delete_verification_error(final_results: list[int]) -> None:
        msg = f"Post-delete ANN verification did not become empty before the deadline, final_results={final_results}"
        raise RuntimeError(msg)

    def _wait_post_delete_ann_ready(self, wait_timeout: float, poll_interval: float) -> None:
        assert self.db is not None

        wait_spfresh_post_delete_catchup = getattr(self.db, "wait_spfresh_post_delete_catchup", None)
        if wait_spfresh_post_delete_catchup is None:
            return

        try:
            barrier_ts = wait_spfresh_post_delete_catchup(wait_timeout, poll_interval)
        except Exception as e:
            log.exception("Post-delete SPFRESH catch-up failed before ANN search")
            raise RuntimeError("Post-delete SPFRESH catch-up failed before ANN search") from e

        if barrier_ts is not None:
            log.info(
                "SPFRESH incremental apply caught up before post-delete ANN search, delete_commit_barrier_ts=%s",
                barrier_ts,
            )

    def _verify_search_after_delete(self) -> tuple[list[int], list[int], float, int, bool]:
        """Record immediate post-delete ANN results, then wait for eventual empty results."""
        search_query = self._get_delete_search_query()
        wait_timeout = float(getattr(self.config.db_case_config, "delete_search_wait_timeout", 30.0))
        poll_interval = float(getattr(self.config.db_case_config, "delete_search_poll_interval", 1.0))

        self._wait_post_delete_ann_ready(wait_timeout, poll_interval)

        immediate_results = self._search_once_after_delete(search_query)
        deadline = time.perf_counter() + wait_timeout
        poll_count = 1
        final_results = immediate_results
        start = time.perf_counter()

        while final_results and time.perf_counter() < deadline:
            time.sleep(poll_interval)
            final_results = self._search_once_after_delete(search_query)
            poll_count += 1

        deadline_exceeded = bool(final_results)
        return immediate_results, final_results, time.perf_counter() - start, poll_count, deadline_exceeded

    @utils.time_it
    def _db_method_task(self, method_name: str) -> OptimizeResult | None:
        with self.db.init():
            method = getattr(self.db, method_name)
            if method_name == "optimize":
                return method(data_size=self.ca.dataset.data.size)
            return method()

    def _run_db_method(self, method_name: str) -> OptimizeResult:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._db_method_task, method_name)
            try:
                raw_result, wall_duration = future.result(timeout=self.ca.optimize_timeout)
                return self._coerce_optimize_result(raw_result, wall_duration)
            except TimeoutError as e:
                log.warning(f"VectorDB {method_name} timeout in {self.ca.optimize_timeout}")
                kill_proc_tree(pids=list(executor._processes.keys()))
                raise PerformanceTimeoutError from e
            except Exception as e:
                log.warning(f"VectorDB {method_name} error: {e}")
                raise e from None

    def _build_spfresh_index(self) -> OptimizeResult:
        return self._run_db_method("build_spfresh_index")

    def _wait_spfresh_incremental_catchup(self) -> OptimizeResult:
        return self._run_db_method("wait_spfresh_incremental_catchup")

    def _run_streaming_case(self) -> Metric:
        log.info("Start streaming case")
        try:
            self._init_read_write_runner()
            m = self.read_write_runner.run_read_write()
        except Exception as e:
            log.warning(f"Failed to run streaming case, reason = {e}")
            traceback.print_exc()
            raise e from None
        else:
            log.info(f"Streaming case got result: {m}")
            return m

    def _run_cloud_insert_case(self) -> Metric:
        assert self.db is not None
        started = time.perf_counter()
        runner_kwargs = {}
        if self.ca.is_multitenant:
            runner_kwargs["tenant_case"] = self.ca
        runner = ConcurrentInsertRunner(
            self.db,
            self.ca.dataset,
            self.normalize,
            self.ca.filters,
            max_workers=self.config.load_concurrency or None,
            batch_size=self.config.insert_batch_size,
            duration=self.ca.duration,
            **runner_kwargs,
        )
        count = runner.task()
        insert_done = time.perf_counter()
        readiness_timeout = self.ca.readiness_timeout
        readiness_poll_interval = self.ca.readiness_poll_interval
        readiness_deadline = None if readiness_timeout is None else time.perf_counter() + readiness_timeout
        with self.db.init():
            status = self.db.poll_insert_readiness(count)
            searchable_started = time.perf_counter()
            while not status["fully_searchable"]:
                if readiness_deadline is not None and time.perf_counter() >= readiness_deadline:
                    msg = (
                        "Cloud insert readiness timed out waiting for fully_searchable "
                        f"after {readiness_timeout}s; last_status={status}"
                    )
                    raise TimeoutError(msg)
                time.sleep(readiness_poll_interval)
                status = self.db.poll_insert_readiness(count)
            indexed_started = time.perf_counter()
            while not status["fully_indexed"]:
                if readiness_deadline is not None and time.perf_counter() >= readiness_deadline:
                    msg = (
                        "Cloud insert readiness timed out waiting for fully_indexed "
                        f"after {readiness_timeout}s; last_status={status}"
                    )
                    raise TimeoutError(msg)
                time.sleep(readiness_poll_interval)
                status = self.db.poll_insert_readiness(count)
        return Metric(
            inserted_count=count,
            insert_rows_per_second=round(count / max(insert_done - started, 0.001), 4),
            insert_completion_seconds=round(insert_done - started, 4),
            searchable_after_insert_seconds=round(indexed_started - searchable_started, 4),
            indexed_after_searchable_seconds=round(time.perf_counter() - indexed_started, 4),
            additional_parameters=status.get("additional_parameters", {}),
        )

    def _init_cold_warm_search_runner(self) -> None:
        if self.normalize:
            test_emb = np.stack(self.ca.dataset.test_data)
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
            self.test_emb = test_emb.tolist()
        else:
            self.test_emb = self.ca.dataset.test_data

        self.cold_warm_search_runner = ColdWarmSearchRunner(
            db=self.db,
            test_data=self.test_emb,
            filters=self.ca.filters,
            k=self.config.case_config.k,
            payload_profile=self.ca.payload_profile,
            query_count=self.ca.query_count,
        )

    def _run_cloud_cold_latency_case(self, drop_old: bool = True) -> Metric:
        log.info("Start cloud cold latency case")
        try:
            self._validate_cloud_cold_latency_config(drop_old)
            m = Metric()
            self._init_cold_warm_search_runner()
            m.additional_parameters = {
                "cold_latency": self.cold_warm_search_runner.run(),
            }
            m.payload_profile = self.ca.payload_profile.value
            m.payload_estimated_bytes_per_query = self._estimated_payload_bytes_per_query()
        except Exception as e:
            log.warning(f"Failed to run cloud cold latency case, reason = {e}")
            traceback.print_exc()
            raise e from None
        else:
            log.info(f"Cloud cold latency case got result: {m}")
            return m

    @utils.time_it
    def _load_data(self, start_offset: int = 0, limit: int | None = None):
        if start_offset == 0 and limit is None:
            return self._load_train_data()
        return self._load_train_data(start_offset=start_offset, limit=limit)

    def _load_train_data(self, start_offset: int = 0, limit: int | None = None):
        """Insert vector or FTS train data concurrently and get insert duration."""
        if self.config.db == DB.TiDB:
            return self._load_tidb_train_data(start_offset, limit)
        if start_offset != 0 or limit is not None:
            raise ValueError("Partial loading is only supported by the TiDB SPFresh workflow")
        try:
            runner_kwargs = {}
            if self.ca.is_multitenant:
                runner_kwargs["tenant_case"] = self.ca
            runner = ConcurrentInsertRunner(
                self.db,
                self.ca.dataset,
                self.normalize,
                self.ca.filters,
                self.ca.load_timeout,
                max_workers=self.config.load_concurrency or None,
                batch_size=self.config.insert_batch_size,
                with_scalar_labels=self.ca.with_scalar_labels,
                workload_kind=self.workload_kind,
                **runner_kwargs,
            )
            return runner.run()
        except Exception as e:
            raise e from None
        finally:
            runner = None

    def _load_tidb_train_data(self, start_offset: int = 0, limit: int | None = None):
        """Insert train data and get the insert_duration"""
        try:
            runner = SerialInsertRunner(
                self.db,
                self.ca.dataset,
                self.normalize,
                self.ca.filters,
                self.ca.load_timeout,
                batch_size=self.config.insert_batch_size,
                start_offset=start_offset,
                limit=limit,
            )
            count, load_state = runner.run()
        except Exception as e:
            raise e from None
        else:
            if hasattr(self.db, "import_load_state"):
                self.db.import_load_state(load_state)
            return count
        finally:
            runner = None

    def _serial_search(self) -> tuple[float, ...]:
        """Performance serial tests, search the entire test data once,
        calculate the recall, serial_latency_p99, serial_latency_p95

        Returns:
            tuple[float, ...]: vector cases return recall, ndcg, p99, p95, p50, recall_at;
                FTS cases return recall, p99, p95, p50.
        """
        try:
            if self.serial_search_runner is None:
                if self.is_fts and getattr(self.ca.dataset, "recall_skipped", False):
                    log.warning(
                        "Skipping FTS serial recall: %s",
                        getattr(self.ca.dataset, "recall_skip_reason", "unknown"),
                    )
                    return (0.0, 0.0, 0.0, 0.0, 0.0)
                msg = "serial search runner is not initialized"
                raise RuntimeError(msg)  # noqa: TRY301
            results, _ = self.serial_search_runner.run()
        except Exception as e:
            log.warning(f"search error: {e!s}, {e}")
            self.stop()
            raise e from e
        else:
            return results

    def _conc_search(self):
        """Performance concurrency tests, search the test data endlessness
        for 30s in several concurrencies

        Returns:
            float: the largest qps in all concurrencies
        """
        try:
            return self.search_runner.run()
        except Exception as e:
            log.warning(f"search error: {e!s}, {e}")
            raise e from None
        finally:
            self.stop()

    @utils.time_it
    def _optimize_task(self) -> None:
        with self.db.init():
            return self.db.optimize(data_size=self.ca.dataset.data.size)

    def _optimize(self) -> OptimizeResult:
        if self.is_fts:
            try:
                with utils.timeout(self.ca.optimize_timeout, PerformanceTimeoutError):
                    raw_result, duration = self._optimize_task()
                    return self._coerce_optimize_result(raw_result, duration)
            except PerformanceTimeoutError:
                log.warning(f"VectorDB optimize timeout in {self.ca.optimize_timeout}")
                raise
        return self._run_db_method("optimize")

    def _init_search_runners(self):
        if self.is_fts:
            return self._init_fts_search_runner()
        return self._init_search_runner()

    def _init_search_runner(self):
        if self.normalize:
            test_emb = np.stack(self.ca.dataset.test_data)
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
            self.test_emb = test_emb.tolist()
        else:
            self.test_emb = self.ca.dataset.test_data

        tenant_labels = self.ca.tenant_labels() if self.ca.is_multitenant else None
        measure_recall = getattr(self.ca, "measure_recall", True)
        gt_df = self.ca.dataset.gt_data if measure_recall else None

        if TaskStage.SEARCH_SERIAL in self.config.stages:
            self.serial_search_runner = SerialSearchRunner(
                db=self.db,
                test_data=self.test_emb,
                ground_truth=gt_df,
                filters=self.ca.filters,
                k=self.config.case_config.k,
                payload_profile=self.ca.payload_profile,
                tenant_labels=tenant_labels,
                measure_recall=measure_recall,
                workload_kind=WorkloadKind.VECTOR,
            )
        if TaskStage.SEARCH_CONCURRENT in self.config.stages:
            self.search_runner = MultiProcessingSearchRunner(
                db=self.db,
                test_data=self.test_emb,
                filters=self.ca.filters,
                concurrencies=self.config.case_config.concurrency_search_config.num_concurrency,
                duration=self.config.case_config.concurrency_search_config.concurrency_duration,
                concurrency_timeout=self.config.case_config.concurrency_search_config.concurrency_timeout,
                k=self.config.case_config.k,
                payload_profile=self.ca.payload_profile,
                tenant_labels=tenant_labels,
                workload_kind=WorkloadKind.VECTOR,
                nq=self.config.case_config.nq,
            )

    def _init_fts_search_runner(self):
        fts_dataset = self.ca.dataset

        if fts_dataset.queries_data is None or fts_dataset.gt_data is None:
            msg = "FTS dataset is missing queries or ground truth. Call prepare() before initializing search."
            raise ValueError(msg)
        test_texts = [q.text for q in fts_dataset.queries_data]
        if len(test_texts) != len(fts_dataset.gt_data):
            msg = f"FTS query count {len(test_texts)} does not match ground truth row count {len(fts_dataset.gt_data)}"
            raise ValueError(msg)

        log.info(f"FTS concurrent test will use {len(test_texts)} queries for testing")
        self.test_texts = test_texts

        if TaskStage.SEARCH_SERIAL in self.config.stages:
            recall_queries = fts_dataset.recall_queries_data
            recall_ground_truth = fts_dataset.recall_gt_data
            if recall_queries is None or recall_ground_truth is None:
                msg = (
                    "FTS dataset is missing recall queries or ground truth. Call prepare() before initializing search."
                )
                raise ValueError(msg)
            if len(recall_queries) != len(recall_ground_truth):
                msg = (
                    f"FTS recall query count {len(recall_queries)} does not match "
                    f"ground truth row count {len(recall_ground_truth)}"
                )
                raise ValueError(msg)
            if fts_dataset.recall_skipped:
                log.warning("FTS serial recall will be skipped: %s", fts_dataset.recall_skip_reason)
                self.serial_search_runner = None
            else:
                recall_test_texts = [q.text for q in recall_queries]
                log.info(f"FTS serial recall will use {len(recall_test_texts)} queries")
                self.serial_search_runner = SerialSearchRunner(
                    db=self.db,
                    test_data=recall_test_texts,
                    ground_truth=recall_ground_truth,
                    filters=self.ca.filters,
                    k=self.config.case_config.k,
                    payload_profile=self.ca.payload_profile,
                    workload_kind=WorkloadKind.FULL_TEXT,
                )
        if TaskStage.SEARCH_CONCURRENT in self.config.stages:
            self.search_runner = MultiProcessingSearchRunner(
                db=self.db,
                test_data=test_texts,
                filters=self.ca.filters,
                concurrencies=self.config.case_config.concurrency_search_config.num_concurrency,
                duration=self.config.case_config.concurrency_search_config.concurrency_duration,
                concurrency_timeout=self.config.case_config.concurrency_search_config.concurrency_timeout,
                k=self.config.case_config.k,
                payload_profile=self.ca.payload_profile,
                workload_kind=WorkloadKind.FULL_TEXT,
            )

    def _init_read_write_runner(self):
        ca: StreamingPerformanceCase = self.ca
        self.read_write_runner = ReadWriteRunner(
            db=self.db,
            dataset=ca.dataset,
            insert_rate=ca.insert_rate,
            search_stages=ca.search_stages,
            optimize_after_write=ca.optimize_after_write,
            read_dur_after_write=ca.read_dur_after_write,
            concurrencies=ca.concurrencies,
            k=self.config.case_config.k,
            normalize=self.normalize,
            batch_size=self.config.insert_batch_size,
        )

    def stop(self):
        if self.search_runner:
            self.search_runner.stop()


DATA_FORMAT = " %-14s | %-12s %-20s %7s | %-10s"
TITLE_FORMAT = (" %-14s | %-12s %-20s %7s | %-10s") % (
    "DB",
    "CaseType",
    "Dataset",
    "Filter",
    "task_label",
)


class TaskRunner(BaseModel):
    run_id: str
    task_label: str
    case_runners: list[CaseRunner]

    def num_cases(self) -> int:
        return len(self.case_runners)

    def num_finished(self) -> int:
        return self._get_num_by_status(RunningStatus.FINISHED)

    def set_finished(self, idx: int) -> None:
        self.case_runners[idx].status = RunningStatus.FINISHED

    def _get_num_by_status(self, status: RunningStatus) -> int:
        return sum([1 for c in self.case_runners if c.status == status])

    def display(self) -> None:
        fmt = [TITLE_FORMAT]
        fmt.append(DATA_FORMAT % ("-" * 11, "-" * 12, "-" * 20, "-" * 7, "-" * 7))

        for f in self.case_runners:
            filters = f.ca.filters.filter_rate

            ds_str = f"{f.ca.dataset.data.name}-{f.ca.dataset.data.label}-{utils.numerize(f.ca.dataset.data.size)}"
            fmt.append(
                DATA_FORMAT
                % (
                    f.config.db_name,
                    f.ca.label.name,
                    ds_str,
                    filters,
                    self.task_label,
                ),
            )

        tmp_logger = logging.getLogger("no_color")
        for f in fmt:
            tmp_logger.info(f)
