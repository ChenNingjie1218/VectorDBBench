import logging
import time
from unittest.mock import Mock, patch

from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.models import (
    DB, IndexType, CaseType, TaskConfig, CaseConfig,
)

log = logging.getLogger(__name__)

class TestBenchRunner:
    def test_wait_shuts_down_executor(self):
        runner = BenchMarkRunner()
        runner.running_task = Mock()
        result_future = runner.result_future = Mock()
        executor = runner.executor = Mock()

        runner.wait()

        result_future.result.assert_called_once_with()
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        assert runner.running_task is None

    def test_shutdown_cancels_running_task_before_executor(self):
        runner = BenchMarkRunner()
        running_task = runner.running_task = Mock()
        executor = runner.executor = Mock()

        with patch.object(runner, "kill_proc_tree") as kill_proc_tree:
            runner.shutdown(cancel=True)

        running_task.case_runners.__iter__.assert_called_once_with()
        kill_proc_tree.assert_called_once_with(timeout=5)
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        assert runner.running_task is None

    def test_get_results(self):
        runner = BenchMarkRunner()

        result = runner.get_results()
        log.info(f"test result: {result}")

    def test_performance_case_whole(self):
        runner = BenchMarkRunner()

        task_config=TaskConfig(
            db=DB.Milvus,
            db_config=DB.Milvus.config(),
            db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
            case_config=CaseConfig(case_id=CaseType.PerformanceSZero),
        )

        runner.run([task_config])
        runner._sync_running_task()
        result = runner.get_results()
        log.info(f"test result: {result}")

    def test_performance_case_clean(self):
        runner = BenchMarkRunner()

        task_config=TaskConfig(
            db=DB.Milvus,
            db_config=DB.Milvus.config(),
            db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
            case_config=CaseConfig(case_id=CaseType.PerformanceSZero),
        )

        runner.run([task_config])
        time.sleep(3)
        runner.stop_running()

    def test_performance_case_no_error(self):
        task_config=TaskConfig(
            db=DB.ZillizCloud,
            db_config=DB.ZillizCloud.config(uri="xxx", user="abc", password="1234"),
            db_case_config=DB.ZillizCloud.case_config_cls()(),
            case_config=CaseConfig(case_id=CaseType.PerformanceSZero),
        )

        t = task_config.copy()
        d = t.json(exclude={'db_config': {'password', 'api_key'}})
        log.info(f"{d}")

        import ujson
        loads = ujson.loads(d)
        log.info(f"{loads}")
