"""Apache Airflow DAG for DriftGuard autonomous drift monitoring and retraining."""

from __future__ import annotations

from datetime import datetime, timedelta

# Import DAG and operators conditionally to support both standalone Python and Airflow environments
try:
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def check_drift_callable(**kwargs) -> str:
    """Checks if dataset drift exceeds critical threshold."""
    from monitoring.drift_detector import evaluate_drift

    drift_res = evaluate_drift(save_reports=True)
    if drift_res.get("drift_detected", False):
        return "retrain_challenger_task"
    return "skip_retrain_task"


def retrain_challenger_callable(**kwargs) -> dict:
    """Trains a new Challenger model."""
    from orchestration.retrain_pipeline import load_retraining_data, train_challenger

    train_df, val_df = load_retraining_data()
    model, metrics, version, run_id = train_challenger(train_df, val_df)
    return {
        "metrics": metrics,
        "version": version,
        "run_id": run_id,
    }


def evaluate_and_promote_callable(**kwargs) -> dict:
    """Evaluates the Challenger vs Champion on the holdout benchmark."""
    import pandas as pd
    from data.split_data import PROCESSED_DIR
    from orchestration.retrain_pipeline import evaluate_and_promote_challenger
    from training.train import ARTIFACTS_DIR
    import joblib

    ti = kwargs.get("ti")
    retrain_info = ti.xcom_pull(task_ids="retrain_challenger_task") if ti else {}

    val_df = pd.read_parquet(PROCESSED_DIR / "val_holdout.parquet")
    challenger_model = joblib.load(ARTIFACTS_DIR / "champion_model.joblib")["model"]
    challenger_metrics = retrain_info.get("metrics", {})
    challenger_version = retrain_info.get("version", "unknown")

    decision = evaluate_and_promote_challenger(
        challenger_model=challenger_model,
        challenger_metrics=challenger_metrics,
        challenger_version=challenger_version,
        val_df=val_df,
    )
    return decision


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "driftguard",
        "depends_on_past": False,
        "start_date": datetime(2026, 1, 1),
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        dag_id="driftguard_autonomous_retrain",
        default_args=default_args,
        description="Autonomous drift-aware model retraining and promotion pipeline",
        schedule_interval=timedelta(hours=6),
        catchup=False,
        tags=["mlops", "fraud-detection", "driftguard"],
    )

    with dag:
        branch_task = BranchPythonOperator(
            task_id="check_drift_and_branch",
            python_callable=check_drift_callable,
        )

        retrain_task = PythonOperator(
            task_id="retrain_challenger_task",
            python_callable=retrain_challenger_callable,
        )

        eval_promote_task = PythonOperator(
            task_id="evaluate_and_promote_task",
            python_callable=evaluate_and_promote_callable,
        )

        skip_task = EmptyOperator(
            task_id="skip_retrain_task",
        )

        branch_task >> [retrain_task, skip_task]
        retrain_task >> eval_promote_task
