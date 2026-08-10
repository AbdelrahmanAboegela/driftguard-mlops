"""Automated concurrent performance benchmark and latency profiling tool for DriftGuard."""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import time
import uuid
import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_payload() -> dict:
    """Generates a realistic test payload."""
    payload = {
        "Time": random.uniform(0, 172800),
        "Amount": round(random.lognormvariate(3.0, 1.2), 2),
        "request_id": str(uuid.uuid4()),
    }
    for i in range(1, 29):
        payload[f"V{i}"] = random.gauss(0.0, 1.0)
    return payload


async def send_single_request(client: httpx.AsyncClient, target_url: str) -> tuple[bool, float]:
    """Sends a single POST request and records round-trip latency in ms."""
    payload = generate_sample_payload()
    t0 = time.perf_counter()
    try:
        resp = await client.post(target_url, json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return (resp.status_code == 200), latency_ms
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return False, latency_ms


async def run_benchmark(
    target_url: str = "http://localhost:8000/predict",
    total_requests: int = 500,
    concurrency: int = 25,
) -> dict:
    """Executes concurrent load testing and computes latency percentiles."""
    logger.info("Starting load test against %s (Total: %d, Concurrency: %d)...", target_url, total_requests, concurrency)

    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(10.0, connect=5.0)

    latencies = []
    success_count = 0
    failure_count = 0

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:

        async def worker():
            nonlocal success_count, failure_count
            async with semaphore:
                success, lat = await send_single_request(client, target_url)
                latencies.append(lat)
                if success:
                    success_count += 1
                else:
                    failure_count += 1

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(total_requests)]
        await asyncio.gather(*tasks)
        total_time_sec = time.perf_counter() - t_start

    lat_arr = np.array(latencies)
    throughput = total_requests / max(1e-6, total_time_sec)

    results = {
        "total_requests": total_requests,
        "success_count": success_count,
        "failure_count": failure_count,
        "total_time_sec": round(total_time_sec, 3),
        "throughput_rps": round(throughput, 2),
        "latency_avg_ms": round(float(np.mean(lat_arr)), 2),
        "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
        "latency_p90_ms": round(float(np.percentile(lat_arr, 90)), 2),
        "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(lat_arr, 99)), 2),
        "latency_min_ms": round(float(np.min(lat_arr)), 2),
        "latency_max_ms": round(float(np.max(lat_arr)), 2),
    }

    print("\n" + "=" * 60)
    print("           DRIFTGUARD LOAD TEST BENCHMARK REPORT")
    print("=" * 60)
    print(f"  Target Endpoint:     {target_url}")
    print(f"  Total Requests:      {results['total_requests']}")
    print(f"  Concurrency:         {concurrency}")
    print(f"  Success / Failed:    {results['success_count']} / {results['failure_count']}")
    print(f"  Duration:            {results['total_time_sec']}s")
    print(f"  * Throughput:        {results['throughput_rps']} req/s")
    print("-" * 60)
    print(f"  p50 (Median) Latency:{results['latency_p50_ms']} ms")
    print(f"  p90 Latency:         {results['latency_p90_ms']} ms")
    print(f"  p95 Latency:         {results['latency_p95_ms']} ms")
    print(f"  p99 Latency:         {results['latency_p99_ms']} ms")
    print(f"  Min / Max Latency:   {results['latency_min_ms']} ms / {results['latency_max_ms']} ms")
    print("=" * 60 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DriftGuard Load Testing & Profiling")
    parser.add_argument("--url", type=str, default="http://localhost:8000/predict", help="Target endpoint")
    parser.add_argument("--requests", type=int, default=300, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent workers")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.requests, args.concurrency))


if __name__ == "__main__":
    main()
