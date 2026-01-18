"""
Notebook 7: Production Monitoring and Optimization

This hands-on notebook covers production deployment considerations for RAG systems:
- Implementing monitoring pipelines
- Designing A/B tests
- Performance profiling and optimization
- Cost-efficiency analysis

Learning Objectives:
- Set up monitoring for production RAG systems
- Design and analyze A/B tests
- Profile performance bottlenecks
- Optimize cost-efficiency trade-offs

Duration: 45-60 minutes
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import time


# ============================================================================
# EXERCISE 1: IMPLEMENTING MONITORING PIPELINE
# ============================================================================

EXERCISE_1_INSTRUCTIONS = """
# Exercise 1: Implementing a Monitoring Pipeline

## Objective
Build a monitoring pipeline that tracks key metrics for a production RAG system.

## Background
Production RAG systems need continuous monitoring to detect:
- Performance degradation (latency increases)
- Quality issues (faithfulness drops)
- Cost overruns (budget exceeded)
- System failures (error rate spikes)

## Your Task
Implement a monitoring system that:
1. Collects metrics from production queries
2. Calculates aggregated statistics (p50, p95, p99 latency)
3. Detects anomalies and triggers alerts
4. Generates monitoring dashboards

## Starter Code
"""


@dataclass
class QueryMetric:
    """Metrics for a single query."""
    query_id: str
    timestamp: datetime
    latency_ms: float
    cost_usd: float
    error: bool
    faithfulness: Optional[float] = None
    relevancy: Optional[float] = None


class MonitoringPipeline:
    """
    Monitoring pipeline for production RAG system.
    
    This class collects and aggregates metrics from production queries.
    """
    
    def __init__(self):
        self.metrics: List[QueryMetric] = []
        self.alert_thresholds = {
            "latency_p95_ms": 2000,
            "error_rate": 0.05,
            "faithfulness": 0.70,
            "cost_per_query_usd": 0.01
        }
    
    def record_query(self, metric: QueryMetric):
        """
        Record metrics for a single query.
        
        Args:
            metric: QueryMetric with query performance data
        """
        self.metrics.append(metric)
    
    def get_latency_percentile(self, percentile: int, window_minutes: int = 60) -> float:
        """
        Calculate latency percentile over time window.
        
        Args:
            percentile: Percentile to calculate (50, 95, 99)
            window_minutes: Time window in minutes
            
        Returns:
            Latency at specified percentile in milliseconds
            
        TODO: Implement this method
        Hints:
        - Filter metrics to time window
        - Extract latency values
        - Calculate percentile using sorted values
        - Handle empty metrics list
        """
        # INTENTIONAL BUG: This implementation doesn't filter by time window
        # Students should fix this to only consider recent metrics
        latencies = [m.latency_ms for m in self.metrics]
        if not latencies:
            return 0.0
        
        latencies.sort()
        index = int(len(latencies) * percentile / 100)
        return latencies[min(index, len(latencies) - 1)]
    
    def get_error_rate(self, window_minutes: int = 60) -> float:
        """
        Calculate error rate over time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Error rate as fraction (0.0 to 1.0)
            
        TODO: Implement this method
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return 0.0
        
        errors = sum(1 for m in recent_metrics if m.error)
        return errors / len(recent_metrics)
    
    def get_average_cost(self, window_minutes: int = 60) -> float:
        """
        Calculate average cost per query over time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Average cost per query in USD
            
        TODO: Implement this method
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return 0.0
        
        total_cost = sum(m.cost_usd for m in recent_metrics)
        return total_cost / len(recent_metrics)
    
    def get_average_quality(self, metric_name: str, window_minutes: int = 60) -> Optional[float]:
        """
        Calculate average quality metric over time window.
        
        Args:
            metric_name: Name of quality metric ("faithfulness" or "relevancy")
            window_minutes: Time window in minutes
            
        Returns:
            Average quality score or None if no data
            
        TODO: Implement this method
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        values = []
        for m in recent_metrics:
            if metric_name == "faithfulness" and m.faithfulness is not None:
                values.append(m.faithfulness)
            elif metric_name == "relevancy" and m.relevancy is not None:
                values.append(m.relevancy)
        
        if not values:
            return None
        
        return sum(values) / len(values)
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check if any metrics exceed alert thresholds.
        
        Returns:
            List of alerts with metric name, value, and threshold
            
        TODO: Implement this method
        Hints:
        - Check latency p95
        - Check error rate
        - Check quality metrics
        - Check cost per query
        - Return list of alerts for metrics exceeding thresholds
        """
        alerts = []
        
        # Check latency
        p95_latency = self.get_latency_percentile(95, window_minutes=5)
        if p95_latency > self.alert_thresholds["latency_p95_ms"]:
            alerts.append({
                "metric": "latency_p95",
                "value": p95_latency,
                "threshold": self.alert_thresholds["latency_p95_ms"],
                "severity": "high"
            })
        
        # Check error rate
        error_rate = self.get_error_rate(window_minutes=5)
        if error_rate > self.alert_thresholds["error_rate"]:
            alerts.append({
                "metric": "error_rate",
                "value": error_rate,
                "threshold": self.alert_thresholds["error_rate"],
                "severity": "critical"
            })
        
        # TODO: Add checks for faithfulness and cost
        
        return alerts
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate data for monitoring dashboard.
        
        Returns:
            Dictionary with dashboard metrics
            
        TODO: Implement this method
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "latency_p50": self.get_latency_percentile(50, window_minutes=60),
            "latency_p95": self.get_latency_percentile(95, window_minutes=60),
            "latency_p99": self.get_latency_percentile(99, window_minutes=60),
            "error_rate": self.get_error_rate(window_minutes=60),
            "avg_cost_per_query": self.get_average_cost(window_minutes=60),
            "avg_faithfulness": self.get_average_quality("faithfulness", window_minutes=60),
            "avg_relevancy": self.get_average_quality("relevancy", window_minutes=60),
            "total_queries_1h": len([m for m in self.metrics if m.timestamp >= datetime.now() - timedelta(hours=1)]),
            "alerts": self.check_alerts()
        }


# Simulate production queries
def simulate_production_traffic(pipeline: MonitoringPipeline, num_queries: int = 1000):
    """
    Simulate production traffic for testing monitoring pipeline.
    
    Args:
        pipeline: MonitoringPipeline to record metrics
        num_queries: Number of queries to simulate
    """
    print(f"Simulating {num_queries} production queries...")
    
    for i in range(num_queries):
        # Simulate query with realistic metrics
        metric = QueryMetric(
            query_id=f"query_{i}",
            timestamp=datetime.now() - timedelta(minutes=random.randint(0, 120)),
            latency_ms=random.gauss(500, 200),  # Mean 500ms, std 200ms
            cost_usd=random.gauss(0.003, 0.001),  # Mean $0.003, std $0.001
            error=random.random() < 0.02,  # 2% error rate
            faithfulness=random.gauss(0.85, 0.10) if random.random() < 0.1 else None,  # 10% sampled
            relevancy=random.gauss(0.88, 0.08) if random.random() < 0.1 else None  # 10% sampled
        )
        
        pipeline.record_query(metric)
    
    print("Simulation complete!")


# Test the monitoring pipeline
def test_monitoring_pipeline():
    """Test the monitoring pipeline implementation."""
    print("=" * 80)
    print("EXERCISE 1: Testing Monitoring Pipeline")
    print("=" * 80)
    
    pipeline = MonitoringPipeline()
    simulate_production_traffic(pipeline, num_queries=1000)
    
    # Generate dashboard
    dashboard = pipeline.generate_dashboard_data()
    
    print("\n📊 Monitoring Dashboard")
    print("-" * 80)
    print(f"Timestamp: {dashboard['timestamp']}")
    print(f"\nLatency Metrics:")
    print(f"  P50: {dashboard['latency_p50']:.1f}ms")
    print(f"  P95: {dashboard['latency_p95']:.1f}ms")
    print(f"  P99: {dashboard['latency_p99']:.1f}ms")
    print(f"\nError Rate: {dashboard['error_rate']:.2%}")
    print(f"Average Cost: ${dashboard['avg_cost_per_query']:.4f}")
    print(f"\nQuality Metrics (sampled):")
    if dashboard['avg_faithfulness']:
        print(f"  Faithfulness: {dashboard['avg_faithfulness']:.2f}")
    if dashboard['avg_relevancy']:
        print(f"  Relevancy: {dashboard['avg_relevancy']:.2f}")
    print(f"\nTotal Queries (1h): {dashboard['total_queries_1h']}")
    
    if dashboard['alerts']:
        print(f"\n🚨 ALERTS ({len(dashboard['alerts'])})")
        for alert in dashboard['alerts']:
            print(f"  [{alert['severity'].upper()}] {alert['metric']}: {alert['value']:.2f} (threshold: {alert['threshold']})")
    else:
        print("\n✅ No alerts")
    
    print("\n" + "=" * 80)


# ============================================================================
# EXERCISE 2: DESIGNING A/B TESTS
# ============================================================================

EXERCISE_2_INSTRUCTIONS = """
# Exercise 2: Designing and Analyzing A/B Tests

## Objective
Design an A/B test to compare two RAG system variants and analyze results.

## Background
A/B testing helps determine which system configuration performs better:
- Embedding model selection
- Chunk size optimization
- LLM model selection
- Prompt template variations

## Your Task
1. Design an A/B test comparing two embedding models
2. Simulate traffic split between variants
3. Collect metrics for both variants
4. Perform statistical significance testing
5. Make a recommendation

## Starter Code
"""


@dataclass
class ABTestMetric:
    """Metrics for A/B test."""
    variant: str
    query_id: str
    timestamp: datetime
    context_precision: float
    latency_ms: float
    cost_usd: float


class ABTestFramework:
    """
    Framework for running A/B tests on RAG systems.
    """
    
    def __init__(self, experiment_id: str, variant_a_name: str, variant_b_name: str):
        self.experiment_id = experiment_id
        self.variant_a_name = variant_a_name
        self.variant_b_name = variant_b_name
        self.metrics_a: List[ABTestMetric] = []
        self.metrics_b: List[ABTestMetric] = []
    
    def assign_variant(self, user_id: str) -> str:
        """
        Assign user to variant using consistent hashing.
        
        Args:
            user_id: User identifier
            
        Returns:
            Variant name ("A" or "B")
            
        TODO: Implement consistent hashing
        Hints:
        - Use hash function on user_id
        - Ensure same user always gets same variant
        - Split traffic 50/50
        """
        # Simple implementation: hash user_id and mod 2
        hash_value = hash(f"{user_id}:{self.experiment_id}")
        return "A" if hash_value % 2 == 0 else "B"
    
    def record_metric(self, variant: str, metric: ABTestMetric):
        """Record metric for a variant."""
        if variant == "A":
            self.metrics_a.append(metric)
        else:
            self.metrics_b.append(metric)
    
    def get_variant_metrics(self, variant: str) -> Dict[str, float]:
        """
        Calculate aggregated metrics for a variant.
        
        Args:
            variant: Variant name ("A" or "B")
            
        Returns:
            Dictionary with aggregated metrics
            
        TODO: Implement this method
        """
        metrics = self.metrics_a if variant == "A" else self.metrics_b
        
        if not metrics:
            return {}
        
        return {
            "avg_context_precision": sum(m.context_precision for m in metrics) / len(metrics),
            "avg_latency_ms": sum(m.latency_ms for m in metrics) / len(metrics),
            "avg_cost_usd": sum(m.cost_usd for m in metrics) / len(metrics),
            "sample_size": len(metrics)
        }
    
    def calculate_statistical_significance(self, metric_name: str = "avg_context_precision") -> Dict[str, Any]:
        """
        Calculate statistical significance of difference between variants.
        
        Args:
            metric_name: Metric to test
            
        Returns:
            Dictionary with p-value and significance
            
        TODO: Implement t-test for significance
        Hints:
        - Extract metric values for both variants
        - Calculate means and standard deviations
        - Perform t-test (simplified version)
        - p-value < 0.05 indicates significance
        """
        metrics_a = self.get_variant_metrics("A")
        metrics_b = self.get_variant_metrics("B")
        
        if not metrics_a or not metrics_b:
            return {"error": "Insufficient data"}
        
        # Simplified significance test (in practice, use scipy.stats.ttest_ind)
        mean_a = metrics_a[metric_name]
        mean_b = metrics_b[metric_name]
        
        # INTENTIONAL BUG: This is a simplified test that doesn't account for variance
        # Students should implement proper t-test
        difference = abs(mean_b - mean_a)
        p_value = 0.03 if difference > 0.05 else 0.15  # Simplified
        
        return {
            "metric": metric_name,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "difference": mean_b - mean_a,
            "improvement_pct": (mean_b - mean_a) / mean_a * 100 if mean_a > 0 else 0,
            "p_value": p_value,
            "is_significant": p_value < 0.05
        }
    
    def generate_recommendation(self) -> str:
        """
        Generate recommendation based on A/B test results.
        
        Returns:
            Recommendation string
            
        TODO: Implement this method
        """
        sig_test = self.calculate_statistical_significance("avg_context_precision")
        
        if "error" in sig_test:
            return "Insufficient data to make recommendation"
        
        if not sig_test["is_significant"]:
            return f"No significant difference detected (p={sig_test['p_value']:.3f}). Continue with variant A."
        
        if sig_test["improvement_pct"] > 0:
            return f"Variant B shows {sig_test['improvement_pct']:.1f}% improvement (p={sig_test['p_value']:.3f}). Recommend promoting variant B to production."
        else:
            return f"Variant A performs better. Continue with variant A."


# Simulate A/B test
def simulate_ab_test(framework: ABTestFramework, num_queries: int = 1000):
    """
    Simulate A/B test with production traffic.
    
    Args:
        framework: ABTestFramework to record metrics
        num_queries: Number of queries to simulate
    """
    print(f"Simulating A/B test with {num_queries} queries...")
    
    for i in range(num_queries):
        user_id = f"user_{i % 200}"  # 200 unique users
        variant = framework.assign_variant(user_id)
        
        # Variant B has slightly better context precision but higher latency
        if variant == "A":
            metric = ABTestMetric(
                variant="A",
                query_id=f"query_{i}",
                timestamp=datetime.now(),
                context_precision=random.gauss(0.82, 0.08),
                latency_ms=random.gauss(450, 50),
                cost_usd=random.gauss(0.003, 0.0005)
            )
        else:
            metric = ABTestMetric(
                variant="B",
                query_id=f"query_{i}",
                timestamp=datetime.now(),
                context_precision=random.gauss(0.88, 0.08),  # Better precision
                latency_ms=random.gauss(420, 50),  # Slightly faster
                cost_usd=random.gauss(0.0028, 0.0005)  # Slightly cheaper
            )
        
        framework.record_metric(variant, metric)
    
    print("Simulation complete!")


# Test A/B testing framework
def test_ab_testing():
    """Test the A/B testing framework."""
    print("=" * 80)
    print("EXERCISE 2: Testing A/B Testing Framework")
    print("=" * 80)
    
    framework = ABTestFramework(
        experiment_id="embedding_model_test_2024_01",
        variant_a_name="text-embedding-ada-002",
        variant_b_name="nv-embed-v2"
    )
    
    simulate_ab_test(framework, num_queries=1000)
    
    # Analyze results
    metrics_a = framework.get_variant_metrics("A")
    metrics_b = framework.get_variant_metrics("B")
    
    print(f"\n📊 A/B Test Results: {framework.experiment_id}")
    print("-" * 80)
    print(f"\nVariant A ({framework.variant_a_name}):")
    print(f"  Sample Size: {metrics_a['sample_size']}")
    print(f"  Avg Context Precision: {metrics_a['avg_context_precision']:.3f}")
    print(f"  Avg Latency: {metrics_a['avg_latency_ms']:.1f}ms")
    print(f"  Avg Cost: ${metrics_a['avg_cost_usd']:.4f}")
    
    print(f"\nVariant B ({framework.variant_b_name}):")
    print(f"  Sample Size: {metrics_b['sample_size']}")
    print(f"  Avg Context Precision: {metrics_b['avg_context_precision']:.3f}")
    print(f"  Avg Latency: {metrics_b['avg_latency_ms']:.1f}ms")
    print(f"  Avg Cost: ${metrics_b['avg_cost_usd']:.4f}")
    
    # Statistical significance
    sig_test = framework.calculate_statistical_significance("avg_context_precision")
    print(f"\n📈 Statistical Analysis:")
    print(f"  Difference: {sig_test['difference']:.3f} ({sig_test['improvement_pct']:.1f}%)")
    print(f"  P-value: {sig_test['p_value']:.3f}")
    print(f"  Significant: {'Yes' if sig_test['is_significant'] else 'No'}")
    
    # Recommendation
    recommendation = framework.generate_recommendation()
    print(f"\n💡 Recommendation:")
    print(f"  {recommendation}")
    
    print("\n" + "=" * 80)


# ============================================================================
# EXERCISE 3: PERFORMANCE PROFILING
# ============================================================================

EXERCISE_3_INSTRUCTIONS = """
# Exercise 3: Performance Profiling and Optimization

## Objective
Profile a RAG pipeline to identify performance bottlenecks and optimize.

## Background
RAG pipelines have multiple stages that contribute to latency:
- Embedding generation: 50-200ms
- Vector search: 10-100ms
- Context augmentation: 5-20ms
- LLM generation: 500-3000ms

## Your Task
1. Profile each stage of the RAG pipeline
2. Identify bottlenecks
3. Propose optimizations
4. Estimate cost savings

## Starter Code
"""


@dataclass
class StageProfile:
    """Performance profile for a pipeline stage."""
    stage_name: str
    avg_latency_ms: float
    p95_latency_ms: float
    avg_cost_usd: float
    percentage_of_total: float


class PerformanceProfiler:
    """
    Profiler for RAG pipeline performance.
    """
    
    def __init__(self):
        self.stage_metrics: Dict[str, List[float]] = {
            "embedding": [],
            "retrieval": [],
            "augmentation": [],
            "generation": []
        }
        self.stage_costs: Dict[str, List[float]] = {
            "embedding": [],
            "retrieval": [],
            "augmentation": [],
            "generation": []
        }
    
    def profile_query(self):
        """
        Profile a single query through the pipeline.
        
        TODO: Implement this method
        Hints:
        - Measure time for each stage
        - Record costs for each stage
        - Store in stage_metrics and stage_costs
        """
        # Simulate embedding stage
        embedding_time = random.gauss(100, 30)
        self.stage_metrics["embedding"].append(embedding_time)
        self.stage_costs["embedding"].append(0.0001)
        
        # Simulate retrieval stage
        retrieval_time = random.gauss(50, 20)
        self.stage_metrics["retrieval"].append(retrieval_time)
        self.stage_costs["retrieval"].append(0.0002)
        
        # Simulate augmentation stage
        augmentation_time = random.gauss(10, 5)
        self.stage_metrics["augmentation"].append(augmentation_time)
        self.stage_costs["augmentation"].append(0.0)
        
        # Simulate generation stage (most expensive)
        generation_time = random.gauss(1200, 300)
        self.stage_metrics["generation"].append(generation_time)
        self.stage_costs["generation"].append(0.0025)
    
    def get_stage_profile(self, stage_name: str) -> StageProfile:
        """
        Get performance profile for a stage.
        
        Args:
            stage_name: Name of stage
            
        Returns:
            StageProfile with metrics
            
        TODO: Implement this method
        """
        latencies = self.stage_metrics[stage_name]
        costs = self.stage_costs[stage_name]
        
        if not latencies:
            return StageProfile(stage_name, 0, 0, 0, 0)
        
        avg_latency = sum(latencies) / len(latencies)
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[p95_index]
        avg_cost = sum(costs) / len(costs)
        
        total_latency = sum(sum(self.stage_metrics[s]) for s in self.stage_metrics) / len(latencies)
        percentage = (avg_latency / total_latency * 100) if total_latency > 0 else 0
        
        return StageProfile(
            stage_name=stage_name,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            avg_cost_usd=avg_cost,
            percentage_of_total=percentage
        )
    
    def identify_bottlenecks(self) -> List[str]:
        """
        Identify performance bottlenecks.
        
        Returns:
            List of bottleneck stages
            
        TODO: Implement this method
        """
        bottlenecks = []
        
        for stage_name in self.stage_metrics.keys():
            profile = self.get_stage_profile(stage_name)
            
            # Consider a stage a bottleneck if it's >40% of total time
            if profile.percentage_of_total > 40:
                bottlenecks.append(stage_name)
        
        return bottlenecks
    
    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """
        Suggest optimizations based on profiling.
        
        Returns:
            List of optimization suggestions
            
        TODO: Implement this method
        """
        optimizations = []
        bottlenecks = self.identify_bottlenecks()
        
        for stage in bottlenecks:
            if stage == "embedding":
                optimizations.append({
                    "stage": stage,
                    "issue": "Embedding generation is slow",
                    "suggestion": "Use smaller embedding model or implement caching",
                    "estimated_savings": "20-30% latency reduction"
                })
            elif stage == "generation":
                optimizations.append({
                    "stage": stage,
                    "issue": "LLM generation is slow and expensive",
                    "suggestion": "Use smaller model for simple queries, implement streaming",
                    "estimated_savings": "30-40% cost reduction"
                })
        
        return optimizations


# Test performance profiling
def test_performance_profiling():
    """Test the performance profiling."""
    print("=" * 80)
    print("EXERCISE 3: Testing Performance Profiling")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    
    # Profile 100 queries
    print("\nProfiling 100 queries...")
    for _ in range(100):
        profiler.profile_query()
    
    # Generate profiles
    print("\n📊 Performance Profile")
    print("-" * 80)
    
    total_latency = 0
    total_cost = 0
    
    for stage_name in ["embedding", "retrieval", "augmentation", "generation"]:
        profile = profiler.get_stage_profile(stage_name)
        total_latency += profile.avg_latency_ms
        total_cost += profile.avg_cost_usd
        
        print(f"\n{stage_name.upper()}:")
        print(f"  Avg Latency: {profile.avg_latency_ms:.1f}ms ({profile.percentage_of_total:.1f}%)")
        print(f"  P95 Latency: {profile.p95_latency_ms:.1f}ms")
        print(f"  Avg Cost: ${profile.avg_cost_usd:.4f}")
    
    print(f"\nTOTAL:")
    print(f"  Avg Latency: {total_latency:.1f}ms")
    print(f"  Avg Cost: ${total_cost:.4f}")
    
    # Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks()
    if bottlenecks:
        print(f"\n🔍 Bottlenecks Identified: {', '.join(bottlenecks)}")
    
    # Suggest optimizations
    optimizations = profiler.suggest_optimizations()
    if optimizations:
        print(f"\n💡 Optimization Suggestions:")
        for opt in optimizations:
            print(f"\n  {opt['stage'].upper()}:")
            print(f"    Issue: {opt['issue']}")
            print(f"    Suggestion: {opt['suggestion']}")
            print(f"    Estimated Savings: {opt['estimated_savings']}")
    
    print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("NOTEBOOK 7: PRODUCTION MONITORING AND OPTIMIZATION")
    print("=" * 80)
    
    # Run all exercises
    test_monitoring_pipeline()
    print("\n")
    test_ab_testing()
    print("\n")
    test_performance_profiling()
    
    print("\n" + "=" * 80)
    print("EXERCISES COMPLETE!")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Fix the intentional bugs in the monitoring pipeline")
    print("2. Implement proper statistical testing in A/B framework")
    print("3. Add more optimization suggestions to the profiler")
    print("4. Experiment with different alert thresholds")
    print("5. Design your own A/B test for a different metric")
