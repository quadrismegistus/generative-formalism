# Statistical Analysis: `stats.py`

The stats module provides comprehensive statistical methods for comparing and analyzing poetic corpora, supporting both descriptive statistics and hypothesis testing.

## Overview

This module implements:
- **Descriptive statistics** - Summarizing distributions of poetic features
- **Comparative analysis** - Statistical tests for differences between corpora
- **Effect size measurement** - Quantifying practical significance of findings
- **Visualization support** - Statistical plots and charts

## Core Functions

### Descriptive Statistics

#### `describe_corpus_statistics(df)`
Comprehensive statistical summary of a poetry corpus.

```python
stats = describe_corpus_statistics(poem_corpus)
# Returns means, medians, distributions for all numerical features
```

#### `calculate_feature_distributions(corpus, features)`
Distribution analysis for specific poetic features.

```python
distributions = calculate_feature_distributions(
    corpus,
    features=['num_lines', 'rhyme_density', 'stress_regularity']
)
```

#### `measure_corpus_diversity(corpus, feature)`
Quantifies diversity/entropy in categorical features.

```python
diversity = measure_corpus_diversity(corpus, 'rhyme_scheme')
# Returns entropy-based diversity measure
```

### Comparative Analysis

#### `compare_corpora_statistical(corpus1, corpus2, features)`
Statistical comparison between two poetry corpora.

```python
comparison = compare_corpora_statistical(
    historical_poems,
    ai_poems,
    features=['rhyme_accuracy', 'meter_regularity', 'line_length']
)
# Returns test statistics, p-values, effect sizes
```

#### `test_distribution_differences(data1, data2, test_type='auto')`
Tests for differences between two distributions.

```python
result = test_distribution_differences(
    historical_line_lengths,
    ai_line_lengths,
    test_type='mann_whitney'  # or 'ttest', 'ks_test', 'auto'
)
```

#### `bootstrap_confidence_intervals(data, statistic='mean', confidence=0.95)`
Bootstrap confidence intervals for robust statistics.

```python
ci = bootstrap_confidence_intervals(
    rhyme_accuracy_scores,
    statistic='median',
    confidence=0.95
)
```

### Effect Size Measurement

#### `calculate_effect_sizes(group1, group2)`
Comprehensive effect size calculations.

```python
effects = calculate_effect_sizes(human_features, ai_features)
# Returns Cohen's d, Glass's delta, Cliff's delta, etc.
```

#### `interpret_effect_size(effect_size, measure='cohens_d')`
Human-readable interpretation of effect sizes.

```python
interpretation = interpret_effect_size(0.8, 'cohens_d')
# Returns: "large effect"
```

### Temporal Analysis

#### `analyze_temporal_trends(corpus_by_period, feature)`
Analyzes how features change over time periods.

```python
trends = analyze_temporal_trends(
    poems_by_century,
    feature='rhyme_scheme_diversity'
)
# Returns trend analysis, significance tests
```

#### `detect_change_points(time_series_data)`
Identifies significant changes in temporal patterns.

```python
change_points = detect_change_points(rhyme_usage_by_decade)
# Returns locations of significant shifts
```

### Advanced Statistical Methods

#### `multivariate_comparison(corpus1, corpus2, features)`
Multivariate statistical comparison between corpora.

```python
result = multivariate_comparison(
    historical_corpus,
    ai_corpus,
    features=['rhyme_density', 'meter_regularity', 'line_length', 'stanza_length']
)
# Returns MANOVA results, discriminant analysis
```

#### `cluster_analysis(corpus, features, n_clusters='auto')`
Cluster poems based on formal features.

```python
clusters = cluster_analysis(
    combined_corpus,
    features=['prosodic_features'],
    n_clusters=5
)
```

#### `principal_component_analysis(corpus, features)`
Dimensionality reduction for poetic features.

```python
pca_result = principal_component_analysis(
    corpus,
    features=prosodic_feature_list
)
# Returns components, explained variance, loadings
```

## Specialized Poetry Statistics

### Rhyme Statistics
#### `analyze_rhyme_distributions(corpus)`
Comprehensive analysis of rhyme patterns and frequencies.

```python
rhyme_stats = analyze_rhyme_distributions(corpus)
# Returns scheme frequencies, rhyme quality distributions, etc.
```

#### `compare_rhyme_accuracy(corpus1, corpus2)`
Statistical comparison of rhyme accuracy between corpora.

```python
comparison = compare_rhyme_accuracy(human_poems, ai_poems)
# Returns significance tests for rhyme quality differences
```

### Metrical Statistics
#### `analyze_meter_patterns(corpus)`
Statistical analysis of metrical patterns and regularity.

```python
meter_stats = analyze_meter_patterns(corpus)
# Returns meter type frequencies, regularity scores, etc.
```

#### `test_metrical_consistency(corpus1, corpus2)`
Compare metrical consistency between corpora.

```python
consistency_test = test_metrical_consistency(
    traditional_sonnets,
    ai_sonnets
)
```

### Formal Feature Statistics
#### `analyze_structural_features(corpus)`
Analysis of structural elements (lines, stanzas, length).

```python
structure_stats = analyze_structural_features(corpus)
# Returns distributions of structural characteristics
```

## Visualization Integration

### Statistical Plotting
#### `plot_distribution_comparison(data1, data2, feature_name)`
Comparative distribution plots with statistical annotations.

```python
plot = plot_distribution_comparison(
    human_rhyme_scores,
    ai_rhyme_scores,
    feature_name='Rhyme Accuracy'
)
```

#### `plot_effect_sizes(comparison_results)`
Visualization of effect sizes across multiple features.

```python
plot = plot_effect_sizes(corpus_comparison_results)
# Forest plot style visualization
```

#### `plot_temporal_trends(trend_analysis)`
Time series plots with trend lines and confidence intervals.

```python
plot = plot_temporal_trends(rhyme_evolution_analysis)
```

### Report Generation
#### `generate_statistical_report(comparison_results)`
Comprehensive statistical report in markdown format.

```python
report = generate_statistical_report(
    human_vs_ai_comparison,
    include_plots=True,
    significance_level=0.05
)
```

## Robustness and Validation

### Multiple Testing Correction
```python
def adjust_p_values(p_values, method='holm'):
    """Adjust p-values for multiple comparisons."""
    # Implements various correction methods
    # Returns adjusted p-values
```

### Non-Parametric Methods
```python
def robust_statistical_tests(data1, data2):
    """Use robust, non-parametric tests when appropriate."""
    # Automatically choose appropriate tests
    # Handle non-normal distributions
```

### Cross-Validation
```python
def cross_validate_findings(corpus, analysis_function, n_folds=5):
    """Cross-validate statistical findings."""
    # Split corpus into folds
    # Repeat analysis on each fold
    # Test consistency of results
```

## Performance Optimization

### Efficient Computation
- **Vectorized operations** - Use numpy/pandas for fast computation
- **Sampling strategies** - Handle large corpora efficiently
- **Caching** - Store expensive statistical computations

### Memory Management
- **Lazy evaluation** - Compute statistics only when needed
- **Chunked processing** - Handle datasets larger than memory
- **Efficient data structures** - Optimize for statistical operations

## Quality Assurance

### Assumption Testing
```python
def test_statistical_assumptions(data, test_type):
    """Test assumptions for statistical tests."""
    # Normality tests
    # Homogeneity of variance
    # Independence checks
```

### Sensitivity Analysis
```python
def sensitivity_analysis(analysis_function, parameter_ranges):
    """Test sensitivity of results to parameter choices."""
    # Vary key parameters
    # Assess result stability
```

This module provides the statistical foundation for making rigorous, evidence-based claims about differences between human and AI-generated poetry.
