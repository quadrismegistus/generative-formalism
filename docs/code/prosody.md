# Prosodic Analysis: `prosody.py`

The prosody module implements computational methods for analyzing the formal features of poetry, including rhythm, meter, stress patterns, and syllable structure.

## Overview

This module provides:
- **Metrical analysis** - Stress pattern detection and meter classification
- **Syllable counting** - Accurate syllabification for English poetry
- **Rhythm measurement** - Quantifying rhythmic regularity and variation
- **Phonetic processing** - Interface to prosodic analysis libraries

## Core Functions

### Basic Prosodic Analysis

#### `analyze_prosody(text)`
Comprehensive prosodic analysis of a poem or text passage.

```python
prosody_data = analyze_prosody(poem_text)
# Returns dict with stress patterns, syllable counts, meter info
```

#### `get_syllable_count(line)`
Accurate syllable counting for poetic lines.

```python
syllables = get_syllable_count("Shall I compare thee to a summer's day?")
# Returns: 10
```

#### `get_stress_pattern(line)`
Identifies stressed and unstressed syllables.

```python
pattern = get_stress_pattern("To be or not to be, that is the question")
# Returns: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] (0=unstressed, 1=stressed)
```

### Metrical Classification

#### `identify_meter(stress_pattern)`
Classifies stress patterns into traditional metrical feet.

```python
meter = identify_meter([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# Returns: "iambic pentameter"
```

#### `analyze_meter_regularity(poem_lines)`
Measures how consistently a poem follows metrical patterns.

```python
regularity = analyze_meter_regularity(sonnet_lines)
# Returns score from 0 (irregular) to 1 (perfectly regular)
```

#### `detect_metrical_substitutions(stress_patterns)`
Identifies variations from the dominant metrical pattern.

```python
substitutions = detect_metrical_substitutions(poem_stress_patterns)
# Returns locations and types of metrical variations
```

### Advanced Analysis

#### `measure_rhythmic_complexity(lines)`
Quantifies the rhythmic complexity of poetic text.

```python
complexity = measure_rhythmic_complexity(poem_lines)
# Returns numerical measure of rhythmic variation
```

#### `analyze_caesura_patterns(lines)`
Detects and analyzes caesuras (pauses) within poetic lines.

```python
caesuras = analyze_caesura_patterns(poem_lines)
# Returns positions and strengths of internal pauses
```

#### `get_prosodic_features(poem_text)`
Extracts comprehensive prosodic feature vector.

```python
features = get_prosodic_features(poem_text)
# Returns dict with dozens of prosodic measurements
```

## Integration with `prosodic` Library

### Configuration
```python
prosodic.USE_CACHE = False      # Disable caching for research
prosodic.LOG_LEVEL = 'CRITICAL' # Minimize output
```

### Core Prosodic Objects
```python
# Create prosodic text object
text_obj = prosodic.Text(poem_text)

# Get syllable-level analysis
syllables = text_obj.syllables()

# Extract stress patterns
stresses = [syll.stress for syll in syllables]
```

## Metrical Pattern Recognition

### Standard Meters
The module recognizes common English meters:
- **Iambic** - unstressed/stressed patterns (˘ /)
- **Trochaic** - stressed/unstressed patterns (/ ˘)
- **Anapestic** - unstressed/unstressed/stressed (˘ ˘ /)
- **Dactylic** - stressed/unstressed/unstressed (/ ˘ ˘)
- **Spondaic** - stressed/stressed (/ /)
- **Pyrrhic** - unstressed/unstressed (˘ ˘)

### Line Length Classifications
- **Monometer** - 1 foot
- **Dimeter** - 2 feet
- **Trimeter** - 3 feet
- **Tetrameter** - 4 feet
- **Pentameter** - 5 feet
- **Hexameter** - 6 feet
- **Heptameter** - 7 feet
- **Octameter** - 8 feet

### Pattern Matching
```python
def classify_line_meter(stress_pattern):
    """Classify a stress pattern into meter and length."""
    # Identify repeating foot patterns
    # Determine line length
    # Return meter classification
```

## Statistical Analysis

### Corpus-Level Metrics
#### `compare_prosodic_distributions(corpus1, corpus2)`
Statistical comparison of prosodic features between corpora.

```python
comparison = compare_prosodic_distributions(
    historical_poems,
    ai_poems,
    features=['stress_regularity', 'syllable_variance', 'meter_consistency']
)
```

#### `measure_prosodic_diversity(corpus)`
Quantifies the range of prosodic patterns in a corpus.

```python
diversity = measure_prosodic_diversity(poem_corpus)
# Returns entropy-based measure of metrical variety
```

### Temporal Analysis
#### `track_prosodic_evolution(corpus_by_period)`
Analyzes how prosodic patterns change over time.

```python
evolution = track_prosodic_evolution(poems_by_century)
# Returns trends in metrical preferences
```

## Performance Optimization

### Caching Strategies
- **Result caching** - Store expensive prosodic analyses
- **Pattern caching** - Cache common stress pattern classifications
- **Batch processing** - Efficient handling of large poem sets

### Parallel Processing
```python
def analyze_corpus_prosody_parallel(poems, n_workers=4):
    """Parallel prosodic analysis for large corpora."""
    # Distribute poems across worker processes
    # Collect and merge results
```

## Validation and Accuracy

### Human Annotation Comparison
```python
def validate_against_human_scansion(poems, human_annotations):
    """Compare automated analysis with expert human scansion."""
    # Calculate agreement rates
    # Identify systematic errors
    # Return accuracy metrics
```

### Cross-Validation
```python
def cross_validate_prosodic_analysis(test_corpus):
    """Test prosodic analysis accuracy across different poem types."""
    # Test on various periods, genres, authors
    # Report performance by category
```

## Error Handling and Edge Cases

### Text Preprocessing
- **Normalization** - Handle archaic spellings and contractions
- **Punctuation** - Manage how punctuation affects scansion
- **Line breaks** - Proper handling of enjambment

### Ambiguous Cases
- **Multiple valid scansions** - Handle metrical ambiguity
- **Dialect variations** - Account for pronunciation differences
- **Historical pronunciation** - Adapt for older texts

This module provides the computational foundation for understanding the formal structure of poetry, enabling systematic comparison between human and AI-generated verse.
