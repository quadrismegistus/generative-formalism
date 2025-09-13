# Rhyme Detection: `rhyme.py`

The rhyme module implements sophisticated algorithms for detecting, analyzing, and classifying rhyme patterns in poetry, supporting both traditional and computational approaches to rhyme analysis.

## Overview

This module provides:
- **Rhyme detection** - Identifying rhyming relationships between words and lines
- **Rhyme scheme analysis** - Classifying formal rhyme patterns (ABAB, ABBA, etc.)
- **Phonetic matching** - Sound-based rhyme detection using prosodic analysis
- **Quality assessment** - Measuring rhyme strength and consistency

## Core Functions

### Basic Rhyme Detection

#### `get_rhyme_for_txt(text)`
Comprehensive rhyme analysis for a complete poem or text.

```python
rhyme_data = get_rhyme_for_txt(sonnet_text)
# Returns detailed analysis including rhyme scheme, pairs, quality scores
```

#### `detect_end_rhymes(lines)`
Identifies rhyming relationships between line endings.

```python
rhyme_pairs = detect_end_rhymes(poem_lines)
# Returns list of (line_i, line_j, rhyme_strength) tuples
```

#### `get_rhyme_scheme(lines)`
Determines the formal rhyme scheme pattern.

```python
scheme = get_rhyme_scheme(sonnet_lines)
# Returns: "ABAB CDCD EFEF GG" for Shakespearean sonnet
```

### Phonetic Analysis

#### `extract_rhyme_phonemes(word)`
Extracts the phonetic ending relevant for rhyme detection.

```python
phonemes = extract_rhyme_phonemes("morning")
# Returns phonetic representation of rhyme-relevant sounds
```

#### `calculate_phonetic_similarity(word1, word2)`
Measures phonetic similarity between potential rhyme words.

```python
similarity = calculate_phonetic_similarity("day", "way")
# Returns similarity score from 0 (no rhyme) to 1 (perfect rhyme)
```

#### `classify_rhyme_type(word1, word2)`
Classifies the type of rhyme relationship.

```python
rhyme_type = classify_rhyme_type("love", "dove")
# Returns: "perfect", "slant", "eye", "none"
```

### Rhyme Scheme Analysis

#### `parse_rhyme_scheme(scheme_string)`
Parses and validates rhyme scheme notation.

```python
parsed = parse_rhyme_scheme("ABAB CDCD EFEF GG")
# Returns structured representation of the scheme
```

#### `identify_common_schemes(rhyme_schemes)`
Identifies standard poetic forms from rhyme schemes.

```python
forms = identify_common_schemes(["ABAB CDCD EFEF GG", "ABBA ABBA CDE CDE"])
# Returns: ["shakespearean_sonnet", "petrarchan_sonnet"]
```

#### `measure_scheme_regularity(detected_scheme, expected_scheme)`
Compares detected rhyme scheme to expected pattern.

```python
regularity = measure_scheme_regularity("ABAB CDCD EFEF GG", "ABAB CDCD EFEF GG")
# Returns adherence score from 0 to 1
```

### Advanced Rhyme Analysis

#### `analyze_rhyme_quality(rhyme_pairs)`
Assesses the quality and strength of rhyme relationships.

```python
quality = analyze_rhyme_quality(detected_rhymes)
# Returns metrics for perfect vs. slant rhymes, consistency, etc.
```

#### `detect_internal_rhymes(lines)`
Identifies rhymes within lines (not just end rhymes).

```python
internal_rhymes = detect_internal_rhymes(poem_lines)
# Returns positions and strengths of internal rhyming
```

#### `measure_rhyme_density(poem_text)`
Quantifies overall rhyme frequency in a text.

```python
density = measure_rhyme_density(poem_text)
# Returns ratio of rhyming to non-rhyming words
```

## Rhyme Type Classification

### Perfect Rhymes
Words with identical sounds from the vowel of the stressed syllable onward:
- **cat/hat**, **light/night**, **remember/December**

### Slant Rhymes (Near Rhymes)
Words with similar but not identical sounds:
- **soul/oil**, **worth/north**, **cloud/proud**

### Eye Rhymes
Words that look like they should rhyme but don't sound the same:
- **love/move**, **bough/through**, **wind/kind**

### Consonance
Words sharing final consonant sounds but different vowels:
- **milk/walk**, **tent/went**, **desk/risk**

### Assonance
Words sharing vowel sounds but different consonants:
- **hear/near**, **fate/rain**, **time/light**

## Statistical Analysis

### Corpus-Level Rhyme Patterns

#### `analyze_rhyme_preferences(corpus)`
Identifies preferred rhyme schemes and patterns in a corpus.

```python
preferences = analyze_rhyme_preferences(historical_corpus)
# Returns frequency of different rhyme schemes
```

#### `compare_rhyme_distributions(corpus1, corpus2)`
Statistical comparison of rhyme patterns between corpora.

```python
comparison = compare_rhyme_distributions(
    human_poems,
    ai_poems,
    metrics=['scheme_diversity', 'perfect_rhyme_ratio', 'scheme_consistency']
)
```

#### `track_rhyme_evolution(corpus_by_period)`
Analyzes how rhyme preferences change over time.

```python
evolution = track_rhyme_evolution(poems_by_century)
# Returns temporal trends in rhyming practices
```

### Quality Metrics

#### `calculate_rhyme_accuracy(predicted_rhymes, true_rhymes)`
Measures accuracy of automated rhyme detection.

```python
accuracy = calculate_rhyme_accuracy(auto_detected, human_annotated)
# Returns precision, recall, F1 scores
```

#### `assess_scheme_consistency(poem_schemes)`
Evaluates how consistently poems follow their declared schemes.

```python
consistency = assess_scheme_consistency(analyzed_poems)
# Returns measures of formal adherence
```

## Integration with Sample Analysis

### Batch Processing
#### `get_rhyme_for_sample(poem_dataframe)`
Analyzes rhyme patterns for an entire corpus sample.

```python
rhyme_results = get_rhyme_for_sample(df_poems)
# Returns DataFrame with rhyme analysis for each poem
```

### Caching and Performance
```python
# Results are cached for expensive operations
@cache
def analyze_poem_rhyme(poem_text):
    """Cached rhyme analysis for individual poems."""
    return detailed_rhyme_analysis(poem_text)
```

## Validation and Testing

### Ground Truth Comparison
```python
def validate_rhyme_detection(test_poems, expert_annotations):
    """Compare automated detection with expert analysis."""
    # Calculate agreement rates
    # Identify systematic errors
    # Report performance metrics
```

### Cross-Linguistic Support
```python
def adapt_rhyme_detection(language="en"):
    """Configure rhyme detection for different languages."""
    # Load language-specific phonetic rules
    # Adjust similarity thresholds
    # Return configured analyzer
```

## Error Handling

### Text Normalization
- **Spelling variations** - Handle archaic and variant spellings
- **Pronunciation changes** - Account for historical pronunciation shifts
- **Dialect differences** - Manage regional pronunciation variations

### Edge Cases
- **Incomplete poems** - Handle fragments and partial texts
- **Mixed languages** - Manage multilingual texts
- **Non-standard forms** - Handle experimental and free verse poetry

This module provides the computational foundation for understanding rhyme patterns in poetry, enabling systematic analysis of one of poetry's most fundamental formal features.
