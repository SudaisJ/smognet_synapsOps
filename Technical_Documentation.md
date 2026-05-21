# 📄 SmogNet: Technical Documentation & Methodology

## 1. System Architecture Overview
The SmogNet intelligence system is an end-to-end data pipeline built entirely in Python using a modular design. It integrates three primary stages:
1.  **Stage 1: Spike Detection** (Time-Series Anomaly Detection)
2.  **Stage 2: Source Classification** (Chemical Fingerprint Modeling)
3.  **Stage 3: Public Alert Generation** (Natural Language Processing)

The frontend is served using **Streamlit**, which provides a responsive, real-time command center interface.

---

## 2. Stage 1: Context-Aware Spike Detection
**Goal:** Detect abnormal pollution spikes in real-time, adjusting for city-specific baselines and seasonal shifts.

### Methodology: Rolling Z-Score
Static thresholds (e.g., $PM_{2.5} > 150 \mu g/m^3$) are prone to high false-positive rates during naturally polluted seasons (like Lahore winters) and false negatives in cleaner seasons.

To solve this, we implemented a **7-Day Rolling Z-Score Filter**:
1.  **Grouping:** Data is grouped by `City`.
2.  **Rolling Baseline:** For every timestamp $t$, we calculate the mean ($\mu$) and standard deviation ($\sigma$) over the preceding 7 days (168 hours).
3.  **Z-Score Calculation:**
    $$ Z_t = \frac{PM2.5_t - \mu_{rolling}}{\sigma_{rolling}} $$
4.  **Anomaly Threshold:** A spike is defined where $Z_t > 2.5$. This ensures we are only flagging statistically significant deviations relative to the city's *recent* context.

---

## 3. Stage 2: Chemical Fingerprint Modeling (Source Classification)
**Goal:** Determine the probable emission source for detected spikes.

### Methodology: Rule-Based Feature Engineering
Given the potential for overlapping pollution signals, we adopted a robust, rule-based inference engine rather than a black-box ML model. This ensures **high interpretability** and **computational efficiency**.

The classification is executed conditionally (only if $Is\_Spike = True$) using the following logic hierarchy:
1.  **Crop Burning:** Triggered by high concentrations of Ammonia ($NH_3$) and Carbon Monoxide ($CO$).
2.  **Vehicular Emissions:** Triggered by dominance of Nitrogen Oxides ($NO$ + $NO_2$), typical of combustion engines.
3.  **Industrial Emissions:** Triggered by severe Sulfur Dioxide ($SO_2$) spikes.
4.  **Dust Storms:** Triggered by an abnormally high ratio of coarse to fine particulate matter ($PM_{10} / PM_{2.5} > 2.0$).
5.  **Mixed Sources:** Fallback class for overlapping signatures.

---

## 4. Stage 3: Public Alert Generation
**Goal:** Translate numerical anomaly data into a non-technical, 3-sentence public health advisory.

### Methodology: Context-Aware Natural Language Generation (NLG)
We utilize a dynamic f-string templating engine (with hooks for direct LLM API integration).
*   **Sentence 1 (Announce):** Dynamically injects the `City`, `PM2.5` severity, and the classified `Source`.
*   **Sentence 2 (Vulnerability):** Identifies at-risk populations.
*   **Sentence 3 (Action):** Applies rule-based conditional logic to recommend protective measures based on the specific source (e.g., N95 masks for Dust Storms, window closure for Crop Burning).

---

## 5. Feature Engineering Assumptions
*   **Data Imputation:** Missing data points are handled using Forward Fill (`ffill`) followed by Backward Fill (`bfill`) to preserve time-series continuity without data leakage.
*   **Baseline Window:** A 7-day window is assumed optimal to capture weekly cyclical trends (e.g., weekday traffic vs. weekend lulls) while remaining responsive to rapid seasonal shifts.
