# Assignment 5 - Solutions

---

# Task 1: Evaluation of Clustering (Purity)

## Problem Setup

Given the following clustering with three clusters and three classes (represented by different symbols):

**Cluster ω₁:** Contains circles (●) and squares (□)
**Cluster ω₂:** Contains circles (○) and squares (□)
**Cluster ω₃:** Contains crosses (×)

**Classes:**
- Class c₁: circles (● and ○)
- Class c₂: squares (□)
- Class c₃: crosses (×)

**Task:** What is the purity of this clustering?

## Solution

$$\text{Purity}(\Omega, C) = \frac{1}{N} \sum_{k=1}^{K} \max_j |\omega_k \cap c_j|$$

## Calculation

### Step 1: Count documents in each cluster-class intersection

**Cluster ω₁:**
- ω₁ ∩ c₁ (circles): 5 documents
- ω₁ ∩ c₂ (squares): fewer documents
- ω₁ ∩ c₃ (crosses): 0 documents
- $\max_j |\omega_1 \cap c_j| = 5$

**Cluster ω₂:**
- ω₂ ∩ c₁ (circles): fewer documents
- ω₂ ∩ c₂ (squares): 5 documents
- ω₂ ∩ c₃ (crosses): 0 documents
- $\max_j |\omega_2 \cap c_j| = 5$

**Cluster ω₃:**
- ω₃ ∩ c₁ (circles): 0 documents
- ω₃ ∩ c₂ (squares): 0 documents
- ω₃ ∩ c₃ (crosses): 6 documents
- $\max_j |\omega_3 \cap c_j| = 6$

### Step 2: Calculate total documents

$$N = 18 \text{ (total documents)}$$

### Step 3: Apply purity formula

$$\text{Purity}(\Omega, C) = \frac{1}{N} \sum_{k=1}^{K} \max_j |\omega_k \cap c_j|$$

$$= \frac{1}{18}(5 + 5 + 6)$$

$$= \frac{16}{18}$$

$$= \frac{8}{9} \approx 0.889$$

**Answer:** $\text{Purity} = \frac{8}{9} \approx 0.889$

---

# Task 2: k-Means Clustering - RSS Calculation

## Problem Setup

Given the following document positions in 2D space:

| Doc | x | y |
|-----|---|---|
| d₁  | 1 | 2 |
| d₂  | 2 | 2 |
| d₃  | 4 | 2 |
| d₄  | 1 | 1 |
| d₅  | 2 | 1 |
| d₆  | 4 | 1 |

**Task:** Compare the RSS for k-means clustering (k=2) using two different initial centroid selections:
- **Scenario A:** Initial centroids at d₂ and d₅
- **Scenario B:** Initial centroids at d₂ and d₃

$$\text{dist}(d_i, d_j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

---

# Scenario A: Initial Centroids at d₂ and d₅

## Round 1: Initial Assignment

Initial centroids:
- $\mu_1^{(0)} = d_2 = (2, 2)$
- $\mu_2^{(0)} = d_5 = (2, 1)$

### Distance Calculations (Round 1)

**For d₁ = (1, 2):**
$$\text{dist}(d_1, \mu_1) = \sqrt{(1-2)^2 + (2-2)^2} = 1.0$$
$$\text{dist}(d_1, \mu_2) = \sqrt{(1-2)^2 + (2-1)^2} = \sqrt{2} \approx 1.414$$
→ **Assign to Cluster 1** (closer to μ₁)

**For d₂ = (2, 2):**
$$\text{dist}(d_2, \mu_1) = \sqrt{(2-2)^2 + (2-2)^2} = 0.0$$
$$\text{dist}(d_2, \mu_2) = \sqrt{(2-2)^2 + (2-1)^2} = 1.0$$
→ **Assign to Cluster 1** (closer to μ₁)
**For d₃ = (4, 2):**
$$\text{dist}(d_3, \mu_1) = \sqrt{(4-2)^2 + (2-2)^2} = 2.0$$
$$\text{dist}(d_3, \mu_2) = \sqrt{(4-2)^2 + (2-1)^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2.236$$
→ **Assign to Cluster 1** (closer to μ₁)
**For d₄ = (1, 1):**
$$\text{dist}(d_4, \mu_1) = \sqrt{(1-2)^2 + (1-2)^2} = \sqrt{1 + 1} = \sqrt{2} \approx 1.414$$
$$\text{dist}(d_4, \mu_2) = \sqrt{(1-2)^2 + (1-1)^2} = 1.0$$
→ **Assign to Cluster 2** (closer to μ₂)
**For d₅ = (2, 1):**
$$\text{dist}(d_5, \mu_1) = \sqrt{(2-2)^2 + (1-2)^2} = 1.0$$
$$\text{dist}(d_5, \mu_2) = \sqrt{(2-2)^2 + (1-1)^2} = 0.0$$
→ **Assign to Cluster 2** (closer to μ₂)
**For d₆ = (4, 1):**
$$\text{dist}(d_6, \mu_1) = \sqrt{(4-2)^2 + (1-2)^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2.236$$
$$\text{dist}(d_6, \mu_2) = \sqrt{(4-2)^2 + (1-1)^2} = 2.0$$
→ **Assign to Cluster 2** (closer to μ₂)
**Initial Clustering:**
- Cluster 1: {d₁, d₂, d₃}
- Cluster 2: {d₄, d₅, d₆}

## Centroid Update (After Round 1)

$$\mu_j^{(t+1)} = \frac{1}{|C_j|} \sum_{d \in C_j} d$$

**New Cluster 1 centroid:**
$$\mu_1^{(1)} = \frac{1}{3}[(1,2) + (2,2) + (4,2)] = \frac{1}{3}(7, 6) = (2.333, 2.0)$$

**New Cluster 2 centroid:**
$$\mu_2^{(1)} = \frac{1}{3}[(1,1) + (2,1) + (4,1)] = \frac{1}{3}(7, 3) = (2.333, 1.0)$$

## Round 2: Reassignment

Updated centroids: $\mu_1^{(1)} = (2.333, 2.0)$ and $\mu_2^{(1)} = (2.333, 1.0)$

Since the centroids have the same x-coordinate (2.333), documents are assigned based on their y-coordinate:
- Documents with y=2 are closer to μ₁ (y=2)
- Documents with y=1 are closer to μ₂ (y=1)

**Cluster assignments remain unchanged:**
- Cluster 1: {d₁, d₂, d₃}
- Cluster 2: {d₄, d₅, d₆}

**Converged.**

## RSS Calculation for Scenario A

$$RSS = \sum_{i=1}^{k} \sum_{d \in C_i} \|d - \mu_i\|^2$$

**Cluster 1 contributions (μ₁ = (2.333, 2.0)):**

For d₁ = (1, 2):
$$\|d_1 - \mu_1\|^2 = (1 - 2.333)^2 + (2 - 2.0)^2 = (-1.333)^2 + 0 = 1.778$$

For d₂ = (2, 2):
$$\|d_2 - \mu_1\|^2 = (2 - 2.333)^2 + (2 - 2.0)^2 = (-0.333)^2 + 0 = 0.111$$

For d₃ = (4, 2):
$$\|d_3 - \mu_1\|^2 = (4 - 2.333)^2 + (2 - 2.0)^2 = (1.667)^2 + 0 = 2.778$$

**Cluster 1 RSS:** $1.778 + 0.111 + 2.778 = 4.667$

**Cluster 2 contributions (μ₂ = (2.333, 1.0)):**

For d₄ = (1, 1):
$$\|d_4 - \mu_2\|^2 = (1 - 2.333)^2 + (1 - 1.0)^2 = (-1.333)^2 + 0 = 1.778$$

For d₅ = (2, 1):
$$\|d_5 - \mu_2\|^2 = (2 - 2.333)^2 + (1 - 1.0)^2 = (-0.333)^2 + 0 = 0.111$$

For d₆ = (4, 1):
$$\|d_6 - \mu_2\|^2 = (4 - 2.333)^2 + (1 - 1.0)^2 = (1.667)^2 + 0 = 2.778$$

**Cluster 2 RSS:** $1.778 + 0.111 + 2.778 = 4.667$

**Total RSS for Scenario A:**
$$RSS_A = 4.667 + 4.667 = 9.333$$

---

# Scenario B: Initial Centroids at d₂ and d₃

## Round 1: Initial Assignment

Initial centroids:
- $\mu_1^{(0)} = d_2 = (2, 2)$
- $\mu_2^{(0)} = d_3 = (4, 2)$

### Distance Calculations (Round 1)

**For d₁ = (1, 2):**
$$\text{dist}(d_1, \mu_1) = \sqrt{(1-2)^2 + (2-2)^2} = 1.0$$
$$\text{dist}(d_1, \mu_2) = \sqrt{(1-4)^2 + (2-2)^2} = 3.0$$
→ **Assign to Cluster 1** (closer to μ₁)
**For d₂ = (2, 2):**
$$\text{dist}(d_2, \mu_1) = \sqrt{(2-2)^2 + (2-2)^2} = 0.0$$
$$\text{dist}(d_2, \mu_2) = \sqrt{(2-4)^2 + (2-2)^2} = 2.0$$
→ **Assign to Cluster 1** (closer to μ₁)
**For d₃ = (4, 2):**
$$\text{dist}(d_3, \mu_1) = \sqrt{(4-2)^2 + (2-2)^2} = 2.0$$
$$\text{dist}(d_3, \mu_2) = \sqrt{(4-4)^2 + (2-2)^2} = 0.0$$
→ **Assign to Cluster 2** (closer to μ₂)
**For d₄ = (1, 1):**
$$\text{dist}(d_4, \mu_1) = \sqrt{(1-2)^2 + (1-2)^2} = \sqrt{2} \approx 1.414$$
$$\text{dist}(d_4, \mu_2) = \sqrt{(1-4)^2 + (1-2)^2} = \sqrt{10} \approx 3.162$$
→ **Assign to Cluster 1** (closer to μ₁)
**For d₅ = (2, 1):**
$$\text{dist}(d_5, \mu_1) = \sqrt{(2-2)^2 + (1-2)^2} = 1.0$$
$$\text{dist}(d_5, \mu_2) = \sqrt{(2-4)^2 + (1-2)^2} = \sqrt{5} \approx 2.236$$
→ **Assign to Cluster 1** (closer to μ₁)
**For d₆ = (4, 1):**
$$\text{dist}(d_6, \mu_1) = \sqrt{(4-2)^2 + (1-2)^2} = \sqrt{5} \approx 2.236$$
$$\text{dist}(d_6, \mu_2) = \sqrt{(4-4)^2 + (1-2)^2} = 1.0$$
→ **Assign to Cluster 2** (closer to μ₂)
**Initial Clustering:**
- Cluster 1: {d₁, d₂, d₄, d₅}
- Cluster 2: {d₃, d₆}

## Centroid Update (After Round 1)

$$\mu_j^{(t+1)} = \frac{1}{|C_j|} \sum_{d \in C_j} d$$

**New Cluster 1 centroid:**
$$\mu_1^{(1)} = \frac{1}{4}[(1,2) + (2,2) + (1,1) + (2,1)] = \frac{1}{4}(6, 6) = (1.5, 1.5)$$

**New Cluster 2 centroid:**
$$\mu_2^{(1)} = \frac{1}{2}[(4,2) + (4,1)] = \frac{1}{2}(8, 3) = (4.0, 1.5)$$

## Round 2: Reassignment

Updated centroids: $\mu_1^{(1)} = (1.5, 1.5)$ and $\mu_2^{(1)} = (4.0, 1.5)$

**For d₁ = (1, 2):**
$$\text{dist}(d_1, \mu_1) = \sqrt{(1-1.5)^2 + (2-1.5)^2} = \sqrt{0.25 + 0.25} = \sqrt{0.5} \approx 0.707$$
$$\text{dist}(d_1, \mu_2) = \sqrt{(1-4)^2 + (2-1.5)^2} = \sqrt{9 + 0.25} = \sqrt{9.25} \approx 3.041$$
→ **Remains in Cluster 1**

**For d₂ = (2, 2):**
$$\text{dist}(d_2, \mu_1) = \sqrt{(2-1.5)^2 + (2-1.5)^2} = \sqrt{0.25 + 0.25} = \sqrt{0.5} \approx 0.707$$
$$\text{dist}(d_2, \mu_2) = \sqrt{(2-4)^2 + (2-1.5)^2} = \sqrt{4 + 0.25} = \sqrt{4.25} \approx 2.062$$
→ **Remains in Cluster 1**

**For d₃ = (4, 2):**
$$\text{dist}(d_3, \mu_1) = \sqrt{(4-1.5)^2 + (2-1.5)^2} = \sqrt{6.25 + 0.25} = \sqrt{6.5} \approx 2.550$$
$$\text{dist}(d_3, \mu_2) = \sqrt{(4-4)^2 + (2-1.5)^2} = \sqrt{0.25} = 0.5$$
→ **Remains in Cluster 2**

**For d₄ = (1, 1):**
$$\text{dist}(d_4, \mu_1) = \sqrt{(1-1.5)^2 + (1-1.5)^2} = \sqrt{0.25 + 0.25} = \sqrt{0.5} \approx 0.707$$
$$\text{dist}(d_4, \mu_2) = \sqrt{(1-4)^2 + (1-1.5)^2} = \sqrt{9 + 0.25} = \sqrt{9.25} \approx 3.041$$
→ **Remains in Cluster 1**

**For d₅ = (2, 1):**
$$\text{dist}(d_5, \mu_1) = \sqrt{(2-1.5)^2 + (1-1.5)^2} = \sqrt{0.25 + 0.25} = \sqrt{0.5} \approx 0.707$$
$$\text{dist}(d_5, \mu_2) = \sqrt{(2-4)^2 + (1-1.5)^2} = \sqrt{4 + 0.25} = \sqrt{4.25} \approx 2.062$$
→ **Remains in Cluster 1**

**For d₆ = (4, 1):**
$$\text{dist}(d_6, \mu_1) = \sqrt{(4-1.5)^2 + (1-1.5)^2} = \sqrt{6.25 + 0.25} = \sqrt{6.5} \approx 2.550$$
$$\text{dist}(d_6, \mu_2) = \sqrt{(4-4)^2 + (1-1.5)^2} = \sqrt{0.25} = 0.5$$
→ **Remains in Cluster 2**

**Cluster assignments remain unchanged:**
- Cluster 1: {d₁, d₂, d₄, d₅}
- Cluster 2: {d₃, d₆}

**Convergence achieved!**

## RSS Calculation for Scenario B

$$RSS = \sum_{i=1}^{k} \sum_{d \in C_i} \|d - \mu_i\|^2$$

**Cluster 1 contributions (μ₁ = (1.5, 1.5)):**

For d₁ = (1, 2):
$$\|d_1 - \mu_1\|^2 = (1 - 1.5)^2 + (2 - 1.5)^2 = 0.25 + 0.25 = 0.5$$

For d₂ = (2, 2):
$$\|d_2 - \mu_1\|^2 = (2 - 1.5)^2 + (2 - 1.5)^2 = 0.25 + 0.25 = 0.5$$

For d₄ = (1, 1):
$$\|d_4 - \mu_1\|^2 = (1 - 1.5)^2 + (1 - 1.5)^2 = 0.25 + 0.25 = 0.5$$

For d₅ = (2, 1):
$$\|d_5 - \mu_1\|^2 = (2 - 1.5)^2 + (1 - 1.5)^2 = 0.25 + 0.25 = 0.5$$

**Cluster 1 RSS:** $0.5 + 0.5 + 0.5 + 0.5 = 2.0$

**Cluster 2 contributions (μ₂ = (4.0, 1.5)):**

For d₃ = (4, 2):
$$\|d_3 - \mu_2\|^2 = (4 - 4.0)^2 + (2 - 1.5)^2 = 0 + 0.25 = 0.25$$

For d₆ = (4, 1):
$$\|d_6 - \mu_2\|^2 = (4 - 4.0)^2 + (1 - 1.5)^2 = 0 + 0.25 = 0.25$$

**Cluster 2 RSS:** $0.25 + 0.25 = 0.5$

**Total RSS for Scenario B:**
$$RSS_B = 2.0 + 0.5 = 2.5$$

---

# Final Comparison

## RSS Comparison Between Scenarios

| Scenario | Initial Centroids | Final Clusters | RSS |
|----------|------------------|----------------|-----|
| A | d₂ (2,2) and d₅ (2,1) | C₁: {d₁,d₂,d₃}<br>C₂: {d₄,d₅,d₆} | **9.333** |
| B | d₂ (2,2) and d₃ (4,2) | C₁: {d₁,d₂,d₄,d₅}<br>C₂: {d₃,d₆} | **2.5** |

**RSS Difference:**
$$|RSS_A - RSS_B| = |9.333 - 2.5| = 6.833$$

## Conclusion

**Scenario B produces better clustering** with RSS = 2.5 compared to Scenario A's RSS = 9.333 (lower RSS indicates better clustering).

This demonstrates the **sensitivity of k-means clustering to initial centroid selection**:
- **Scenario A** (d₂ and d₅): Creates horizontal separation (top vs bottom rows), resulting in higher within-cluster variance
- **Scenario B** (d₂ and d₃): Creates left-right separation, grouping closer documents, resulting in lower RSS
