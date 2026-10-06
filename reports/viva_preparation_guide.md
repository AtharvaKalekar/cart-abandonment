# 🎓 Master Viva Preparation Guide: Case Study No. 59 (Cart Abandonment Analysis)

This document contains everything you need to confidently explain and defend your Machine Learning project in front of an examiner.

---

## 1. Problem Statement (PS)
* **Topic:** Case Study No. 59 – Cart Abandonment Analysis (B.Tech CSE Semester V).
* **Goal:** Predict whether an online shopping session will end in **abandonment** (non-purchase) using real-time session behavior, and segment abandoning users to implement targeted retention interventions.
* **Business Value:** Identifying abandoners before they leave allows e-commerce platforms to trigger micro-interventions (e.g., targeted discounts, exit-intent popups, cart save reminders), maximizing revenue while minimizing unnecessary coupon expenditure.

---

## 2. Solution Approach & Architecture
We implemented an end-to-end 3-tier ML system:
1. **Supervised Binary Classification:** Predicts session abandonment probability $P(Y=1 \mid X)$. We trained two operational variants:
   * **Early-Session Model (No `PageValues`):** Used during active browsing before basket value is calculated.
   * **Full Model (With `PageValues`):** Used at final checkout step.
2. **Cost-Aware Threshold Optimization:** Replaced the default static threshold ($t=0.5$) with an out-of-fold cost-optimal threshold ($t^*$) that minimizes asymmetric business loss ($C_{FP} \neq C_{FN}$).
3. **Unsupervised Behavioral Segmentation:** Applied $K$-Means clustering on log-transformed session metrics of abandoning sessions to identify 3 distinct user personas (*Light Browsers*, *Bouncers*, *Deep Browsers*).

---

## 3. Dataset Used
* **Dataset Name:** UCI Online Shoppers Purchasing Intention Dataset (Sakar et al., 2018).
* **Total Sessions:** 12,330 raw sessions $\rightarrow$ **12,205 session instances after exact deduplication**.
* **Features:** 17 input features (10 numerical, 7 categorical) + 1 binary target (`Revenue` / `Is_Abandoned`).
* **Class Balance:**
  * **Abandoners (`Is_Abandoned = 1` / `Revenue = False`):** 10,297 sessions (**84.37%**)
  * **Purchasers (`Is_Abandoned = 0` / `Revenue = True`):** 1,908 sessions (**15.63%**)
* **Important Nuance for Viva:** The dataset tracks **session non-conversion**, which is a broader proxy for cart abandonment since individual item-level add-to-cart logs are unobserved in session-level aggregates.

---

## 4. Data Cleaning & Integrity Audit
1. **Deduplication:** Identified and removed **125 exact duplicate rows** (reducing total rows from 12,330 to 12,205).
2. **Missing Values:** 0 missing values found across all columns.
3. **Outlier Treatment:** Checked duration features (`Administrative_Duration`, `ProductRelated_Duration`). Used log-transformation ($\log(1+x)$) for clustering algorithms to handle extreme right skewness without artificially deleting valid long-browsing customer sessions.

---

## 5. Feature Engineering & Preprocessing Pipeline
* **Numerical Pipeline (`StandardScaler`):** Standardizes continuous features to zero mean and unit variance:
  $$z = \frac{x - \mu}{\sigma}$$
* **Categorical Pipeline (`OneHotEncoder`):** Converts nominal variables (`Month`, `VisitorType`, `Weekend`) into binary dummy variables.
* **Target Encoding:** Mapped boolean `Revenue` $\rightarrow$ `Is_Abandoned` ($1 = \text{Abandoned}$, $0 = \text{Purchased}$).
* **Feature Ablation (`PageValues`):** `PageValues` measures the average value of pages visited prior to transaction. Since `PageValues > 0` leaks target information (almost exclusively non-zero for buyers), we created an **Early-Session Model** excluding `PageValues` to prevent data leakage in real-time prediction.

---

## 6. Machine Learning Models, Mathematical Formulas & Hyperparameters

### Model 1: Logistic Regression (Linear Baseline)
* **Mathematical Formula:** Computes log-odds of abandonment:
  $$P(Y=1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_i X_i)}}$$
* **Loss Function:** Binary Cross-Entropy (Log Loss):
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
* **Hyperparameters:** `solver='lbfgs'`, `C=1.0`, `max_iter=1000`.

---

### Model 2: K-Nearest Neighbors (KNN - Distance Baseline)
* **Mathematical Formula:** Classification based on majority vote of $K$ nearest Euclidean neighbors:
  $$d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$
  $$\hat{P}(Y=1 \mid x) = \frac{1}{K} \sum_{i \in \mathcal{N}_K(x)} I(y_i = 1)$$
* **Hyperparameters:** `n_neighbors=5`, `weights='uniform'`, `metric='minkowski'` ($p=2$).

---

### Model 3: Decision Tree Classifier
* **Mathematical Formula:** Recursive binary partitioning maximizing Information Gain via Gini Impurity:
  $$I_G(p) = 1 - \sum_{k=1}^K p_k^2 = 1 - (p_0^2 + p_1^2)$$
  $$\text{Gain} = I_G(\text{Parent}) - \left( \frac{N_{\text{left}}}{N} I_G(\text{Left}) + \frac{N_{\text{right}}}{N} I_G(\text{Right}) \right)$$
* **Hyperparameters:** `criterion='gini'`, `max_depth=None`, `min_samples_split=2`, `random_state=42`.

---

### Model 4: Random Forest Classifier (Bagging Ensemble)
* **Mathematical Formula:** Ensemble of $B$ bootstrap decision trees aggregated by majority voting:
  $$\hat{P}(Y=1 \mid X) = \frac{1}{B} \sum_{b=1}^B T_b(X)$$
* **Hyperparameters:** `n_estimators=100`, `max_features='sqrt'`, `bootstrap=True`, `random_state=42`.

---

### Model 5: HistGradientBoostingClassifier (Champion Model 🏆)
* **Mathematical Formula:** Sequential gradient boosting over decision trees using binning. At step $m$:
  $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$
  where $h_m(x)$ is trained to predict the negative gradient (pseudo-residuals) of log loss:
  $$r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}} = y_i - \hat{p}_i$$
* **Hyperparameters:** `learning_rate=0.1`, `max_iter=100`, `early_stopping=True`, `random_state=42`.

---

## 7. Synthetic Minority Over-sampling Technique (SMOTE)
* **Why SMOTE?** Target class is imbalanced (84.4% abandoners vs 15.6% purchasers). Standard classifiers tend to favor the majority class.
* **How SMOTE Works:**
  1. For each sample $x_i$ in the minority class (Purchasers), find its $K$ nearest minority neighbors in feature space.
  2. Randomly select one neighbor $x_{zi}$.
  3. Create a synthetic sample along the line segment connecting them:
     $$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i) \quad \text{where } \lambda \sim U(0, 1)$$
* **CRITICAL VIVA GOTCHA:** SMOTE was applied **ONLY inside cross-validation training folds** (`imblearn.pipeline.Pipeline`). Applying SMOTE to the entire dataset before splitting causes severe **data leakage** and inflated test scores!

---

## 8. Cost Analysis & Threshold Tuning ($t^*$)
Default classification uses a probability threshold of $t=0.5$. In business, costs are asymmetric:
* **False Positive (FP):** Intervention offered to a user who would have purchased anyway $\rightarrow$ Cost $C_{FP} = \$2$ (Discount / Coupon cost).
* **False Negative (FN):** Missed abandoner (no intervention) $\rightarrow$ Cost $C_{FN} = \$25$ (Lost margin).

### Expected Total Business Cost Formula:
$$\text{Cost}(t) = C_{FP} \cdot FP(t) + C_{FN} \cdot FN(t)$$

### Optimal Threshold Selection ($t^*$):
We tune threshold $t^*$ on out-of-fold cross-validation predictions:
$$t^* = \arg\min_{t \in [0, 1]} \text{Cost}(t)$$
* **Result:** Optimal threshold $t^* \approx 0.12 - 0.13$. Lowering the threshold boosts **Recall** for abandoners, reducing costly False Negatives and maximizing net business savings.

---

## 9. Which Model is Best and Why?
### 🏆 Winner: **Gradient Boosting Classifier (HistGradientBoosting)**

### Quantitative Performance Summary:
* **ROC-AUC:** `0.934` (Full Model) / `0.775` (Early-Session Model)
* **Macro F1-Score:** `79.46%`
* **Abandoner F1:** `93.92%`
* **Accuracy:** `89.64%`

### Why Gradient Boosting won over other models:
1. **Handles Non-Linearities & Interactions:** Captures complex non-linear relationships between browsing duration (`ProductRelated_Duration`), exit rate (`ExitRates`), and page count without manual feature interaction terms.
2. **Gradient-Driven Optimization:** Unlike Random Forest (which builds independent trees), Gradient Boosting sequentially corrects errors from prior trees, resulting in sharper decision boundaries.
3. **Histogram Binning:** HistGradientBoosting groups continuous variables into integer-valued bins (255 bins), drastically speeding up computation and reducing sensitivity to slight noise.

---

## 10. Behavioral Clustering & Personas ($K$-Means)
Applied $K$-Means ($k=3$) on log-transformed duration & rate metrics for abandoning sessions:
1. **Cluster 0 — Light Browsers (40.9%):** Low duration (~8 pages, ~262s), low exit rate.
2. **Cluster 1 — Bouncers (7.2%):** Very high bounce/exit rates (~18%), minimal engagement (~2 pages, 19s). Immediate drop-off.
3. **Cluster 2 — Deep Browsers (52.0%):** High page views (~49 pages), long browsing time (~1,875s). Interested but hesitant buyers.

---

## 11. Core Examiner Q&A Cheat Sheet (Top Trick Questions)

#### Q1: "Why is Accuracy a poor metric for this problem?"
* **Answer:** With 84.4% base abandonment, a dummy model that predicts "Abandonment" for every session achieves 84.4% accuracy while being completely useless for business decision-making. That's why we evaluate **Macro F1-Score**, **Purchaser F1**, **PR-AUC**, and **Cost Savings**.

#### Q2: "What is the difference between your Early-Session Model and Full Model?"
* **Answer:** The **Full Model** includes `PageValues`, achieving ROC-AUC `0.934`. However, `PageValues` is only populated late in the funnel when items are added. The **Early-Session Model** excludes `PageValues` (ROC-AUC `0.775`), allowing real-time intervention while the user is still browsing.

#### Q3: "Did you apply SMOTE before or after splitting your train/test sets?"
* **Answer:** **Strictly after splitting**, and inside the training cross-validation folds using `imbalanced-learn` pipelines. Applying SMOTE before splitting duplicates synthetic samples into the test set, creating artificial data leakage.

#### Q4: "Why did you use Log-Transformation before K-Means?"
* **Answer:** Session duration features (`ProductRelated_Duration`) exhibit severe right skewness with extreme long-tail outliers. $K$-Means uses Euclidean distance, which is sensitive to scale and skewness. $\log(1 + x)$ normalizes distributions so Euclidean distances accurately reflect behavioral similarity.

---

## 12. End Summary (30-Second Elevator Pitch)
> "In this project, we built an end-to-end Machine Learning solution for Case Study 59 (Cart Abandonment Analysis) using 12,205 deduplicated e-commerce sessions. We developed both an early-session classification model and a full-session model using HistGradientBoosting, achieving a top ROC-AUC of 0.934 and a Macro F1 of 79.5%. By tuning the decision threshold to cost ratio ($C_{FP}=\$2, C_{FN}=\$25$), we minimized asymmetric business loss. Finally, we used K-Means clustering to discover 3 distinct abandoner personas, enabling automated, targeted retention strategies."
