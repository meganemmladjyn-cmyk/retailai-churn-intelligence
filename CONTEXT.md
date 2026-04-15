# CONTEXT.md — Business & Domain Context

## Churn definition

> **A customer is considered churned if they have made no purchase in the last 90 days.**

This threshold is derived from the median purchase frequency observed in the synthetic dataset.
A customer who has not purchased within 90 days is labelled `is_churned = 1`.

---

## Customer personas

### Persona 1 — Active customer
- **Profile:** Purchases every 2–4 weeks, high email open rate (> 40%), low support ticket count.
- **days_since_last_purchase:** < 30
- **Churn risk:** Low (score < 0.33)
- **Recommended action:** Standard loyalty programme, upsell campaign.

### Persona 2 — At-risk customer
- **Profile:** Last purchase 30–89 days ago, declining purchase frequency, moderate email engagement.
- **days_since_last_purchase:** 30–89
- **Churn risk:** Medium (score 0.33–0.66)
- **Recommended action:** Personalised re-engagement email with a discount offer.

### Persona 3 — Lost customer
- **Profile:** No purchase for 90+ days, very low email open rate (< 10%), high return rate.
- **days_since_last_purchase:** ≥ 90
- **Churn risk:** High (score > 0.66)
- **Recommended action:** Win-back campaign with a significant incentive (free shipping, 20 % off).

---

## Target audience

| Audience | Key expectation |
|---|---|
| **PME client** | Actionable churn alerts, simple API integration, clear recommendations per customer |
| **Recruiter** | Clean code, proper ML pipeline, production-ready deployment (Docker + CI/CD) |
| **OpenClassrooms jury** | End-to-end project: EDA → feature engineering → model → API → explainability (SHAP) |

---

## Business metrics

| Metric | Target | Rationale |
|---|---|---|
| **AUC-ROC** | > 0.85 | Overall discrimination power of the model |
| **Recall (churned class)** | Maximise | Missing a churner (false negative) costs more than a false alarm |
| **Precision (churned class)** | Secondary | Acceptable level of false positives to keep retention costs reasonable |
| **Churn rate (synthetic)** | ~25 % | Realistic for SME e-commerce; class imbalance handled via `scale_pos_weight` |

> **Recall > Precision** — It is better to alert on a customer who was not going to churn
> than to miss one who was.
