# 🧾 Healthcare Invoice Processing: OCR + LLM System

Production-grade invoice processing system achieving **99% field-level accuracy** and **73% reduction in processing time** for a US healthcare provider.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **Note:** This repository documents the architecture, design decisions, and lessons learned from a production system. Code examples are illustrative and anonymized to respect client confidentiality.

---

## 🎯 Business Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Field-Level Accuracy** | 91% (RAG) | 99% (Fine-tuned) | **+8 percentage points** |
| **Processing Time** | 8+ minutes/invoice | <2 minutes/invoice | **73% reduction** |
| **Automation Rate** | Manual review required | 77% fully automated | **77% no-touch processing** |
| **Cost Savings** | Baseline | $340K annually | **Significant ROI** |
| **Data Privacy** | External API calls | On-premises | **Full compliance** |

**Production Timeline:** 6 months running in production, processing 10,000 invoices/month

---

## 🏗️ Architecture Overview

### The Journey: RAG → Fine-Tuning

We started with **Azure OpenAI + RAG** (91% accuracy) but hit a ceiling at 93% despite extensive prompt engineering. Three critical issues pushed us to rebuild:

1. **Healthcare compliance:** Patient-adjacent data flowing through external APIs
2. **Cost at scale:** API costs adding up at 10K invoices/month
3. **Accuracy plateau:** 700 invoices/month still had errors (7% error rate)

### Final System Architecture

```
┌─────────────────┐
│  Invoice PDF/   │
│  Image Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Preprocessing Layer            │
│  - Rotation correction          │
│  - Adaptive thresholding        │
│  - Region detection             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  PaddleOCR                      │
│  - Text extraction              │
│  - Multi-column layout handling │
│  - Table structure detection    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  BioBERT (Medical NER)          │
│  - Drug name recognition        │
│  - CPT/ICD code extraction      │
│  - Clinical term parsing        │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  LLaMA 3.2 (QLoRA Fine-tuned)   │
│  - Structured field extraction  │
│  - Confidence scoring           │
│  - LoRA Rank: 8                 │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Three-Tier Validation          │
│  Tier 1: Format validation      │
│  Tier 2: Confidence (>0.95)     │
│  Tier 3: Business rules         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Output / Human Review Queue    │
│  - 77% fully automated          │
│  - 23% routed for review        │
└─────────────────────────────────┘
```

---

## 🔧 Tech Stack

### Core Components

**OCR:** PaddleOCR 2.7+
- Chosen after benchmarking against Tesseract and EasyOCR
- **12% higher accuracy** on 500-invoice test corpus
- Superior performance on multi-column layouts and tables

**Medical NER:** BioBERT
- Pre-trained on PubMed abstracts
- Essential for medical terminology (drug names, CPT codes)
- General-purpose NER models were "hopeless" on healthcare terms

**LLM:** LLaMA 3.2 (Fine-tuned with QLoRA)
- Self-hosted for data privacy and cost efficiency
- QLoRA training on T4 GPUs (~7x cheaper than A100 full fine-tuning)
- **LoRA Rank 8** (sweet spot after testing ranks 4-32)

**Infrastructure:**
- Python 3.10+
- PyTorch 2.0+
- HuggingFace Transformers
- PostgreSQL (transaction management, audit trail)

---

## 💡 Key Technical Decisions

### 1. Why PaddleOCR Over Tesseract?

**Testing methodology:** Benchmarked on 500 real invoices

| OCR Engine | Accuracy | Multi-Column | Tables | Speed |
|------------|----------|--------------|--------|-------|
| Tesseract | 76% | Poor (merged columns) | Poor | Fast |
| EasyOCR | 79% | Medium | Medium | Slow |
| **PaddleOCR** | **88%** | **Excellent** | **Excellent** | Medium |

**Decision:** PaddleOCR's 12% accuracy advantage outweighed the steeper learning curve.

### 2. Why BioBERT for Medical Entities?

General-purpose NER models failed catastrophically on healthcare terminology:
- "Metformin 500mg" → parsed as single blob (no drug/dosage separation)
- CPT codes → missed entirely
- Drug names with unusual spellings → constant errors

BioBERT, pre-trained on PubMed, understood medical terminology out of the box. **This wasn't marginal - it was the difference between working and not working.**

### 3. Why Fine-Tuning Over RAG?

| Approach | Accuracy | Privacy | Cost | Maintenance |
|----------|----------|---------|------|-------------|
| RAG (Azure OpenAI) | 91-93% ceiling | External API ❌ | Per-request fees | Easy template updates |
| **Fine-tuning (LLaMA 3.2)** | **99%** | **On-premises ✅** | **One-time training** | **Retraining for template changes** |

**Trade-off accepted:** Higher operational overhead (retraining) for 8% accuracy gain + compliance + cost savings.

### 4. Why QLoRA Over Full Fine-Tuning?

**Cost comparison:**
- Full fine-tuning: A100 GPU required (~$200/hr cloud cost)
- QLoRA: T4 GPU sufficient (~$30/hr cloud cost)
- **7x cost reduction**

**Accuracy comparison:**
- Full fine-tuning: 99.2% accuracy
- QLoRA (Rank 8): 99.0% accuracy
- **Trade-off: 0.2% accuracy loss for 7x cost savings ✅**

### 5. Three-Tier Validation Strategy

**Tier 1 - Format Validation:**
- Regex checks for invoice numbers, dates, monetary values
- Catches obvious extraction errors

**Tier 2 - Confidence Scoring:**
- Threshold: 0.95
- Below threshold → human review queue
- Result: 77% fully automated, 23% reviewed

**Tier 3 - Business Rules:**
- Does total = sum of line items?
- Are procedure codes valid for this provider type?
- Cross-field consistency checks

**Combined result:** 91% (RAG) → 99% (fine-tuned + validation)

---

## 🚧 Problems We Didn't Anticipate

### Week 2: OCR Quality Issues

**Problem:** 30% of invoices producing garbage output

**Root cause:** 
- Scanned at odd angles
- Low resolution / fax artifacts
- Handwritten annotations overlapping printed text

**Solution:** Added preprocessing layer
- Rotation correction
- Adaptive thresholding for low-contrast scans
- Region detection to separate printed text from handwritten notes

**Impact:** **15% accuracy improvement** from preprocessing alone

**Lesson:** OCR is half the problem. Never assume clean input.

---

### Week 4: Vendor Format Memorization

**Problem:** Test accuracy 96%, but production accuracy 81% on new vendors

**Root cause:** Model memorized vendor-specific patterns
- Vendor A: total always bottom-right
- Vendor B: specific date format
- Vendor C: particular line item layout

**Solution:**
- Vendor-agnostic preprocessing
- Diverse training batches (stratified by vendor)
- **Evaluation on held-out vendors** (not just held-out invoices)

**Result:** Vendor generalization gap reduced from 15% → 4%

**Lesson:** Standard train/val splits can hide serious generalization problems. Test on held-out groups (vendors, time periods, etc.), not just random samples.

---

### Week 6: Minority Class Crushing

**Problem:** Overall accuracy 96%, but minority classes at 0.68 F1

**Breakdown:**
- Standard line items (88% of data): 0.98 F1 ✅
- Compound medications (4%): 0.68 F1 ❌
- DME rentals (4%): 0.71 F1 ❌
- Bundled service codes (4%): 0.66 F1 ❌

**Why this matters:** 1 in 3 compound medications extracted incorrectly

**Solution:**
- Weighted cross-entropy loss (inverse class frequency)
  - Compound medications: weight 8.2
  - Standard line items: weight 1.0
- Stratified validation splits (ensure rare classes in every fold)

**Result:** Minority class F1 improved from 0.68 → 0.81

**Lesson:** Monitor class-level metrics, not just aggregate accuracy.

---

### Month 3: Production Incident (Template Drift)

**Problem:** Accuracy dropped from 99% → 91% on one vendor overnight

**Root cause:** Major vendor updated invoice template
- Field positions shifted
- Date format changed: MM/DD/YYYY → YYYY-MM-DD
- Model trained on old format

**Response:**
1. Immediately route vendor to human review queue
2. Collect corrected examples over next few days
3. Retrain model with new format
4. Redeploy

**Recovery time:** ~1 week

**Prevention implemented:**
- Daily vendor-level accuracy monitoring
- Documented retraining process
- Automated alerts for accuracy drops

**Lesson:** Fine-tuning has operational overhead that RAG doesn't. Template changes require retraining. Monitor vendor-level metrics, not just overall accuracy.

---

## 📊 Results After 6 Months in Production

### Accuracy Metrics
- **99% field-level accuracy** (up from 91% with RAG)
- **77% fully automated** (no human touch required)
- **Minority class F1: 0.81** (up from 0.68)

### Performance Metrics
- **Processing time:** 8+ minutes → <2 minutes per invoice (**73% reduction**)
- **Same-day processing** (replaced 3-5 day backlog)
- **10,000 invoices/month** sustained volume

### Business Impact
- **$340K annual savings** in processing costs
- **Zero data leaving environment** (healthcare compliance achieved)
- **Stable production:** 6 months running, accuracy holding steady

---

## 🎓 What Actually Mattered

After 6 months, these five decisions made the difference:

1. **Moving from RAG to fine-tuning** - Broke through 93% ceiling to 99%
2. **BioBERT for medical NER** - Domain-specific pre-training was essential, not optional
3. **Cross-vendor validation from day one** - Caught generalization issues early
4. **Weighted loss for class imbalance** - Simple technique, huge impact on minority classes
5. **Vendor-level accuracy monitoring** - Caught template drift within 24 hours

---

## 📖 Documentation

- [Architecture Deep Dive](docs/architecture.md) - Detailed system design and component decisions
- [Fine-Tuning Guide](docs/fine-tuning.md) - QLoRA implementation, hyperparameters, training process
- [Production Lessons](docs/lessons-learned.md) - What worked, what didn't, what we'd do differently
- [Validation Strategy](docs/validation.md) - Three-tier validation implementation
- [Monitoring & Maintenance](docs/monitoring.md) - Vendor-level tracking, retraining pipeline

---

## 📝 Related Writing

I've written extensively about building this system:

- **[Medium Article: How We Built a 99% Accurate Invoice Processing System](https://medium.com/@vaibhav-rathi)** - Full technical story from RAG to fine-tuning
- **Dev.to:** Cross-posted technical deep-dive
- **LinkedIn:** Production ML lessons learned

---

## 🔮 What I'd Do Differently

If starting this project again:

1. **Budget preprocessing time upfront** - OCR quality issues cost us a week. Preprocessing drove more accuracy gains than model tuning.

2. **Test on held-out groups immediately** - Not just held-out samples, but held-out *vendors*. This catches generalization failures that random splits miss.

3. **Plan for template changes from day one** - With fine-tuning, vendor template changes require retraining. Build monitoring and retraining pipeline upfront, not after first incident.

---

## 💬 Key Takeaway

> "The model was maybe 30% of this project. The rest was preprocessing, validation logic, monitoring, and maintenance. The unsexy work - rotation correction, class-level metrics, vendor-specific monitoring, retraining pipelines - is what separates demos from production."

Fine-tuning gave us better accuracy than RAG ever could. It also gave us operational overhead RAG didn't have. Every template change means collecting new data and retraining. 

**For us, the trade-off was worth it:** 99% accuracy vs 91%, plus data privacy and lower per-inference costs.

---

## 👤 About

Built by **Vaibhav Rathi** while at R Systems (2021-2025), working with a US healthcare provider.

**Background:** Senior Data Scientist at Fractal Analytics with 8+ years building production ML/LLM systems. Previous work includes content generation pipelines (4x throughput), multi-agent healthcare systems (50% efficiency gains), and recommendation engines (86% CTR improvement).

**Currently:** Building LLM-powered automation systems for enterprise clients at Fractal Analytics.

## 📧 Connect

- **LinkedIn:** [linkedin.com/in/vaibhav-rathi-ai](https://linkedin.com/in/vaibhav-rathi-ai)
- **Email:** vaibhav.rathi1223@gmail.com
- **Medium:** Technical writing on production ML systems
- **Open to:** Staff Data Scientist / Lead Data Scientist / Senior MLE roles

---

## ⭐ Support This Work

If you found this documentation useful, please **star this repository**! It helps others discover production ML patterns and best practices.

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

**Disclaimer:** This repository documents the architecture and approach used in a production system. Code examples are illustrative and have been anonymized to respect client confidentiality. All metrics and technical details are from actual production deployment.
