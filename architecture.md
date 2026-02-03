# Architecture Deep Dive

## Table of Contents
1. [High-Level System Design](#high-level-system-design)
2. [Component Selection Rationale](#component-selection-rationale)
3. [Data Flow](#data-flow)
4. [Key Technical Decisions](#key-technical-decisions)
5. [System Requirements](#system-requirements)
6. [Performance Characteristics](#performance-characteristics)

---

## High-Level System Design

### Evolution: From RAG to Fine-Tuning

**Phase 1: RAG-Based System (Months 1-3)**
- Azure OpenAI GPT-3.5
- Vendor templates in vector database
- In-context learning with template examples
- **Result:** 91-93% accuracy ceiling

**Phase 2: Fine-Tuned System (Months 4-9)**
- Self-hosted LLaMA 3.2
- QLoRA fine-tuning on invoice-output pairs
- Domain-specific medical NER
- **Result:** 99% accuracy sustained

### Why We Rebuilt

Three critical issues with RAG approach:

**1. Accuracy Plateau**
```
Month 1: 91% accuracy (initial RAG deployment)
Month 2: 92% accuracy (better templates)
Month 3: 93% accuracy (extensive prompt engineering)
Month 4: Still 93% (hit ceiling)
```
- At 10,000 invoices/month: 700 invoices with errors
- 700 invoices requiring manual review/correction
- Automation wasn't as "automatic" as it seemed

**2. Healthcare Compliance**
- Patient-adjacent information in invoices
- Every API call sent data to external service (Azure)
- Compliance team increasingly concerned
- HIPAA implications of external data processing

**3. Cost at Scale**
- Per-request API costs adding up
- 10K invoices × 2 API calls each = 20K requests/month
- Recurring monthly expense vs one-time training cost

---

## Component Selection Rationale

### 1. OCR Engine: PaddleOCR

**Benchmark Test:** 500 real healthcare invoices

**Test Conditions:**
- Mix of scan qualities (high-res, fax-quality, photos)
- Various layouts (single-column, multi-column, tables)
- Different invoice types (itemized, summary, mixed)

**Results:**

| Engine | Accuracy | Multi-Column | Tables | Speed | Learning Curve |
|--------|----------|--------------|--------|-------|----------------|
| Tesseract 5.0 | 76% | Poor | Poor | 0.3s/page | Easy |
| EasyOCR | 79% | Medium | Medium | 1.2s/page | Easy |
| **PaddleOCR** | **88%** | **Excellent** | **Excellent** | 0.7s/page | **Steep** |

**Specific PaddleOCR Advantages:**
- Better at multi-column invoice layouts (didn't merge columns like Tesseract)
- Superior table boundary detection
- Handled rotated text better
- Better language model for medical terminology

**Trade-offs Accepted:**
- Steeper learning curve (less community support)
- Slightly slower than Tesseract
- More complex configuration

**Decision:** 12% accuracy gain worth the complexity.

---

### 2. Medical NER: BioBERT

**Why Domain-Specific NER Was Non-Negotiable**

**Test with General-Purpose NER (spaCy, BERT-base):**

Example invoice line:
```
"Metformin 500mg - qty 90 - $45.00 - NDC 12345-678-90"
```

**General NER output:**
```
Entity: "Metformin 500mg" (single blob, no separation)
Dosage: Not detected
NDC code: Missed entirely
```

**BioBERT output:**
```
Drug: "Metformin"
Dosage: "500mg"
Quantity: "90"
NDC: "12345-678-90"
```

**Real Impact:**
- CPT codes (Current Procedural Terminology): General models missed 47%
- Drug names with unusual spellings: General models 31% error rate
- Compounded medications: General models couldn't parse at all

**BioBERT Advantages:**
- Pre-trained on PubMed abstracts (medical literature)
- Understood clinical terminology out of the box
- Recognized drug names, procedure codes, medical abbreviations
- No fine-tuning needed for basic medical entity recognition

**Decision:** General-purpose NER was "hopeless" on healthcare terms. BioBERT was the difference between working and not working.

---

### 3. LLM: LLaMA 3.2 with QLoRA Fine-Tuning

**Model Selection Process:**

**Options Considered:**
1. Continue with GPT-3.5 API (status quo)
2. GPT-4 API (more expensive, more accurate)
3. Self-hosted open-source LLM (LLaMA, Mistral, etc.)

**Decision Matrix:**

| Option | Accuracy | Privacy | Cost (10K/mo) | Maintenance | Latency |
|--------|----------|---------|---------------|-------------|---------|
| GPT-3.5 API | 91-93% | External ❌ | ~$400/mo | Easy | <1s |
| GPT-4 API | ~95% (est) | External ❌ | ~$4000/mo | Easy | 1-2s |
| **Self-hosted LLaMA** | **99%** | **On-prem ✅** | **One-time** | **Complex** | **2-3s** |

**Why LLaMA 3.2 Specifically:**
- 3B parameters (sweet spot for T4 GPU inference)
- Strong instruction-following capabilities
- Proven performance on structured extraction tasks
- Open weights (full control, on-premises deployment)

**Fine-Tuning vs RAG:**

```
RAG Approach:
- Model sees templates in context
- Generalizes from examples
- Limited by context window
- Accuracy ceiling: 93%

Fine-Tuning Approach:
- Model learns task directly
- Parameters updated with task-specific data
- No context window limitation
- Achieved: 99% accuracy
```

---

### 4. QLoRA vs Full Fine-Tuning

**GPU Cost Comparison:**

| Approach | GPU Required | Cloud Cost | Training Time | Accuracy |
|----------|-------------|------------|---------------|----------|
| Full Fine-tuning | A100 (40GB) | ~$200/hr | 8 hours | 99.2% |
| **QLoRA** | **T4 (16GB)** | **~$30/hr** | **8 hours** | **99.0%** |

**Cost Calculation:**
- Full fine-tuning: 8 hours × $200 = **$1,600**
- QLoRA: 8 hours × $30 = **$240**
- **Savings: $1,360 (85% reduction)**

**Accuracy Trade-off:**
- Full fine-tuning: 99.2%
- QLoRA: 99.0%
- **Difference: 0.2% (2 in 1000 invoices)**

**Decision:** 0.2% accuracy loss not worth 7x cost increase.

**QLoRA Hyperparameters (After Experimentation):**

Tested LoRA ranks: 4, 8, 16, 32

| Rank | Params Trainable | Accuracy | Training Time | Inference Speed |
|------|------------------|----------|---------------|-----------------|
| 4 | 0.05% | 97.1% | 6 hrs | Fast |
| **8** | **0.1%** | **99.0%** | **8 hrs** | **Fast** |
| 16 | 0.2% | 99.1% | 10 hrs | Medium |
| 32 | 0.4% | 98.8% (overfitting!) | 12 hrs | Slow |

**Rank 8 was the sweet spot:**
- Rank 4: Underfitted on complex line items (bundled services)
- Rank 16+: Diminishing returns on accuracy, slower inference
- Rank 32: Actually started overfitting (validation loss increased)

**Final QLoRA Configuration:**
```python
# LoRA Config
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj"]

# Training
epochs = 3
batch_size = 4
learning_rate = 1e-4
warmup_steps = 100
gradient_accumulation_steps = 4
```

---

### 5. Three-Tier Validation Layer

**Why Validation Was Critical:**

Even at 99% model accuracy, we needed validation because:
1. Healthcare invoices are high-stakes (billing disputes, compliance)
2. Some errors are obvious (e.g., invoice date in future)
3. Confidence scores help route edge cases to humans

**Tier 1: Format Validation (Regex-based)**

```python
# Example validation rules
validation_rules = {
    'invoice_number': r'^INV-\d{6,8}$',
    'date': r'^\d{2}/\d{2}/\d{4}$',
    'amount': r'^\$?\d{1,6}\.\d{2}$',
    'ndc_code': r'^\d{5}-\d{3}-\d{2}$',
}
```

**Catches:**
- Malformed invoice numbers
- Invalid date formats
- Amounts without decimal places
- Incorrect NDC code structure

**Impact:** Caught 3% of extraction errors

---

**Tier 2: Confidence Scoring**

**Threshold Tuning:**

Tested thresholds from 0.85 to 0.99 on validation set:

| Threshold | Auto-Processed | Human Review | Final Accuracy |
|-----------|----------------|--------------|----------------|
| 0.85 | 92% | 8% | 97.2% |
| 0.90 | 85% | 15% | 98.5% |
| **0.95** | **77%** | **23%** | **99.0%** |
| 0.97 | 68% | 32% | 99.1% |
| 0.99 | 51% | 49% | 99.2% |

**Decision:** 0.95 threshold
- 77% fully automated (acceptable automation rate)
- 23% routed to human review (manageable volume)
- 99% final accuracy (target achieved)

**Confidence Score Distribution (Production):**

```
0.99-1.00: 45% of invoices (very high confidence)
0.95-0.99: 32% of invoices (auto-processed)
0.90-0.95: 15% of invoices (human review)
0.00-0.90: 8% of invoices (human review + flagged)
```

---

**Tier 3: Business Rules Validation**

```python
# Example business rules
def validate_business_rules(invoice):
    checks = []
    
    # Total should equal sum of line items
    calculated_total = sum(item.amount for item in invoice.line_items)
    if abs(invoice.total - calculated_total) > 0.01:
        checks.append("Total mismatch")
    
    # Date should not be in future
    if invoice.date > datetime.now():
        checks.append("Future date")
    
    # CPT codes should be valid for provider type
    for code in invoice.cpt_codes:
        if not is_valid_cpt_for_provider(code, invoice.provider_type):
            checks.append(f"Invalid CPT: {code}")
    
    # Quantity should be positive
    for item in invoice.line_items:
        if item.quantity <= 0:
            checks.append("Invalid quantity")
    
    return checks
```

**Impact:** Caught 2% of errors that passed format and confidence checks

---

**Combined Validation Results:**

```
Model extraction: 99.0% accuracy
+ Tier 1 (Format): +0.3% (caught malformed fields)
+ Tier 2 (Confidence): Routed 23% to review (prevented errors)
+ Tier 3 (Business): +0.2% (caught inconsistencies)
= Final system: 99.5% effective accuracy in production
```

---

## Data Flow

### End-to-End Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT: Invoice PDF/Image                                │
│    - From email attachments                                 │
│    - Scanned documents                                      │
│    - Faxes (converted to PDF)                               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING                                            │
│    - Rotation correction (deskew)                           │
│    - Adaptive thresholding (enhance contrast)               │
│    - Region detection (separate handwritten from printed)   │
│                                                              │
│    Impact: +15% accuracy improvement                        │
│    Lesson: OCR is half the problem, preprocessing critical  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. OCR: PaddleOCR                                           │
│    - Text extraction                                        │
│    - Multi-column layout handling                           │
│    - Table structure detection                              │
│                                                              │
│    Output: Raw text with positional information             │
│    Accuracy: 88% (text correctly extracted)                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. MEDICAL NER: BioBERT                                     │
│    - Drug name recognition                                  │
│    - CPT/ICD/NDC code extraction                            │
│    - Clinical term parsing                                  │
│                                                              │
│    Output: Tagged entities with medical context             │
│    Why needed: General NER hopeless on medical terms        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. STRUCTURED EXTRACTION: LLaMA 3.2 (QLoRA)                │
│    - Extract fields into JSON schema                        │
│    - Field-level confidence scores                          │
│    - Handle complex line items                              │
│                                                              │
│    Input: OCR text + BioBERT entities                       │
│    Output: Structured JSON + confidence per field           │
│    Accuracy: 99% field-level                                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. VALIDATION LAYER                                         │
│    ┌─────────────────────────────────────────────┐          │
│    │ Tier 1: Format Validation (Regex)          │          │
│    │ - Invoice number format                    │          │
│    │ - Date format                              │          │
│    │ - Amount format                            │          │
│    │ Catches: ~3% of errors                     │          │
│    └─────────────────────────────────────────────┘          │
│                        ▼                                     │
│    ┌─────────────────────────────────────────────┐          │
│    │ Tier 2: Confidence Scoring                 │          │
│    │ - Threshold: 0.95                          │          │
│    │ - Above 0.95: Auto-process (77%)           │          │
│    │ - Below 0.95: Human review (23%)           │          │
│    └─────────────────────────────────────────────┘          │
│                        ▼                                     │
│    ┌─────────────────────────────────────────────┐          │
│    │ Tier 3: Business Rules                     │          │
│    │ - Total = sum of line items?               │          │
│    │ - Valid CPT codes for provider?            │          │
│    │ - Date not in future?                      │          │
│    │ Catches: ~2% of remaining errors           │          │
│    └─────────────────────────────────────────────┘          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. OUTPUT ROUTING                                           │
│                                                              │
│   ┌─────────────────┐              ┌─────────────────┐      │
│   │ Auto-Processed  │              │ Human Review    │      │
│   │ (77%)           │              │ Queue (23%)     │      │
│   │                 │              │                 │      │
│   │ - Conf > 0.95   │              │ - Conf < 0.95   │      │
│   │ - All validations│              │ - Validation   │      │
│   │   passed         │              │   failures      │      │
│   │                 │              │                 │      │
│   │ → PostgreSQL    │              │ → Review UI     │      │
│   │ → Downstream    │              │ → Corrections   │      │
│   │   systems       │              │ → Retraining    │      │
│   └─────────────────┘              └─────────────────┘      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Processing Time Breakdown

**Per Invoice (Average):**

| Stage | Time | % of Total |
|-------|------|------------|
| Preprocessing | 0.2s | 10% |
| OCR (PaddleOCR) | 0.7s | 35% |
| BioBERT NER | 0.3s | 15% |
| LLaMA extraction | 0.6s | 30% |
| Validation | 0.1s | 5% |
| Database write | 0.1s | 5% |
| **Total** | **2.0s** | **100%** |

**Before (Manual):** 8+ minutes per invoice  
**After (Automated):** <2 seconds per invoice  
**Speedup:** 240x faster (99.6% time reduction)

---

## Key Technical Decisions

### Decision Log

#### Decision 1: RAG → Fine-Tuning
**Date:** Month 4  
**Context:** RAG hit 93% accuracy ceiling, couldn't improve further

**Options Considered:**
1. Keep RAG, accept 93% accuracy
2. Try GPT-4 API (more expensive)
3. Rebuild with fine-tuned open-source model

**Decision:** Fine-tune LLaMA 3.2 with QLoRA

**Rationale:**
- RAG ceiling: 93% → Fine-tuning: 99% (+6% gain)
- Healthcare compliance: On-premises data
- Cost: One-time training vs recurring API fees
- Control: Full model ownership

**Trade-offs:**
- Initial setup time: 4 weeks
- Operational overhead: Retraining for template changes
- Infrastructure: GPU for training, CPU for inference

**Result:** 8% accuracy improvement, compliance achieved, costs reduced

---

#### Decision 2: PaddleOCR Over Tesseract
**Date:** Month 1  
**Context:** Need OCR engine for text extraction

**Options Considered:**
1. Tesseract (most popular, easy)
2. EasyOCR (moderate accuracy)
3. PaddleOCR (high accuracy, complex)
4. Cloud OCR APIs (Google Vision, AWS Textract)

**Decision:** PaddleOCR

**Rationale:**
- 12% accuracy advantage over Tesseract on test set
- Superior multi-column and table handling
- No external API dependencies (healthcare compliance)

**Trade-offs:**
- Steeper learning curve
- Less community support than Tesseract
- Slightly slower (0.7s vs 0.3s per page)

**Result:** 88% OCR accuracy vs 76% with Tesseract

---

#### Decision 3: BioBERT for Medical NER
**Date:** Month 2  
**Context:** General NER models failing on medical terminology

**Options Considered:**
1. spaCy (general-purpose)
2. BERT-base (general-purpose)
3. BioBERT (medical domain)
4. Custom NER training

**Decision:** BioBERT

**Rationale:**
- Pre-trained on medical literature (PubMed)
- Understood drug names, CPT codes, medical terms out-of-box
- General models "hopeless" on healthcare terminology
- No fine-tuning needed

**Trade-offs:**
- Larger model size than general NER
- Medical domain specific (not reusable for other projects)

**Result:** Medical entity recognition essential for system to work

---

#### Decision 4: QLoRA vs Full Fine-Tuning
**Date:** Month 4  
**Context:** GPU budget constraints (no A100 access)

**Options Considered:**
1. Full fine-tuning (A100 required)
2. LoRA (moderate efficiency)
3. QLoRA (maximum efficiency)
4. Adapter-based fine-tuning

**Decision:** QLoRA with rank 8

**Rationale:**
- 7x cheaper than full fine-tuning ($240 vs $1,600)
- Only 0.2% accuracy loss (99.0% vs 99.2%)
- Fits on T4 GPU (accessible, affordable)
- 0.1% of parameters trainable (fast, efficient)

**Trade-offs:**
- Slightly lower accuracy than full fine-tuning
- Required careful hyperparameter tuning
- Less community knowledge than standard fine-tuning

**Result:** 99% accuracy achieved within GPU budget

---

#### Decision 5: Confidence Threshold 0.95
**Date:** Month 5  
**Context:** Balancing automation vs accuracy

**Options Considered:**
- 0.85 (92% auto, 97.2% accuracy)
- 0.90 (85% auto, 98.5% accuracy)
- 0.95 (77% auto, 99.0% accuracy)
- 0.97 (68% auto, 99.1% accuracy)

**Decision:** 0.95 threshold

**Rationale:**
- 77% automation acceptable (7,700 invoices/month)
- 23% human review manageable (2,300 invoices/month)
- 99% accuracy target achieved
- Higher threshold = diminishing returns

**Trade-offs:**
- 23% still need review (not fully automated)
- Review team still required
- Balance between automation and accuracy

**Result:** 99% final accuracy with 77% automation

---

## System Requirements

### Hardware

**Development/Training:**
- GPU: NVIDIA T4 (16GB VRAM) or better
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 100GB (models, datasets, logs)

**Production/Inference:**
- CPU inference acceptable (batch processing)
- 16+ cores recommended for throughput
- RAM: 64GB+ for model loading
- Storage: 50GB (models, audit trail)

**Scaling Considerations:**
- CPU inference: ~2s per invoice
- At 10K/month: Can process in ~5.5 hours (nightly batch)
- GPU inference: ~0.5s per invoice (if real-time needed)

### Software Stack

**Core Dependencies:**
```
python>=3.10
torch>=2.0.0
transformers>=4.30.0
paddleocr>=2.7.0
peft>=0.4.0
bitsandbytes>=0.41.0  # For QLoRA
```

**Supporting Libraries:**
```
opencv-python>=4.8.0  # Image preprocessing
pillow>=10.0.0        # Image handling
psycopg2>=2.9.0       # PostgreSQL connector
fastapi>=0.100.0      # API server (if needed)
```

**Infrastructure:**
```
PostgreSQL 14+        # Database
Docker + Kubernetes   # Deployment
Prometheus + Grafana  # Monitoring
```

---

## Performance Characteristics

### Accuracy Metrics

**Overall System:**
- Field-level accuracy: 99.0%
- End-to-end accuracy (all fields correct): 97.0%
- Human review rate: 23%

**Per Field Type:**

| Field | Accuracy | Avg Confidence | Notes |
|-------|----------|----------------|-------|
| Invoice Number | 99.8% | 0.98 | Highly structured |
| Date | 99.5% | 0.97 | Multiple formats handled |
| Total Amount | 99.2% | 0.96 | Critical field |
| Vendor Details | 99.1% | 0.96 | Address parsing |
| Line Items | 98.5% | 0.94 | Most complex |
| CPT/ICD Codes | 98.8% | 0.95 | BioBERT essential |

**Minority Class Performance:**

| Class | % of Data | F1 Score | Notes |
|-------|-----------|----------|-------|
| Standard line items | 88% | 0.98 | Easy cases |
| Compound medications | 4% | 0.81 | Weighted loss helped |
| DME rentals | 4% | 0.83 | Multi-line spanning |
| Bundled services | 4% | 0.79 | Complex grouping |

---

### Production Stability (6 Months)

**Uptime & Reliability:**
- System uptime: 99.9%
- Zero critical failures
- Zero rollbacks
- Mean time to resolution (issues): <2 hours

**Accuracy Over Time:**
```
Month 1: 95% (initial deployment, learning phase)
Month 2: 97% (first retraining with corrections)
Month 3: 98% (second retraining, incident handling)
Month 4: 99% (stable performance achieved)
Month 5: 99% (maintained)
Month 6: 99% (maintained)
```

**Vendor Coverage:**
- Started: 15 vendor formats in training
- Month 6: 32 vendor formats learned
- New vendor handling: <1 week to production quality

---

### Cost Analysis

**One-Time Costs:**
- Development: 4 weeks engineering time
- Training: $240 (8 hours T4 GPU)
- Testing & validation: 2 weeks
- **Total initial investment:** ~6 weeks + $240

**Recurring Costs (Monthly):**
- Infrastructure: CPU servers for inference
- Database: PostgreSQL managed service
- Monitoring: Prometheus + Grafana
- Retraining: Quarterly ($240 × 4 = $960/year)

**Cost Comparison (10K invoices/month):**

| Approach | Monthly Cost | Annual Cost |
|----------|--------------|-------------|
| Manual processing | Baseline | Baseline |
| GPT-3.5 API | ~$400 | ~$4,800 |
| Fine-tuned LLaMA | One-time $240 | $960 (retraining) |

**Savings:** $340K annually in processing costs

---

### Scaling Characteristics

**Current Load:** 10,000 invoices/month

**Batch Processing (CPU):**
- Processing rate: 1,800 invoices/hour (2s each)
- Nightly batch: Complete in 5.5 hours
- Headroom: Can scale to 30K/month with current setup

**Real-Time Processing (GPU):**
- Processing rate: 7,200 invoices/hour (0.5s each)
- Can handle peak loads and real-time requests
- More expensive infrastructure

**Horizontal Scaling:**
- Stateless design (can parallelize)
- Add workers to increase throughput
- Database becomes bottleneck >50K/month

---

## Conclusion

This architecture represents 6 months of production evolution. Key insights:

1. **Preprocessing matters more than model choice** - 15% accuracy gain from image quality
2. **Domain-specific components essential** - BioBERT non-negotiable for medical terms
3. **Fine-tuning breaks RAG ceiling** - 91% → 99% accuracy gain
4. **Validation layers catch model errors** - 3-tier validation critical
5. **Monitor vendor-level metrics** - Aggregate accuracy hides problems

The system is 30% model, 70% everything else - preprocessing, validation, monitoring, and maintenance.
