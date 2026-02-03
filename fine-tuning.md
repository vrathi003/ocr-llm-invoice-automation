# Fine-Tuning Guide: QLoRA Implementation

## Table of Contents
1. [Why Fine-Tuning Over RAG](#why-fine-tuning-over-rag)
2. [QLoRA vs Full Fine-Tuning](#qlora-vs-full-fine-tuning)
3. [Training Data Preparation](#training-data-preparation)
4. [QLoRA Configuration](#qlora-configuration)
5. [Training Process](#training-process)
6. [Evaluation Methodology](#evaluation-methodology)
7. [Production Deployment](#production-deployment)

---

## Why Fine-Tuning Over RAG

### The RAG Ceiling

Our initial RAG-based system achieved 91-93% accuracy but couldn't improve further:

**Month-by-Month RAG Performance:**
```
Month 1: 91% (initial deployment)
  ↓ Added better templates
Month 2: 92% (+1%)
  ↓ Extensive prompt engineering
Month 3: 93% (+1%)
  ↓ More examples, better chunking
Month 4: Still 93% (hit ceiling)
```

**Why RAG Hit a Ceiling:**

1. **Template Generalization Problem**
   - RAG: "Here's Vendor A's template, extract from this invoice"
   - Works when invoice matches template exactly
   - Fails on layout variations, unexpected fields, format changes
   
2. **Context Window Limitations**
   - Maximum examples in context: ~10-15 templates
   - 32 vendor formats in production
   - Can't fit all variations in context
   
3. **No Parameter Updates**
   - Model parameters frozen
   - Learning happens only through in-context examples
   - Can't deeply learn invoice extraction patterns

### Why Fine-Tuning Broke Through

**Fine-Tuning Advantages:**

1. **Direct Task Learning**
   - Parameters updated specifically for invoice extraction
   - Model learns patterns across all training examples
   - Not limited by context window
   
2. **Better Generalization**
   - Learns underlying structure, not specific templates
   - Generalizes to new vendors better
   - Cross-vendor validation gap: 15% (RAG) → 4% (fine-tuned)
   
3. **Accuracy Improvement**
   - RAG ceiling: 93%
   - Fine-tuned: 99%
   - **+6 percentage point improvement**

**Trade-offs Accepted:**

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Setup Time** | Immediate | 4 weeks |
| **Accuracy** | 91-93% | 99% |
| **Template Changes** | 30 min update | Retraining cycle |
| **Data Privacy** | External API | On-premises ✅ |
| **Cost** | Per-request | One-time training |

**Decision:** 6% accuracy gain + privacy + cost savings worth the operational overhead.

---

## QLoRA vs Full Fine-Tuning

### GPU Requirements Comparison

**Full Fine-Tuning:**
- Requires: NVIDIA A100 (40GB VRAM)
- All model parameters updated
- Memory intensive (loads full precision model)
- Cloud cost: ~$200/hour
- Training time: 8 hours
- **Total cost: $1,600**

**QLoRA (Quantized Low-Rank Adaptation):**
- Requires: NVIDIA T4 (16GB VRAM)
- Only low-rank adapters updated (0.1% of parameters)
- Quantizes base model to 4-bit (memory efficient)
- Cloud cost: ~$30/hour
- Training time: 8 hours
- **Total cost: $240**

**Cost Savings: 85% ($1,360)**

---

### Accuracy Comparison

We tested both approaches on same dataset (1000 invoices):

| Approach | Field Accuracy | End-to-End Accuracy | Inference Speed |
|----------|----------------|---------------------|-----------------|
| Full Fine-tuning | 99.2% | 97.2% | 0.6s |
| **QLoRA** | **99.0%** | **97.0%** | **0.6s** |

**Accuracy Gap: 0.2%** (2 additional errors per 1000 invoices)

**Decision:** 0.2% accuracy loss not worth 7x cost increase.

---

### How QLoRA Works

**Standard Fine-Tuning:**
```
Base Model (3B parameters)
     ↓ Update all parameters
Fine-Tuned Model (3B parameters)
```

**QLoRA:**
```
Base Model (3B parameters, quantized to 4-bit)
     ↓ Freeze base model
     ↓ Add trainable low-rank adapters
Base Model + LoRA Adapters (3B + 3M = 0.1% trainable)
```

**Key Innovations:**
1. **4-bit Quantization:** Base model stored in 4-bit (reduces memory 4x)
2. **Low-Rank Adapters:** Only adapters trained (reduces trainable params ~1000x)
3. **Same Output Quality:** Minimal accuracy loss vs full fine-tuning

---

### LoRA Rank Experimentation

We tested ranks from 4 to 32 to find optimal trade-off:

| Rank | Params Trainable | Training Time | Accuracy | Inference Speed | Overfitting? |
|------|------------------|---------------|----------|-----------------|--------------|
| 4 | 1.5M (0.05%) | 6 hrs | 97.1% | Fast ✅ | No, underfitting |
| **8** | **3M (0.1%)** | **8 hrs** | **99.0%** | **Fast ✅** | **No** |
| 16 | 6M (0.2%) | 10 hrs | 99.1% | Medium | No |
| 32 | 12M (0.4%) | 12 hrs | 98.8% | Slow ❌ | Yes ⚠️ |

**Findings:**

**Rank 4 (Too Low):**
- Underfitted on complex cases
- Couldn't handle bundled service codes
- Simple line items: OK
- Complex line items: Poor

**Rank 8 (Sweet Spot):**
- Best accuracy (99.0%)
- Fast inference
- No overfitting
- Good on all invoice types

**Rank 16 (Diminishing Returns):**
- Only 0.1% better than rank 8
- 25% slower training
- Slower inference
- Not worth the cost

**Rank 32 (Overfitting):**
- Validation loss increased (sign of overfitting)
- Memorized training examples
- Poor generalization to new vendors
- Slowest inference

**Decision: Rank 8** - Best balance of accuracy, speed, and generalization.

---

## Training Data Preparation

### Data Collection

**Sources:**
- Historical invoices: 8,000 invoices (past 2 years)
- Current production: 2,000 invoices (recent months)
- **Total: 10,000 invoices**

**Vendor Distribution:**
- 15 major vendors: 7,500 invoices (75%)
- 25 minor vendors: 2,500 invoices (25%)
- Ensures diverse format coverage

---

### Labeling Process

**Initial Labeling (Manual):**
- Week 1-2: Labeled 500 invoices manually
- Domain expert (medical billing) + ML engineer
- Created labeling guidelines document
- Established schema for structured output

**Schema Example:**
```json
{
  "invoice_number": "INV-12345678",
  "date": "2023-09-15",
  "vendor": {
    "name": "Medical Supplies Inc",
    "address": "123 Health St, Boston MA 02101"
  },
  "line_items": [
    {
      "description": "Metformin 500mg",
      "quantity": 90,
      "unit_price": 0.50,
      "total": 45.00,
      "ndc_code": "12345-678-90",
      "cpt_code": null
    }
  ],
  "subtotal": 45.00,
  "tax": 0.00,
  "total": 45.00
}
```

**Scaled Labeling (Semi-Automated):**
- Weeks 3-6: Labeled remaining 9,500 invoices
- Used RAG system to generate initial labels
- Human reviewed and corrected
- Much faster than full manual labeling

**Quality Control:**
- 10% double-labeled (two independent labelers)
- Inter-annotator agreement: 97%
- Disagreements reviewed by senior medical biller

---

### Data Splits

**Training Set:** 8,000 invoices (80%)
- All major vendors represented
- Stratified by vendor (ensure proportional distribution)

**Validation Set:** 1,000 invoices (10%)
- Same vendor distribution as training
- Used for hyperparameter tuning
- Monitored for overfitting

**Test Set:** 1,000 invoices (10%)
- **Critical:** Held-out vendors (not in training)
- Tests cross-vendor generalization
- Never seen during training

**Why Held-Out Vendors in Test Set:**

Week 4 taught us this lesson the hard way:
- Random split: 96% accuracy (great!)
- Held-out vendors: 81% accuracy (disaster!)
- Model had memorized vendor formats

**New Split Strategy:**
- Training: Vendors A-L (12 vendors)
- Validation: Mix of A-L + M-O (15 vendors)
- Test: Vendors P-Z (completely new, 5 vendors)

Result: More realistic accuracy estimate, caught generalization issues early.

---

### Data Augmentation

**Challenges:**
- Limited data from some vendors
- Need diversity to prevent overfitting
- Real-world variability (scan quality, rotations, etc.)

**Augmentation Techniques:**

1. **OCR Noise Injection**
   - Simulated OCR errors (character swaps)
   - Added scanning artifacts
   - Purpose: Robustness to poor scan quality
   
2. **Field Permutation**
   - Swapped field positions
   - Purpose: Prevent position-based memorization
   
3. **Vendor Anonymization**
   - Replaced vendor names with placeholders
   - Purpose: Force model to learn structure, not vendor names

**Impact:** Validation accuracy improved 2% with augmentation.

---

## QLoRA Configuration

### Final Configuration

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load base model (quantized)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# LoRA Config
lora_config = LoraConfig(
    r=8,                          # LoRA rank (sweet spot from experiments)
    lora_alpha=16,                # Scaling factor (2 * rank)
    lora_dropout=0.05,            # Dropout for regularization
    bias="none",                  # Don't train biases
    task_type="CAUSAL_LM",        # Causal language modeling
    target_modules=[              # Which modules to add adapters to
        "q_proj",                 # Query projection
        "v_proj",                 # Value projection
    ],
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)

# Check trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

# Output: Trainable: 3,145,728 / 3,007,823,872 (0.10%)
```

---

### Hyperparameter Rationale

**LoRA Rank (r=8):**
- Lower rank: Underfits complex patterns
- Higher rank: Overfits, slower inference
- 8 = Sweet spot from experiments

**LoRA Alpha (16):**
- Scaling factor for adapter outputs
- Rule of thumb: 2 × rank
- Controls adapter influence on base model

**LoRA Dropout (0.05):**
- Regularization to prevent overfitting
- 5% dropout during training
- Tested 0.0, 0.05, 0.1 (0.05 worked best)

**Target Modules (q_proj, v_proj):**
- Only attention query and value projections
- Most impactful for our task
- Adding k_proj, o_proj didn't improve accuracy

---

## Training Process

### Training Configuration

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    # Output
    output_dir="./llama-invoice-qlora",
    
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # Effective batch size: 4 * 4 = 16
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    
    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=8,
    
    # Logging
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    
    # Hardware
    fp16=True,                         # Mixed precision training
    dataloader_num_workers=4,
)
```

### Training Duration

**Hardware:** Single T4 GPU (16GB VRAM)

**Timeline:**
- Epoch 1: 2.5 hours (learning, high loss)
- Epoch 2: 2.5 hours (improving, loss decreasing)
- Epoch 3: 2.5 hours (fine-tuning, loss plateau)
- **Total: ~8 hours**

**Why 3 Epochs:**
- Epoch 1: Validation loss still decreasing
- Epoch 2: Validation loss still improving (slower)
- Epoch 3: Validation loss plateaus
- Epoch 4: Validation loss starts increasing (overfitting)

**Epoch Selection:**
```
Epoch  Train Loss  Val Loss  Val Accuracy
  1      0.42       0.38        95.2%
  2      0.28       0.25        97.8%
  3      0.19       0.22        99.0%  ← Selected
  4      0.15       0.24        98.5%  (overfitting)
```

---

### Learning Rate Scheduling

**Cosine Schedule with Warmup:**

```
Learning Rate
  │
1e-4│     ╱──╲
    │    ╱    ╲
    │   ╱      ╲___
    │  ╱           ╲___
    │ ╱                ╲___
0   │╱                     ╲___
    └───────────────────────────→ Steps
    0   100          4000        8000

    Warmup  Peak    Decay       End
```

**Why Cosine:**
- Gradual warmup prevents instability
- Cosine decay better than linear
- Final low LR fine-tunes without disrupting

**Why 1e-4 Peak:**
- Too high (1e-3): Training unstable, loss spikes
- Too low (1e-5): Training too slow, doesn't converge
- 1e-4: Goldilocks zone

---

### Gradient Accumulation

**Challenge:** T4 GPU only fits batch size 4

**Solution:** Accumulate gradients over 4 steps

```
Step 1: Forward pass, compute gradients (don't update)
Step 2: Forward pass, accumulate gradients
Step 3: Forward pass, accumulate gradients
Step 4: Forward pass, accumulate gradients, UPDATE
```

**Effective batch size:** 4 (device) × 4 (accumulation) = 16

**Why 16:**
- Smaller batch: Noisy gradients, unstable
- Larger batch: Slower convergence
- 16: Good balance

---

## Evaluation Methodology

### Metrics

**Field-Level Accuracy:**
```python
def field_level_accuracy(predictions, ground_truth):
    """
    Compare each field individually
    """
    correct_fields = 0
    total_fields = 0
    
    for pred, gt in zip(predictions, ground_truth):
        # Invoice number
        if pred['invoice_number'] == gt['invoice_number']:
            correct_fields += 1
        total_fields += 1
        
        # Date
        if pred['date'] == gt['date']:
            correct_fields += 1
        total_fields += 1
        
        # Total amount
        if abs(pred['total'] - gt['total']) < 0.01:  # Floating point tolerance
            correct_fields += 1
        total_fields += 1
        
        # Line items (more complex)
        for pred_item, gt_item in zip(pred['line_items'], gt['line_items']):
            if pred_item['description'] == gt_item['description']:
                correct_fields += 1
            total_fields += 1
            # ... (check other line item fields)
    
    return correct_fields / total_fields
```

**Result:** 99.0% field-level accuracy

---

**End-to-End Accuracy:**
```python
def end_to_end_accuracy(predictions, ground_truth):
    """
    Invoice is correct only if ALL fields match
    """
    correct_invoices = 0
    
    for pred, gt in zip(predictions, ground_truth):
        if invoices_match(pred, gt):  # Every field correct
            correct_invoices += 1
    
    return correct_invoices / len(predictions)
```

**Result:** 97.0% end-to-end accuracy

---

**F1 Score (Per Class):**

For minority classes (compound medications, DME, bundled services):

```python
from sklearn.metrics import f1_score, classification_report

# Extract line item types
y_true = [item['type'] for invoice in ground_truth for item in invoice['line_items']]
y_pred = [item['type'] for invoice in predictions for item in invoice['line_items']]

print(classification_report(y_true, y_pred))

"""
                          precision  recall  f1-score  support
standard_medication          0.99     0.99      0.99     8800
compound_medication          0.82     0.80      0.81      400
dme_rental                   0.85     0.81      0.83      400
bundled_service              0.78     0.80      0.79      400

                accuracy                       0.97    10000
               macro avg       0.86     0.85     0.86    10000
            weighted avg       0.97     0.97     0.97    10000
"""
```

**Minority Class Improvement:**
- Before weighted loss: F1 = 0.68
- After weighted loss: F1 = 0.81
- **+13 points improvement**

---

### Cross-Vendor Evaluation

**Critical Test:** How well does model generalize to unseen vendors?

**Held-Out Vendor Test Set:**
- 5 vendors never seen in training
- 1,000 invoices total
- True measure of generalization

**Results:**

| Vendor | Invoices | Field Accuracy | Notes |
|--------|----------|----------------|-------|
| Vendor P | 300 | 97.2% | Similar layout to training vendors |
| Vendor Q | 250 | 95.8% | Very different layout |
| Vendor R | 200 | 98.1% | Standard format |
| Vendor S | 150 | 93.5% | Complex bundled services |
| Vendor T | 100 | 96.4% | Handwritten annotations |
| **Average** | **1000** | **96.2%** | **4% gap from training** |

**Cross-Vendor Gap:**
- Training vendors: 99.0%
- Held-out vendors: 96.2%
- **Gap: 2.8%**

Much better than initial:
- Week 4 (before fix): 15% gap
- After vendor-agnostic training: 2.8% gap

---

### Confidence Calibration

**Goal:** Ensure confidence scores reflect actual accuracy

**Calibration Plot:**

```
Accuracy
  │
100%│        ●
    │       ●
    │      ●
 99%│     ●
    │    ●
 95%│   ●
    │  ●
    │ ●
    │●
  0%└──────────────────────→ Confidence
    0.5  0.7  0.9  0.95 0.99
```

**Findings:**
- Confidence 0.95-1.0: 99.2% actual accuracy ✅ (well-calibrated)
- Confidence 0.90-0.95: 97.8% actual accuracy ✅
- Confidence 0.80-0.90: 94.1% actual accuracy ✅
- Confidence <0.80: 87.3% actual accuracy ⚠️ (underconfident)

**Threshold Selection:**
- Set threshold at 0.95
- Above 0.95: 99.2% accuracy (auto-process)
- Below 0.95: Route to human review

---

## Production Deployment

### Model Export

**Save LoRA Adapters:**
```python
# Save adapters (only 3M parameters, ~12MB)
model.save_pretrained("./invoice-llama-adapters")

# Load in production
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    "./invoice-llama-adapters"
)
```

**Why Adapters Only:**
- Base model (3B params): 6GB storage
- Adapters (3M params): 12MB storage
- Can share base model across projects, swap adapters

---

### Inference Optimization

**Batch Processing:**
```python
# Process invoices in batches (nightly run)
batch_size = 32
for batch in batches(invoices, batch_size):
    # Extract text (OCR + NER)
    texts = [extract_text(inv) for inv in batch]
    
    # Batch inference (much faster than one-by-one)
    predictions = model.generate(
        texts,
        max_new_tokens=512,
        do_sample=False,  # Deterministic for production
        temperature=0.0,
    )
    
    # Post-process
    results = [parse_output(pred) for pred in predictions]
```

**Throughput:**
- Single invoice: 0.6s
- Batch of 32: 8s (0.25s per invoice)
- **2.4x speedup with batching**

---

### Monitoring

**Daily Metrics:**
```python
# Track per-vendor accuracy
daily_metrics = {
    'overall_accuracy': 0.99,
    'per_vendor_accuracy': {
        'Vendor A': 0.992,
        'Vendor B': 0.988,
        'Vendor C': 0.914,  # ⚠️ Alert triggered!
    },
    'confidence_distribution': {
        '0.95-1.00': 0.77,  # Auto-processed
        '0.90-0.95': 0.15,  # Review queue
        '0.00-0.90': 0.08,  # Flagged
    },
    'throughput': 10234,  # Invoices processed
}
```

**Alerts:**
- Vendor accuracy drops >5%: Immediate alert
- Overall accuracy drops below 98%: Investigation
- Review queue exceeds 30%: Capacity alert

---

### Retraining Pipeline

**When to Retrain:**
1. **Quarterly scheduled retraining**
   - Incorporate 3 months of corrections
   - Maintain model freshness
   
2. **Vendor template changes**
   - Accuracy drop detected
   - Collect 100+ new format examples
   - Retrain with new data
   
3. **New vendor onboarding**
   - Collect 200+ labeled invoices
   - Add to training set
   - Retrain to include new vendor

**Retraining Process:**
```bash
# Collect new labeled data
python collect_corrections.py --last-90-days

# Merge with existing training set
python merge_datasets.py

# Retrain (same config as original)
python train_qlora.py --config invoice_config.yaml

# Validate on held-out vendors
python evaluate.py --test-set held_out_vendors.json

# Deploy if validation passes
if [ validation_accuracy > 0.98 ]; then
    python deploy.py --model ./new-adapters
fi
```

**Cost:** $240 per retraining (8 hours T4)
**Frequency:** Quarterly + as-needed
**Annual cost:** ~$960

---

## Lessons Learned

### What Worked Well

1. **QLoRA saved 85% of cost with minimal accuracy loss**
   - Fine-tuning accessible on consumer GPUs
   - Production-quality results without A100
   
2. **Rank 8 was sweet spot**
   - Lower: Underfitting
   - Higher: Overfitting, slower
   - Testing ranks 4-32 was worth it
   
3. **Held-out vendor evaluation caught generalization issues**
   - Random split hid 15% accuracy gap
   - Vendor-stratified split essential
   
4. **Weighted loss fixed minority class problem**
   - Simple technique, huge impact
   - F1 improved from 0.68 → 0.81

### What We'd Do Differently

1. **Test on held-out groups from day 1**
   - Not just random samples
   - Whatever natural grouping exists (vendors, time periods, etc.)
   
2. **Budget more time for hyperparameter search**
   - Finding rank 8 took 3 days
   - Could have started with systematic sweep
   
3. **Plan retraining pipeline before first deployment**
   - Built after first incident
   - Should have been part of initial design

---

## Conclusion

Fine-tuning with QLoRA broke through the RAG ceiling (91% → 99%) while remaining affordable and practical.

**Key Takeaways:**
- QLoRA makes fine-tuning accessible (T4 vs A100)
- Proper evaluation (held-out vendors) essential
- Hyperparameter tuning (rank, LR) matters
- Plan for maintenance (retraining) from day 1

The model is 30% of the project. The rest is data prep, evaluation, monitoring, and maintenance.
