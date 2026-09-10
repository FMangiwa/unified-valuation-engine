# Unified Market Intelligence & Valuation Engine

An end-to-end AI engineering portfolio project that discovers online deals, extracts structured product information, estimates fair market value using multiple AI valuation models, combines the predictions with an ensemble, measures model agreement, and identifies potential bargains.

> **Portfolio project:** The primary goal is to demonstrate AI engineering architecture and implementation rather than production-grade price accuracy.

---

## Project Overview

The system automatically processes deal listings and produces a valuation such as:

| Signal | Example |
|---|---:|
| Listed Price | $450 |
| Modal QLoRA | $600 |
| GPT OSS 120B | $849.99 |
| GPT-5 Mini | $699 |
| Ensemble Fair Value | $699 |
| Discount | 35.6% |
| Confidence | Low — models disagree |
| Verdict | 🔥 High Margin Bargain |

The system deliberately keeps the valuation models independent. The QLoRA model provides a learned historical price signal, while the frontier models provide additional contextual valuation signals.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │     DealNews RSS    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Web Page Scraping  │
                         │  + Product Parsing  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │     GPT-5 Mini        │
                        │ Structured Extraction │
                        └───────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Canonical Product   │
                         │      Schema         │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
           ┌────────────┐   ┌──────────────┐  ┌────────────┐
           │ Modal      │   │ Groq         │  │ GPT-5 Mini │
           │ QLoRA      │   │ GPT OSS 120B │  │ Valuation  │
           │ Rev2       │   │ Valuation    │  │            │
           └─────┬──────┘   └──────┬───────┘  └──────┬─────┘
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │ Median Ensemble     │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
              ┌────────────────┐       ┌────────────────┐
              │ Discount / Deal│       │ Model          │
              │ Detection      │       │ Confidence     │
              └───────┬────────┘       └────────────────┘
                      │
                      ▼
              ┌────────────────┐
              │ Gradio UI      │
              │ + Notifications│
              └────────────────┘
```

---

## Demo

Watch the application demo:

`umive_demo.mp4`

The demo shows the end-to-end workflow from deal discovery and product extraction through AI valuation, ensemble scoring, and deal detection.

---

## Key Engineering Features

### 1. Automated Deal Discovery

The scraper consumes DealNews RSS feeds and retrieves the corresponding deal pages.

The pipeline extracts:

- Category
- Condition
- Product title
- Features/specifications
- Listed price
- Deal URL

The extracted information is converted into a canonical product representation before being sent to the valuation models.

---

### 2. LLM-Based Structured Extraction

GPT-5 Mini is used for product extraction rather than relying only on brittle HTML selectors.

This allows the system to turn inconsistent deal-page content into a predictable structure.

Example:

```text
Product: [Category: Laptop]
[Condition: New]
Lenovo V14 G4 FHD 14" Intel Core i7 Laptop
| Features: Intel Core i7, 14" FHD ...
```

---

### 3. QLoRA Fine-Tuned Price Model

A Llama 3.2 3B model was fine-tuned using QLoRA.

The training pipeline uses:

- 4-bit quantization
- LoRA adapters
- Unsloth
- Modal H100 training
- A train/validation/test split
- A dedicated held-out test set
- Merged 16-bit model export

The deployed model is served through Modal.

### What the QLoRA represents

The QLoRA model should not be interpreted as a perfect real-time market oracle.

It represents a **learned historical price signal** based on the price distribution represented in the training data.

Therefore, a QLoRA prediction below the current listed price does not necessarily mean the model is simply "wrong." It can indicate that the learned historical price signal is lower than the current listing.

This is one reason the project combines multiple valuation signals rather than relying exclusively on QLoRA.

---

## 4. Multi-Model Valuation

The system evaluates the same normalized product through three independent valuation paths:

### Modal QLoRA

A custom fine-tuned Llama 3.2 3B model.

Purpose:

- Demonstrate custom model training
- Demonstrate QLoRA
- Demonstrate cloud GPU training
- Provide an independent learned price signal
- Provide a self-hosted/fallback valuation path

### Groq GPT OSS 120B

A large open-weight model accessed through Groq.

Purpose:

- Independent valuation signal
- Fast inference
- Model diversity

### GPT-5 Mini

A frontier model used for:

- Product extraction
- Independent valuation
- Strong contextual reasoning

---

## 5. Ensemble Valuation

The system uses the **median** of successful model predictions.

Example:

```text
QLoRA       $600
GPT OSS     $849.99
GPT-5 Mini  $699
             ↓
Median      $699
```

The median was chosen because it is less sensitive to a single unusually high or low model prediction.

Failed providers are excluded rather than replacing their result with an artificial price.

---

## 6. Confidence Scoring

Confidence is based on the agreement between the valuation models.

Examples:

```text
3 models close together
        ↓
High — strong agreement
```

```text
Models have moderate spread
        ↓
Medium — moderate agreement
```

```text
Models have substantial disagreement
        ↓
Low — models disagree
```

If only one valuation provider is available:

```text
Low — 1 model
```

This separates two concepts that should not be confused:

**Deal strength**

> How far below estimated value is the listed price?

**Model confidence**

> How much do the independent valuation models agree?

A deal can therefore have a strong estimated discount while still having low confidence.

---

## 7. Deal Detection

The system calculates:

```text
Discount % =
(Fair Value - Listed Price) / Fair Value × 100
```

A configurable threshold is used to flag potential bargains.

The system can also classify stronger opportunities using a stricter threshold.

---

## 8. Graceful Degradation

The architecture is designed so that a single provider failure does not necessarily stop the entire application.

For example:

```text
GPT-5 Mini unavailable
        │
        ▼
QLoRA + GPT OSS
        │
        ▼
Ensemble still available
```

Or:

```text
Only QLoRA available
        │
        ▼
QLoRA-only valuation
        │
        ▼
Low confidence
```

This is an important part of the architecture because external model APIs and cloud services can fail independently.

---

# Evaluation

Rev2 includes a dedicated held-out evaluation set.

Current recorded result:

| Metric | Rev2 |
|---|---:|
| MAE | $39.81 |
| MAPE | 36.66% |
| Within ±20% | 38.56% |
| Within ±30% | 51.96% |
| Test examples | 10,000 |

These numbers should be interpreted as experimental model metrics rather than evidence of production-grade market valuation accuracy.

The purpose of the evaluation pipeline is to demonstrate that the model is tested on data separated from training and that model performance is measured quantitatively.

---

# Why This Project Matters

The project is intentionally broader than a single price-prediction model.

It demonstrates an end-to-end AI engineering workflow:

```text
Data ingestion
      ↓
Web scraping
      ↓
LLM structured extraction
      ↓
Dataset preparation
      ↓
QLoRA fine-tuning
      ↓
Cloud GPU training
      ↓
Model versioning
      ↓
Model deployment
      ↓
Multi-model inference
      ↓
Ensemble
      ↓
Confidence estimation
      ↓
Deal classification
      ↓
UI / notification
```

The valuation model is one component inside the larger system.

---

# Project Files

| File | Purpose |
|---|---|
| `app.py` | Gradio application and UI |
| `agents.py` | Autonomous deal-scanning orchestration |
| `scraper.py` | DealNews scraping and LLM extraction |
| `pricer_qlora.py` | QLoRA, Groq and GPT-5 Mini inference |
| `prompt_config.py` | Centralized prompts and input/output formatting |
| `dataset_modal.py` | Dataset construction and train/validation/test generation |
| `train_modal.py` | Modal H100 QLoRA training |
| `modal_deploy.py` | Modal inference deployment |
| `config.py` | Model, API and application configuration |

---

# Training Pipeline

The QLoRA model is trained on product metadata with price targets.

The Rev2 dataset is split into:

```text
80% Training
10% Validation
10% Test
```

The held-out test set is kept separate from the training process.

The training pipeline uses a Llama 3.2 3B base model with QLoRA adapters and exports a merged model for inference.

---

# Running the Application

## 1. Clone

```bash
git clone https://github.com/FMangiwa/unified-valuation-engine.git
cd unified-valuation-engine
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Modal authentication is configured separately through the Modal CLI.

---

# Modal Deployment

Authenticate with Modal:

```bash
modal setup
```

Deploy the inference service:

```bash
modal deploy modal_deploy.py
```

The application expects the deployed Modal service to be available under the configured Rev2 application name.

Make sure the deployed app name and the name used by `pricer_qlora.py` match.

---

# Start the Gradio Application

```bash
python app.py
```

The UI provides two primary workflows:

### Automated Scanner

Runs the deal discovery pipeline and evaluates multiple listings.

### Single Item Evaluator

Allows a product description and listed price to be evaluated directly.

Example:

```text
Product:
Lenovo V14 G4 FHD 14" Intel Core i7 Laptop

Listed Price:
$450
```

Example output:

```text
Modal QLoRA             $600
Groq GPT OSS 120B       $849.99
GPT-5 Mini              $699
Ensemble                $699
Discount                35.6%
Confidence              Low — models disagree
Verdict                 High Margin Bargain
```

---

# Design Decisions

## Why use multiple models?

Different models can provide different valuation signals.

Rather than assuming one model is always correct, the system exposes disagreement and uses it as a confidence signal.

## Why use the median?

A median ensemble is simple, explainable, and less sensitive to an individual outlier.

## Why keep QLoRA?

The QLoRA model demonstrates the complete custom-model engineering workflow:

```text
Dataset → Fine-tuning → Evaluation → Export → Deployment → Inference
```

It also provides an independent valuation signal and fallback path.

## Why not force QLoRA to match the frontier models?

Because they represent different sources of information.

QLoRA learns from its training distribution, while frontier models can reason over the current product description and context.

The disagreement itself is useful information.

---

# Limitations

This is a portfolio/experimental system.

Important limitations include:

- Historical training prices are not guaranteed to represent current fair market value.
- Amazon metadata prices are not a perfect ground-truth definition of fair value.
- Product variants can have materially different prices.
- Used/new/refurbished conditions can affect valuation.
- MSRP, historical sale price and current market value are different concepts.
- LLM valuations can be inconsistent.
- Model outputs can contain outliers.
- External APIs and Modal can become unavailable.
- DealNews pages and RSS feeds can change.
- The ensemble does not guarantee that a deal is genuinely profitable.

Therefore, the system should be viewed as a **deal-discovery and valuation-assistance system**, not an automated purchasing or investment decision-maker.

---

# Portfolio Objective

The primary objective of this project is to demonstrate practical AI engineering skills.

The project showcases:

- Python application architecture
- LLM integration
- Structured outputs
- Web scraping
- Dataset engineering
- QLoRA fine-tuning
- GPU training
- Modal cloud deployment
- Model versioning
- Multi-model orchestration
- Ensemble inference
- Confidence estimation
- Graceful fallbacks
- Gradio UI
- Automated deal detection
- Push notifications
- Quantitative evaluation

The model's price accuracy is treated as an experimental result, while the **engineering architecture and complete working pipeline are the primary portfolio deliverables**.

---

## License

See `LICENSE`.

````