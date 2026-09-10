import gradio as gr
from engine.agents import AutonomousPlanningAgent

# Initialize autonomous planning agent instance
agent = AutonomousPlanningAgent()


def run_pipeline(progress=gr.Progress(track_tqdm=True)):
    """Runs deal discovery and returns all model predictions in the output table."""
    progress(0.1, desc="🔍 Step 1/3: Fetching RSS feeds & scanning listings...")

    raw_listings = agent.scraper.fetch_latest_listings(
        memory_urls=agent.memory_urls
    )

    if not raw_listings:
        progress(1.0, desc="✅ Complete. No new deals found.")
        return []

    total_items = len(raw_listings)
    results_table = []

    progress(0.2, desc=f"⚡ Step 2/3: Evaluating {total_items} items across all pricer models...")

    for idx, item in enumerate(raw_listings, start=1):
        deal_url = item.get("url", "")
        description = item.get("description", "")
        listed_price = float(item.get("listed_price", 0.0))

        current_pct = 0.2 + (idx / total_items) * 0.7
        progress(
            current_pct,
            desc=f"⚡ [{idx}/{total_items}] Evaluating models for: {description[:35]}..."
        )

        if deal_url:
            agent.memory_urls.append(deal_url)

        # Predict prices across all models using predict_all()
        estimates = agent.pricer.predict_all(description)
        modal_val = estimates["modal_qlora"]
        groq_val = estimates["groq_gpt-oss-120b"]
        gpt_val = estimates["gpt_5_mini"]
        estimated_value = estimates["ensemble_est"]
        confidence = estimates.get("confidence", "Unavailable")

        if estimated_value > 0 and listed_price > 0:
            discount_pct = round(((estimated_value - listed_price) / estimated_value) * 100, 1)

            results_table.append([
                description[:80],
                f"${listed_price:.2f}",
                f"${modal_val:.2f}",
                f"${groq_val:.2f}",
                f"${gpt_val:.2f}",
                f"${estimated_value:.2f}",
                f"{discount_pct}%",
                confidence,
                f"[Open Deal ↗]({deal_url})"
            ])

            # Trigger notification if listed price is at least 20% below ensemble value
            if listed_price <= (estimated_value * 0.80):
                agent.notifier.send_notification(
                    title="🎯 High-Margin Deal Identified!",
                    message=f"Listed: ${listed_price} | Ensemble Val: ${estimated_value} ({discount_pct}% off)\nLink: {deal_url}",
                    url=deal_url
                )

    progress(1.0, desc=f"✅ Finished! Processed {len(results_table)} items.")
    return results_table


def evaluate_single_item(description: str, listed_price: float): #, progress=gr.Progress(track_tqdm=True)
    """Evaluates a single item across all available valuation providers."""
    if not description.strip():
        return (
            "$0.00", "$0.00", "$0.00", "$0.00",
            "0%", "Unavailable",
            "Please enter a valid description."
        )

    # progress(0.05, desc="🔍 Preparing product description...")

    # progress(0.20, desc="🧠 Step 1/3: Running Modal QLoRA...")
    estimates = agent.pricer.predict_all(description)

    # progress(0.55, desc="☁️ Step 2/3: Running cloud valuation providers...")
    modal_val = estimates["modal_qlora"]
    groq_val = estimates["groq_gpt-oss-120b"]
    gpt_val = estimates["gpt_5_mini"]

    # progress(0.80, desc="📊 Step 3/3: Calculating ensemble, discount & confidence...")
    estimated_value = estimates["ensemble_est"]
    confidence = estimates.get("confidence", "Unavailable")

    if estimated_value > 0 and listed_price > 0:
        discount_pct = round(
            ((estimated_value - listed_price) / estimated_value) * 100, 1
        )
        verdict = (
            "🔥 High Margin Bargain!"
            if listed_price <= (estimated_value * 0.80)
            else "Fair Value / Standard Price"
        )
    else:
        discount_pct = 0.0
        verdict = "Unable to calculate discount."

    # progress(1.0, desc="✅ Valuation complete!")

    return (
        f"${modal_val:.2f}" if modal_val > 0 else "$0.00",
        f"${groq_val:.2f}" if groq_val > 0 else "$0.00",
        f"${gpt_val:.2f}" if gpt_val > 0 else "$0.00",
        f"${estimated_value:.2f}",
        f"{discount_pct}%",
        confidence,
        verdict,
    )


# --- Gradio Multi-Tab Interface ---
with gr.Blocks(title="Market Intelligence & Valuation Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Unified Market Intelligence & Valuation Engine")
    gr.Markdown("Autonomous deal scanning, multi-model price valuation, and provider benchmarks.")
    
    with gr.Tabs():
        # TAB 1: Discovery Pipeline
        with gr.TabItem("📡 Automated Deal Scanner"):
            scan_btn = gr.Button("🚀 Run Discovery Cycle", variant="primary")

            # Headers updated to match 8 output columns
            opportunities_table = gr.Dataframe(
                headers=[
                    "Description", 
                    "Listed Price", 
                    "Modal QLoRA", 
                    "GPT OSS 120b", 
                    "GPT-5 Mini", 
                    "Ensemble Val", 
                    "Discount",
                    "Confidence",
                    "URL"
                ],
                wrap=True,
                column_widths=[
                    300, 100, 110, 110, 110, 110, 90, 190, 130
                ],
                datatype=[
                    "markdown", "number", "number", "number", "number",
                    "number", "str", "str", "markdown"
                ],
                interactive=False
            )

            scan_btn.click(
                fn=run_pipeline,
                inputs=[],
                outputs=[opportunities_table]
            )

        # TAB 2: Single Evaluator
        with gr.TabItem("🔍 Single Item Evaluator"):
            with gr.Row():
                with gr.Column():
                    input_desc = gr.Textbox(
                        label="Product Description",
                        placeholder="e.g., Lenovo IdeaPad Slim 5 16' Touchscreen Laptop AMD Ryzen 7 16GB RAM 512GB SSD",
                        lines=3
                    )
                    input_price = gr.Number(label="Listed Price ($)", value=299.99)
                    eval_btn = gr.Button("Calculate True Value", variant="primary")

                with gr.Column():
                    out_modal = gr.Textbox(label="Modal QLoRA Prediction", interactive=False)
                    out_groq = gr.Textbox(label="Groq GPT OSS 120b Prediction", interactive=False)
                    out_gpt = gr.Textbox(label="GPT-5 Mini Prediction", interactive=False)
                    out_ensemble = gr.Textbox(label="Final Ensemble Valuation", interactive=False)
                    out_disc = gr.Textbox(label="Calculated Discount %", interactive=False)
                    out_confidence = gr.Textbox(label="Model Confidence", interactive=False)
                    out_verdict = gr.Textbox(label="Analysis Verdict", interactive=False)

            eval_btn.click(
                fn=evaluate_single_item,
                inputs=[input_desc, input_price],
                outputs=[
                    out_modal, out_groq, out_gpt, out_ensemble,
                    out_disc, out_confidence, out_verdict
                ]
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True)