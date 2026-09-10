import modal
import config
import logging
import statistics
from openai import OpenAI

import prompt_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QLoRAPricer:
    def __init__(self, model_id=config.HF_FINE_TUNED_MODEL):
        self.model_id = model_id
        self.openai_client = OpenAI(api_key=getattr(config, "OPENAI_API_KEY", None))
        self.groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=getattr(config, "GROQ_API_KEY", None)
        )

    def _ensure_formatted_input(self, formatted_input: str) -> str:
        """Ensures non-empty formatted input from scraper.py."""
        if not formatted_input or not isinstance(formatted_input, str):
            return prompt_config.format_input(
                "Electronics", "New", "Standard Item", "Standard specifications"
            )
        return formatted_input.strip()

    def predict_with_gpt(self, formatted_input: str) -> float:
        """Valuation using OpenAI gpt-5-mini model with full structured input."""
        clean_input = self._ensure_formatted_input(formatted_input)
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a precise retail valuation engine. {prompt_config.INSTRUCTION} "
                            "Output ONLY a raw numerical float representing USD price."
                        )
                    },
                    {"role": "user", "content": clean_input}
                ]
            )
            content = response.choices[0].message.content.strip()
            val = prompt_config.parse_price(content)
            if val is not None:
                return val
        except Exception as e:
            logger.error(f"[GPT-5-Mini Pricer Exception]: {e}")

        return 0.0

    def predict_with_groq(self, formatted_input: str) -> float:
        """Fallback cloud valuation using Groq API passing full structured input."""
        api_key = getattr(config, "GROQ_API_KEY", None)
        if not api_key:
            logger.warning("[Groq Warning]: GROQ_API_KEY is not configured.")
            return 0.1

        clean_input = self._ensure_formatted_input(formatted_input)

        try:
            response = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a price prediction engine. {prompt_config.INSTRUCTION} "
                            "Output ONLY the raw numeric dollar value (e.g. 159.99). "
                            "Do not output words, reasoning, or currency symbols."
                        )
                    },
                    {"role": "user", "content": clean_input}
                ],
                temperature=0.1,
                max_tokens=150,
                reasoning_effort="low"
            )

            content = (response.choices[0].message.content or "").strip()
            val = prompt_config.parse_price(content)
            if val is not None:
                return val if val >= 0 else 0.0

        except Exception as e:
            logger.error(f"[Groq SDK Parsing Error]: {e}")

        return 0.0

    def predict_with_modal(self, formatted_input: str) -> float:
        """Triggers remote serverless GPU execution on Modal using the exact training prompt format.

        Sends only the structured product description -- modal_deploy.py's
        ModelInferenceService.predict() is responsible for prepending
        prompt_config.INSTRUCTION and applying prompt_config.SYSTEM_PROMPT, so
        this stays a single source of truth rather than being duplicated here.
        """
        clean_input = self._ensure_formatted_input(formatted_input)
        try:
            ModelInferenceService = modal.Cls.from_name("unified-valuation-engine-rev2", "ModelInferenceService")
            service = ModelInferenceService()

            predicted_price = service.predict.remote(product_description=clean_input)
            return float(predicted_price)

        except Exception as e:
            raise RuntimeError(f"[Modal Remote Call Exception]: Remote valuation failed - {e}") from e

    def predict_all(self, formatted_input: str) -> dict:
        """
        Runs all providers independently.

        Failed providers return 0 and are excluded from the ensemble.
        Confidence is based on the agreement/spread of the providers that
        actually returned a valid valuation.
        """
        clean_input = self._ensure_formatted_input(formatted_input)

        # Each provider is isolated so one API failure does not kill the whole pipeline.
        try:
            modal_val = round(self.predict_with_modal(clean_input), 2)
        except Exception as e:
            logger.error(f"[Modal QLoRA Exception]: {e}")
            modal_val = 0.0

        try:
            groq_val = round(self.predict_with_groq(clean_input), 2)
        except Exception as e:
            logger.error(f"[Groq Exception]: {e}")
            groq_val = 0.0

        try:
            gpt_val = round(self.predict_with_gpt(clean_input), 2)
        except Exception as e:
            logger.error(f"[GPT-5 Mini Exception]: {e}")
            gpt_val = 0.0

        values = [
            ("QLoRA", modal_val),
            ("GPT OSS 120B", groq_val),
            ("GPT-5 Mini", gpt_val),
        ]
        valid_vals = [value for _, value in values if value > 0]

        ensemble_val = (
            round(statistics.median(valid_vals), 2)
            if valid_vals else 0.0
        )

        # Confidence is intentionally simple and explainable:
        # tighter model agreement => higher confidence.
        if len(valid_vals) == 0:
            confidence = "Unavailable"
        elif len(valid_vals) == 1:
            confidence = "Low — 1 model"
        else:
            median_val = statistics.median(valid_vals)
            spread_pct = (
                ((max(valid_vals) - min(valid_vals)) / median_val) * 100
                if median_val > 0 else 100.0
            )

            if len(valid_vals) == 3 and spread_pct <= 10:
                confidence = "High — strong agreement"
            elif spread_pct <= 25:
                confidence = "Medium — moderate agreement"
            else:
                confidence = "Low — models disagree"

        return {
            "modal_qlora": modal_val,
            "groq_gpt-oss-120b": groq_val,
            "gpt_5_mini": gpt_val,
            "ensemble_est": ensemble_val,
            "confidence": confidence,
        }

    def predict_price(self, formatted_input: str) -> float:
        """Main entry point returning the single ensemble estimated value."""
        results = self.predict_all(formatted_input)
        return results["ensemble_est"]
