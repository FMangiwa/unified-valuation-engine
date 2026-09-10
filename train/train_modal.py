import modal

app = modal.App("llama-3b-price-engine-trainer-rev2")
image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("git")
    .pip_install(
        "torch>=2.6.0","triton","transformers>=4.48.0","datasets",
        "peft>=0.18.0","trl>=0.12.0","accelerate","bitsandbytes",
        "numpy","huggingface_hub","xformers"
    )
    .pip_install("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    .add_local_python_source("prompt_config")
)

@app.function(image=image,gpu="H100",timeout=14400,secrets=[modal.Secret.from_dotenv()])
def train():
    import os, time, json
    import numpy as np, torch
    from datasets import load_dataset
    from huggingface_hub import login, create_tag
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from trl import SFTTrainer
    from transformers import TrainingArguments, EarlyStoppingCallback
    import prompt_config

    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token: login(token=token)

    model, tokenizer=FastLanguageModel.from_pretrained(
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        max_seq_length=1024, load_in_4bit=True
    )
    model=FastLanguageModel.get_peft_model(
        model,r=32,lora_alpha=64,lora_dropout=0.0,bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_rslora=False,init_lora_weights=True
    )
    tokenizer=get_chat_template(tokenizer,chat_template="llama-3.2")

    ds=load_dataset("faxoi/valuation-engine-Rev2")
    def fmt(examples):
        texts=[]
        for inp,out in zip(examples["input"],examples["output"]):
            msgs=[
                {"role":"system","content":prompt_config.SYSTEM_PROMPT},
                {"role":"user","content":prompt_config.build_user_turn(inp)},
                {"role":"assistant","content":prompt_config.format_target(out)},
            ]
            texts.append(tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False))
        return {"text":texts}

    train_ds=ds["train"].map(fmt,batched=True)
    val_ds=ds["validation"].map(fmt,batched=True)
    test_ds=ds["test"]

    training_args = TrainingArguments(
        output_dir="/tmp/checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=3200,
        warmup_ratio=0.05,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_strategy="steps",
        eval_steps=800,
        save_strategy="steps",
        save_steps=800,
        per_device_eval_batch_size=16,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to="none"
    )
    
    # args=TrainingArguments(
    #     output_dir="/tmp/checkpoints",per_device_train_batch_size=4,
    #     gradient_accumulation_steps=2,warmup_ratio=0.05,num_train_epochs=2,
    #     learning_rate=1e-4,lr_scheduler_type="cosine",optim="paged_adamw_32bit",
    #     weight_decay=0.01,fp16=not torch.cuda.is_bf16_supported(),
    #     bf16=torch.cuda.is_bf16_supported(),eval_strategy="steps",eval_steps=250,
    #     save_strategy="steps",save_steps=250,save_total_limit=2,
    #     load_best_model_at_end=True,metric_for_best_model="eval_loss",
    #     greater_is_better=False,logging_steps=10,report_to="none"
    # )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ]
    )
    
    # trainer=SFTTrainer(
    #     model=model,processing_class=tokenizer,train_dataset=train_ds,
    #     eval_dataset=val_ds,dataset_text_field="text",max_seq_length=2048,args=args,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    # )
    
    trainer=train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    trainer.train()

    FastLanguageModel.for_inference(model)
    preds,actuals=[],[]
    for row in test_ds:
        prompt=tokenizer.apply_chat_template(
            prompt_config.build_messages(row["input"]),tokenize=False,add_generation_prompt=True
        )
        x=tokenizer([prompt],return_tensors="pt").to("cuda")
        with torch.no_grad():
            y=model.generate(**x,max_new_tokens=16,do_sample=False)
        text=tokenizer.decode(y[0][x.input_ids.shape[-1]:],skip_special_tokens=True)
        p=prompt_config.parse_price(text); a=prompt_config.parse_price(row["output"])
        if p is not None and a is not None:
            preds.append(p);actuals.append(a)

    p=np.array(preds); a=np.array(actuals)
    mae=float(np.mean(np.abs(p-a)))
    mape=float(np.mean(np.abs(p-a)/np.maximum(a,1))*100)
    within20=float(np.mean(np.abs(p-a)/np.maximum(a,1)<=.20)*100)
    within30=float(np.mean(np.abs(p-a)/np.maximum(a,1)<=.30)*100)

    metrics={"mae":mae,"mape_pct":mape,"within20_pct":within20,"within30_pct":within30,"n":len(p)}
    print("=== REV2 HELD-OUT TEST ===")
    print(json.dumps(metrics,indent=2))

    repo="faxoi/llama-3.2-3b-price-engine-Rev2"
    model.push_to_hub_merged(repo,tokenizer,save_method="merged_16bit",token=token)
    tag=f"rev2-mae{mae:.0f}-w20{within20:.0f}-{time.strftime('%Y%m%d-%H%M%S')}"
    if token: create_tag(repo,tag=tag,token=token)
    print("Validated model tag:",tag)
    return metrics
