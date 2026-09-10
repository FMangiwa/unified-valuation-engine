import modal

app = modal.App("dataset4valuation-rev2")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets", "pandas", "huggingface_hub", "transformers>=4.48.0")
    .add_local_python_source("prompt_config")
)
volume = modal.Volume.from_name("valuation-dataset-cache", create_if_missing=True)

def synthetic_cases():
    import prompt_config
    rows = [
        ("Power Bank","New","Anker 20000mAh Ultra-High Capacity Portable Charger",
         ["20,000mAh capacity","dual USB output","trickle charging mode"],29.99),
        ("Power Station","New","Eco-Worthy 2048Wh LiFePO4 Portable Solar Generator",
         ["2048Wh","2400W inverter","LiFePO4"],1050),
        ("Power Station","New","Jackery Explorer 300 Portable Power Station",
         ["290Wh","300W AC","solar compatible"],249.99),
        ("Laptop","Refurbished","Apple MacBook Pro 14.2-inch 2021 M1 Pro",
         ["M1 Pro","32GB RAM","512GB SSD"],999),
        ("Desktop","New","Firebat F1 AMD Ryzen 7 Mini PC",
         ["Ryzen 7","16GB DDR5","512GB NVMe"],420),
    ]
    return [{
        "instruction": prompt_config.INSTRUCTION,
        "input": prompt_config.format_input(c,cnd,t,f),
        "output": prompt_config.format_target(p),
    } for c,cnd,t,f,p in rows]

@app.function(image=image, volumes={"/data": volume}, timeout=3600, secrets=[modal.Secret.from_dotenv()])
def prepare_and_upload_dataset(target_count=100000):
    import os, json, random, re
    from datasets import Dataset, DatasetDict
    from huggingface_hub import hf_hub_download, login
    import prompt_config

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token: login(token=token)

    path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename="raw/meta_categories/meta_Electronics.jsonl",
        repo_type="dataset",
    )

    rows = synthetic_cases()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= target_count: break
            try: item=json.loads(line)
            except: continue
            title=item.get("title","")
            price=item.get("price")
            if not isinstance(title,str) or len(title.strip())<10 or price in (None,""): continue
            try: p=float(str(price).replace("$","").replace(",","").strip())
            except: continue
            if not 5 <= p <= 5000: continue
            cats=item.get("categories") or []
            cat=str(cats[-1]) if isinstance(cats,list) and cats else "Electronics"
            cond="Refurbished" if any(x in title.lower() for x in ["refurbished","renewed"]) else "New"
            feats=[str(x).strip() for x in (item.get("features") or [])[:3] if len(str(x).strip())>5]
            rows.append({
                "instruction":prompt_config.INSTRUCTION,
                "input":prompt_config.format_input(cat,cond,title,feats),
                "output":prompt_config.format_target(p),
            })

    # Shuffle BEFORE splitting. The old pipeline evaluated the first 500 rows
    # of a sample that was also the first 2,500 rows of the training corpus.
    random.Random(42).shuffle(rows)

    n=len(rows)
    train_end=int(n*0.80)
    val_end=int(n*0.90)
    ds=DatasetDict({
        "train":Dataset.from_list(rows[:train_end]),
        "validation":Dataset.from_list(rows[train_end:val_end]),
        "test":Dataset.from_list(rows[val_end:]),
    })
    repo="faxoi/valuation-engine-Rev2"
    if token: ds.push_to_hub(repo, token=token)
    print(f"Uploaded {n} rows: train={train_end}, validation={val_end-train_end}, test={n-val_end}")
    return {"repo":repo,"rows":n}
