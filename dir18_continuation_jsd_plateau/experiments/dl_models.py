from huggingface_hub import snapshot_download
for repo, rev in [("EleutherAI/pythia-1.4b-deduped","step143000"),
                  ("EleutherAI/pythia-1.4b-deduped","step0"),
                  ("EleutherAI/pythia-410m-deduped","step143000")]:
    p = snapshot_download(repo, revision=rev, allow_patterns=["*.json","*.bin","*.safetensors","*.txt"])
    print("OK", repo, rev, p, flush=True)
