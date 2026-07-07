"""CPU PCA diagnostics for Qwen last-token activations: top-1 variance fraction,
participation ratio, d90/d95/d99 — the massive-activation-dimension characterization."""
import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); C=os.path.join(HERE,"cache"); R=os.path.join(HERE,"results")
out=[]
for L in (2,10):
    x=np.load(os.path.join(C,f"acts_qwen_L{L}.npy")).astype(np.float64)
    x=x-x.mean(0,keepdims=True)
    cov=(x.T@x)/(x.shape[0]-1)
    ev=np.linalg.eigvalsh(cov)[::-1]; ev=np.clip(ev,0,None)
    frac=ev/ev.sum(); cum=np.cumsum(frac)
    pr=(ev.sum()**2)/(ev**2).sum()
    d=lambda q:int(np.searchsorted(cum,q)+1)
    out.append({"layer":L,"top1_frac":float(frac[0]),"top5_frac":float(cum[4]),
                "pca_pr":float(pr),"d90":d(.90),"d95":d(.95),"d99":d(.99),"d_model":2048})
    print(out[-1])
json.dump(out,open(os.path.join(R,"qwen_pca_diag.json"),"w"),indent=1)
