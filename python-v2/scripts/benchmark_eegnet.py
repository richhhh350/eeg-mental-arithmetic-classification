"""Synthetic-only timing before freezing the stage3 compute budget."""
import json
import time
import torch
from eeg_v2.eegnet import EEGNet, seed_all, train_epoch

results = []
for threads in (1, 2, 4):
    seed_all(42, threads)
    x = torch.randn(1440, 19, 256)
    y = torch.arange(1440) % 2
    model = EEGNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(42)
    for epoch in range(2):
        start = time.perf_counter()
        loss = train_epoch(model, optimizer, x, y, generator, batch_size=128)
        results.append({"threads": threads, "epoch": epoch + 1,
                        "seconds": time.perf_counter() - start, "loss": loss})
        print(json.dumps(results[-1]), flush=True)
print(json.dumps({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                  "parameters": sum(p.numel() for p in model.parameters()), "timing": results}))
