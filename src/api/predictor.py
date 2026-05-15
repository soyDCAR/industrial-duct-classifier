import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

import structlog
import torch
from PIL import Image

from model import MultiEfficientNet, get_transforms

logger = structlog.get_logger()

MAX_CLASS = 7
_executor = ThreadPoolExecutor(max_workers=4)


class ModelPredictor:
    def __init__(
        self,
        model: MultiEfficientNet,
        idx_to_class_d: dict,
        idx_to_class_o: dict,
        device: torch.device,
    ):
        self.model = model
        self.idx_to_class_d = idx_to_class_d
        self.idx_to_class_o = idx_to_class_o
        self.device = device
        self.transform = get_transforms(train=False)

    @classmethod
    def load(cls, model_path: str, mapping_path: str) -> "ModelPredictor":
        with open(mapping_path) as f:
            data = json.load(f)
        idx_to_class_d = {int(k): v for k, v in data["idx_to_class_d"].items()}
        idx_to_class_o = {int(k): v for k, v in data["idx_to_class_o"].items()}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiEfficientNet(len(idx_to_class_d), len(idx_to_class_o))
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device).eval()

        logger.info(
            "model_loaded",
            device=str(device),
            classes_d=len(idx_to_class_d),
            classes_o=len(idx_to_class_o),
        )
        return cls(model, idx_to_class_d, idx_to_class_o, device)

    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        t0 = time.perf_counter()
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)

        with torch.no_grad():
            out_d, out_o = self.model(tensors)
            probs_d = torch.softmax(out_d, dim=1)
            probs_o = torch.softmax(out_o, dim=1)
            preds_d = torch.argmax(out_d, dim=1)
            preds_o = torch.argmax(out_o, dim=1)

        per_image_ms = (time.perf_counter() - t0) / len(images) * 1000
        results = []
        for i in range(len(images)):
            d_idx = int(preds_d[i].item())
            o_idx = int(preds_o[i].item())
            d_val = self.idx_to_class_d[d_idx]
            o_val = self.idx_to_class_o[o_idx]

            results.append(
                {
                    "d_total": f"d{d_val}" if d_val < MAX_CLASS else "d7+",
                    "o_occupied": f"o{o_val}" if o_val < MAX_CLASS else "o7+",
                    "v_vacant": str(max(d_val - o_val, 0)),
                    "confidence_d": round(float(probs_d[i, d_idx].item()), 4),
                    "confidence_o": round(float(probs_o[i, o_idx].item()), 4),
                    "latency_ms": round(per_image_ms, 2),
                }
            )
        return results


class DynamicBatchQueue:
    """Collects individual predict requests and processes them in batches every max_wait_ms."""

    def __init__(
        self, predictor: ModelPredictor, max_batch_size: int = 32, max_wait_ms: float = 50.0
    ):
        self.predictor = predictor
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_event_loop().create_task(self._process_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def predict(self, image: Image.Image, request_id: str) -> dict:
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((image, future, request_id))
        return await future

    async def _process_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            batch: list[tuple] = []

            # Block until at least one item arrives
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self.max_wait_ms / 1000)
                batch.append(item)
            except asyncio.TimeoutError:
                continue

            # Collect additional items within the window without blocking
            deadline = loop.time() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            images = [b[0] for b in batch]
            futures = [b[1] for b in batch]

            try:
                results = await loop.run_in_executor(
                    _executor, self.predictor.predict_batch, images
                )
                for fut, result in zip(futures, results):
                    if not fut.cancelled():
                        fut.set_result(result)
            except Exception as exc:
                for fut in futures:
                    if not fut.cancelled():
                        fut.set_exception(exc)
