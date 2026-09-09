import torch  # noqa: F401  (preloads CUDA/cuDNN DLLs before onnxruntime)
import numpy as np
import onnxruntime as ort

MODEL_PATH = "./models/voice_detector.onnx"
AI_INDEX = 1
HUMAN_INDEX = 0

_session = None


def _get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print(f"[core.main] Model loaded on: {_session.get_providers()[0]}")
    return _session


def predict(input_values: np.ndarray):
    """
    input_values: numpy float32 array, shape (1, sequence_length)
    returns: (ai_probability, human_probability)
    """
    session = _get_session()

    logits = session.run(None, {"input_values": input_values})[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    ai_probability = float(probs[0][AI_INDEX])
    human_probability = float(probs[0][HUMAN_INDEX])

    return ai_probability, human_probability