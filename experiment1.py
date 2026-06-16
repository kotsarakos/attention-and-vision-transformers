"""
Part 1 — Scaled Dot-Product Attention in NumPy.

    - English Sentence: “Maria isn’t here”
    - Greek Sentence: “∆εν είναι εδώ η Μαρία”

    1. Load matrices and report the relevant dimensions
       (d_model, T, T', d_k, d_v).
    2. Implement scaled_dot_product_attention(Q, K, V) using ONLY NumPy.
    3. Run the function for self-attention (English sentence) and for
       cross-attention (Greek -> English) using the given projection
       matrices. Print the shapes of Q, K, V, A, Y and check them.
    4. Verify the outputs against torch.nn.functional.scaled_dot_product_attention
       with absolute tolerance 1e-5 (np.allclose).
    5. Plot the attention matrices A as two heatmaps with matplotlib.

Cross-attention convention used here:
    "from Greek to English" means queries come from the Greek sentence
    and keys/values come from the English sentence. Hence
        Q in R^{T_q x d_k},    T_q  = 5  (Greek)
        K in R^{T_kv x d_k},   T_kv = 3  (English)
        V in R^{T_kv x d_v}
        A in R^{T_q x T_kv}   (5 x 3)
        Y in R^{T_q x d_v}    (5 x 6)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent
ATOL = 1e-5

ENGLISH_SENTENCE = ["Maria", "isn't", "here"]
GREEK_SENTENCE = ["Δεν", "είναι", "εδώ", "η", "Μαρία"]


# Generic helpers (used by every subtask)

def load_vocab(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def embed(E: NDArray[np.float32], vocab: dict[str, int], tokens: list[str]) -> NDArray[np.float32]:
    """
    Return a (T, d_model) matrix with the row-embeddings of the tokens.
    """
    return E[[vocab[t] for t in tokens]]


def banner(title: str) -> None:
    line = "=" * 68
    print(f"\n{line}\n{title}\n{line}")


def assert_shape(name: str, arr: np.ndarray, expected: tuple[int, ...]) -> None:
    """
    Assert that arr.shape == expected, print a message, and raise an AssertionError if not.
    """
    ok = arr.shape == expected
    tag = "OK  " if ok else "FAIL"
    print(f"  [{tag}] {name:>1}.shape = {str(arr.shape):<10}  expected {expected}")
    assert ok, f"shape mismatch for {name}: got {arr.shape}, expected {expected}"


# Subtask 1 — Dimensions

def subtask_1(
    E: NDArray[np.float32],
    projections: dict[str, NDArray[np.float32]],
    X_en: NDArray[np.float32],
    X_el: NDArray[np.float32],
) -> tuple[int, int, int]:
    banner("Subtask 1 — Matrix dimensions")
    vocab_size, d_model = E.shape
    d_k = projections["W_Q_self"].shape[1]
    d_v = projections["W_V_self"].shape[1]

    print(f"  Embedding matrix   E.shape = {E.shape}   (|V|={vocab_size})")
    print(f"  d_model            = {d_model}")
    print(f"  d_k (queries/keys) = {d_k}")
    print(f"  d_v (values)       = {d_v}")

    print("\n  Self-attention (English):")
    print(f"    T = T' = {len(ENGLISH_SENTENCE)},  X_en.shape = {X_en.shape}")

    print("\n  Cross-attention (Greek -> English):")
    print(f"    queries  side : Greek,    T_q  = {len(GREEK_SENTENCE)}")
    print(f"    keys/val side : English,  T_kv = {len(ENGLISH_SENTENCE)}")
    print(f"    X_el.shape = {X_el.shape}, X_en.shape = {X_en.shape}")

    print("\n  Projection matrices:")
    for name, mat in projections.items():
        print(f"    {name:<10s} shape = {mat.shape}")

    return d_model, d_k, d_v


# Subtask 2 — Scaled dot-product attention in NumPy

def softmax(x: NDArray[np.float32], axis: int = -1) -> NDArray[np.float32]:
    """
    Numerically stable row-wise softmax.
    """
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x_shift)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: NDArray[np.float32],
    K: NDArray[np.float32],
    V: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Scaled dot-product attention (Vaswani et al., 2017):

        A = softmax_row( Q K^T / sqrt(d_k) )
        Y = A V

    Shapes:
        Q : (T_q,  d_k)
        K : (T_kv, d_k)
        V : (T_kv, d_v)
    Returns:
        Y : (T_q, d_v)
        A : (T_q, T_kv)
    """
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    A = softmax(scores, axis=-1)
    Y = A @ V
    return Y, A


# Subtask 3 — Run attention and check shapes

def subtask_3(
    X_en: NDArray[np.float32],
    X_el: NDArray[np.float32],
    projections: dict[str, NDArray[np.float32]],
    d_k: int,
    d_v: int,
) -> tuple[tuple, tuple]:
    banner("Subtask 3 — Q, K, V, A, Y shapes")
    T_en = len(ENGLISH_SENTENCE)
    T_el = len(GREEK_SENTENCE)

    print("\n[Self-attention — English]")
    Q_s = X_en @ projections["W_Q_self"]
    K_s = X_en @ projections["W_K_self"]
    V_s = X_en @ projections["W_V_self"]
    Y_s, A_s = scaled_dot_product_attention(Q_s, K_s, V_s)
    assert_shape("Q", Q_s, (T_en, d_k))
    assert_shape("K", K_s, (T_en, d_k))
    assert_shape("V", V_s, (T_en, d_v))
    assert_shape("A", A_s, (T_en, T_en))
    assert_shape("Y", Y_s, (T_en, d_v))

    print("\n[Cross-attention — Greek -> English]")
    # queries from Greek, keys / values from English
    Q_c = X_el @ projections["W_Q_cross"]
    K_c = X_en @ projections["W_K_cross"]
    V_c = X_en @ projections["W_V_cross"]
    Y_c, A_c = scaled_dot_product_attention(Q_c, K_c, V_c)
    assert_shape("Q", Q_c, (T_el, d_k))
    assert_shape("K", K_c, (T_en, d_k))
    assert_shape("V", V_c, (T_en, d_v))
    assert_shape("A", A_c, (T_el, T_en))
    assert_shape("Y", Y_c, (T_el, d_v))

    return (Q_s, K_s, V_s, A_s, Y_s), (Q_c, K_c, V_c, A_c, Y_c)


# Subtask 4 — Verify against PyTorch

def torch_reference(
    Q: NDArray[np.float32],
    K: NDArray[np.float32],
    V: NDArray[np.float32],
    d_k: int,
) -> NDArray[np.float32]:
    Q_t = torch.from_numpy(np.ascontiguousarray(Q, dtype=np.float32))
    K_t = torch.from_numpy(np.ascontiguousarray(K, dtype=np.float32))
    V_t = torch.from_numpy(np.ascontiguousarray(V, dtype=np.float32))
    Y_t = F.scaled_dot_product_attention(Q_t, K_t, V_t, scale=1.0 / np.sqrt(d_k))
    return Y_t.numpy()


def subtask_4(self_outputs: tuple, cross_outputs: tuple, d_k: int) -> None:
    banner("Subtask 4 — Verify against PyTorch (atol = 1e-5)")
    Q_s, K_s, V_s, _, Y_s = self_outputs
    Q_c, K_c, V_c, _, Y_c = cross_outputs

    Y_s_t = torch_reference(Q_s, K_s, V_s, d_k)
    Y_c_t = torch_reference(Q_c, K_c, V_c, d_k)

    self_ok = bool(np.allclose(Y_s, Y_s_t, atol=ATOL))
    cross_ok = bool(np.allclose(Y_c, Y_c_t, atol=ATOL))
    self_diff = float(np.max(np.abs(Y_s - Y_s_t)))
    cross_diff = float(np.max(np.abs(Y_c - Y_c_t)))

    print(f"  Self-attention  : np.allclose = {self_ok},  max|Δ| = {self_diff:.2e}")
    print(f"  Cross-attention : np.allclose = {cross_ok},  max|Δ| = {cross_diff:.2e}")

    np.set_printoptions(precision=4, suppress=True)
    print("\n  Y_self  (NumPy)  :"); print(Y_s)
    print("\n  Y_self  (PyTorch):"); print(Y_s_t)
    print("\n  Y_cross (NumPy)  :"); print(Y_c)
    print("\n  Y_cross (PyTorch):"); print(Y_c_t)


# Subtask 5 — Heatmaps

def plot_attention(
    A: NDArray[np.float32],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(A, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Keys")
    ax.set_ylabel("Queries")
    ax.set_title(title)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = float(A[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def subtask_5(A_self: NDArray[np.float32], A_cross: NDArray[np.float32]) -> None:
    banner("Subtask 5 — Heatmaps of attention matrices A")
    p_self = ROOT / "attention_self.png"
    p_cross = ROOT / "attention_cross.png"

    plot_attention(A_self,  ENGLISH_SENTENCE, ENGLISH_SENTENCE,
                   "Self-attention (English)",            p_self)
    plot_attention(A_cross, GREEK_SENTENCE, ENGLISH_SENTENCE,
                   "Cross-attention (Greek -> English)",  p_cross)

    print(f"  Saved: {p_self.name}")
    print(f"  Saved: {p_cross.name}")
    print(f"  Row sums (self)  = {A_self.sum(axis=1)}    (should all equal 1.0)")
    print(f"  Row sums (cross) = {A_cross.sum(axis=1)}   (should all equal 1.0)")


# Entry point
def main() -> None:
    E = np.load(ROOT / "E.npy")
    projections = {
        "W_Q_self":  np.load(ROOT / "W_Q_self.npy"),
        "W_K_self":  np.load(ROOT / "W_K_self.npy"),
        "W_V_self":  np.load(ROOT / "W_V_self.npy"),
        "W_Q_cross": np.load(ROOT / "W_Q_cross.npy"),
        "W_K_cross": np.load(ROOT / "W_K_cross.npy"),
        "W_V_cross": np.load(ROOT / "W_V_cross.npy"),
    }
    vocab = load_vocab(ROOT / "vocab.json")

    X_en = embed(E, vocab, ENGLISH_SENTENCE)
    X_el = embed(E, vocab, GREEK_SENTENCE)

    _, d_k, d_v = subtask_1(E, projections, X_en, X_el)
    self_outputs, cross_outputs = subtask_3(X_en, X_el, projections, d_k, d_v)
    subtask_4(self_outputs, cross_outputs, d_k)
    subtask_5(self_outputs[3], cross_outputs[3])


if __name__ == "__main__":
    main()
