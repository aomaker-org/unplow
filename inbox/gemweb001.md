Treating a Large Language Model (LLM) as compressed data—and writing code to decompress it—sits at the intersection of algorithmic information theory and autoregressive sequence modeling. Depending on what you treat as the "archive," this problem splits into two distinct mathematical paradigms:

1. **Lossy Static Weight Decompression (Unpacking the Training Corpus from $\theta$):** Treating the static neural network parameters $\theta$ themselves as a self-contained, lossy archive of the pre-training corpus $D$, and extracting memorized sequences from the weights **with zero external bitstream**.
2. **Lossless Autoregressive Arithmetic Decompression (Unpacking a Bitstream via $P_{\theta}$):** Treating the LLM as a frozen probabilistic measure $P_{\theta}(x_t \mid x_{1..t-1})$ paired with an **Arithmetic Coder**, where a compact bitstream (or even an arbitrary seed/hash) is decompressed into full text by navigating the model's cumulative token probability intervals.

---

## Theoretical Foundations & Attribution

The equivalence between language modeling and data compression rests on four foundational bodies of research:

* **Shannon's Source Coding Theorem (Claude Shannon, 1948 & 1951):** In *A Mathematical Theory of Communication* (1948) and *Prediction and Entropy of Printed English* (1951), Shannon proved that the optimal code length for a symbol $x_t$ given prior context $x_{1..t-1}$ under a model $P_{\theta}$ is $-\log_2 P_{\theta}(x_t \mid x_{1..t-1})$ bits. Minimizing standard cross-entropy loss during LLM pre-training is mathematically identical to minimizing the expected bit-length required to losslessly encode the training dataset.
* **Solomonoff Induction & The Hutter Prize (Ray Solomonoff, 1964; Andrey Kolmogorov, 1965; Marcus Hutter, 2005):** Hutter formalized the link between general intelligence and lossless compression by demonstrating that finding the shortest program that outputs a dataset $D$ (Kolmogorov complexity) requires learning the causal rules governing $D$.
* **Neural Weights as Implicit Archives (Ilya Sutskever, 2023; Mor Geva et al., 2021):** In his 2023 Simons Institute lecture (*An Observation on Generalization*), Sutskever framed unsupervised training as joint compression: the total description length of a dataset $D$ given model weights $\theta$ is the size of the weights $\vert{}\theta\vert{}$ plus the cross-entropy reconstruction residuals $\sum \mathcal{L}_{\theta}(D)$. Because $\vert{}\theta\vert{}$ (e.g., 3 GB for a 1.5B FP16 model) is orders of magnitude smaller than $\vert{}D\vert{}$ (terabytes of text), the model compresses general patterns into circuits while storing verbatim low-entropy facts inside MLP layers acting as associative key-value memories (**Geva et al., 2021**, *Transformer Feed-Forward Layers Are Key-Value Memories*).
* **Training Data Extraction vs. Arithmetic Coding:**
* **Nicholas Carlini, Florian Tramèr, Eric Wallace, et al. (2021):** In *Extracting Training Data from Large Language Models* (USENIX Security 2021), Carlini et al. demonstrated how to "decompress" verbatim training data directly out of static weights $\theta$ by sampling from the model and ranking outputs by the ratio of model cross-entropy to intrinsic `zlib` entropy.
* **Jorma Rissanen (1976); Ian Witten, Radford Neal, & John Cleary (1987); Fabrice Bellard (2019–2021); Grégoire Delétang et al. (2023):** Rissanen invented Arithmetic Coding, which Witten, Neal, and Cleary operationalized into fixed-point integer interval splitting. Bellard (`NNCP`, `ts_zip`) and **Delétang, Ruoss, et al. (2023/ICLR 2024)** at Google DeepMind (*Language Modeling Is Compression*) proved that pairing frozen pre-trained LLMs with an arithmetic coder produces a state-of-the-art general-purpose lossless compressor and decompressor.



| Dimension | Paradigm 1: Static Weight Decompression (Carlini et al., 2021) | Paradigm 2: Arithmetic Decompression (Delétang et al., 2023) |
| --- | --- | --- |
| **What is the compressed payload?** | The model weights $\theta$ alone (0 external bytes required) | A compact bitstream $B \in \{0,1\}^k$ (plus model weights $\theta$ as the dictionary) |
| **Fidelity** | **Lossy** reconstruction of the pre-training corpus (exact only on memorized basins) | **100% Lossless** reconstruction of encoded documents (or exact bijective sampling on random bits) |
| **Core Mechanism** | Temperature-decayed basin exploration + Perplexity-to-`zlib` filtering | 32-bit fixed-point interval partitioning over quantized Softmax CDFs |
| **Information Metric** | Memorization score $\mathcal{M}(x) = H_{\text{zlib}}(x) / \mathcal{L}_{\theta}(x)$ | Compression rate $\frac{1}{T}\sum_{t=1}^T -\log_2 P_{\theta}(x_t \mid x_{1..t-1})$ bits/token |

---

## Paradigm 1: Decompressing Static Weights $\theta$ (Training Corpus Extraction)

When you have only the LLM weights and want to unpack what the model memorized during pre-training, naive greedy decoding from the Beginning-of-Sequence (`BOS`) token immediately collapses into generic, repetitive loops (e.g., `"The The The..."`).

To systematically decompress the parameter manifold, we implement the two-stage pipeline introduced by **Carlini et al. (2021)**:

1. **Temperature-Decayed Basin Exploration:** For the first $k$ tokens (the prefix probe), we sample with a high temperature ($\tau_{\text{high}} = 1.2$) or top-$k$ sampling so the trajectory jumps into diverse semantic basins across the weight matrices. After step $k$, we decay the temperature sharply ($\tau_{\text{low}} = 0.2$) so the trajectory locks onto the highest-probability memorized path within that basin.
2. **Intrinsic vs. Model Entropy Calibration:** A sequence $x_{1..T}$ is genuinely memorized in the weights (rather than trivial repetition) if the LLM finds it highly predictable (low cross-entropy loss $\mathcal{L}_{\theta}(x)$ in bits), **while** a model-free syntactic compressor like `zlib` (**Ziv & Lempel, 1977**) finds it complex and incompressible (high $H_{\text{zlib}}(x)$). We also compute **Carlini's Lowercase Calibration Ratio**, $\mathcal{L}_{\theta}(\text{lower}(x)) / \mathcal{L}_{\theta}(x)$: if case-folding a sequence dramatically *increases* the LLM's perplexity, the weights have memorized the exact casing of a specific proper noun, code block, or license header.

```python
import zlib
from dataclasses import dataclass
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ExtractedRecord:
    text: str
    model_bits_per_byte: float
    zlib_bits_per_byte: float
    memorization_score: float
    casing_sensitivity: float


class WeightDecompressor:
    """
    Unpacks memorized training sequences directly from static LLM weights
    using temperature-decayed basin probing and Carlini et al. (2021)
    zlib-to-perplexity entropy calibration.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def compute_sequence_bits(self, text: str) -> float:
        """
        Computes the exact Shannon information content (in bits) assigned to `text`
        by the LLM: sum_{t} -log_2 P_theta(x_t | x_{<t}).
        """
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = enc.input_ids
        if input_ids.shape[1] < 2:
            return float("inf")

        outputs = self.model(input_ids)
        # Shift logits and targets for causal next-token evaluation
        shift_logits = outputs.logits[:, :-1, :].contiguous().to(torch.float32)
        shift_labels = input_ids[:, 1:].contiguous()

        # Cross-entropy in nats, converted to base-2 bits via 1 / ln(2)
        loss_nats = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        return float(loss_nats.item() / torch.log(torch.tensor(2.0)).item())

    @torch.no_grad()
    def probe_weight_basins(
        self,
        num_samples: int = 16,
        seq_len: int = 128,
        explore_steps: int = 12,
        temp_explore: float = 1.15,
        temp_lock: float = 0.25,
        top_k: int = 60,
        seed_prefixes: List[str] = None,
    ) -> List[ExtractedRecord]:
        """
        Probes the LLM weight space by dispersing into diverse semantic basins
        during `explore_steps` and then annealing into low-temperature extraction.
        """
        if not seed_prefixes:
            # Diverse structural anchors to jump into distinct parameter subspaces
            seed_prefixes = [
                "Copyright (c) ",
                "def ",
                "The following is a verbatim ",
                "According to the official ",
            ]

        records: List[ExtractedRecord] = []

        for i in range(num_samples):
            prefix = seed_prefixes[i % len(seed_prefixes)]
            input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)
            past_key_values = None
            curr_ids = input_ids

            for step in range(seq_len):
                outputs = self.model(
                    input_ids=curr_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :].to(torch.float32)

                # High temperature early to jump basins; low temperature later to lock onto memorized paths
                temp = temp_explore if step < explore_steps else temp_lock
                scaled_logits = logits / temp

                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                curr_ids = next_token

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            raw_bytes = decoded_text.encode("utf-8")
            num_bytes = max(len(raw_bytes), 1)

            # 1. Model compressed description length (bits per byte)
            model_total_bits = self.compute_sequence_bits(decoded_text)
            model_bpb = model_total_bits / num_bytes

            # 2. Intrinsic syntactic entropy via zlib (bits per byte)
            zlib_total_bits = len(zlib.compress(raw_bytes, level=9)) * 8.0
            zlib_bpb = zlib_total_bits / num_bytes

            # 3. Lowercase perturbation ratio (detects verbatim casing memorization)
            lower_bits = self.compute_sequence_bits(decoded_text.lower())
            casing_ratio = (lower_bits / max(model_total_bits, 1e-6))

            # Carlini et al. memorization score: high zlib entropy + low model perplexity
            memorization_score = zlib_bpb / max(model_bpb, 1e-6)

            records.append(
                ExtractedRecord(
                    text=decoded_text,
                    model_bits_per_byte=round(model_bpb, 3),
                    zlib_bits_per_byte=round(zlib_bpb, 3),
                    memorization_score=round(memorization_score, 3),
                    casing_sensitivity=round(casing_ratio, 3),
                )
            )

        # Sort descending by memorization score (highest ratio of intrinsic complexity to model surprise)
        records.sort(key=lambda r: r.memorization_score, reverse=True)
        return records


if __name__ == "__main__":
    # Can use "gpt2" for quick CPU testing or "Qwen/Qwen2.5-1.5B"
    extractor = WeightDecompressor(model_name="gpt2")
    extracted = extractor.probe_weight_basins(num_samples=4, seq_len=64)
    for idx, rec in enumerate(extracted, 1):
        print(f"\n--- Decompressed Weight Basin #{idx} ---")
        print(f"Memorization Score (Zlib/LLM): {rec.memorization_score} "
              f"(LLM: {rec.model_bits_per_byte} bpb | Zlib: {rec.zlib_bits_per_byte} bpb | "
              f"Case Ratio: {rec.casing_sensitivity})")
        print(rec.text)

```

---

## Paradigm 2: Lossless Arithmetic Compression & Decompression Engine

In the second paradigm (**Delétang et al., 2023; Bellard, 2021**), the LLM's forward pass acts as a dynamic coordinate system over the unit interval $[0, 1)$.

### How Fixed-Point LLM Arithmetic Decompression Works

At each token index $t$, the LLM takes the previously decompressed tokens $x_{1..t-1}$ and outputs a logit vector $\mathbf{z}_t \in \mathbb{R}^{\vert{}\mathcal{V}\vert{}}$.

1. **Quantized Positive CDF Construction:** Standard IEEE-754 floating-point softmax probabilities cannot be used directly because rounding differences or zero-probability tokens ($P(v) = 0$) collapse the coder's interval width to zero. Instead, we map $\text{softmax}(\mathbf{z}_t)$ onto an exact integer **Cumulative Distribution Function (CDF)** array $C[0..\vert{}\mathcal{V}\vert{}]$ with total frequency mass $M = 2^{24}$ ($16,777,216$), guaranteeing every token $v \in \mathcal{V}$ has frequency $C[v+1] - C[v] \ge 1$.
2. **32-Bit Interval Partitioning (Witten, Neal, & Cleary, 1987):** We maintain an active integer range $[L, H]$ inside $[0, 2^{32}-1]$ and a 32-bit window $V$ sliding along the compressed bitstream. At step $t$, with span $S = H - L + 1$, the decoder projects $V$ into the LLM's CDF space:

$$\text{scaled\_target} = \left\lfloor \frac{(V - L + 1)\cdot M - 1}{S} \right\rfloor$$



Using binary search (`np.searchsorted`), the decoder finds the unique token $x_t$ satisfying $C[x_t] \le \text{scaled\_target} \lt C[x_t + 1]$, narrows $[L, H]$ to that token's sub-interval, shifts out matching leading bits (renormalization), and feeds $x_t$ into the LLM's KV-cache to obtain the distribution for $x_{t+1}$.

### Bonus Property: "Generative Decompression" of Arbitrary Bitstreams

Because Arithmetic Coding establishes an exact measure-preserving bijection between uniformly distributed bitstreams in $[0, 1)$ and the LLM's joint probability space $P_{\theta}(X)$, feeding **any arbitrary byte sequence** (such as a cryptographic SHA-256 hash or random bytes) into `decompress_bits()` below deterministically "decompresses" those bytes into coherent text sampled proportionally to the LLM's learned manifold.

```python
import numpy as np
import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# 32-bit Fixed-Point Arithmetic Coding Constants (Witten, Neal, & Cleary, 1987)
PRECISION_BITS = 32
MAX_VAL = (1 << PRECISION_BITS) - 1          # 0xFFFFFFFF
HALF = 1 << (PRECISION_BITS - 1)             # 0x80000000
QUARTER = 1 << (PRECISION_BITS - 2)          # 0x40000000
THREE_QUARTERS = 3 << (PRECISION_BITS - 2)   # 0xC0000000
TOTAL_FREQ = 1 << 24                         # 16,777,216 (fits vocabularies > 150k with >=1 count/token)


class LLMArithmeticCodec:
    """
    Lossless Compressor and Decompressor powered by a Causal LLM's next-token
    probability distribution (Delétang et al., 2023; Bellard, 2021).
    """

    def __init__(self, model_name: str = "gpt2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use float32 during arithmetic coding to guarantee deterministic logit evaluation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        self.bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )

    @staticmethod
    def logits_to_integer_cdf(logits: torch.Tensor) -> np.ndarray:
        """
        Converts raw 1D token logits into a strictly monotonic integer CDF
        where C[0] = 0, C[V] = TOTAL_FREQ, and C[i+1] - C[i] >= 1 for all i.
        """
        probs = torch.softmax(logits.to(torch.float64), dim=-1).cpu().numpy()
        vocab_size = probs.shape[0]
        remaining_mass = TOTAL_FREQ - vocab_size
        if remaining_mass <= 0:
            raise ValueError(f"TOTAL_FREQ ({TOTAL_FREQ}) must exceed vocab_size ({vocab_size}).")

        # Every token gets at least 1 unit of frequency mass to prevent zero-width collapse
        freqs = np.floor(probs * remaining_mass).astype(np.int64) + 1
        remainder = TOTAL_FREQ - int(np.sum(freqs))
        freqs[np.argmax(probs)] += remainder

        cdf = np.zeros(vocab_size + 1, dtype=np.int64)
        np.cumsum(freqs, out=cdf[1:])
        return cdf

    @torch.no_grad()
    def compress(self, text: str) -> Tuple[bytes, int]:
        """
        Compresses `text` into a compact byte payload using the LLM as the predictor.
        Returns (compressed_bytes, num_tokens).
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(token_ids)

        low = 0
        high = MAX_VAL
        pending_bits = 0
        output_bits: List[int] = []

        def emit_bit(bit: int):
            nonlocal pending_bits
            output_bits.append(bit)
            while pending_bits > 0:
                output_bits.append(1 - bit)
                pending_bits -= 1

        curr_input = torch.tensor([[self.bos_token_id]], device=self.device)
        past_key_values = None

        for token in token_ids:
            outputs = self.model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            cdf = self.logits_to_integer_cdf(outputs.logits[0, -1, :])

            span = high - low + 1
            c_low = int(cdf[token])
            c_high = int(cdf[token + 1])

            high = low + (span * c_high) // TOTAL_FREQ - 1
            low = low + (span * c_low) // TOTAL_FREQ

            while True:
                if high < HALF:
                    emit_bit(0)
                elif low >= HALF:
                    emit_bit(1)
                    low -= HALF
                    high -= HALF
                elif low >= QUARTER and high < THREE_QUARTERS:
                    pending_bits += 1
                    low -= QUARTER
                    high -= QUARTER
                else:
                    break
                low = (low << 1) & MAX_VAL
                high = ((high << 1) | 1) & MAX_VAL

            curr_input = torch.tensor([[token]], device=self.device)

        # Flush final interval state
        pending_bits += 1
        if low < QUARTER:
            emit_bit(0)
        else:
            emit_bit(1)

        # Pack bits into bytes
        while len(output_bits) % 8 != 0:
            output_bits.append(0)
        byte_array = bytearray()
        for i in range(0, len(output_bits), 8):
            byte_val = 0
            for b in output_bits[i : i + 8]:
                byte_val = (byte_val << 1) | b
            byte_array.append(byte_val)

        return bytes(byte_array), num_tokens

    @torch.no_grad()
    def decompress(self, compressed_bytes: bytes, num_tokens: int) -> str:
        """
        Decompresses a byte payload back into text by stepping through the exact same
        LLM next-token CDFs. Also works as a deterministic 'manifold decompressor'
        when given arbitrary/random bytes!
        """
        # Unpack bytes into bit stream
        bits: List[int] = []
        for byte_val in compressed_bytes:
            for shift in range(7, -1, -1):
                bits.append((byte_val >> shift) & 1)

        bit_idx = 0

        def read_bit() -> int:
            nonlocal bit_idx
            if bit_idx < len(bits):
                b = bits[bit_idx]
                bit_idx += 1
                return b
            return 0  # Zero-padding beyond EOF

        low = 0
        high = MAX_VAL
        value = 0
        for _ in range(PRECISION_BITS):
            value = (value << 1) | read_bit()

        decoded_tokens: List[int] = []
        curr_input = torch.tensor([[self.bos_token_id]], device=self.device)
        past_key_values = None

        for _ in range(num_tokens):
            outputs = self.model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            cdf = self.logits_to_integer_cdf(outputs.logits[0, -1, :])

            span = high - low + 1
            # Invert the encoder's floor projection to locate the target cumulative frequency
            scaled_target = ((value - low + 1) * TOTAL_FREQ - 1) // span
            token = int(np.searchsorted(cdf, scaled_target, side="right") - 1)
            decoded_tokens.append(token)

            c_low = int(cdf[token])
            c_high = int(cdf[token + 1])
            high = low + (span * c_high) // TOTAL_FREQ - 1
            low = low + (span * c_low) // TOTAL_FREQ

            while True:
                if high < HALF:
                    pass
                elif low >= HALF:
                    value -= HALF
                    low -= HALF
                    high -= HALF
                elif low >= QUARTER and high < THREE_QUARTERS:
                    value -= QUARTER
                    low -= QUARTER
                    high -= QUARTER
                else:
                    break
                low = (low << 1) & MAX_VAL
                high = ((high << 1) | 1) & MAX_VAL
                value = ((value << 1) | read_bit()) & MAX_VAL

            curr_input = torch.tensor([[token]], device=self.device)

        return self.tokenizer.decode(decoded_tokens, clean_up_tokenization_spaces=False)


if __name__ == "__main__":
    codec = LLMArithmeticCodec(model_name="gpt2")

    # 1. Lossless Round-Trip Compression & Decompression
    original_text = (
        "Information theory establishes that any autoregressive language model "
        "capable of predicting the next token defines an optimal lossless compressor."
    )
    payload, n_tokens = codec.compress(original_text)
    reconstructed = codec.decompress(payload, n_tokens)

    raw_len = len(original_text.encode("utf-8"))
    comp_len = len(payload)
    print(f"Original Size:      {raw_len} bytes")
    print(f"LLM Compressed:     {comp_len} bytes ({comp_len * 8 / raw_len:.2f} bits/byte)")
    print(f"Exact Match:        {original_text == reconstructed}")
    print(f"Reconstructed Text: {reconstructed}\n")

    # 2. Decompressing an Arbitrary 16-Byte Seed directly through the LLM Manifold
    import hashlib
    seed_bytes = hashlib.sha256(b"unpack_the_latent_manifold").digest()[:16]
    unpacked_manifold_text = codec.decompress(seed_bytes, num_tokens=30)
    print("Decompressed from 16-byte SHA-256 Digest:")
    print(unpacked_manifold_text)

```

---

## Critical Engineering Nuances When Scaling

* **Hardware Non-Determinism in Floating-Point MatMuls:** In Paradigm 2, the arithmetic decoder requires that the LLM produce the **exact** same integer CDF bit-for-bit during decompression as it did during compression. Because parallel CUDA reductions (`torch.matmul` in FP16/BF16) are non-associative across different GPU architectures, production neural compressors like **Fabrice Bellard's `ts_zip**` and **DeepMind's implementation** either run inference in deterministic FP32 (`torch.use_deterministic_algorithms(True)`) or use integer-quantized inference kernels (such as `llama.cpp` / GGUF `Q8_0` deterministic CPU/GPU kernels).
* **KV-Cache Context Window Sliding:** If the sequence being decompressed exceeds the LLM's context window $W$, you cannot simply flush the entire KV-cache at token $W+1$ without degrading compression ratios back to unigram entropy. Instead, slide the KV-cache window in overlapping strides (e.g., retaining the most recent $W/2$ tokens in `past_key_values`) identically in both encoder and decoder.
