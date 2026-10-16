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
