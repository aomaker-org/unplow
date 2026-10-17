# Path: unplow/tests/test_ctw_coder.py
"""Unit tests for Phase 6 BayesianCTSMixerCDF.

Tests Vovk/CTS posterior weight shifting toward bigram expert on repetition,
reset behavior, and lossless .unp roundtrip under Bit 3 (FLAG_BAYES_MIXER = 0x08).
"""

import numpy as np
import torch
from unplow.ctw_coder import FLAG_BAYES_MIXER, TOTAL_FREQ, BayesianCTSMixerCDF
from unplow.paradigm2 import LLMArithmeticCodec


def test_bayesian_cts_posterior_shifts_to_bigram_on_repetition():
    vocab_size = 128
    mixer = BayesianCTSMixerCDF(
        vocab_size=vocab_size,
        init_weights=(0.86, 0.05, 0.09),
        learning_rate=1.0,
        switching_alpha=0.02,
    )
    initial_bi_weight = float(mixer.weights[2])
    flat_logits = torch.zeros(vocab_size, dtype=torch.float32)

    # Repeatedly feed the bigram pattern (12 -> 34) and verify Expert 2 weight grows
    for _ in range(6):
        cdf_a = mixer.logits_to_bayes_mixed_cdf(flat_logits)
        assert cdf_a[0] == 0 and cdf_a[-1] == TOTAL_FREQ
        mixer.update(12)

        cdf_b = mixer.logits_to_bayes_mixed_cdf(flat_logits)
        assert np.all(np.diff(cdf_b) >= 1)
        mixer.update(34)

    final_bi_weight = float(mixer.weights[2])
    assert final_bi_weight > initial_bi_weight * 2.5
    assert len(mixer.trajectory) == 12


def test_bayesian_cts_mixer_lossless_container_roundtrip():
    codec = LLMArithmeticCodec(model_name="mock")
    text = "struct packet { uint32_t id; }; struct packet { uint32_t id; };"
    container_bytes = codec.compress_to_container(
        text, domain_id=3, use_bayes_mixer=True
    )
    reconstructed, meta = codec.decompress_from_container(container_bytes)
    assert reconstructed == text
    assert meta.domain_id == 3
    assert bool(meta.flags & FLAG_BAYES_MIXER) is True
    assert meta.bayes_mixer is True
    assert meta.is_bayes_mixer_enabled is True


def test_bayesian_cts_mixer_reset_and_active_weights():
    mixer = BayesianCTSMixerCDF(vocab_size=64)
    w0, w1, w2 = mixer._active_weights()
    assert w0 == 1.0 and w1 == 0.0 and w2 == 0.0

    mixer.update(5)
    w0, w1, w2 = mixer._active_weights()
    assert w1 > 0.0

    mixer.reset()
    assert len(mixer.history) == 0
    assert len(mixer.trajectory) == 0
    w0, w1, w2 = mixer._active_weights()
    assert w0 == 1.0 and w1 == 0.0 and w2 == 0.0

# end of file: unplow/tests/test_ctw_coder.py
