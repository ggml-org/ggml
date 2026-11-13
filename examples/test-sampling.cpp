#include "common.h"

#include <cstdio>

int main() {
    const std::vector<std::vector<float>> cases = {
        { 1.0f, 3.0f, 2.0f },
        { -3.0f, -1.0f, -2.0f },
        { -1.0f, 0.0f, -2.0f },
        { 1.0f, 3.0f, 3.0f },
        { 3.0f, 1.0f, 2.0f },
        { 1.0f, 2.0f, 3.0f },
        { 0.0f },
    };
    const std::vector<gpt_vocab::id> expected = { 1, 1, 1, 1, 0, 2, 0 };

    for (size_t i = 0; i < cases.size(); ++i) {
        gpt_vocab vocab;
        for (size_t j = 0; j < cases[i].size(); ++j) {
            vocab.id_to_token[j] = std::to_string(j);
        }

        for (double temp : { 0.0, -1.0 }) {
            for (int top_k : { 1, 2, 40 }) {
                for (double top_p : { 0.1, 1.0 }) {
                    for (unsigned seed : { 0u, 42u, 1234u }) {
                        std::mt19937 rng(seed);
                        const std::mt19937 rng_before = rng;
                        const gpt_vocab::id id = gpt_sample_top_k_top_p(
                            vocab, cases[i].data(), top_k, top_p, temp, rng);
                        if (id != expected[i] || rng != rng_before) {
                            fprintf(stderr, "case %zu: temp=%g top_k=%d top_p=%g seed=%u: expected %d, got %d, rng changed=%d\n",
                                    i, temp, top_k, top_p, seed, expected[i], id, rng != rng_before);
                            return 1;
                        }
                    }
                }
            }
        }

        // Positive-temperature top-k=1 must still select the maximum logit.
        std::mt19937 rng(42);
        const gpt_vocab::id id = gpt_sample_top_k_top_p(vocab, cases[i].data(), 1, 1.0, 0.8, rng);
        if (cases[i][id] != cases[i][expected[i]]) {
            fprintf(stderr, "case %zu: positive-temperature top-k=1 selected %d\n", i, id);
            return 1;
        }
    }

    return 0;
}
