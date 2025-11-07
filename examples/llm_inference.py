"""
Large Language Model Inference with ONNX Runtime + TensorRT
Optimized text generation with Llama 3, Qwen, Mistral models
"""

import numpy as np
import onnxruntime as ort
from typing import List, Optional
import time
from transformers import AutoTokenizer


class LLMInference:
    """
    LLM Inference Engine with TensorRT optimization

    Supports:
    - Llama 3.2, 3.3
    - Qwen 2.5
    - Mistral
    - DeepSeek
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        use_tensorrt: bool = True,
        use_fp16: bool = True,
        max_length: int = 2048
    ):
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length

        # Setup providers
        providers = self._setup_providers(use_tensorrt, use_fp16)

        print(f"Loading model: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=providers)

        print("Model loaded successfully!")
        self._print_model_info()

    def _setup_providers(self, use_tensorrt: bool, use_fp16: bool) -> List:
        """Configure execution providers"""

        providers = []

        if use_tensorrt:
            trt_options = {
                'device_id': 0,
                'trt_max_workspace_size': 8 * 1024 * 1024 * 1024,  # 8GB
                'trt_fp16_enable': use_fp16,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_llm_cache',
                'trt_timing_cache_enable': True,
                'trt_force_sequential_engine_build': False,
            }
            providers.append(('TensorrtExecutionProvider', trt_options))
            print("TensorRT enabled for LLM inference")

        providers.extend([
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ])

        return providers

    def _print_model_info(self):
        """Print model input/output information"""
        print("\n=== Model Information ===")
        print("Inputs:")
        for inp in self.session.get_inputs():
            print(f"  - {inp.name}: {inp.shape} ({inp.type})")
        print("Outputs:")
        for out in self.session.get_outputs():
            print(f"  - {out.name}: {out.shape} ({out.type})")
        print("=" * 30 + "\n")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition

        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]

        print(f"Prompt: {prompt}")
        print(f"Generating {max_new_tokens} tokens...")

        # Generation loop
        generated_tokens = []
        total_time = 0

        for i in range(max_new_tokens):
            start_time = time.time()

            # Prepare inputs
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": np.ones_like(input_ids)
            }

            # Run inference
            outputs = self.session.run(None, ort_inputs)
            logits = outputs[0]  # [batch, seq_len, vocab_size]

            # Get next token logits
            next_token_logits = logits[0, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if generated_tokens:
                for token_id in set(generated_tokens):
                    next_token_logits[token_id] /= repetition_penalty

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < np.partition(
                    next_token_logits, -top_k
                )[-top_k]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_indices = np.argsort(next_token_logits)[::-1]
                sorted_logits = next_token_logits[sorted_indices]
                cumulative_probs = np.cumsum(self._softmax(sorted_logits))

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample next token
            probs = self._softmax(next_token_logits)
            next_token = np.random.choice(len(probs), p=probs)

            # Update input_ids
            input_ids = np.concatenate([
                input_ids,
                np.array([[next_token]])
            ], axis=1)

            generated_tokens.append(next_token)

            # Calculate metrics
            iteration_time = (time.time() - start_time) * 1000
            total_time += iteration_time

            # Print progress
            if (i + 1) % 10 == 0:
                avg_time = total_time / (i + 1)
                tokens_per_sec = 1000 / avg_time
                print(f"Generated {i+1}/{max_new_tokens} tokens | "
                      f"Avg: {avg_time:.2f}ms/token | "
                      f"Speed: {tokens_per_sec:.2f} tokens/sec")

            # Check for EOS token
            if next_token == self.tokenizer.eos_token_id:
                print("EOS token generated, stopping...")
                break

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Print statistics
        avg_latency = total_time / len(generated_tokens)
        print(f"\n=== Generation Stats ===")
        print(f"Tokens generated: {len(generated_tokens)}")
        print(f"Total time: {total_time:.2f}ms")
        print(f"Average latency: {avg_latency:.2f}ms/token")
        print(f"Throughput: {1000/avg_latency:.2f} tokens/sec")
        print("=" * 30 + "\n")

        return generated_text

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


def main():
    """Example usage"""

    # Initialize LLM
    llm = LLMInference(
        model_path="models/llama-3.2-1b.onnx",
        tokenizer_path="meta-llama/Llama-3.2-1B",
        use_tensorrt=True,
        use_fp16=True
    )

    # Example prompts
    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate fibonacci numbers:",
        "What are the benefits of using ONNX Runtime with TensorRT?",
    ]

    for prompt in prompts:
        print("\n" + "="*60)
        generated = llm.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9
        )
        print(f"\nGenerated:\n{generated}")
        print("="*60)


if __name__ == "__main__":
    main()
