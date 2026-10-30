LLMInf

A lightweight Python library for simplifying LLM inference with Hugging Face Transformers.

Features

- Simple interface for text generation.
- Support for Hugging Face models.
- Configurable generation parameters.
- Optional conversation memory.
- Easy memory management.

Installation

Install directly from GitHub:

pip install git+https://github.com/OSMPRODUCTION87/LLMInf.git

Usage

from LLMInf import LLM

bot = LLM("model_name")

response = bot.g("Hello!")
print(response)

Requirements

- Python 3.9+
- PyTorch
- Transformers

License

See "LICENSE" (LICENSE) for usage terms.

Author

FID
