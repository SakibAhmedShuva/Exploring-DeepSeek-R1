# Exploring DeepSeek-R1 Language Models with Transformers

This repository contains a Jupyter Notebook (`DeepSeek_R1.ipynb`) demonstrating various ways to interact with **DeepSeek-R1 series** language models (e.g., `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` as explored in the notebook) using the Hugging Face `transformers` library.

The notebook explores:
*   **Basic Text Generation:** Using the `pipeline` for quick text generation.
*   **System Prompting for Structured Output:** Specifically, forcing JSON-only responses for tasks like Named Entity Recognition (NER).
*   **4-bit Quantization:** Loading and running models in 4-bit precision using `bitsandbytes` for reduced memory footprint.
*   **Manual Model and Tokenizer Interaction:** Directly using `AutoModelForCausalLM` and `AutoTokenizer` for more control.
*   **Chat Templating:** Applying chat templates for conversational AI.
*   **Streaming Output:** Generating text token by token for a more interactive experience using `TextStreamer`.
*   **Performance Measurement:** Basic timing of generation tasks.

## 🚀 Notebook: `DeepSeek_R1.ipynb`

This notebook is structured to guide you through different aspects of using a DeepSeek-R1 model. While the examples in the notebook use `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, the techniques are generally applicable to other models in the DeepSeek-R1 series.

### Key Sections:

1.  **Setup & Installation:**
    *   Installs necessary libraries: `transformers`, `accelerate`, `bitsandbytes`, `torch`.
2.  **Pipeline-based Text Generation:**
    *   Initializes the `text-generation` pipeline with an example DeepSeek-R1 model.
    *   Demonstrates how to format messages (system prompt, user query) for the pipeline.
    *   Highlights the importance of `return_full_text=False` for chat interactions.
    *   Shows an example of NER extraction with strict JSON output rules defined in a system prompt.
3.  **4-bit Quantization with Manual Model Loading:**
    *   Defines a `BitsAndBytesConfig` for 4-bit quantization (nf4, bfloat16 compute, double quantization).
    *   Loads the tokenizer and model manually using `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` with the quantization config (example shown with `DeepSeek-R1-Distill-Qwen-14B`).
    *   Applies chat templating manually.
    *   Performs text generation using `model.generate()`.
    *   Demonstrates decoding only the newly generated tokens.
    *   Includes examples for both NER extraction and a chatbot persona ("Health PRO AI").
4.  **Streaming Output:**
    *   Uses the 4-bit quantized model.
    *   Initializes `TextStreamer` from `transformers`.
    *   Passes the `streamer` object to `model.generate()` to print tokens as they are generated.
    *   Shows the assembled response after streaming is complete.

### Observations from the Notebook (using `DeepSeek-R1-Distill-Qwen-14B` as an example):

*   DeepSeek-R1 models can follow complex system prompts to produce structured JSON output, although they might sometimes include a "thought process" (e.g., `<think>...</think>`) which may need to be filtered.
*   4-bit quantization significantly reduces the memory required to run these models, making them more accessible on consumer GPUs (like the T4 used in Colab).
*   The generation speed with 4-bit quantization for the example model in the notebook was observed to be around 2 minutes for longer outputs on a T4 GPU.
*   Streaming output provides a much better user experience for interactive applications.
*   Proper chat templating is crucial for getting coherent responses in a conversational context with these models.
*   Model outputs from the DeepSeek-R1 series (and many LLMs) might include special tokens or end-of-sequence tokens that may need to be handled or stripped during post-processing.

## 🛠️ Prerequisites

*   Python 3.8+
*   PyTorch
*   Hugging Face Transformers, Accelerate, BitsAndBytes, SentencePiece
*   A CUDA-enabled GPU is highly recommended, especially for 4-bit quantization (the notebook was run on a T4 GPU).

## ⚙️ Setup & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SakibAhmedShuva/Exploring-DeepSeek-R1.git
    cd Exploring-DeepSeek-R1
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    It's recommended to create a `requirements.txt` file with the following content:
    ```
    transformers
    accelerate
    bitsandbytes
    torch
    sentencepiece
    ```
    Then install using:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, you can run the installation cells within the Jupyter Notebook.

4.  **Hugging Face Hub Token (Optional but Recommended):**
    The notebook outputs indicate warnings about a missing `HF_TOKEN`. While public models can often be downloaded without it, authenticating with the Hugging Face Hub is good practice and might be required for certain models or features.
    *   Create a token in your Hugging Face settings tab: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    *   You can then log in via the CLI:
        ```bash
        huggingface-cli login
        ```
    *   Or, if using Google Colab, set it as a secret.

5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook DeepSeek_R1.ipynb
    ```
    Or, if you prefer JupyterLab:
    ```bash
    jupyter lab DeepSeek_R1.ipynb
    ```
    If you are using Google Colab, simply upload the notebook and run the cells. Ensure you have a GPU runtime selected.
    Remember to **change the `model_id` variable** in the notebook if you wish to experiment with a different DeepSeek-R1 variant.

## 📝 Notes

*   "DeepSeek-R1" refers to a series of models. The example model used in the notebook, `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, is a distilled version. Other DeepSeek-R1 variants may have different characteristics, sizes, and resource requirements.
*   When using quantization, ensure your `bitsandbytes` installation is compatible with your CUDA version.
*   The `torch_dtype` (e.g., `torch.bfloat16` or `torch.float16`) should be chosen based on your GPU's capabilities. `bfloat16` is generally preferred on newer GPUs (Ampere architecture and later).
*   The notebook outputs sometimes show a warning: `Sliding Window Attention is enabled but not implemented for sdpa; unexpected results may be encountered.` This is an informational message from the library and its relevance may depend on the specific DeepSeek-R1 model architecture.

## 💡 Potential Future Work

*   Fine-tuning a DeepSeek-R1 model on a specific task.
*   Comparing performance (speed, output quality, VRAM usage) of different DeepSeek-R1 variants.
*   Comparing different quantization methods (e.g., GGUF, AWQ) if applicable to DeepSeek-R1 models.
*   Building a simple web application (e.g., using Gradio or Streamlit) to interact with a chosen DeepSeek-R1 model.
*   More in-depth analysis of the "thought process" output and how to mitigate or leverage it.
*   Experimenting with different decoding strategies (temperature, top_p, top_k, repetition penalty) for various tasks.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
