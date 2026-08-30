"""
Dialogue Summarizer
====================

A modular dialogue summarization engine built on top of the FLAN-T5
sequence-to-sequence transformer model. Supports zero-shot, one-shot,
and few-shot inference strategies with configurable generation parameters.

Model : google/flan-t5-base (248M parameters)
Dataset : knkarthick/dialogsum (13,460 dialogue-summary pairs)
"""

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig


class DialogueSummarizer:
    """
    End-to-end dialogue summarization pipeline.

    This class encapsulates model loading, dataset management, prompt
    construction, and text generation. It exposes explicit methods for
    each inference strategy so the caller can compare results without
    dealing with raw tokenizer or model internals.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    model : AutoModelForSeq2SeqLM
        The loaded sequence-to-sequence language model.
    tokenizer : AutoTokenizer
        The corresponding tokenizer (fast variant).
    dataset : DatasetDict
        The DialogSum dataset split into train / validation / test.
    """

    DEFAULT_MODEL = "google/flan-t5-base"
    DEFAULT_DATASET = "knkarthick/dialogsum"

    # ------------------------------------------------------------------
    # Prompt templates
    # ------------------------------------------------------------------
    #   FLAN-T5 was fine-tuned on many prompt templates.  The two used
    #   here are:
    #     1. A plain instruction ("Summarize the following conversation.")
    #     2. A FLAN-native template ("Dialogue: ... What was going on?")
    #   Template (2) tends to produce slightly better results because it
    #   matches one of the templates the model saw during training.
    # ------------------------------------------------------------------

    INSTRUCTION_TEMPLATE = (
        "Summarize the following conversation.\n\n"
        "{dialogue}\n\n"
        "Summary:\n"
    )

    FLAN_TEMPLATE = (
        "Dialogue:\n\n"
        "{dialogue}\n\n"
        "What was going on?\n"
    )

    FLAN_TEMPLATE_WITH_ANSWER = (
        "Dialogue:\n\n"
        "{dialogue}\n\n"
        "What was going on?\n"
        "{summary}\n\n\n"
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, model_name: str = DEFAULT_MODEL, dataset_name: str = DEFAULT_DATASET):
        """
        Load the model, tokenizer, and dataset.

        Parameters
        ----------
        model_name : str
            HuggingFace hub identifier for the model.
        dataset_name : str
            HuggingFace hub identifier for the dataset.
        """
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        print(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name)

        print("All resources loaded successfully.\n")

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def encode(self, text: str):
        """
        Tokenize a string into input-ID tensors.

        The tokenizer splits the raw text into sub-word tokens, maps each
        token to its vocabulary index, and returns a PyTorch tensor ready
        for the model.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        dict
            Dictionary with at least ``input_ids`` as a key.
        """
        return self.tokenizer(text, return_tensors="pt")

    def decode(self, token_ids):
        """
        Convert a tensor of token IDs back to a human-readable string.

        Parameters
        ----------
        token_ids : torch.Tensor
            1-D tensor of vocabulary indices.

        Returns
        -------
        str
            Decoded text with special tokens removed.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def demonstrate_tokenization(self, sentence: str = "What time is it, Tom?"):
        """
        Print the encoded token IDs and the round-trip decoded text for
        a given sentence.  Useful for verifying that the tokenizer is
        working correctly and for understanding the sub-word vocabulary.

        Parameters
        ----------
        sentence : str
            Any short sentence to tokenize.
        """
        encoded = self.encode(sentence)
        decoded = self.decode(encoded["input_ids"][0])

        print("Tokenization Demo")
        print("-" * 50)
        print(f"  Original  : {sentence}")
        print(f"  Token IDs : {encoded['input_ids'][0].tolist()}")
        print(f"  Decoded   : {decoded}")
        print()

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def _generate(self, prompt: str, generation_config: GenerationConfig = None) -> str:
        """
        Internal generation routine.

        Parameters
        ----------
        prompt : str
            The full prompt string (including any examples).
        generation_config : GenerationConfig, optional
            Custom generation parameters. Defaults to ``max_new_tokens=50``.

        Returns
        -------
        str
            The generated text.
        """
        if generation_config is None:
            generation_config = GenerationConfig(max_new_tokens=50)

        inputs = self.encode(prompt)
        output_ids = self.model.generate(
            inputs["input_ids"],
            generation_config=generation_config,
        )
        return self.decode(output_ids[0])

    # ------------------------------------------------------------------
    # Inference strategies
    # ------------------------------------------------------------------

    def summarize_without_prompt(self, dialogue: str, **gen_kwargs) -> str:
        """
        Feed the raw dialogue directly into the model without any
        instruction.  This is a baseline to show what happens when the
        model receives no task guidance.

        Parameters
        ----------
        dialogue : str
            The raw dialogue text.

        Returns
        -------
        str
            Model output (usually not a proper summary).
        """
        config = GenerationConfig(max_new_tokens=50, **gen_kwargs)
        return self._generate(dialogue, generation_config=config)

    def zero_shot(self, dialogue: str, template: str = "flan", **gen_kwargs) -> str:
        """
        Zero-shot inference: provide an instruction prompt with no
        examples.

        Parameters
        ----------
        dialogue : str
            The dialogue to summarize.
        template : str
            ``"instruction"`` for the plain instruction template or
            ``"flan"`` for the FLAN-native template.

        Returns
        -------
        str
            The generated summary.
        """
        fmt = self.INSTRUCTION_TEMPLATE if template == "instruction" else self.FLAN_TEMPLATE
        prompt = fmt.format(dialogue=dialogue)
        config = GenerationConfig(max_new_tokens=50, **gen_kwargs)
        return self._generate(prompt, generation_config=config)

    def one_shot(self, example_index: int, target_index: int, split: str = "test", **gen_kwargs) -> str:
        """
        One-shot inference: provide a single dialogue-summary example
        before the target dialogue.

        Parameters
        ----------
        example_index : int
            Index of the example to use as the demonstration.
        target_index : int
            Index of the dialogue to summarize.
        split : str
            Dataset split to use.

        Returns
        -------
        str
            The generated summary.
        """
        return self.few_shot([example_index], target_index, split=split, **gen_kwargs)

    def few_shot(self, example_indices: list, target_index: int, split: str = "test", **gen_kwargs) -> str:
        """
        Few-shot inference: provide multiple dialogue-summary examples
        before the target dialogue.

        Parameters
        ----------
        example_indices : list[int]
            Indices of the examples to include in the prompt.
        target_index : int
            Index of the dialogue to summarize.
        split : str
            Dataset split to use.

        Returns
        -------
        str
            The generated summary.
        """
        prompt = self._build_few_shot_prompt(example_indices, target_index, split)
        config = GenerationConfig(max_new_tokens=50, **gen_kwargs)
        return self._generate(prompt, generation_config=config)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_few_shot_prompt(self, example_indices: list, target_index: int, split: str = "test") -> str:
        """
        Assemble a few-shot prompt from the dataset.

        Each example is rendered with the FLAN template that includes
        the reference summary.  The final target dialogue is rendered
        without a summary so the model must generate one.

        Parameters
        ----------
        example_indices : list[int]
            Indices of complete examples (dialogue + summary).
        target_index : int
            Index of the dialogue to summarize.
        split : str
            Dataset split.

        Returns
        -------
        str
            The assembled prompt.
        """
        prompt = ""
        for idx in example_indices:
            entry = self.dataset[split][idx]
            prompt += self.FLAN_TEMPLATE_WITH_ANSWER.format(
                dialogue=entry["dialogue"],
                summary=entry["summary"],
            )

        target_dialogue = self.dataset[split][target_index]["dialogue"]
        prompt += self.FLAN_TEMPLATE.format(dialogue=target_dialogue)
        return prompt

    # ------------------------------------------------------------------
    # Dataset access
    # ------------------------------------------------------------------

    def get_dialogue(self, index: int, split: str = "test") -> str:
        """Return the raw dialogue text at *index* in *split*."""
        return self.dataset[split][index]["dialogue"]

    def get_summary(self, index: int, split: str = "test") -> str:
        """Return the human-written reference summary at *index* in *split*."""
        return self.dataset[split][index]["summary"]
