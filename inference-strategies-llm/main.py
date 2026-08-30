"""
main.py -- Dialogue Summarization Demo
=======================================

Entry point that demonstrates all inference strategies supported by
the DialogueSummarizer class:

    1. Tokenization round-trip
    2. Raw inference (no prompt engineering)
    3. Zero-shot inference with an instruction prompt
    4. Zero-shot inference with a FLAN-T5 prompt template
    5. One-shot inference
    6. Few-shot inference
    7. Generation configuration exploration (temperature sampling)

Run:
    python main.py
"""

from summarizer import DialogueSummarizer
from transformers import GenerationConfig


DIVIDER = "-" * 80


def section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_comparison(label: str, human_summary: str, model_output: str):
    """Print a side-by-side comparison of human vs. model summaries."""
    print(DIVIDER)
    print(f"HUMAN SUMMARY:\n  {human_summary}\n")
    print(f"{label}:\n  {model_output}")
    print(DIVIDER)
    print()


def main():
    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------
    summarizer = DialogueSummarizer()

    # Sample indices from the test split
    sample_indices = [40, 200]
    target_index = 200

    # ==================================================================
    # 1. Tokenization Demo
    # ==================================================================
    section("1 -- Tokenization")
    summarizer.demonstrate_tokenization("What time is it, Tom?")

    # ==================================================================
    # 2. Raw Inference (No Prompt Engineering)
    # ==================================================================
    section("2 -- Inference Without Prompt Engineering")

    for i, idx in enumerate(sample_indices, start=1):
        dialogue = summarizer.get_dialogue(idx)
        human = summarizer.get_summary(idx)
        output = summarizer.summarize_without_prompt(dialogue)

        print(f"Example {i}")
        print(f"INPUT DIALOGUE:\n{dialogue}\n")
        print_comparison("MODEL OUTPUT (no prompt)", human, output)

    # ==================================================================
    # 3. Zero-Shot -- Instruction Prompt
    # ==================================================================
    section("3 -- Zero-Shot Inference (Instruction Prompt)")

    for i, idx in enumerate(sample_indices, start=1):
        dialogue = summarizer.get_dialogue(idx)
        human = summarizer.get_summary(idx)
        output = summarizer.zero_shot(dialogue, template="instruction")

        print(f"Example {i}")
        print_comparison("ZERO-SHOT (instruction)", human, output)

    # ==================================================================
    # 4. Zero-Shot -- FLAN-T5 Template
    # ==================================================================
    section("4 -- Zero-Shot Inference (FLAN-T5 Template)")

    for i, idx in enumerate(sample_indices, start=1):
        dialogue = summarizer.get_dialogue(idx)
        human = summarizer.get_summary(idx)
        output = summarizer.zero_shot(dialogue, template="flan")

        print(f"Example {i}")
        print_comparison("ZERO-SHOT (FLAN template)", human, output)

    # ==================================================================
    # 5. One-Shot Inference
    # ==================================================================
    section("5 -- One-Shot Inference")

    human = summarizer.get_summary(target_index)
    output = summarizer.one_shot(example_index=40, target_index=target_index)

    print(f"Example dialogue used : index 40")
    print(f"Target dialogue       : index {target_index}")
    print_comparison("ONE-SHOT", human, output)

    # ==================================================================
    # 6. Few-Shot Inference
    # ==================================================================
    section("6 -- Few-Shot Inference")

    human = summarizer.get_summary(target_index)
    output = summarizer.few_shot(
        example_indices=[40, 80, 120],
        target_index=target_index,
    )

    print(f"Example dialogues used : indices 40, 80, 120")
    print(f"Target dialogue        : index {target_index}")
    print_comparison("FEW-SHOT", human, output)

    # ==================================================================
    # 7. Generation Configuration Exploration
    # ==================================================================
    section("7 -- Generation Configuration (Temperature)")

    temperatures = [0.1, 0.5, 1.0]
    human = summarizer.get_summary(target_index)

    for temp in temperatures:
        output = summarizer.few_shot(
            example_indices=[40, 80, 120],
            target_index=target_index,
            do_sample=True,
            temperature=temp,
        )
        print(f"Temperature = {temp}")
        print_comparison(f"FEW-SHOT (temp={temp})", human, output)

    # ------------------------------------------------------------------
    section("Done")
    print("All inference strategies completed successfully.")


if __name__ == "__main__":
    main()
