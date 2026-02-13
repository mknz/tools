# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "llm",
#     "llm-ollama",
# ]
# ///
import argparse
import os

import llm


WORDS_CUTOFF = 200


def process(text: str, model_id: str) -> str:
    """Process input text with LLM."""
    prompt_text = f'以下のテキストを日本語に翻訳してください。「だ・である」調を用いること。改行等を適切に補うようにし、Markdown形式で出力すること。原文の意味を保ちつつ自然な日本語に翻訳し、また訳抜けがないように留意してください。文中に出てくる数字については半角で統一してください。説明なしで結果のみを出力すること。\n---\n{text}'
    model = llm.get_model(model_id)
    return model.prompt(prompt_text).text()


def count_words(text: str) -> int:
    """Counts the number of words in a given text."""
    return len(text.split())


def process_and_save(buffer, output_dir, file_index, config):
    source_text = "".join(buffer)

    # Save source text to file
    source_filename = os.path.join(output_dir, f"original_{file_index:03}.txt")
    with open(source_filename, "w", encoding="utf-8") as out_file:
        out_file.write(source_text)

    # Translate
    try:
        output_text = process(source_text, config['model_id'])
    except Exception:
        print(f'failed: {file_index}')
        return

    output_filename = os.path.join(output_dir, f"trans_{file_index:03}.txt")
    with open(output_filename, "w", encoding="utf-8") as out_file:
        out_file.write(output_text)

    print(file_index)


def split_and_process_file(config: dict):
    """Reads a file, accumulates lines until given number of words
    and sentence ending with period (.), processes them, and writes
    output files.
    """
    filepath = config['filepath']
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()

    output_dir = os.path.dirname(filepath) or "."
    buffer = []
    word_count = 0
    file_index = 1

    for line in lines:
        buffer.append(line)
        word_count += count_words(line)
        if (word_count >= WORDS_CUTOFF) and buffer[-1].strip()[-1] == '.':
            process_and_save(buffer, output_dir, file_index, config)

            # Reset buffer and counters
            buffer.clear()
            word_count = 0
            file_index += 1

    # Process remaining text if any
    if buffer:
        process_and_save(buffer, output_dir, file_index, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="Path to the input text file")
    parser.add_argument("--model")
    args = parser.parse_args()
    config = {
        'filepath': args.filepath,
        'model_id': args.model,
    }
    split_and_process_file(config)


if __name__ == "__main__":
    main()
