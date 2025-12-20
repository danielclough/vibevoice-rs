#!/usr/bin/env bash
# Run streaming model voices with their corresponding i18n text examples
#
# Usage:
#   ./run_i18n_voices.sh           # Run all voices
#   ./run_i18n_voices.sh all       # Run all voices (explicit)
#   ./run_i18n_voices.sh in-man    # Run Indonesian male voice
#   ./run_i18n_voices.sh en-woman  # Run all English female voices
#   ./run_i18n_voices.sh de        # Run all German voices
#   ./run_i18n_voices.sh Frank     # Run voice by name

set -e

OUTPUT_DIR="output/i18n"
VOICES_DIR="voices/streaming_model"
TEXT_DIR="text_examples/i18n"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build release binary first
echo "Building release binary..."
cargo build --release -p vibevoice-cli

BINARY="./target/release/vibevoice"

# Array of all voice/text pairs
declare -a ALL_VOICES=(
    # English
    "en-Frank_man"
    "en-Carter_man"
    "en-Mike_man"
    "en-Davis_man"
    "en-Grace_woman"
    "en-Emma_woman"
    # German
    "de-Spk0_man"
    "de-Spk1_woman"
    # Portuguese
    "pt-Spk0_woman"
    "pt-Spk1_man"
    # Japanese
    "jp-Spk0_man"
    "jp-Spk1_woman"
    # Italian
    "it-Spk0_woman"
    "it-Spk1_man"
    # Dutch
    "nl-Spk0_man"
    "nl-Spk1_woman"
    # Spanish
    "sp-Spk0_woman"
    "sp-Spk1_man"
    # Polish
    "pl-Spk0_man"
    "pl-Spk1_woman"
    # French
    "fr-Spk0_man"
    "fr-Spk1_woman"
    # Korean
    "kr-Spk0_woman"
    "kr-Spk1_man"
    # Indonesian
    "in-Samuel_man"
)

# Filter voices based on arguments
declare -a VOICES=()
if [[ $# -eq 0 ]] || [[ "$1" == "all" ]]; then
    # No args or "all": run all voices
    VOICES=("${ALL_VOICES[@]}")
else
    # Filter by pattern(s)
    # Supports: "in-man" (lang-gender), "en" (lang only), "woman" (gender only), "Frank" (name)
    for pattern in "$@"; do
        pattern_lower=$(echo "$pattern" | tr '[:upper:]' '[:lower:]')
        for voice in "${ALL_VOICES[@]}"; do
            voice_lower=$(echo "$voice" | tr '[:upper:]' '[:lower:]')
            match=false

            # Check if pattern has lang-gender format (e.g., "in-man", "en-woman")
            if [[ "$pattern_lower" == *-* ]]; then
                lang_pattern="${pattern_lower%%-*}"
                gender_pattern="${pattern_lower##*-}"
                # Match if voice starts with lang and ends with gender
                if [[ "$voice_lower" == "${lang_pattern}-"* && "$voice_lower" == *"_${gender_pattern}" ]]; then
                    match=true
                fi
            fi

            # Also check simple substring match
            if [[ "$voice_lower" == *"$pattern_lower"* ]]; then
                match=true
            fi

            if [[ "$match" == true ]]; then
                # Avoid duplicates
                if [[ ! " ${VOICES[*]} " =~ " ${voice} " ]]; then
                    VOICES+=("$voice")
                fi
            fi
        done
    done
fi

if [[ ${#VOICES[@]} -eq 0 ]]; then
    echo "No voices matched pattern(s): $*"
    echo ""
    echo "Available voices:"
    for v in "${ALL_VOICES[@]}"; do
        echo "  $v"
    done
    exit 1
fi

echo "========================================"
echo "Running ${#VOICES[@]} voice(s)"
echo "========================================"
echo ""

SUCCESS=0
FAILED=0

for voice in "${VOICES[@]}"; do
    VOICE_FILE="$VOICES_DIR/${voice}.safetensors"
    TEXT_FILE="$TEXT_DIR/${voice}.txt"
    OUTPUT_FILE="$OUTPUT_DIR/${voice}.wav"

    if [[ ! -f "$VOICE_FILE" ]]; then
        echo "SKIP: Voice file not found: $VOICE_FILE"
        ((++FAILED))
        continue
    fi

    if [[ ! -f "$TEXT_FILE" ]]; then
        echo "SKIP: Text file not found: $TEXT_FILE"
        ((++FAILED))
        continue
    fi

    echo "Processing: $voice"
    echo "  Voice: $VOICE_FILE"
    echo "  Text:  $TEXT_FILE"
    echo "  Output: $OUTPUT_FILE"

    # Read text from file and pass via --text (realtime model doesn't support --script)
    TEXT_CONTENT=$(cat "$TEXT_FILE")
    if $BINARY --model realtime \
        --voice "$VOICE_FILE" \
        --text "$TEXT_CONTENT" \
        --output "$OUTPUT_FILE"; then
        echo "  SUCCESS"
        ((++SUCCESS))
    else
        echo "  FAILED"
        ((++FAILED))
    fi
    echo ""
done

echo "========================================"
echo "Complete: $SUCCESS succeeded, $FAILED failed"
echo "Output files in: $OUTPUT_DIR"
echo "========================================"
