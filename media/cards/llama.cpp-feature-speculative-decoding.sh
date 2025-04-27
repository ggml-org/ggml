#!/bin/bash

cd "$(dirname "$0")"

source common.sh

echo -e "
 ${CC}   ${CT}ggml-org/llama.cpp                 feature  

                ${CT}   Speculative Decoding   ${CC}

    ${CC}perf: ${CA}improved text-generation speed ${CC}
    ${CC}uses: ${CA}code generation, structured output, summarization ${CC}

    ${CC}sizes: ${CA}4B, 12B, 27B
    ${CC}capab: ${CA}Text, Tools, Vision

    ${CC}attn:   ${CA}SWA 1:4 ${CC}  |  head size: ${CA}256 ${CC}
    ${CC}vision: ${CA}SigLIP ${CC}   |  ${CC}extra:     ${CA}QAT ${CC}


    ${TC}                                                ${CC}
    ${TC} > llama-server                                 ${CC}
    ${TC}     ${TC}-hf  ggml-org/Qwen2.5-Coder-14B-Instruct-Q8_0-GGUF   ${CC}
    ${TC}     ${TO}-hfd ggml-org/Qwen2.5-Coder-1.5B-Instruct-Q8_0-GGUF  ${CC}
    ${TC}                                                ${CC}
" > card.txt

cat card.txt

echo -e "${C0}"

if [ $do_img -eq 1 ]; then
    cat card.txt | textimg --background 238,238,238,255 -F $p_fs -o output.png -f ../fonts/ProFontWinTweaked/ProFontWindows.ttf

    gg_add_frame output.png

    cp output.png $(basename "$0" .sh).png

    printf "Image saved as $(basename "$0" .sh).png\n"
fi

exit 0
