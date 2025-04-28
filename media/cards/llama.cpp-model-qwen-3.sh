#!/bin/bash

cd "$(dirname "$0")"

source common.sh

echo -e "
 ${CC}   ${CT}llama.cpp                                   model   

                      ${CT}   Qwen3   ${CC}

       ${CC}author:  ${CA}Qwen Team, Alibaba Group ${CC}
       ${CC}license: ${CA}Apache-2.0 ${CC}

       ${CC}capab: ${CA}Reasoning (On/Off), Instructions ${CC}
       ${CC}       ${CA}Translation, 100+ Languages ${CC}

       ${CC}sizes:
       ${CC} - dense: ${CA}0.6B, 1.7B, 4B, 8B, 14B, 32B
       ${CC} - MoE:   ${CA}30B-A3B, 235B-A22B

       ${CC}attn: ${CA}Grouped-Query  ${CC}|  head size: ${CA}128 ${CC}
       ${CC}exps: ${CA}128 (8 active) ${CC}|  ctx  size: ${CA}32k/128k ${CC}


    ${TC}                                                ${CC}
    ${TC} > llama-server ${TA}-hf ggml-org/Qwen3-30B-A3B-GGUF ${CC}
    ${TC}                                                ${CC}" > card.txt

cat card.txt

echo -e "${C0}"

if [ $do_img -eq 1 ]; then
    cat card.txt | textimg --background 238,238,238,0 -F $p_fs -o output.png -f ../fonts/ProFontWinTweaked/ProFontWindows.ttf

    gg_add_frame output.png

    magick output.png -resize 1200x $(basename "$0" .sh).png

    printf "Image saved as $(basename "$0" .sh).png\n"
fi

exit 0
