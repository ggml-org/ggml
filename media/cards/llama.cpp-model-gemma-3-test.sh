#!/bin/bash

cd "$(dirname "$0")"

source common.sh

echo -e "
 ${CC}   ${CT}llama.cpp                         model   

                ${CT}   Gemma 3   ${CC}

    ${CC}creator: ${CA}Google ${CC}
    ${CC}license: ${CA}Gemma Terms of Use ${CC}

    ${CC}sizes: ${CA}4B, 12B, 27B
    ${CC}capab: ${CA}Text, Tools, Vision

    ${CC}attn:   ${CA}SWA 1:4 ${CC}  |  head size: ${CA}256 ${CC}
    ${CC}vision: ${CA}SigLIP ${CC}   |  ${CC}extra:     ${CA}QAT ${CC}

    ${TC}                                      ${CC}
    ${TC} > llama-cli    ${TA}-hf ggml-org/gemma-3  ${CC}
    ${TC} > llama-server ${TA}-hf ggml-org/gemma-3  ${CC}
    ${TC}                                      ${CC}
" > card.txt

cat card.txt

echo -e "${C0}"

if [ $do_img -eq 1 ]; then
    cat card.txt | textimg --background 238,238,238,0 -F $p_fs -o output.png -f ../fonts/ProFontWinTweaked/ProFontWindows.ttf

    gg_add_frame output.png

    cp output.png $(basename "$0" .sh).png

    printf "Image saved as $(basename "$0" .sh).png\n"
fi

exit 0
