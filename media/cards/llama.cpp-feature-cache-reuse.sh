#!/bin/bash

cd "$(dirname "$0")"

source common.sh

echo -e "
 ${CC}   ${CT}ggml-org/llama.cpp                         feature  

                   ${CT}   Cache reuse   ${CC}

    ${CA}An advanced technique for reducing the
    prompt-processing time by \"shifting\" chunks
    of the previous context to new positions. ${CC}

    ${TC}                                                ${CC}
    ${TH} # prompt 0 (cached)                            ${CC}
    ${TC} ${TA}AAA${TC}B${TA}CCCC${TC}DDD${TA}EE${TC}F${TA}GG${TC}HHH${TA}III${TC}xxx                      ${CC}
    ${TC}                                                ${CC}
    ${TH} # prompt 1 (reuse from prompt 0)               ${CC}
    ${TC} ${TA}AAACCCCEEGGIII${TC}yyy                              ${CC}
    ${TC}                                                ${CC}

    ${CC}uses: ${CA}partial context updates ${CC}
    ${CC}req:  ${CA}RoPE encoding ${CC}

    ${CC}${CL}https://github.com/ggml-org/llama.cpp/pull/9866 ${CC}

    ${TC}                                                ${CC}
    ${TC} > llama-server ${TA}--cache-reuse 256 ${TC}[...]         ${CC}
    ${TC}                                                ${CC}
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
