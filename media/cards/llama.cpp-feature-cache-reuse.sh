#!/bin/bash

cd "$(dirname "$0")"

#CT="\x1b[48;2;155;255;255m\x1b[38;2;029;161;242m"
CT="\x1b[49m\x1b[0m\x1b[38;2;000;000;000m"
CR="\x1b[49m\x1b[0m\x1b[38;2;205;008;008m"
CL="\x1b[49m\x1b[0m\x1b[38;2;035;008;165m"

CC="\x1b[49m\x1b[0m\x1b[38;2;080;080;080m"
TC="\x1b[49m\x1b[1m\x1b[48;2;000;000;000m\x1b[38;2;255;225;255m"
TA="\x1b[49m\x1b[1m\x1b[48;2;000;000;000m\x1b[38;2;110;225;110m"
TH="\x1b[49m\x1b[1m\x1b[48;2;000;000;000m\x1b[38;2;160;160;160m"
TO="\x1b[49m\x1b[1m\x1b[48;2;000;000;000m\x1b[38;2;238;138;070m"
TW="\x1b[49m\x1b[1m\x1b[48;2;000;000;000m\x1b[38;2;255;255;255m"
CG="\x1b[49m\x1b[0m\x1b[38;2;000;255;124m"
CY="\x1b[49m\x1b[0m\x1b[38;2;255;238;110m"
CO="\x1b[49m\x1b[0m\x1b[38;2;208;058;050m"
#CA="\x1b[49m\x1b[0m\x1b[38;2;035;008;165m"
CA="\x1b[49m\x1b[0m\x1b[38;2;000;000;000m"

#CT="\x1b[49m\x1b[0m\x1b[38;2;029;161;242m"
#CR="\x1b[49m\x1b[0m\x1b[38;2;255;048;048m"
#
#CC="\x1b[49m\x1b[0m\x1b[38;2;255;255;255m"
#CB="\x1b[49m\x1b[1m\x1b[38;2;255;255;255m"
#CG="\x1b[49m\x1b[0m\x1b[38;2;000;255;124m"
#CY="\x1b[49m\x1b[0m\x1b[38;2;255;238;110m"

p_s="2.0"

p_fs=$(bc <<< "20*${p_s}")
p_fs=${p_fs%.*}

p_sw=$(bc <<< "1.5*${p_s}")

# line height
p_lh=$(bc <<< "0.8*(1.2*${p_fs})")
p_lh=${p_lh%.*}

echo "line heigh = ${p_lh}"

# frame
p_fx0=$(bc <<< "0.50*${p_lh}")
p_fy0=$(bc <<< "0.90*${p_lh}")

p_fr=$(bc <<< "8.0*${p_s}")

echo -e "
 ${CC}   ${CT}ggml-org/llama.cpp                         feature  

                   ${CT}   Cache reuse   ${CC}

    ${CA}An advanced technique for reducing the
    prompt-processing time by \"shifting\" chunks
    of the previous context to new positions. ${CC}

    ${TC}                                                ${CC}
    ${TH} # prompt 0 (cached)                            ${CC}
    ${TW} ${TA}AAA${TW}B${TA}CCCC${TW}DDD${TA}EE${TW}F${TA}GG${TW}HHH${TA}III${TW}xxx                      ${CC}
    ${TC}                                                ${CC}
    ${TH} # prompt 1 (reuse from prompt 0)               ${CC}
    ${TW} ${TA}AAACCCCEEGGIII${TW}yyy                              ${CC}
    ${TC}                                                ${CC}

    ${CC}uses: ${CA}partial context updates ${CC}
    ${CC}req:  ${CA}RoPE encoding ${CC}

    ${CC}${CL}https://github.com/ggml-org/llama.cpp/pull/9866 ${CC}

    ${TC}                                                ${CC}
    ${TC} > llama-server ${TA}--cache-reuse 256 ${TC}[...]         ${CC}
    ${TC}                                                ${CC}
" | textimg --background 238,238,238,255 -F $p_fs -o output.png -f ../fonts/ProFontWinTweaked/ProFontWindows.ttf

x0=$(bc <<< "${p_fx0}")
y0=$(bc <<< "${p_fy0}")

x1="%[fx:w - $(bc <<< "${p_fx0} + 4*${p_s}")]"
y1="%[fx:h - $(bc <<< "${p_fy0}")]"


magick output.png \
    -format "
        roundrectangle ${x0},${y0} ${x1},${y1} ${p_fr},${p_fr};
        " \
    info: > frame.mvg

magick output.png -border 0 -alpha transparent \
    -background none -fill none -stroke black -strokewidth $p_sw \
    -draw "@frame.mvg"    frame.png

x0=$(bc <<< "${p_fx0}")
y0=$(bc <<< "${p_fy0}")

x1="%[fx:w - $(bc <<< "${p_fx0} + 3*${p_s}")]"
y1="%[fx:h - $(bc <<< "${p_fy0} - 1*${p_s}")]"

x2="%[fx:w - $(bc <<< "${p_fx0} + 2*${p_s}")]"
y2="%[fx:h - $(bc <<< "${p_fy0} - 2*${p_s}")]"

x3="%[fx:w - $(bc <<< "${p_fx0} + 1*${p_s}")]"
y3="%[fx:h - $(bc <<< "${p_fy0} - 3*${p_s}")]"

x4="%[fx:w - $(bc <<< "${p_fx0} + 0*${p_s}")]"
y4="%[fx:h - $(bc <<< "${p_fy0} - 4*${p_s}")]"

magick output.png \
    -format "
        roundrectangle ${x0},${y0} ${x1},${y1} ${p_fr},${p_fr};
        roundrectangle ${x0},${y0} ${x2},${y2} ${p_fr},${p_fr};
        roundrectangle ${x0},${y0} ${x3},${y3} ${p_fr},${p_fr};
        roundrectangle ${x0},${y0} ${x4},${y4} ${p_fr},${p_fr};
        " \
    info: > shadow.mvg

magick output.png -border 0 -alpha transparent \
    -background none -fill none -stroke gray -strokewidth $p_sw \
    -draw "@shadow.mvg"    shadow.png

x0=$(bc <<< "${p_fx0}")
y0=$(bc <<< "${p_fy0} + 1.50*${p_lh}")

x1="%[fx:w - $(bc <<< "${p_fx0} + 4*${p_s}")]"
y1=${y0}

magick output.png \
    -format "
        line           ${x0},${y0} ${x1},${y1};
        " \
    info: > title.mvg

magick output.png -border 0 -alpha transparent \
    -background none -fill none -stroke gray -strokewidth $p_sw \
    -draw "@title.mvg"    title.png

#magick output.png -border 0 -alpha transparent \
#    -background none -fill white -stroke none -strokewidth 0 \
#    -draw "@frame.mvg"    frame-mask.png

    #frame-mask.png -compose DstIn -composite \
magick output.png -alpha set -bordercolor none -border 0  \
    shadow.png -compose Over -composite \
    title.png  -compose Over -composite \
    frame.png  -compose Over -composite \
    output.png

x0=$(bc <<< "${p_fx0} + 0.10*${p_lh}")
y0=$(bc <<< "${p_fy0} + 0.05*${p_lh}")

lw=$(bc <<< "1.5*${p_lh}")
lh=$(bc <<< "1.5*${p_lh}")

composite -geometry ${lw}x${lh}+${x0}+${y0} ../logo/ggml-logo-transparent.png output.png output.png
