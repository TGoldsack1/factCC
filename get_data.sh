#!/bin/bash
# Author: Tomas Goldsack


fileid="1kOG8kxqxuTSBCaEZg20CdSz37Y2CJy3A"
filename="st2_dev-data.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

